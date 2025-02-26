import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import initialize_output_weights, get_in_features_2d, get_in_features_3d

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Inputspace(nn.Module):

    def __init__(self, vision_range, time_steps):
        """
        A PyTorch Module that represents the input space of a neural network.
        """
        super(Inputspace, self).__init__()

        self.vision_range = vision_range
        self.time_steps = time_steps

        pos_out_features = 16
        self.pos_dense1 = nn.Linear(in_features=2, out_features=pos_out_features)
        initialize_output_weights(self.pos_dense1, 'hidden')

        # EXPLORE VIEW CONVOLUTION LAYERS
        layers_dict = [
            {'padding': (0, 0), 'dilation': (1, 1), 'kernel_size': (5, 5), 'stride': (2, 2),
             'in_channels': self.time_steps, 'out_channels': 4},
            {'padding': (0, 0), 'dilation': (1, 1), 'kernel_size': (3, 3), 'stride': (1, 1), 'in_channels': 4,
             'out_channels': 8},
        ]
        self.explore_conv1 = nn.Conv2d(in_channels=layers_dict[0]['in_channels'],
                                       out_channels=layers_dict[0]['out_channels'],
                                       kernel_size=layers_dict[0]['kernel_size'], stride=layers_dict[0]['stride'],
                                       padding=layers_dict[0]['padding'])
        self.explore_conv2 = nn.Conv2d(in_channels=layers_dict[1]['in_channels'],
                                       out_channels=layers_dict[1]['out_channels'],
                                       kernel_size=layers_dict[1]['kernel_size'], stride=layers_dict[1]['stride'],
                                       padding=layers_dict[1]['padding'])

        in_f = get_in_features_2d(h_in=30, w_in=30, layers_dict=layers_dict)

        features_explore = in_f * layers_dict[1]['out_channels']
        explore_out_features = 64

        self.explore_flat1 = nn.Linear(in_features=features_explore, out_features=explore_out_features)

        # FIRE VIEW CONVOLUTION LAYERS
        self.fire_conv1 = nn.Conv2d(in_channels=layers_dict[0]['in_channels'],
                                       out_channels=layers_dict[0]['out_channels'],
                                       kernel_size=layers_dict[0]['kernel_size'], stride=layers_dict[0]['stride'],
                                       padding=layers_dict[0]['padding'])
        self.fire_conv2 = nn.Conv2d(in_channels=layers_dict[1]['in_channels'],
                                       out_channels=layers_dict[1]['out_channels'],
                                       kernel_size=layers_dict[1]['kernel_size'], stride=layers_dict[1]['stride'],
                                       padding=layers_dict[1]['padding'])

        in_f = get_in_features_2d(h_in=30, w_in=30, layers_dict=layers_dict)

        features_fire = in_f * layers_dict[1]['out_channels']
        fire_out_features = 64

        self.fire_flat1 = nn.Linear(in_features=features_fire, out_features=fire_out_features)

        self.flatten = nn.Flatten()

        input_features = pos_out_features * self.time_steps + explore_out_features + fire_out_features
        self.out_features = 32
        mid_features = 64

        self.input_dense1 = nn.Linear(in_features=input_features, out_features=mid_features)
        initialize_output_weights(self.input_dense1, 'hidden')
        self.input_dense2 = nn.Linear(in_features=mid_features, out_features=self.out_features)
        initialize_output_weights(self.input_dense2, 'hidden')

    @staticmethod
    def prepare_tensor(states):
        exploration_map, fire_map, position = states

        if isinstance(exploration_map, np.ndarray):
            exploration_map = torch.tensor(exploration_map, dtype=torch.float32).to(device)

        if isinstance(fire_map, np.ndarray):
            fire_map = torch.tensor(fire_map, dtype=torch.float32).to(device)

        if isinstance(position, np.ndarray):
            position = torch.tensor(position, dtype=torch.float32).to(device)

        return exploration_map, fire_map, position

    def forward(self, states):
        exploration_map, fire_map, position = self.prepare_tensor(states)

        pos = F.relu(self.pos_dense1(position))
        pos = self.flatten(pos)

        explore = F.relu(self.explore_conv1(exploration_map))
        explore = F.relu(self.explore_conv2(explore))
        explore = self.flatten(explore)
        explore = F.relu(self.explore_flat1(explore))

        fire = F.relu(self.fire_conv1(fire_map))
        fire = F.relu(self.fire_conv2(fire))
        fire = self.flatten(fire)
        fire = F.relu(self.fire_flat1(fire))

        concat_tensor = torch.cat((pos, fire, explore), dim=1)

        concat_tensor = F.relu(self.input_dense1(concat_tensor))
        concat_tensor = F.relu(self.input_dense2(concat_tensor))
        output_vision = torch.flatten(concat_tensor, start_dim=1)

        return output_vision


class Actor(nn.Module):
    """
    A PyTorch Module that represents the actor network of a PPO agent.
    """
    def __init__(self, vision_range, time_steps):
        super(Actor, self).__init__()
        self.Inputspace = Inputspace(vision_range, time_steps=time_steps)
        self.in_features = self.Inputspace.out_features
        # Mu
        self.mu_goal = nn.Linear(in_features=self.in_features, out_features=2)
        initialize_output_weights(self.mu_goal, 'actor')

        # Logstd
        self.log_std = nn.Parameter(torch.zeros(2, ))

    def forward(self, states):
        x = self.Inputspace(states)
        mu_goal = torch.tanh(self.mu_goal(x))
        std = torch.exp(self.log_std)
        var = torch.pow(std, 2)

        return mu_goal, var


class Critic(nn.Module):
    """
    A PyTorch Module that represents the critic network of a PPO agent.
    """
    def __init__(self, vision_range, time_steps):
        super(Critic, self).__init__()
        self.Inputspace = Inputspace(vision_range, time_steps=time_steps)
        self.in_features = self.Inputspace.out_features

        # Value
        self.value = nn.Linear(in_features=self.in_features, out_features=1)
        initialize_output_weights(self.value, 'critic')

    def forward(self, states):
        x = self.Inputspace(states)
        value = self.value(x)
        return value

class RNDModel(nn.Module):
    def __init__(self, vision_range, time_steps):
        super(RNDModel, self).__init__()

        self.predictor = Inputspace(vision_range, time_steps)
        self.target = Inputspace(vision_range, time_steps)

        # Set parameters in target network to be non-trainable
        for param in self.target.parameters():
            param.requires_grad = False

    def get_intrinsic_reward(self, obs):
        with torch.no_grad():
            tgt_features = self.target(obs)
        pred_features = self.predictor(obs)
        intrinsic_rewards = (tgt_features - pred_features).pow(2).mean(dim=1)
        return intrinsic_rewards

    def forward(self, states):
        target_features = self.target(states)
        predictor_features = self.predictor(states)

        return target_features, predictor_features