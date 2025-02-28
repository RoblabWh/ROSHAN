import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import initialize_output_weights, get_in_features_2d, get_in_features_3d

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Inputspace(nn.Module):

    def __init__(self, vision_range, map_size, time_steps):
        """
        A PyTorch Module that represents the input space of a neural network.
        """
        super(Inputspace, self).__init__()

        self.vision_range = vision_range
        self.map_size = map_size
        self.time_steps = time_steps

        # EXPLORE VIEW CONVOLUTION LAYERS
        # d_in, h_in, w_in
        layers_dict = [
            {'padding': (0, 0, 0), 'dilation': (1, 1, 1), 'kernel_size': (1, 3, 3), 'stride': (2, 2, 2),
             'in_channels': self.time_steps, 'out_channels': 4},
            {'padding': (0, 0, 0), 'dilation': (1, 1, 1), 'kernel_size': (1, 2, 2), 'stride': (1, 1, 1), 'in_channels': 4,
             'out_channels': 8},
        ]
        self.explore_conv1 = nn.Conv3d(in_channels=layers_dict[0]['in_channels'],
                                       out_channels=layers_dict[0]['out_channels'],
                                       kernel_size=layers_dict[0]['kernel_size'], stride=layers_dict[0]['stride'],
                                       padding=layers_dict[0]['padding'])
        self.explore_conv2 = nn.Conv3d(in_channels=layers_dict[1]['in_channels'],
                                       out_channels=layers_dict[1]['out_channels'],
                                       kernel_size=layers_dict[1]['kernel_size'], stride=layers_dict[1]['stride'],
                                       padding=layers_dict[1]['padding'])

        in_f = get_in_features_3d(h_in=self.map_size, w_in=self.map_size, d_in=self.time_steps, layers_dict=layers_dict)

        features_explore = in_f * layers_dict[1]['out_channels']

        self.flatten = nn.Flatten()

        input_features = features_explore
        self.out_features = 32
        mid_features = 64

        self.input_dense1 = nn.Linear(in_features=input_features, out_features=mid_features)
        initialize_output_weights(self.input_dense1, 'hidden')
        self.input_dense2 = nn.Linear(in_features=mid_features, out_features=self.out_features)
        initialize_output_weights(self.input_dense2, 'hidden')

    @staticmethod
    def prepare_tensor(states):
        all_maps = states

        if isinstance(all_maps, tuple):
            all_maps = all_maps[0]

        if isinstance(all_maps, np.ndarray):
            all_maps = torch.tensor(all_maps, dtype=torch.float32).to(device)

        return all_maps

    def forward(self, states):
        all_maps = self.prepare_tensor(states)

        explore = F.relu(self.explore_conv1(all_maps))
        explore = F.relu(self.explore_conv2(explore))
        explore = self.flatten(explore)

        explore = F.relu(self.input_dense1(explore))
        explore = F.relu(self.input_dense2(explore))
        output_vision = torch.flatten(explore, start_dim=1)

        return output_vision


class Actor(nn.Module):
    """
    A PyTorch Module that represents the actor network of a PPO agent.
    """
    def __init__(self, vision_range, map_size, time_steps):
        super(Actor, self).__init__()
        self.Inputspace = Inputspace(vision_range, map_size, time_steps=time_steps)
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
    def __init__(self, vision_range, map_size, time_steps):
        super(Critic, self).__init__()
        self.Inputspace = Inputspace(vision_range, map_size, time_steps=time_steps)
        self.in_features = self.Inputspace.out_features

        # Value
        self.value = nn.Linear(in_features=self.in_features, out_features=1)
        initialize_output_weights(self.value, 'critic')

    def forward(self, states):
        x = self.Inputspace(states)
        value = self.value(x)
        return value

class RNDModel(nn.Module):
    def __init__(self, vision_range, map_size, time_steps):
        super(RNDModel, self).__init__()

        self.predictor = Inputspace(vision_range, map_size, time_steps)
        self.target = Inputspace(vision_range, map_size, time_steps)

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