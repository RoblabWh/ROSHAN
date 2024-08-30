import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Bernoulli
import torch

torch.autograd.set_detect_anomaly(True)

from utils import initialize_hidden_weights, initialize_output_weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Inputspace(nn.Module):

    def __init__(self, vision_range, time_steps):
        """
        A PyTorch Module that represents the input space of a neural network.
        """
        super(Inputspace, self).__init__()

        # Initialize Adaptive Pooling layers
        # self.terrain_adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fire_adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        dim_x = 100
        dim_y = 100
        d_in = time_steps
        #self.map_adaptive_pool = nn.AdaptiveAvgPool3d((dim_y, dim_x, 1))

        # layers_dict = [
        #     {'padding': (1, 1, 1), 'dilation': (1, 1, 1), 'kernel_size': (3, 3, 3), 'stride': (1, 1, 1)},
        #     {'padding': (1, 1, 1), 'dilation': (1, 1, 1), 'kernel_size': (3, 3, 3), 'stride': (1, 1, 1)},
        #     # {'padding': (0, 0, 0), 'dilation': (1, 1, 1), 'kernel_size': (3, 3, 3), 'stride': (2, 2, 2)},
        # ]
        # self.terrain_conv1 = nn.Conv3d(in_channels=17, out_channels=16, kernel_size=layers_dict[0]['kernel_size'],
        #                                stride=layers_dict[0]['stride'], padding=layers_dict[0]['padding'])
        # self.terrain_bn1 = nn.BatchNorm3d(16)
        # self.terrain_conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=layers_dict[1]['kernel_size'],
        #                                stride=layers_dict[1]['stride'], padding=layers_dict[1]['padding'])
        # self.terrain_bn2 = nn.BatchNorm3d(32)
        # # self.terrain_conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=layers_dict[2]['kernel_size'],
        # #                                stride=layers_dict[2]['stride'])
        # # self.terrain_bn3 = nn.BatchNorm3d(32)
        # in_f = self.get_in_features_3d(h_in=vision_range, w_in=vision_range, d_in=d_in, layers_dict=layers_dict)

        layers_dict = [
            {'padding': (1, 1), 'dilation': (1, 1), 'kernel_size': (3, 3), 'stride': (1, 1)},
            {'padding': (1, 1), 'dilation': (1, 1), 'kernel_size': (3, 3), 'stride': (1, 1)},
            # {'padding': (0, 0, 0), 'dilation': (1, 1, 1), 'kernel_size': (3, 3, 3), 'stride': (2, 2, 2)},
        ]
        self.terrain_conv1 = nn.Conv2d(in_channels=d_in, out_channels=4, kernel_size=layers_dict[0]['kernel_size'],
                                       stride=layers_dict[0]['stride'], padding=layers_dict[0]['padding'])
        self.terrain_bn1 = nn.BatchNorm2d(4)
        self.terrain_conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=layers_dict[1]['kernel_size'],
                                       stride=layers_dict[1]['stride'], padding=layers_dict[1]['padding'])
        self.terrain_bn2 = nn.BatchNorm2d(8)
        # self.terrain_conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=layers_dict[2]['kernel_size'],
        #                                stride=layers_dict[2]['stride'])
        # self.terrain_bn3 = nn.BatchNorm3d(32)
        in_f = self.get_in_features_2d(h_in=vision_range, w_in=vision_range, layers_dict=layers_dict)

        # layers_dict = [
        #     {'padding': (1, 1), 'dilation': (1, 1), 'kernel_size': (3, 3), 'stride': (1, 1)},
        #     {'padding': (1, 1), 'dilation': (1, 1), 'kernel_size': (3, 3), 'stride': (1, 1)},
        #     {'padding': (0, 0), 'dilation': (1, 1), 'kernel_size': (3, 3), 'stride': (2, 2)},
        # ]
        #
        # self.terrain_conv1 = nn.Conv2d(in_channels=17, out_channels=8, kernel_size=layers_dict[0]['kernel_size'],
        #                                stride=layers_dict[0]['stride'], padding=layers_dict[0]['padding'])
        # self.terrain_bn1 = nn.BatchNorm2d(8)
        # self.terrain_conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=layers_dict[1]['kernel_size'],
        #                                stride=layers_dict[1]['stride'], padding=layers_dict[1]['padding'])
        # self.terrain_bn2 = nn.BatchNorm2d(16)
        # self.terrain_conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=layers_dict[2]['kernel_size'],
        #                                stride=layers_dict[2]['stride'])
        # self.terrain_bn3 = nn.BatchNorm2d(32)
        #
        # in_f = self.get_in_features_2d(h_in=vision_range, w_in=vision_range, layers_dict=layers_dict)

        features_terrain = in_f * 8 # 8 is the number of output channels

        # layers_dict = [
        #     {'padding': (1, 1, 1), 'dilation': (1, 1, 1), 'kernel_size': (3, 3, 3), 'stride': (1, 1, 1)},
        #     {'padding': (1, 1, 1), 'dilation': (1, 1, 1), 'kernel_size': (3, 3, 3), 'stride': (1, 1, 1)},
        #     {'padding': (0, 0, 0), 'dilation': (1, 1, 1), 'kernel_size': (3, 3, 3), 'stride': (2, 2, 2)},
        # ]
        # self.fire_conv1 = nn.Conv3d(in_channels=1, out_channels=8, padding=layers_dict[0]['padding'],
        #                             kernel_size=layers_dict[0]['kernel_size'], stride=layers_dict[0]['stride'])
        # self.fire_bn1 = nn.BatchNorm3d(8)
        # self.fire_conv2 = nn.Conv3d(in_channels=8, out_channels=16, padding=layers_dict[1]['padding'],
        #                             kernel_size=layers_dict[1]['kernel_size'], stride=layers_dict[1]['stride'])
        # self.fire_bn2 = nn.BatchNorm3d(16)
        # self.fire_conv3 = nn.Conv3d(in_channels=16, out_channels=32, padding=layers_dict[2]['padding'],
        #                             kernel_size=layers_dict[2]['kernel_size'], stride=layers_dict[2]['stride'])
        # self.fire_bn3 = nn.BatchNorm3d(32)
        # # initialize_hidden_weights(self.fire_conv1)
        # in_f = self.get_in_features_3d(h_in=vision_range, w_in=vision_range, d_in=d_in, layers_dict=layers_dict)

        layers_dict = [
            {'padding': (1, 1), 'dilation': (1, 1), 'kernel_size': (3, 3), 'stride': (1, 1)},
            {'padding': (1, 1), 'dilation': (1, 1), 'kernel_size': (3, 3), 'stride': (1, 1)},
            #{'padding': (0, 0), 'dilation': (1, 1), 'kernel_size': (3, 3), 'stride': (2, 2)},
        ]
        self.fire_conv1 = nn.Conv2d(in_channels=d_in, out_channels=4, padding=layers_dict[0]['padding'],
                                    kernel_size=layers_dict[0]['kernel_size'], stride=layers_dict[0]['stride'])
        self.fire_bn1 = nn.BatchNorm2d(4)
        self.fire_conv2 = nn.Conv2d(in_channels=4, out_channels=8, padding=layers_dict[1]['padding'],
                                    kernel_size=layers_dict[1]['kernel_size'], stride=layers_dict[1]['stride'])
        self.fire_bn2 = nn.BatchNorm2d(8)
        # self.fire_conv3 = nn.Conv2d(in_channels=16, out_channels=32, padding=layers_dict[2]['padding'],
        #                             kernel_size=layers_dict[2]['kernel_size'], stride=layers_dict[2]['stride'])
        # self.fire_bn3 = nn.BatchNorm2d(32)
        # initialize_hidden_weights(self.fire_conv1)
        in_f = self.get_in_features_2d(h_in=vision_range, w_in=vision_range, layers_dict=layers_dict)
        features_fire = in_f * 8 # 32 is the number of output channels

        # layers_dict = [
        #     {'padding': (0, 0, 0), 'dilation': (1, 1, 1), 'kernel_size': (3, 3, 3), 'stride': (1, 1, 1)}
        # ]
        # self.map_conv1 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=layers_dict[0]['kernel_size'], stride=layers_dict[0]['stride'])

        # layers_dict = [
        #     {'padding': (0, 0), 'dilation': (1, 1), 'kernel_size': (3, 3), 'stride': (1, 1)}
        # ]
        # self.map_conv1 = nn.Conv2d(in_channels=d_in, out_channels=1, kernel_size=layers_dict[0]['kernel_size'], stride=layers_dict[0]['stride'])

        # initialize_hidden_weights(self.map_conv1)
        # features_map = (int(dim_x * dim_y)) * 1 # 2 is the number of output channels

        self.flatten = nn.Flatten()

        vel_out_features = 16
        self.vel_dense1 = nn.Linear(in_features=2, out_features=8)
        initialize_hidden_weights(self.vel_dense1)
        self.vel_dense2 = nn.Linear(in_features=8, out_features=vel_out_features)
        initialize_hidden_weights(self.vel_dense2)
        # self.vel_dense3 = nn.Linear(in_features=32, out_features=vel_out_features)
        # initialize_hidden_weights(self.vel_dense3)

        position_out_features = 8
        self.position_dense1 = nn.Linear(in_features=2, out_features=16)
        initialize_hidden_weights(self.position_dense1)
        self.position_dense2 = nn.Linear(in_features=16, out_features=position_out_features)
        initialize_hidden_weights(self.position_dense2)
        # self.position_dense3 = nn.Linear(in_features=32, out_features=position_out_features)
        # initialize_hidden_weights(self.position_dense3)

        # four is the number of timeframes TODO make this dynamic
        terrain_out_features = 32
        fire_out_features = 32
        # map_out_features = 32

        self.terrain_flat1 = nn.Linear(in_features=features_terrain, out_features=64)
        initialize_hidden_weights(self.terrain_flat1)
        self.terrain_flat2 = nn.Linear(in_features=64, out_features=terrain_out_features)
        initialize_hidden_weights(self.terrain_flat2)
        # self.terrain_flat3 = nn.Linear(in_features=128, out_features=terrain_out_features)
        # initialize_hidden_weights(self.terrain_flat3)

        self.fire_flat1 = nn.Linear(in_features=features_fire, out_features=64)
        initialize_hidden_weights(self.fire_flat1)
        self.fire_flat2 = nn.Linear(in_features=64, out_features=fire_out_features)
        initialize_hidden_weights(self.fire_flat2)
        # self.fire_flat3 = nn.Linear(in_features=128, out_features=fire_out_features)
        # initialize_hidden_weights(self.fire_flat3)

        # self.map_flat1 = nn.Linear(in_features=features_map, out_features=64)
        # initialize_hidden_weights(self.map_flat1)
        # self.map_flat2 = nn.Linear(in_features=64, out_features=32)
        # initialize_hidden_weights(self.map_flat2)
        # self.map_flat3 = nn.Linear(in_features=128, out_features=map_out_features)
        # initialize_hidden_weights(self.map_flat3)

        # self.input_dense1 = nn.Linear(in_features=input_features, out_features=64)
        # initialize_hidden_weights(self.input_dense1)
        input_features = terrain_out_features + fire_out_features + (position_out_features + vel_out_features) * time_steps
        self.input_dense1 = nn.Linear(in_features=input_features, out_features=128)
        initialize_hidden_weights(self.input_dense1)
        self.input_dense2 = nn.Linear(in_features=128, out_features=256)
        initialize_hidden_weights(self.input_dense2)
        # self.input_dense3 = nn.Linear(in_features=128, out_features=256)
        # initialize_hidden_weights(self.input_dense3)

    def get_in_features_2d(self, h_in, w_in, layers_dict):
        for layer in layers_dict:
            padding = layer['padding']
            dilation = layer['dilation']
            kernel_size = layer['kernel_size']
            stride = layer['stride']

            h_in = ((h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0]) + 1
            w_in = ((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1]) + 1

        return h_in * w_in

    def get_in_features_3d(self, h_in, w_in, d_in, layers_dict):
        for layer in layers_dict:
            padding = layer['padding']
            dilation = layer['dilation']
            kernel_size = layer['kernel_size']
            stride = layer['stride']

            d_in = ((d_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0]) + 1
            h_in = ((h_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1]) + 1
            w_in = ((w_in + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) // stride[2]) + 1

        return d_in * h_in * w_in

    def prepare_tensor(self, states):
        terrain, fire_status, velocity, position = states
        #terrain, fire_status, velocity, maps, position = states

        # if isinstance(terrain, np.ndarray):
        #     terrain = torch.tensor(terrain, dtype=torch.int64).to(device)
        # else:
        #     terrain = terrain.to(dtype=torch.int64)
        # terrain = F.one_hot(terrain, num_classes=17)
        # terrain = terrain.permute(0, 4, 1, 2, 3)
        # terrain = terrain.to(dtype=torch.float32)

        if isinstance(terrain, np.ndarray):
            terrain = torch.tensor(terrain, dtype=torch.float32).to(device)

        if isinstance(fire_status, np.ndarray):
            fire_status = torch.tensor(fire_status, dtype=torch.float32).to(device)
        #fire_status = fire_status.unsqueeze(1)

        if isinstance(velocity, np.ndarray):
            velocity = torch.tensor(velocity, dtype=torch.float32).to(device)

        # if isinstance(maps, np.ndarray):
        #     maps = torch.tensor(maps, dtype=torch.float32).to(device)
        #maps = maps.unsqueeze(1)

        if isinstance(position, np.ndarray):
            position = torch.tensor(position, dtype=torch.float32).to(device)

        return terrain, fire_status, velocity, position
        #return terrain, fire_status, velocity, maps, position

    def forward(self, states):
        #terrain, fire_status, velocity, maps, position = self.prepare_tensor(states)
        terrain, fire_status, velocity, position = self.prepare_tensor(states)
        terrain = F.relu(self.terrain_bn1(self.terrain_conv1(terrain)))
        terrain = F.relu(self.terrain_bn2(self.terrain_conv2(terrain)))
        # terrain = F.relu(self.terrain_bn3(self.terrain_conv3(terrain)))
        terrain = self.flatten(terrain)
        terrain = F.relu(self.terrain_flat1(terrain))
        terrain = F.relu(self.terrain_flat2(terrain))
        # terrain = F.relu(self.terrain_flat3(terrain))

        fire_status = F.relu(self.fire_bn1(self.fire_conv1(fire_status)))
        fire_status = F.relu(self.fire_bn2(self.fire_conv2(fire_status)))
        # fire_status = F.relu(self.fire_bn3(self.fire_conv3(fire_status)))
        fire_status = self.flatten(fire_status)
        fire_status = F.relu(self.fire_flat1(fire_status))
        fire_status = F.relu(self.fire_flat2(fire_status))
        # fire_status = F.relu(self.fire_flat3(fire_status))
        #
        # maps = F.relu(self.map_conv1(maps))
        # maps = self.map_adaptive_pool(maps)
        # maps = self.flatten(maps)
        # maps = F.relu(self.map_flat1(maps))
        # maps = F.relu(self.map_flat2(maps))
        # maps = F.relu(self.map_flat3(maps))

        position = F.relu(self.position_dense1(position))
        position = F.relu(self.position_dense2(position))
        # position = F.relu(self.position_dense3(position))
        position = self.flatten(position)

        velocity = F.relu(self.vel_dense1(velocity))
        velocity = F.relu(self.vel_dense2(velocity))
        # velocity = F.relu(self.vel_dense3(velocity))
        velocity = self.flatten(velocity)

        concated_input = torch.cat((terrain, fire_status, velocity, position), dim=1)
        # concated_input = torch.cat((terrain, fire_status, velocity, maps, position), dim=1)
        input_dense = F.relu(self.input_dense1(concated_input))
        input_dense = F.relu(self.input_dense2(input_dense))
        # input_dense = F.relu(self.input_dense3(input_dense))

        return input_dense


class Actor(nn.Module):
    """
    A PyTorch Module that represents the actor network of a PPO agent.
    """
    def __init__(self, vision_range, time_steps):
        super(Actor, self).__init__()
        self.Inputspace = Inputspace(vision_range, time_steps=time_steps)

        # Mu
        # self.pre_mu = nn.Linear(in_features=256, out_features=128)
        self.mu = nn.Linear(in_features=256, out_features=2)
        # self.pre_water = nn.Linear(in_features=256, out_features=128)
        self.water_emit = nn.Linear(in_features=256, out_features=1)
        # initialize_output_weights(self.pre_mu, 'actor')
        # initialize_output_weights(self.pre_water, 'actor')
        initialize_output_weights(self.mu, 'actor')
        initialize_output_weights(self.water_emit, 'actor')

        # Logstd
        self.log_std = nn.Parameter(torch.zeros(3, ))

    def forward(self, states):
        x = self.Inputspace(states)
        # mu = F.relu(self.pre_mu(x))
        mu = torch.tanh(self.mu(x))
        # water_emit_value = F.relu(self.pre_water(x))
        water_emit_value = torch.sigmoid(self.water_emit(x))
        actions = torch.cat((mu, water_emit_value), dim=1)
        std = torch.exp(self.log_std)
        var = torch.pow(std, 2)

        return actions, var


class Critic(nn.Module):
    """
    A PyTorch Module that represents the critic network of a PPO agent.
    """
    def __init__(self, vision_range, time_steps):
        super(Critic, self).__init__()
        self.Inputspace = self.Inputspace = Inputspace(vision_range, time_steps=time_steps)
        # Value
        # self.pre_value = nn.Linear(in_features=256, out_features=128)
        self.value = nn.Linear(in_features=256, out_features=1)
        initialize_output_weights(self.value, 'critic')
        # initialize_output_weights(self.pre_value, 'critic')

    def forward(self, states):
        x = self.Inputspace(states)
        # value = F.tanh(self.pre_value(x))
        value = self.value(x)
        return value


class ActorCritic(nn.Module):
    """
    A PyTorch Module that represents the actor-critic network of a PPO agent.
    """
    def __init__(self, vision_range, time_steps):
        super(ActorCritic, self).__init__()
        self.actor_cnt = 0
        self.critic_cnt = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(vision_range, time_steps).to(device)
        self.critic = Critic(vision_range,time_steps).to(device)

        # TODO statische var testen
        #self.logstds_param = nn.Parameter(torch.full((n_actions,), 0.1))
        #self.action_var = torch.full((action_dim, ), action_std * action_std).to(device)

    def act(self, state):
        """
        Returns an action sampled from the actor's distribution and the log probability of that action.

        :param states: A tuple of the current lidar scan, orientation to goal, distance to goal, and velocity.
        :return: A tuple of the sampled action and the log probability of that action.
        """

        # TODO: check if normalization of states is necessary
        # was suggested in: Implementation_Matters in Deep RL: A Case Study on PPO and TRPO
        with torch.no_grad():
            action_mean, action_var = self.actor(state)

            action_mean_velocity = action_mean[:, :2].to('cpu')
            action_var_velocity = action_var[:2].to('cpu')
            action_mean_water_emit = action_mean[:, 2].to('cpu')

            cov_mat = torch.diag(action_var_velocity)
            dist_velocity = MultivariateNormal(action_mean_velocity, cov_mat)
            dist_water = Bernoulli(action_mean_water_emit)
            ## logging of actions
            #self.logger.add_actor_output(action_mean.mean(0)[0].item(), action_mean.mean(0)[1].item(), action_var[0].item(), action_var[1].item())

            action_velocity = dist_velocity.sample()
            action_velocity = torch.clip(action_velocity, -1, 1)
            action_water = dist_water.sample()

            action_logprob_velocity = dist_velocity.log_prob(action_velocity)
            action_logprob_water = dist_water.log_prob(action_water)
            action = torch.cat([action_velocity, action_water.unsqueeze(dim=1)], dim=1)
            combined_logprob = action_logprob_velocity + action_logprob_water

            return action.detach().numpy(), combined_logprob.detach().numpy()

    def act_certain(self, state):
        """
        Returns an action from the actor's distribution without sampling.

        :param states: A tuple of the current lidar scan, orientation to goal, distance to goal, and velocity.
        :return: The action from the actor's distribution.
        """
        with torch.no_grad():
            action_mean, _ = self.actor(state)

        return action_mean.detach().cpu().numpy()

    def evaluate(self, state, action):
        """
        Returns the log probability of the given action, the value of the given state, and the entropy of the actor's
        distribution.

        :param state: A tuple of the current lidar scan, orientation to goal, distance to goal, and velocity.
        :param action: The action to evaluate.
        :return: A tuple of the log probability of the given action, the value of the given state, and the entropy of the
        actor's distribution.
        """

        state_value = self.critic(state)

        action_mean, action_var = self.actor(state)

        action_mean_velocity = action_mean[:, :2]
        action_var_velocity = action_var[:2]
        action_mean_water_emit = action_mean[:, 2]

        cov_mat = torch.diag(action_var_velocity)
        dist_velocity = MultivariateNormal(action_mean_velocity, cov_mat)
        dist_water = Bernoulli(action_mean_water_emit)

        action_logprob_velocity = dist_velocity.log_prob(action[:, :2])
        action_logprob_water = dist_water.log_prob(action[:, 2])
        # action_logprob_water = dist_water.log_prob(action[:, 2].view(1, -1))
        combined_logprob = action_logprob_velocity + action_logprob_water

        dist_entropy_velocity = dist_velocity.entropy()
        dist_entropy_water = dist_water.entropy()
        combined_entropy = dist_entropy_velocity + dist_entropy_water

        return combined_logprob.to(device), torch.squeeze(state_value), combined_entropy.to(device)
