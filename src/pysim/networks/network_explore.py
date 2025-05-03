import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from networks.network_fly import DeterministicActor
from utils import initialize_output_weights, get_in_features_2d, get_in_features_3d

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNNSpatialEncoder(nn.Module):
    def __init__(self, in_channels, out_features):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, out_features),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        return self.conv_net(x)

class TemporalEncoder(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size=in_features, hidden_size=hidden_size, batch_first=True)

    def forward(self, x):
        # x: (batch_size, time_steps, features)
        _, h_n = self.gru(x)
        return h_n.squeeze(0)  # final hidden state

class Inputspace(nn.Module):

    def __init__(self, drone_count, map_size, time_steps):
        """
        A PyTorch Module that represents the input space of a neural network.
        """
        super(Inputspace, self).__init__()

        self.drone_count = drone_count
        self.time_steps = time_steps
        spatial_feature_size = 128
        temporal_feature_size = 128

        # CNN Encoders
        self.agent_position_encoder = CNNSpatialEncoder(drone_count, spatial_feature_size)
        self.exploration_map_encoder = CNNSpatialEncoder(1, spatial_feature_size)

        # Temporal Encoders
        self.agent_temporal_encoder = TemporalEncoder(spatial_feature_size, temporal_feature_size)
        self.map_temporal_encoder = TemporalEncoder(spatial_feature_size, temporal_feature_size)

        # Final linear layer to merge features
        self.final_feature_size = temporal_feature_size * 2
        self.merge_layer = nn.Sequential(
            nn.Linear(self.final_feature_size, 256),
            nn.ReLU(),
        )

        self.out_features = 256

    @staticmethod
    def prepare_tensor(states):
        agent_positions, exploration_maps = states

        if isinstance(agent_positions, np.ndarray):
            agent_positions = torch.tensor(agent_positions, dtype=torch.float32).to(device)

        if isinstance(exploration_maps, np.ndarray):
            exploration_maps = torch.tensor(exploration_maps, dtype=torch.float32).to(device)

        return agent_positions, exploration_maps

    def forward(self, states):
        agent_positions, exploration_maps = self.prepare_tensor(states)

        # Encode spatial per timestep
        agent_feats = []
        map_feats = []
        for t in range(self.time_steps):
            # Agent positions at t: (batch, drone_count, H, W)
            agent_t = agent_positions[:, :, t, :, :].permute(1,0,2,3)
            agent_feat_t = self.agent_position_encoder(agent_t)
            agent_feats.append(agent_feat_t.unsqueeze(1))  # (batch, 1, feat)

            # Exploration map at t: (batch, 1, H, W)
            map_t = exploration_maps[:, :, t, :][0:1]
            map_feat_t = self.exploration_map_encoder(map_t)
            map_feats.append(map_feat_t.unsqueeze(1))

        # Stack over time
        agent_feats = torch.cat(agent_feats, dim=1)  # (batch, time_steps, feat)
        map_feats = torch.cat(map_feats, dim=1)  # (batch, time_steps, feat)

        # Temporal encoding
        agent_temporal_feat = self.agent_temporal_encoder(agent_feats)  # (batch, temporal_feat)
        map_temporal_feat = self.map_temporal_encoder(map_feats)  # (batch, temporal_feat)

        # Concatenate and merge
        combined = torch.cat([agent_temporal_feat, map_temporal_feat], dim=-1)
        out = self.merge_layer(combined)

        return out  # (batch, out_features)
        # self.drone_count = drone_count
        # self.map_size = map_size
        # self.time_steps = time_steps
        #
        # # EXPLORE VIEW CONVOLUTION LAYERS
        # # d_in, h_in, w_in
        # layers_dict = [
        #     {'padding': (0, 0, 0), 'dilation': (1, 1, 1), 'kernel_size': (1, 3, 3), 'stride': (1, 2, 2),
        #      'in_channels': self.time_steps, 'out_channels': 4},
        #     {'padding': (0, 0, 0), 'dilation': (1, 1, 1), 'kernel_size': (1, 2, 2), 'stride': (1, 1, 1),
        #      'in_channels': 4, 'out_channels': 8},
        # ]
        # self.drone_view_conv1 = nn.Conv3d(in_channels=layers_dict[0]['in_channels'],
        #                                out_channels=layers_dict[0]['out_channels'],
        #                                kernel_size=layers_dict[0]['kernel_size'],
        #                                stride=layers_dict[0]['stride'],
        #                                padding=layers_dict[0]['padding'])
        # self.drone_view_conv2 = nn.Conv3d(in_channels=layers_dict[1]['in_channels'],
        #                                out_channels=layers_dict[1]['out_channels'],
        #                                kernel_size=layers_dict[1]['kernel_size'],
        #                                stride=layers_dict[1]['stride'],
        #                                padding=layers_dict[1]['padding'])
        #
        # in_f = get_in_features_3d(h_in=self.map_size, w_in=self.map_size, d_in=self.drone_count, layers_dict=layers_dict)
        #
        # features_explore = in_f * layers_dict[1]['out_channels']
        #
        # self.flatten = nn.Flatten()
        #
        # input_features = features_explore
        # self.out_features = 32
        # mid_features = 64
        #
        # self.input_dense1 = nn.Linear(in_features=input_features, out_features=mid_features)
        # initialize_output_weights(self.input_dense1, 'hidden')
        # self.input_dense2 = nn.Linear(in_features=mid_features, out_features=self.out_features)
        # initialize_output_weights(self.input_dense2, 'hidden')

    # def forward(self, states):
    #     all_total_views, all_explore_maps = self.prepare_tensor(states)
    #
    #     explore = F.relu(self.explore_conv1(all_maps))
    #     explore = F.relu(self.explore_conv2(explore))
    #     explore = self.flatten(explore)
    #
    #     explore = F.relu(self.input_dense1(explore))
    #     explore = F.relu(self.input_dense2(explore))
    #     output_vision = torch.flatten(explore, start_dim=1)
    #
    #     return output_vision


class Actor(nn.Module):
    """
    A PyTorch Module that represents the actor network of a PPO agent.
    """
    def __init__(self, vision_range, drone_count, map_size, time_steps):
        super(Actor, self).__init__()
        self.Inputspace = Inputspace(drone_count=drone_count, map_size=map_size, time_steps=time_steps)
        self.mu_goal = nn.Linear(in_features=self.Inputspace.out_features, out_features=2)
        initialize_output_weights(self.mu_goal, 'actor')

        # Logstd
        self.log_std = nn.Parameter(torch.zeros(2, ))

    def forward(self, states):
        x = self.Inputspace(states)
        mu_goal = torch.tanh(self.mu_goal(x))
        std = torch.exp(self.log_std)
        var = torch.pow(std, 2)

        return mu_goal, var

class DeterministicActor(nn.Module):
    """
    A PyTorch Module that represents the actor network of a deterministic agent.
    """
    def __init__(self, vision_range, drone_count, map_size, time_steps):
        super(DeterministicActor, self).__init__()
        self.Inputspace = Inputspace(drone_count=drone_count, map_size=map_size, time_steps=time_steps)
        self.in_features = self.Inputspace.out_features

        # Mu
        self.l1 = nn.Linear(in_features=self.in_features, out_features=400)
        initialize_output_weights(self.l1, 'actor')
        self.l2 = nn.Linear(in_features=400, out_features=300)
        initialize_output_weights(self.l2, 'actor')
        self.l3 = nn.Linear(in_features=300, out_features=2)
        initialize_output_weights(self.l3, 'actor')

    def forward(self, states):
        x = self.Inputspace(states)
        mu_goal = torch.tanh(self.l1(x))
        mu_goal = torch.tanh(self.l2(mu_goal))
        mu_goal = torch.tanh(self.l3(mu_goal))

        return mu_goal

class Critic(nn.Module):
    """
    A PyTorch Module that represents the critic network of a PPO agent.
    """
    def __init__(self, vision_range, drone_count, map_size, time_steps):
        super(Critic, self).__init__()
        self.Inputspace = Inputspace(drone_count=drone_count, map_size=map_size, time_steps=time_steps)
        self.in_features = self.Inputspace.out_features

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward.")

class CriticPPO(Critic):
    """
    A PyTorch Module that represents the critic network of a PPO agent.
    """
    def __init__(self, vision_range, drone_count, map_size, time_steps):
        super(CriticPPO, self).__init__(vision_range, drone_count, map_size, time_steps)

        # Value
        self.value = nn.Linear(in_features=self.in_features, out_features=1)
        initialize_output_weights(self.value, 'critic')

    def forward(self, states):
        x = self.Inputspace(states)
        value = self.value(x)
        return value

class OffPolicyCritic(Critic):
    """
    A PyTorch Module that represents the critic network of an IQL agent.
    """
    def __init__(self, vision_range, drone_count, map_size, time_steps, action_dim):
        super(OffPolicyCritic, self).__init__(vision_range, drone_count, map_size, time_steps)

        # Value
        self.fc1 = nn.Linear(self.in_features + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q_value = nn.Linear(256, 1)
        initialize_output_weights(self.q_value, 'critic')

    def forward(self, state, action):
        x = self.Inputspace(state)
        x = torch.cat([x, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q = self.q_value(x)
        return q

class RNDModel(nn.Module):
    def __init__(self, vision_range, drone_count, map_size, time_steps):
        super(RNDModel, self).__init__()

        self.predictor = Inputspace(drone_count, map_size, time_steps)
        self.target = Inputspace(drone_count, map_size, time_steps)

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

class Value(nn.Module):
    """
    A PyTorch Module that represents the value network of an IQL agent.
    It estimates V(s), the state value.
    """
    def __init__(self, vision_range, drone_count, map_size, time_steps):
        super(Value, self).__init__()
        self.Inputspace = Inputspace(drone_count=drone_count, map_size=map_size, time_steps=time_steps)
        self.in_features = self.Inputspace.out_features

        # Simple MLP head for value estimation
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.v_value = nn.Linear(256, 1)

        initialize_output_weights(self.v_value, 'value')

    def forward(self, state):
        x = self.Inputspace(state)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        v = self.v_value(x)
        return v