import os
import torch.nn as nn
import torch

if os.getenv("PYTORCH_DETECT_ANOMALY", "").lower() in ("1", "true"):
    torch.autograd.set_detect_anomaly(True)

class CNNSpatialEncoder(nn.Module):
    def __init__(self, in_channels, out_features):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Downsample (W,H)/2

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Downsample again (W,H)/4

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # Fixed spatial dimension
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.flatten(x)
        features = self.fc(x)
        return features

class Inputspace(nn.Module):

    def __init__(self, drone_count, map_size, time_steps):
        """
        A PyTorch Module that represents the input space of a neural network.
        """
        super(Inputspace, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.drone_count = drone_count
        self.time_steps = time_steps

        self.out_features = 256

        # CNN Encoders
        spatial_outfeatures = 128
        position_outfeatures = 128
        in_features = self.drone_count * self.time_steps + self.time_steps
        self.feature_extractor = CNNSpatialEncoder(self.time_steps, spatial_outfeatures)
        self.mlp_pos = nn.Sequential(
            nn.Linear(self.time_steps, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, position_outfeatures)
        )
        # Final linear layer to merge features
        self.merge_layer = nn.Sequential(
            nn.Linear(spatial_outfeatures + position_outfeatures, self.out_features),
            nn.ReLU(inplace=True),
        )

    def _ensure_tensor(self, x, dtype=torch.float32):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=dtype)
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)
        return x

    def prepare_tensor(self, states):
        agent_positions, exploration_maps = states
        agent_positions = self._ensure_tensor(agent_positions)
        exploration_maps = self._ensure_tensor(exploration_maps)
        return agent_positions, exploration_maps

    def forward(self, states):
        agent_positions, exploration_maps = self.prepare_tensor(states)
        x = self.feature_extractor(exploration_maps)
        B, D, T, F = agent_positions.shape
        agent_positions_flat = agent_positions.reshape(B, D * T, F)  # Flatten the positions
        position_emb = self.mlp_pos(agent_positions_flat)
        x = torch.cat([position_emb, x], dim=1)
        x = self.merge_layer(x)
        # agent_positions_flat = agent_positions.reshape(B, D * T, W, H)
        # exploration_map_flat = exploration_maps
        # input_tensor = torch.cat([agent_positions_flat, exploration_map_flat], dim=1)  # (B, D*T + T, W, H)
        #
        # x = self.feature_extractor(input_tensor)  # (B, spatial_outfeatures)
        # x = self.merge_layer(x)  # (B, out_features)
        return x  # (B, out_features)


class Actor(nn.Module):
    """
    A PyTorch Module that represents the actor network of a PPO agent.
    """
    def __init__(self, vision_range, drone_count, map_size, time_steps):
        super(Actor, self).__init__()
        self.Inputspace = Inputspace(drone_count=drone_count, map_size=map_size, time_steps=time_steps)
        self.mu_goal = nn.Linear(in_features=self.Inputspace.out_features, out_features=2)
        self.mu_goal._init_gain = 0.1

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
        self.l2 = nn.Linear(in_features=400, out_features=300)
        self.l3 = nn.Linear(in_features=300, out_features=2)
        self.l3._init_gain = 0.1

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
        self.value._init_gain = 1.0

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
        self.q_value._init_gain = 1.0

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
        self.v_value._init_gain = 1.0

    def forward(self, state):
        x = self.Inputspace(state)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        v = self.v_value(x)
        return v