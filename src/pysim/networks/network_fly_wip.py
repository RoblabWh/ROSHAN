import os
import torch.nn as nn
import torch.nn.functional as F
import torch

if os.getenv("PYTORCH_DETECT_ANOMALY", "").lower() in ("1", "true"):
    torch.autograd.set_detect_anomaly(True)

class NeighborEncoder(nn.Module):
    """Permutation-invariant neighbor set encoder with masking."""
    def __init__(self, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        # in_features = [dx, dy, dvx, dvy, dist]
        hidden = 64
        self.out_dim = 64
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, self.out_dim)
        )
        if use_attention:
            self.attn = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim), nn.Tanh(),
                nn.Linear(self.out_dim, 1)  # scalar score
            )

    def forward(self, neigh_feats, mask):
        """
        neigh_feats: (B, K, F) where F = [dx, dy, dvx, dvy, dist]
        mask: (B, K) bool; True for valid neighbors
        """
        B, K, Fdim = neigh_feats.shape
        h = self.mlp(neigh_feats)                        # (B, K, D)
        # zero-out invalid rows
        h = h * mask.unsqueeze(-1)                       # (B, K, D)
        if self.use_attention:
            scores = self.attn(h).squeeze(-1)            # (B, K)
            scores = scores.masked_fill(~mask, -1e9)
            w = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled = (w * h).sum(dim=1)                  # (B, D)
        else:
            # mean pool over valid neighbors; avoid /0
            denom = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
            pooled = h.sum(dim=1) / denom                # (B, D)
        return pooled  # (B, D)

class Inputspace(nn.Module):

    def __init__(self, vision_range, time_steps):
        """
        A PyTorch Module that represents the input space of a neural network.
        """
        super(Inputspace, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.vision_range = vision_range
        self.time_steps = time_steps

        hidden = 64
        out = 64

        self.neigh_encoder = NeighborEncoder(use_attention=True)

        self.self_mlp = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Linear(hidden + self.neigh_encoder.out_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out), nn.ReLU(inplace=True),
        )

        self.out_features = hidden * self.time_steps

        self.flatten = nn.Flatten()

    def prepare_tensor(self, states):
        velocity, delta_goal = states

        if not torch.is_tensor(velocity):
            velocity = torch.as_tensor(velocity, dtype=torch.float32)

        if not torch.is_tensor(delta_goal):
            delta_goal = torch.as_tensor(delta_goal, dtype=torch.float32)

        # Only move if needed
        if velocity.device != self.device:
            velocity = velocity.to(self.device)
        if delta_goal.device != self.device:
            delta_goal = delta_goal.to(self.device)

        return velocity, delta_goal

    def forward(self, states):
        velocity, delta_goal = self.prepare_tensor(states)

        delta_position = delta_goal - velocity
        x = torch.cat((velocity, delta_position), dim=-1)
        x = self.self_mlp(x)  # (B, T, H)
        output_vision = torch.flatten(x, start_dim=1)

        return output_vision


class Actor(nn.Module):
    """
    A PyTorch Module that represents the actor network of a PPO agent.
    """
    def __init__(self, vision_range, drone_count, map_size, time_steps):
        super(Actor, self).__init__()
        self.Inputspace = Inputspace(vision_range, time_steps=time_steps)
        self.in_features = self.Inputspace.out_features
        # Mu
        self.mu_move = nn.Linear(in_features=self.in_features, out_features=2)
        self.mu_move._init_gain = 0.1

        # Logstd
        self.log_std = nn.Parameter(torch.zeros(2, ))

    def forward(self, states):
        x = self.Inputspace(states)
        mu_move = torch.tanh(self.mu_move(x))
        std = torch.exp(self.log_std)
        var = torch.pow(std, 2)

        return mu_move, var

class DeterministicActor(nn.Module):
    """
    A PyTorch Module that represents the actor network of a deterministic agent.
    """
    def __init__(self, vision_range, drone_count, map_size, time_steps):
        super(DeterministicActor, self).__init__()
        self.Inputspace = Inputspace(vision_range, time_steps=time_steps)
        self.in_features = self.Inputspace.out_features

        # Mu
        self.l1 = nn.Linear(in_features=self.in_features, out_features=400)
        self.l2 = nn.Linear(in_features=400, out_features=300)
        self.l3 = nn.Linear(in_features=300, out_features=2)
        self.l3._init_gain = 0.1

    def forward(self, states):
        x = self.Inputspace(states)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        movement = torch.tanh(self.l3(x))
        return movement


class Critic(nn.Module):
    """
    A PyTorch Module that represents the critic network of a PPO agent.
    """
    def __init__(self, vision_range, drone_count, map_size, time_steps):
        super(Critic, self).__init__()
        self.Inputspace = Inputspace(vision_range, time_steps=time_steps)
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

        # Q1 architecture
        self.l1 = nn.Linear(self.in_features + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.l3._init_gain = 1.0

        # Q2 architecture
        self.l4 = nn.Linear(self.in_features + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
        self.l6._init_gain = 1.0

    def forward(self, state, action):
        x = self.Inputspace(state)
        x = torch.cat([x, action], dim=1)

        q1 = F.relu(self.l1(x))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(x))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        x = self.Inputspace(state)
        x = torch.cat([x, action], dim=1)

        q1 = F.relu(self.l1(x))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class Value(nn.Module):
    """
    A PyTorch Module that represents the value network of an IQL agent.
    It estimates V(s), the state value.
    """
    def __init__(self, vision_range, drone_count, map_size, time_steps):
        super(Value, self).__init__()
        self.Inputspace = Inputspace(vision_range, time_steps=time_steps)
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