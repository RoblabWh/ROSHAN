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
        # in_features = [dx, dy] if this doesn't work try: [dx, dy, dvx, dvy, dist]
        hidden = 256
        self.out_dim = 64
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, self.out_dim)
        )
        if use_attention:
            self.attn = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim), nn.Tanh(),
                nn.Linear(self.out_dim, 1)  # scalar score
            )
        # optional regularization helps stability
        self.post_ln = nn.LayerNorm(self.out_dim)

    @staticmethod
    def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # scores: (N, K), mask: (N, K) bool
        scores = scores.masked_fill(~mask, float('-inf'))
        # subtract max for numerical stability on valid elements
        maxes = torch.amax(scores, dim=dim, keepdim=True)
        scores = scores - maxes
        # when all are -inf (no valid), softmax would be NaN; guard by filling zeros
        exp = torch.exp(scores)
        exp = exp * mask.float()
        denom = exp.sum(dim=dim, keepdim=True).clamp_min(1e-9)
        return exp / denom

    def forward(self, neigh_feats: torch.Tensor, mask: torch.Tensor):
        B, T, K, F = neigh_feats.shape
        x = neigh_feats.reshape(B*T, K, F)
        m = mask.reshape(B*T, K)

        h = self.mlp(x)                         # (BT, K, D)
        h = h * m.unsqueeze(-1).float()         # zero invalid
        has_any = m.any(dim=1)                  # (BT,)

        pooled = torch.zeros(h.size(0), h.size(-1), device=h.device, dtype=h.dtype)
        if self.use_attention:
            scores = self.attn(h).squeeze(-1)   # (BT, K)
            if has_any.any():
                idx = has_any.nonzero(as_tuple=True)[0]
                w = self.masked_softmax(scores[idx], m[idx], dim=1).unsqueeze(-1)  # (N, K, 1)
                pooled[idx] = (w * h[idx]).sum(dim=1)                               # (N, D)
        else:
            denom = m.sum(dim=1).clamp(min=1).unsqueeze(-1).float()
            pooled = h.sum(dim=1) / denom

        pooled = self.post_ln(pooled)
        # append no-neighbor flag as an extra feature
        no_neigh_flag = (~has_any).float().unsqueeze(-1)
        pooled = torch.cat([pooled, no_neigh_flag], dim=-1)  # (BT, D+1)
        return pooled.reshape(B, T, -1)

class Inputspace(nn.Module):

    def __init__(self, vision_range, time_steps):
        """
        A PyTorch Module that represents the input space of a neural network.
        """
        super(Inputspace, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.vision_range = vision_range
        self.time_steps = time_steps

        hidden = 256
        out = 64

        self.neigh_encoder = NeighborEncoder(use_attention=True)

        self.self_mlp = nn.Sequential(
            nn.Linear(9, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Linear(hidden + self.neigh_encoder.out_dim + 1, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out), nn.ReLU(inplace=True),
        )

        self.out_features = out * self.time_steps

        self.flatten = nn.Flatten()

    def prepare_tensor(self, states):
        self_id, velocity, delta_goal, cos_sin_goal, speed, distance_to_goal, distances_to_others, distances_mask = states

        if not torch.is_tensor(velocity):
            velocity = torch.as_tensor(velocity, dtype=torch.float32)

        if not torch.is_tensor(delta_goal):
            delta_goal = torch.as_tensor(delta_goal, dtype=torch.float32)

        if not torch.is_tensor(cos_sin_goal):
            cos_sin_goal = torch.as_tensor(cos_sin_goal, dtype=torch.float32)

        if not torch.is_tensor(speed):
            speed = torch.as_tensor(speed, dtype=torch.float32)

        if not torch.is_tensor(distance_to_goal):
            distance_to_goal = torch.as_tensor(distance_to_goal, dtype=torch.float32)

        if not torch.is_tensor(distances_to_others):
            distances_to_others = torch.as_tensor(distances_to_others, dtype=torch.float32)

        if not torch.is_tensor(distances_mask):
            distances_mask = torch.as_tensor(distances_mask, dtype=torch.bool)

        if not torch.is_tensor(self_id):
            self_id = torch.as_tensor(self_id, dtype=torch.int64)

        # Only move if needed
        if velocity.device != self.device:
            velocity = velocity.to(self.device)
        if delta_goal.device != self.device:
            delta_goal = delta_goal.to(self.device)
        if cos_sin_goal.device != self.device:
            cos_sin_goal = cos_sin_goal.to(self.device)
        if speed.device != self.device:
            speed = speed.to(self.device)
        if distance_to_goal.device != self.device:
            distance_to_goal = distance_to_goal.to(self.device)
        if distances_to_others.device != self.device:
            distances_to_others = distances_to_others.to(self.device)
        if distances_mask.device != self.device:
            distances_mask = distances_mask.to(self.device)
        if self_id != self.device:
            self_id = self_id.to(self.device)

        return self_id, velocity, delta_goal, cos_sin_goal, speed, distance_to_goal, distances_to_others, distances_mask

    def forward(self, states):
        self_id, velocity, delta_goal, cos_sin_goal, speed, distance_to_goal, distances_to_others, distances_mask = self.prepare_tensor(states)

        tensors = [self_id.unsqueeze(2), velocity, delta_goal, cos_sin_goal, speed.unsqueeze(2), distance_to_goal.unsqueeze(2)]
        feats = torch.cat(tensors, dim=-1)

        x = self.self_mlp(feats)  # (B, T, H)
        y = self.neigh_encoder(distances_to_others, distances_mask)
        fuse = torch.cat((x, y), dim=-1)
        z = self.fuse(fuse)

        output_vision = torch.flatten(z, start_dim=1)

        return output_vision


class Actor(nn.Module):
    """
    A PyTorch Module that represents the actor network of a PPO agent.
    """
    def __init__(self, vision_range, drone_count, map_size, time_steps, manual_decay=False):
        super(Actor, self).__init__()
        self.Inputspace = Inputspace(vision_range, time_steps=time_steps)
        self.in_features = self.Inputspace.out_features
        # Mu
        self.mu_move = nn.Linear(in_features=self.in_features, out_features=2)
        self.mu_move._init_gain = 0.1

        # Logstd
        self.log_std = nn.Parameter(torch.zeros(2, ), requires_grad=not manual_decay)

    def forward(self, states):
        x = self.Inputspace(states)
        mu_move = torch.tanh(self.mu_move(x))
        std = torch.exp(self.log_std)
        var = torch.pow(std, 2)

        return mu_move, var


class Critic(nn.Module):
    """
    A PyTorch Module that represents the critic network of a PPO agent.
    """
    def __init__(self, vision_range, drone_count, map_size, time_steps, inputspace=None):
        super(Critic, self).__init__()
        self.Inputspace = Inputspace(vision_range, time_steps=time_steps) if not inputspace else inputspace
        self.in_features = self.Inputspace.out_features

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward.")

class CriticPPO(Critic):
    """
    A PyTorch Module that represents the critic network of a PPO agent.
    """
    def __init__(self, vision_range, drone_count, map_size, time_steps, inputspace=None):
        super(CriticPPO, self).__init__(vision_range, drone_count, map_size, time_steps)

        # Value
        self.value = nn.Linear(in_features=self.in_features, out_features=1)
        self.value._init_gain = 1.0

    def forward(self, states):
        x = self.Inputspace(states)
        value = self.value(x)
        return value

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