import os
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import get_device

if os.getenv("PYTORCH_DETECT_ANOMALY", "").lower() in ("1", "true"):
    torch.autograd.set_detect_anomaly(True)

class Inputspace(nn.Module):

    def __init__(self, drone_dim, time_steps):
        super().__init__()

        self.device = get_device()
        self.drone_dim = drone_dim
        hidden_dim = 64

        # Embeddings with activation
        self.pos_emb = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU())
        self.goal_emb = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU())
        self.fire_emb = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU())
        self.vel_emb = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU())
        self.id_emb = nn.Embedding(drone_dim, hidden_dim)

        # Fusion MLP: cat(pos, goal, id, vel) -> hidden
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU()
        )

        # Drone self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(hidden_dim)

        # Drone->Fire cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)

        self.out_features = hidden_dim
        self.register_buffer('_agent_ids', torch.arange(drone_dim))

    def _ensure_tensor(self, x, dtype=torch.float32):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=dtype)
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)
        return x

    @staticmethod
    def _select_last_timestep(x):
        """Reduce 4D/5D tensors to 3D by selecting the last timestep."""
        if x.dim() == 5:
            return x[:, -1, -1, :, :]
        if x.dim() == 4:
            return x[:, -1, :, :]
        return x

    def prepare_tensor(self, states):
        if hasattr(states, 'keys') or isinstance(states, dict):
            # Dict-based path (new schema pipeline)
            drone_states = self._ensure_tensor(states["drone_positions"])
            goal_positions = self._ensure_tensor(states["goal_positions"])
            fire_states = self._select_last_timestep(self._ensure_tensor(states["fire_positions"]))
        else:
            # Legacy tuple path
            drone_states, goal_positions, fire_states = states
            drone_states = self._ensure_tensor(drone_states)
            goal_positions = self._ensure_tensor(goal_positions)
            fire_states = self._select_last_timestep(self._ensure_tensor(fire_states))

        # Extract velocity before collapsing timesteps
        if drone_states.dim() == 4 and drone_states.shape[1] >= 2:
            velocity = drone_states[:, -1] - drone_states[:, -2]  # (B, N, 2)
            drone_pos = drone_states[:, -1]  # (B, N, 2)
        else:
            drone_states = self._select_last_timestep(drone_states)
            velocity = torch.zeros_like(drone_states)
            drone_pos = drone_states

        goal_positions = self._select_last_timestep(goal_positions)

        batch_size, n_drones, _ = drone_pos.shape
        agent_ids = self._agent_ids.unsqueeze(0).expand(batch_size, -1)
        return drone_pos, goal_positions, fire_states, agent_ids, velocity

    def forward(self, states, mask=None):
        drone_pos, goal_pos, fire_states, agent_ids, velocity = self.prepare_tensor(states)

        # Dynamic mask (fixes hardcoded size-2 default)
        if mask is None:
            mask = torch.zeros(fire_states.shape[0], fire_states.shape[1],
                               dtype=torch.bool, device=self.device)

        # Embed
        pos_e = self.pos_emb(drone_pos)
        goal_e = self.goal_emb(goal_pos)
        fire_e = self.fire_emb(fire_states)
        vel_e = self.vel_emb(velocity)
        id_e = self.id_emb(agent_ids)

        # Fusion MLP (replaces additive collapse)
        drone_emb = self.fusion(torch.cat([pos_e, goal_e, id_e, vel_e], dim=-1))

        # Self-attention among drones + residual + LayerNorm
        self_out, _ = self.self_attn(drone_emb, drone_emb, drone_emb)
        drone_emb = self.self_attn_norm(drone_emb + self_out)

        # Cross-attention: drones attend to fires + residual + LayerNorm
        cross_out, attn_weights = self.cross_attn(
            query=drone_emb, key=fire_e, value=fire_e,
            key_padding_mask=mask
        )
        drone_repr = self.cross_attn_norm(drone_emb + cross_out)

        return drone_repr, fire_e, attn_weights

class AttentionActor(nn.Module):
    def __init__(self, vision_range, drone_count, map_size, time_steps, manual_decay):
        super().__init__()
        self.Inputspace = Inputspace(drone_dim=drone_count, time_steps=time_steps)
        hidden_dim = self.Inputspace.out_features

        # Learnable temperature for exploration control
        self.log_temperature = nn.Parameter(torch.zeros(1))

        # Pointer-network projection heads
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, states, masks=None):
        drone_repr, fire_repr, _ = self.Inputspace(states, masks)

        # Pointer-network style scoring
        q = self.query_proj(drone_repr)   # (B, N, D)
        k = self.key_proj(fire_repr)       # (B, F, D)
        d = q.shape[-1] ** 0.5
        temperature = self.log_temperature.exp().clamp(min=0.01)
        logits = torch.bmm(q, k.transpose(1, 2)) / (d * temperature)  # (B, N, F)

        # Build effective mask: always mask groundstation at index 0
        if masks is None:
            masks = torch.zeros(fire_repr.shape[0], fire_repr.shape[1],
                               dtype=torch.bool, device=fire_repr.device)
        else:
            masks = masks.clone()
        masks[:, 0] = True  # groundstation/wait token
        logits = logits.masked_fill(masks.unsqueeze(1), float('-inf'))

        return logits  # raw logits — Categorical(logits=...) applies softmax

class Critic(nn.Module):
    def __init__(self, vision_range, drone_count, map_size, time_steps, inputspace=None):
        super().__init__()
        self.Inputspace_1 = Inputspace(drone_count, time_steps=time_steps) if not inputspace else inputspace
        self.in_features = self.Inputspace_1.out_features

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward.")

class CriticPPO(nn.Module):
    def __init__(self, vision_range, drone_count, map_size, time_steps, inputspace=None):
        super().__init__()
        self.Inputspace_1 = Inputspace(drone_count, time_steps=time_steps) if not inputspace else inputspace
        hidden_dim = self.Inputspace_1.out_features
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.value_head[-1]._init_gain = 1.0

    def forward(self, states, masks=None):
        drone_repr, _, _ = self.Inputspace_1(states, masks)  # (B, N, 64)
        pooled = drone_repr.mean(dim=1)  # (B, 64) — pool drones before value head
        return self.value_head(pooled)    # (B, 1)

class OffPolicyCritic(Critic):
    def __init__(self, vision_range, drone_count, map_size, time_steps, action_dim, inputspace=None):
        super().__init__(vision_range, drone_count, map_size, time_steps, inputspace)
        self.Inputspace_2 = Inputspace(drone_count, time_steps=time_steps) if not inputspace else inputspace

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
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)

        return q1, q2

    def Q1(self, state, action):
        x, _, _ = self.Inputspace_1(state)
        x = x.mean(dim=1)  # pool drones: (B, N, D) -> (B, D)
        x = torch.cat([x, action], dim=1)

        q1 = F.relu(self.l1(x))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

    def Q2(self, state, action):
        x, _, _ = self.Inputspace_2(state)
        x = x.mean(dim=1)
        x = torch.cat([x, action], dim=1)

        q2 = F.relu(self.l4(x))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q2

class Value(nn.Module):
    def __init__(self, vision_range, drone_count, map_size, time_steps, inputspace):
        super().__init__()
        self.Inputspace = Inputspace(drone_count, time_steps=time_steps) if not inputspace else inputspace
        self.in_features = self.Inputspace.out_features

        # Simple MLP head for value estimation
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.v_value = nn.Linear(256, 1)
        self.v_value._init_gain = 1.0

    def forward(self, state):
        x, _, _ = self.Inputspace(state)
        x = x.mean(dim=1)  # pool drones
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        v = self.v_value(x)
        return v
