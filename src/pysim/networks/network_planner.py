"""Minimal MLP-based pointer network for the PlannerAgent.

DeepSets-style: per-fire MLP encoder + per-drone MLP encoder + pairwise dot-product
logits + INDEPENDENT Categorical per drone. Variable-size fire sets are handled via
per-element MLP encoding plus a padding mask — no attention.

Replaces the prior transformer + autoregressive pointer-network in
``network_planner_attn.py``. The full attention-based implementation is preserved
under that filename; flip ``planner_agent.py``'s import to revert.

Public API (must stay stable for ``planner_agent.py`` / ``algorithms/``):
- ``PointerActor.__init__(vision_range, drone_count, map_size, time_steps, manual_decay)``
- ``PointerActor.forward(states, masks=None, actions_idx=None, deterministic=False)
  -> (actions_idx (B,N), log_probs (B,N), entropy (B,N))``
- ``PointerActor.log_temperature`` — required by ``ppo.py`` for logging
- ``self.Inputspace`` attribute — required for ``share_encoder=True``
"""
import os
import torch
import torch.nn as nn

from utils import get_device

if os.getenv("PYTORCH_DETECT_ANOMALY", "").lower() in ("1", "true"):
    torch.autograd.set_detect_anomaly(True)


class Inputspace(nn.Module):
    """Encodes the planner's observation dict into per-drone and per-fire embeddings."""

    def __init__(self, drone_dim, time_steps):
        super().__init__()
        del time_steps  # unused — we always read the last timestep
        self.device = get_device()
        self.drone_dim = drone_dim
        hidden_dim = 64

        # Per-drone MLP: cat(pos(2), prev_goal(2), water(1)) → hidden
        self.drone_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        # Per-fire MLP: pos(2) → hidden
        self.fire_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        # GS marker added to fire-embedding index 0 so coincident-coordinate fires
        # don't alias the groundstation. Init zero → no-op until learned.
        self.is_gs_emb = nn.Parameter(torch.zeros(hidden_dim))

        # Global context — three small encoders, summed and broadcast-added to drones.
        # Same pattern as the prior Inputspace (just without the attention pieces around it).
        self.fire_count_emb = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU())
        self.fire_centroid_emb = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU())
        self.wind_emb = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU())

        self.out_features = hidden_dim

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

    def forward(self, states, mask=None):
        drone_pos = self._select_last_timestep(self._ensure_tensor(states["drone_positions"]))
        goal_pos = self._select_last_timestep(self._ensure_tensor(states["goal_positions"]))
        fire_pos = self._select_last_timestep(self._ensure_tensor(states["fire_positions"]))
        drone_water = self._select_last_timestep(self._ensure_tensor(states["drone_water"]))

        # C++ emits fire_positions_mask (True = valid) when fires are RELATIONAL.
        # Convert to the network's convention (True = padded/invalid).
        fire_mask = None
        if mask is None and "fire_positions_mask" in states:
            fm = self._ensure_tensor(states["fire_positions_mask"], dtype=torch.bool)
            if fm.dim() == 3:
                fm = fm[:, -1, :]
            fire_mask = ~fm
        elif mask is not None:
            fire_mask = mask

        # Global context: fire_globals → [count(1), centroid(2), wind(2)]
        fire_globals = self._ensure_tensor(states["fire_globals"])
        if fire_globals.dim() == 3:
            fire_globals = fire_globals[:, -1, :]
        fire_count = fire_globals[:, 0:1]
        fire_centroid = fire_globals[:, 1:3]
        wind = fire_globals[:, 3:5]

        # Per-drone embedding: cat(pos, prev_goal, water) → MLP
        drone_feat = torch.cat([drone_pos, goal_pos, drone_water], dim=-1)
        drone_emb = self.drone_mlp(drone_feat)  # (B, N, H)

        # Per-fire embedding + GS marker on index 0
        fire_emb = self.fire_mlp(fire_pos)  # (B, F, H)
        fire_emb = fire_emb.clone()
        fire_emb[:, 0, :] = fire_emb[:, 0, :] + self.is_gs_emb

        # Global context: broadcast-add to drone embeddings
        global_ctx = (
            self.fire_count_emb(fire_count)
            + self.fire_centroid_emb(fire_centroid)
            + self.wind_emb(wind)
        ).unsqueeze(1)  # (B, 1, H)
        drone_emb = drone_emb + global_ctx

        return drone_emb, fire_emb, fire_mask


class PointerActor(nn.Module):
    """Pointer policy: per-drone Categorical over fires via dot-product logits."""

    def __init__(self, vision_range, drone_count, map_size, time_steps, manual_decay):
        super().__init__()
        del vision_range, map_size, manual_decay  # unused
        self.Inputspace = Inputspace(drone_dim=drone_count, time_steps=time_steps)
        hidden_dim = self.Inputspace.out_features
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        # Required by ppo.py:531 (unconditional read for TensorBoard logging).
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def _logits_and_dist(self, states, masks):
        drone_emb, fire_emb, fire_mask = self.Inputspace(states, masks)
        # If caller supplied a mask, prefer it; otherwise fall back to the C++ one;
        # otherwise treat all fires as valid.
        if masks is None:
            masks = fire_mask
        if masks is None:
            masks = torch.zeros(
                fire_emb.shape[0], fire_emb.shape[1],
                dtype=torch.bool, device=fire_emb.device,
            )

        q = self.q_proj(drone_emb)  # (B, N, H)
        k = self.k_proj(fire_emb)   # (B, F, H)
        d_sqrt = q.shape[-1] ** 0.5
        # Symmetric clamp prevents exp() overflow into either tail.
        temperature = self.log_temperature.clamp(-3.0, 3.0).exp()
        logits = torch.bmm(q, k.transpose(1, 2)) / (d_sqrt * temperature)  # (B, N, F)

        # Guarantee at least one valid action per row — unmask GS (index 0) on
        # fully-padded rows to keep Categorical finite.
        safe_masks = masks.clone()
        all_masked = safe_masks.all(dim=-1)
        if all_masked.any():
            safe_masks[all_masked, 0] = False
        logits = logits.masked_fill(safe_masks.unsqueeze(1).expand_as(logits), float('-inf'))

        return torch.distributions.Categorical(logits=logits)

    def forward(self, states, masks=None, actions_idx=None, deterministic=False):
        dist = self._logits_and_dist(states, masks)

        if actions_idx is not None:
            # PPO evaluate path: recompute log-prob and entropy for given indices.
            return actions_idx, dist.log_prob(actions_idx), dist.entropy()

        if deterministic:
            actions = dist.logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        return actions, dist.log_prob(actions), dist.entropy()


class PointerCritic(nn.Module):
    """PPO state-value critic: mean-pool drone embeddings → MLP → scalar."""

    def __init__(self, vision_range, drone_count, map_size, time_steps, inputspace=None):
        super().__init__()
        del vision_range, map_size  # unused
        self.Inputspace_1 = (
            Inputspace(drone_count, time_steps=time_steps) if inputspace is None else inputspace
        )
        hidden_dim = self.Inputspace_1.out_features
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.value_head[-1]._init_gain = 1.0

    def forward(self, states, masks=None):
        drone_emb, _, _ = self.Inputspace_1(states, masks)  # (B, N, H)
        pooled = drone_emb.mean(dim=1)  # (B, H)
        return self.value_head(pooled)  # (B, 1)


class PointerOffPolicyCritic(nn.Module):
    """Twin-Q critic for IQL/TD3. cat(pooled drone_emb, action) → 2× MLP heads."""

    def __init__(self, vision_range, drone_count, map_size, time_steps, action_dim, inputspace=None):
        super().__init__()
        del vision_range, map_size  # unused
        self.Inputspace_1 = (
            Inputspace(drone_count, time_steps=time_steps) if inputspace is None else inputspace
        )
        self.Inputspace_2 = (
            Inputspace(drone_count, time_steps=time_steps) if inputspace is None else inputspace
        )
        hidden_dim = self.Inputspace_1.out_features

        self.l1 = nn.Linear(hidden_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.l3._init_gain = 1.0

        self.l4 = nn.Linear(hidden_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
        self.l6._init_gain = 1.0

    def _pooled(self, inputspace, state):
        drone_emb, _, _ = inputspace(state)
        return drone_emb.mean(dim=1)  # (B, H)

    def forward(self, state, action):
        return self.Q1(state, action), self.Q2(state, action)

    def Q1(self, state, action):
        x = torch.cat([self._pooled(self.Inputspace_1, state), action], dim=1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.l3(x)

    def Q2(self, state, action):
        x = torch.cat([self._pooled(self.Inputspace_2, state), action], dim=1)
        x = torch.relu(self.l4(x))
        x = torch.relu(self.l5(x))
        return self.l6(x)


class PointerValue(nn.Module):
    """IQL state-value head: mean-pool drone_emb → MLP → scalar."""

    def __init__(self, vision_range, drone_count, map_size, time_steps, inputspace):
        super().__init__()
        del vision_range, map_size  # unused
        self.Inputspace = (
            Inputspace(drone_count, time_steps=time_steps) if inputspace is None else inputspace
        )
        hidden_dim = self.Inputspace.out_features
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.v_value = nn.Linear(256, 1)
        self.v_value._init_gain = 1.0

    def forward(self, state):
        drone_emb, _, _ = self.Inputspace(state)
        x = drone_emb.mean(dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.v_value(x)
