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
        # Type marker added to the GS entry (index 0 of fire_positions) so K_GS has a
        # learnable direction distinct from any real-fire key. Init zero → no-op at start.
        self.is_gs_emb = nn.Parameter(torch.zeros(hidden_dim))
        self.vel_emb = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU())
        self.water_emb = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU())
        self.fire_count_emb = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU())
        self.fire_centroid_emb = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU())
        self.wind_emb = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU())
        self.id_emb = nn.Embedding(drone_dim, hidden_dim)

        # Fusion MLP: cat(pos, goal, id, vel, water) -> hidden
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
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
        drone_states = self._ensure_tensor(states["drone_positions"])
        goal_positions = self._ensure_tensor(states["goal_positions"])
        fire_states = self._select_last_timestep(self._ensure_tensor(states["fire_positions"]))
        # drone_water: SET group with bulk_dims=1 → shape (B, T, N, 1). Collapse to (B, N, 1).
        drone_water = self._select_last_timestep(self._ensure_tensor(states["drone_water"]))

        # C++ emits fire_positions_mask (True = valid) when fires are RELATIONAL.
        # Convert to the network's convention (True = padded/invalid).
        fire_mask = None
        if "fire_positions_mask" in states:
            fire_mask = self._ensure_tensor(states["fire_positions_mask"], dtype=torch.bool)
            # Shape from rl_handler: (B, max_T, K). Select last timestep → (B, K).
            if fire_mask.dim() == 3:
                fire_mask = fire_mask[:, -1, :]
            fire_mask = ~fire_mask  # valid → padded

        # Fire globals: FIXED group → (B, T, 5), select last timestep → (B, 5)
        # Layout: [fire_count(1), fire_centroid(2), wind(2)]
        fire_globals = self._ensure_tensor(states["fire_globals"])
        if fire_globals.dim() == 3:
            fire_globals = fire_globals[:, -1, :]
        fire_count = fire_globals[:, 0:1]     # (B, 1)
        fire_centroid = fire_globals[:, 1:3]  # (B, 2)
        wind = fire_globals[:, 3:5]           # (B, 2)

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
        return drone_pos, goal_positions, fire_states, agent_ids, velocity, drone_water, fire_count, fire_centroid, wind, fire_mask

    def forward(self, states, mask=None):
        (drone_pos, goal_pos, fire_states, agent_ids, velocity, drone_water,
         fire_count, fire_centroid, wind, fire_mask) = self.prepare_tensor(states)

        # Prefer the explicit mask from C++ (RELATIONAL fire_positions_mask).
        # Fall back to the caller-supplied mask, then to all-valid.
        if mask is None:
            mask = fire_mask
        if mask is None:
            mask = torch.zeros(fire_states.shape[0], fire_states.shape[1],
                               dtype=torch.bool, device=self.device)

        # Embed
        pos_e = self.pos_emb(drone_pos)
        goal_e = self.goal_emb(goal_pos)
        fire_e = self.fire_emb(fire_states)
        fire_e = fire_e.clone()
        fire_e[:, 0, :] = fire_e[:, 0, :] + self.is_gs_emb
        vel_e = self.vel_emb(velocity)
        water_e = self.water_emb(drone_water)
        id_e = self.id_emb(agent_ids)

        # Fusion MLP (replaces additive collapse)
        drone_emb = self.fusion(torch.cat([pos_e, goal_e, id_e, vel_e, water_e], dim=-1))

        # Inject global fire context (broadcast-add over all drones)
        fire_count_e = self.fire_count_emb(fire_count)          # (B, 64)
        fire_centroid_e = self.fire_centroid_emb(fire_centroid) # (B, 64)
        wind_e = self.wind_emb(wind)                            # (B, 64)
        global_ctx = (fire_count_e + fire_centroid_e + wind_e).unsqueeze(1)  # (B, 1, 64)
        drone_emb = drone_emb + global_ctx

        # Self-attention among drones + residual + LayerNorm
        self_out, _ = self.self_attn(drone_emb, drone_emb, drone_emb)
        drone_emb = self.self_attn_norm(drone_emb + self_out)

        # Cross-attention: drones attend to fires + residual + LayerNorm.
        # need_weights=False routes through SDPA, which handles fully-masked
        # rows without NaN (see comment above) and is the faster kernel.
        cross_out, _ = self.cross_attn(
            query=drone_emb, key=fire_e, value=fire_e,
            key_padding_mask=mask, need_weights=False,
        )
        drone_repr = self.cross_attn_norm(drone_emb + cross_out)

        # Return the effective fire mask alongside so downstream heads
        # (pointer-network logits) can reuse it without re-resolving.
        return drone_repr, fire_e, mask

class PointerActor(nn.Module):
    def __init__(self, vision_range, drone_count, map_size, time_steps, manual_decay):
        super().__init__()
        self.Inputspace = Inputspace(drone_dim=drone_count, time_steps=time_steps)
        hidden_dim = self.Inputspace.out_features

        # Learnable temperature for exploration control
        self.log_temperature = nn.Parameter(torch.zeros(1))

        # Pointer-network projection heads
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)

        # Learnable soft penalty for duplicate assignments in autoregressive decoding.
        # Starts at exp(0)=1.0 (mild); PPO tunes upward if duplicate-assignment is truly costly.
        # A higher init (e.g. 1.6) crowds out the groundstation in multi-drone batches before
        # the policy can learn the refuel pattern.
        self.log_assignment_penalty = nn.Parameter(torch.tensor(0.0))

        # Inductive bias on the GS logit: bias = gs_water_bias * (1 - water_frac).
        # At water=1 contributes 0; at water=0 contributes gs_water_bias. Init zero so
        # behavior matches the unbiased pointer-network at step 0; PPO tunes magnitude.
        self.gs_water_bias = nn.Parameter(torch.tensor(0.0))

        # Stickiness bias on logits matching each drone's *previous* goal coordinate.
        # The previous-action signal already lives in observations as goal_positions
        # (BuildAgentState pulls it from FlyAgent.last_state.goal_position), but the
        # autoregressive decoder otherwise has to infer "this fire == my last goal"
        # from continuous coordinates joined through two embedding paths. Making it
        # an explicit (drone, fire) match mask + learnable bias gives PPO a direct
        # gradient handle to commit to assignments and counter Categorical sampling
        # noise. Init zero → no-op at step 0; PPO learns sign and magnitude.
        self.stickiness_bias = nn.Parameter(torch.tensor(0.0))
        # Match tolerance for the previous-goal coordinate vs. fire-position comparison.
        # Both sides go through the same (2*p/norm_map - 1) normalization, so equality
        # holds modulo float precision; 1e-3 is well below the smallest inter-cell gap
        # (2/max_dim ≈ 0.025 for a 80-cell map) so collisions across distinct cells
        # are impossible at any realistic map size.
        self.stickiness_match_eps = 1e-3

    def _compute_base_logits(self, states, masks=None):
        """Compute base pointer-network logits and effective mask."""
        drone_repr, fire_repr, resolved_mask = self.Inputspace(states, masks)

        q = self.query_proj(drone_repr)   # (B, N, D)
        k = self.key_proj(fire_repr)       # (B, F, D)
        d = q.shape[-1] ** 0.5
        # Bound the learned temperature on both sides. Without an upper clamp
        # exp(log_temperature) can explode and flatten logits into a near-uniform
        # distribution, effectively killing exploration signal.
        temperature = self.log_temperature.clamp(-3.0, 3.0).exp()
        base_logits = torch.bmm(q, k.transpose(1, 2)) / (d * temperature)  # (B, N, F)

        # Water-conditioned bias on the GS logit (index 0). Lifts GS specifically when
        # the drone is low on water; init zero so this is a no-op until PPO learns.
        dw = self.Inputspace._ensure_tensor(states["drone_water"])
        dw = self.Inputspace._select_last_timestep(dw)  # (B, N, 1) or (B, N)
        if dw.dim() == 3:
            dw = dw.squeeze(-1)
        gs_bias = self.gs_water_bias * (1.0 - dw)  # (B, N)
        base_logits = base_logits.clone()
        base_logits[..., 0] = base_logits[..., 0] + gs_bias

        # Stickiness bias: explicit (drone, fire) match between each drone's previous
        # planner-assigned goal coordinate and each fire's coordinate. The decoder no
        # longer has to infer the join through two embeddings — the match becomes a
        # direct logit boost the policy can lean on to commit to assignments.
        # Both coordinate sides come from the same (2*p/norm_map - 1) normalization,
        # so equality is exact modulo float precision.
        goal_pos_raw = self.Inputspace._ensure_tensor(states["goal_positions"])
        goal_pos = self.Inputspace._select_last_timestep(goal_pos_raw)  # (B, N, 2)
        fire_pos_raw = self.Inputspace._ensure_tensor(states["fire_positions"])
        fire_pos = self.Inputspace._select_last_timestep(fire_pos_raw)  # (B, F, 2)
        diff = (goal_pos.unsqueeze(2) - fire_pos.unsqueeze(1)).abs()  # (B, N, F, 2)
        is_prev = (diff < self.stickiness_match_eps).all(dim=-1).to(base_logits.dtype)
        base_logits = base_logits + self.stickiness_bias * is_prev

        # Reuse the mask Inputspace resolved (caller mask → C++ mask → all-valid
        # fallback, plus the groundstation unmask guard for fully-masked rows).
        masks = resolved_mask.clone()
        if not getattr(self, 'allow_groundstation', True):
            masks[:, 0] = True
            # After forcibly masking groundstation, re-check fully-masked rows.
            all_masked = masks.all(dim=-1)
            if all_masked.any():
                masks[all_masked, 0] = False

        return base_logits, masks

    def forward(self, states, masks=None, actions_idx=None, deterministic=False):
        """Autoregressive decoding over drones.

        Args:
            states: observation dict
            masks: (B, F) bool mask for invalid fires
            actions_idx: (B, N) pre-determined action indices (evaluate mode).
                         If None, actions are sampled (or argmaxed if deterministic).
            deterministic: if True and actions_idx is None, use argmax instead of sampling.

        Returns:
            actions_idx: (B, N) fire index per drone
            log_probs: (B, N) per-drone log probabilities
            entropy: (B, N) per-drone entropy
        """
        base_logits, masks = self._compute_base_logits(states, masks)
        B, N, F = base_logits.shape
        device = base_logits.device

        # Symmetric bound — unbounded exp() otherwise lets the penalty diverge
        # into numerical overflow when combined with -inf masking.
        assignment_penalty = self.log_assignment_penalty.clamp(-3.0, 3.0).exp()
        assignment_counts = torch.zeros(B, F, device=device)

        all_actions = torch.zeros(B, N, dtype=torch.long, device=device)
        all_log_probs = torch.zeros(B, N, device=device)
        all_entropy = torch.zeros(B, N, device=device)

        # Running mask of already-assigned fires. Groundstation (index 0) is
        # reusable across drones — do not add it to the hard-mask so multiple
        # drones can recharge. Non-groundstation fires are hard-masked after
        # selection so sample() and argmax() both respect pointer semantics.
        used_mask = torch.zeros(B, F, dtype=torch.bool, device=device)

        for i in range(N):
            logits_i = base_logits[:, i, :]  # (B, F)
            # Soft-penalize by count (keeps gradient smooth during PPO update)
            logits_i = logits_i - assignment_penalty * assignment_counts
            # Hard-mask invalid fires AND previously-assigned non-groundstation fires
            effective_mask = masks | used_mask
            # Re-guarantee groundstation fallback if everything is masked
            all_masked = effective_mask.all(dim=-1)
            if all_masked.any():
                effective_mask = effective_mask.clone()
                effective_mask[all_masked, 0] = False
            logits_i = logits_i.masked_fill(effective_mask, float('-inf'))

            dist_i = torch.distributions.Categorical(logits=logits_i)

            if actions_idx is not None:
                a_i = actions_idx[:, i]
            elif deterministic:
                a_i = logits_i.argmax(dim=-1)
            else:
                a_i = dist_i.sample()

            all_actions[:, i] = a_i
            all_log_probs[:, i] = dist_i.log_prob(a_i)
            all_entropy[:, i] = dist_i.entropy()

            # Update soft-penalty counts — skip groundstation so multiple drones may recharge
            # simultaneously without accumulating the duplicate-assignment prior. Matches the
            # reusable-groundstation semantics in used_mask below.
            non_gs = a_i != 0
            assignment_counts = assignment_counts.scatter_add(
                1, a_i.unsqueeze(1), non_gs.float().unsqueeze(1)
            )
            # Hard-mask non-groundstation assignments for subsequent drones
            if non_gs.any():
                used_mask = used_mask.clone()
                rows = torch.nonzero(non_gs, as_tuple=False).squeeze(-1)
                used_mask[rows, a_i[non_gs]] = True

        # Diagnostic: non-GS actions should be unique per batch row (hard-mask invariant).
        # Warn once per process if violated — the C++ SameGoalPenalty fires every step, and
        # we want to know whether the decoder is at fault or if duplicates arise in the
        # Python→C++ handoff or the coord mapping.
        if not getattr(PointerActor, "_uniqueness_warned", False):
            with torch.no_grad():
                # Mask GS picks with -1 so they don't count as duplicates among themselves.
                masked = torch.where(all_actions == 0, torch.full_like(all_actions, -1), all_actions)
                for b in range(B):
                    row = masked[b]
                    non_gs_picks = row[row >= 0]
                    if non_gs_picks.numel() != torch.unique(non_gs_picks).numel():
                        import sys
                        print(f"[PointerActor][WARN] duplicate non-GS action indices in batch {b}: "
                              f"{all_actions[b].tolist()} (actions_idx={'provided' if actions_idx is not None else 'sampled'})",
                              file=sys.stderr, flush=True)
                        PointerActor._uniqueness_warned = True
                        break

        return all_actions, all_log_probs, all_entropy

class Critic(nn.Module):
    def __init__(self, vision_range, drone_count, map_size, time_steps, inputspace=None):
        super().__init__()
        self.Inputspace_1 = Inputspace(drone_count, time_steps=time_steps) if not inputspace else inputspace
        self.in_features = self.Inputspace_1.out_features

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward.")

class PointerCritic(nn.Module):
    def __init__(self, vision_range, drone_count, map_size, time_steps, inputspace=None):
        super().__init__()
        self.Inputspace_1 = Inputspace(drone_count, time_steps=time_steps) if not inputspace else inputspace
        hidden_dim = self.Inputspace_1.out_features
        # Attention pooling: learnable query attends over drone representations
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pool_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.value_head[-1]._init_gain = 1.0

    def forward(self, states, masks=None):
        drone_repr, _, _ = self.Inputspace_1(states, masks)  # (B, N, 64)
        pooled, _ = self.pool_attn(
            self.pool_query.expand(drone_repr.shape[0], -1, -1),
            drone_repr, drone_repr
        )
        pooled = pooled.squeeze(1)  # (B, 64)
        return self.value_head(pooled)    # (B, 1)

class PointerOffPolicyCritic(Critic):
    def __init__(self, vision_range, drone_count, map_size, time_steps, action_dim, inputspace=None):
        super().__init__(vision_range, drone_count, map_size, time_steps, inputspace)
        self.Inputspace_2 = Inputspace(drone_count, time_steps=time_steps) if not inputspace else inputspace
        hidden_dim = self.in_features

        # Attention pooling for Q1 and Q2
        self.pool_query_1 = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pool_attn_1 = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.pool_query_2 = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pool_attn_2 = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

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
        pooled, _ = self.pool_attn_1(
            self.pool_query_1.expand(x.shape[0], -1, -1), x, x
        )
        x = pooled.squeeze(1)  # (B, D)
        x = torch.cat([x, action], dim=1)

        q1 = F.relu(self.l1(x))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

    def Q2(self, state, action):
        x, _, _ = self.Inputspace_2(state)
        pooled, _ = self.pool_attn_2(
            self.pool_query_2.expand(x.shape[0], -1, -1), x, x
        )
        x = pooled.squeeze(1)  # (B, D)
        x = torch.cat([x, action], dim=1)

        q2 = F.relu(self.l4(x))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q2

class PointerValue(nn.Module):
    def __init__(self, vision_range, drone_count, map_size, time_steps, inputspace):
        super().__init__()
        self.Inputspace = Inputspace(drone_count, time_steps=time_steps) if not inputspace else inputspace
        self.in_features = self.Inputspace.out_features
        hidden_dim = self.in_features

        # Attention pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pool_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        # Simple MLP head for value estimation
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.v_value = nn.Linear(256, 1)
        self.v_value._init_gain = 1.0

    def forward(self, state):
        x, _, _ = self.Inputspace(state)
        pooled, _ = self.pool_attn(
            self.pool_query.expand(x.shape[0], -1, -1), x, x
        )
        x = pooled.squeeze(1)  # (B, D)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        v = self.v_value(x)
        return v
