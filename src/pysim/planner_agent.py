from networks.network_planner import PointerActor, PointerCritic, PointerOffPolicyCritic, PointerValue
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
import torch
import firesim
from agent import Agent
from utils import get_device

# Per-drone goal-commitment thresholds (in normalized [-1, 1] coord space).
# AT_GOAL: drone counts as "arrived" when within ~1 cell (2/max_dim ≈ 0.025 for an
# 80-cell map; doubled for slack). MATCH: prev_goal coordinate is considered to
# match a current fire when within float-precision distance.
_LOCK_AT_GOAL_EPS = 0.05
_LOCK_MATCH_EPS = 1e-3


class PlannerAgent(Agent):
    def __init__(self, num_drones):
        super().__init__()
        self.name = "planner_agent"
        self.short_name = "planner"
        self.hierarchy_level = "high"
        self.low_level_steps = 200
        self.use_intrinsic_reward = False
        self.rnd_model = None
        self.optimizer = None
        self.MSE_loss = nn.MSELoss()
        self.action_dim = (num_drones, 2)
        self.num_drones = num_drones
        # Water feasibility threshold (fraction in [0, 1]). When a drone's tank is
        # below this, the pointer decoder masks all non-groundstation targets and the
        # commitment lock releases, so the planner is forced to route it to refuel.
        # 0.0 = disabled (only truly-empty drones affected). Set from config in
        # agent_builder for planner_agent; gated on use_water_limit.
        self.water_refuel_threshold = 0.0
        # Per-drone goal-commitment state. False on the very first decision after
        # ResetEnvironment so the planner can actually assign initial goals (FlyAgent
        # spawns are independent of initial goal coordinates — see rl_handler.cpp:225-237).
        self._first_decision_done = False
        # Eval-only heuristic baseline instead of the pointer network. Set from config
        # in agent_builder; heuristic_method picks the assignment rule
        # ("greedy" | "hungarian", see greedy_actions / hungarian_actions).
        self.heuristic_goals = False
        self.heuristic_method = "greedy"

    def get_num_agents(self, num_agents):
        return 1

    def notify_episode_reset(self):
        """Called by agent_handler at episode boundaries (alongside fsc.reset()).
        Resets per-episode commitment state so the first decision after reset
        honors the planner's output regardless of distance-to-prev-goal.
        """
        self._first_decision_done = False

    def apply_commitment(self, state, sampled_actions):
        """Per-drone goal-commitment override.

        For each drone i, if its previous goal is still valid (matches a current
        fire/groundstation coordinate) AND the drone has not yet arrived at it,
        override the planner's sampled action with the previous goal coordinate.
        Drones whose previous goal is invalid (fire extinguished) or who have
        arrived are left free for the planner to redirect.

        On the very first decision after Reset, all drones are forced free
        (locked = False) so initial goals can be assigned.

        Parameters
        ----------
        state : dict
            Observation dict (B=1 at action time). Required keys:
            ``drone_positions`` (..., N, 2), ``goal_positions`` (..., N, 2),
            ``fire_positions`` (..., K, 2). Last timestep is selected.
        sampled_actions : np.ndarray
            Shape (1, N, 2) — coordinates returned by the actor (post _idx_to_coords).

        Returns
        -------
        overridden_actions : np.ndarray
            Same shape as sampled_actions; locked drones replaced with prev_goal coords.
        locked_mask : np.ndarray
            Shape (1, N) bool — True where the drone is locked (action overridden).
        """
        sampled = np.asarray(sampled_actions, dtype=np.float32)  # (1, N, 2)

        # Pull last-timestep slices from the observation. Mirrors Inputspace._select_last_timestep
        # (network_planner.py): collapse 4D/5D state arrays down to (B, X, 2) by taking the
        # last entry of any time dimension.
        def _last(arr):
            a = np.asarray(arr)
            while a.ndim > 3:
                a = a[:, -1]
            return a  # (B, X, 2)

        drone_pos = _last(state["drone_positions"])     # (1, N, 2)
        prev_goal = _last(state["goal_positions"])      # (1, N, 2)
        fire_pos = _last(state["fire_positions"])       # (1, K, 2)

        # First decision of the episode: honor everything, set the flag, return early.
        if not self._first_decision_done:
            self._first_decision_done = True
            locked_mask = np.zeros(prev_goal.shape[:2], dtype=bool)  # (1, N)
            return sampled, locked_mask

        # at_goal_i: drone i is within EPS_AT_GOAL of its prev_goal coord
        at_goal = np.linalg.norm(drone_pos - prev_goal, axis=-1) < _LOCK_AT_GOAL_EPS  # (1, N)

        # prev_goal_valid_i: prev_goal coord matches some current fire (incl. GS at idx 0)
        # diff: (1, N, K, 2) → distances (1, N, K) → min over K (1, N)
        diff = prev_goal[:, :, None, :] - fire_pos[:, None, :, :]
        nearest_dist = np.linalg.norm(diff, axis=-1).min(axis=-1)  # (1, N)
        prev_goal_valid = nearest_dist < _LOCK_MATCH_EPS

        locked_mask = (~at_goal) & prev_goal_valid  # (1, N)

        # Water-feasibility release: if a drone is below the refuel threshold and its
        # previous goal is a *fire* (not the groundstation), free it so the planner —
        # whose pointer decoder now masks all non-GS targets for low-water drones — can
        # redirect it to refuel. Without this, the override would re-inject the old fire
        # coordinate, which the actor's water mask forbids, yielding -inf log-probs /
        # NaN when PPO re-evaluates the stored action. Drones already headed to the GS
        # (prev_goal == GS) stay committed so they finish the refuel trip.
        threshold = float(getattr(self, "water_refuel_threshold", 0.0))
        if threshold > 0.0 and "drone_water" in state:
            water = _last(state["drone_water"])              # (1, N, 1) or (1, N)
            water = np.asarray(water, dtype=np.float32)
            if water.ndim == 3:
                water = water[..., 0]                        # (1, N)
            low_water = water < threshold                    # (1, N)
            gs_coord = fire_pos[:, 0:1, :]                   # (1, 1, 2) — GS is index 0
            prev_is_gs = np.linalg.norm(prev_goal - gs_coord, axis=-1) < _LOCK_MATCH_EPS
            release = low_water & (~prev_is_gs)              # (1, N)
            locked_mask = locked_mask & (~release)

        # Override locked drones' coords with prev_goal. Free drones keep sampled.
        overridden = np.where(locked_mask[..., None], prev_goal, sampled)
        return overridden.astype(np.float32, copy=False), locked_mask

    def _baseline_setup(self, state):
        """Shared preamble of the heuristic baselines (eval-only).

        Returns ``(drone_pos (N,2), fire_pos (K,2), free, fires, actions (N,2))``:
        ``free`` = drones not forced to refuel, ``fires`` = valid, unassigned fire
        indices (groundstation index 0 and fires held by committed drones excluded,
        mirroring the pointer decoder's commitment seed), ``actions`` = every drone
        defaulted to the groundstation. Reads the same observation dict as the
        network, so eval_ground_truth_fires applies identically.
        """
        def _last(arr):
            a = np.asarray(arr)
            while a.ndim > 3:
                a = a[:, -1]
            return a

        drone_pos = _last(state["drone_positions"]).astype(np.float32)[0]  # (N, 2)
        fire_pos = _last(state["fire_positions"]).astype(np.float32)[0]    # (K, 2)
        n = drone_pos.shape[0]
        gs = fire_pos[0]

        valid = np.ones(fire_pos.shape[0], dtype=bool)
        if "fire_positions_mask" in state:
            valid = np.asarray(_last(state["fire_positions_mask"])).reshape(-1) > 0.5
        valid[0] = False  # GS is the fallback, not a fire target

        # Exclude fires a committed drone is already flying to (same lock rule as
        # apply_commitment: not yet at prev_goal). One-step over-reservation on the
        # first decision after reset is harmless — mirrors _commitment_seed.
        prev_goal = _last(state["goal_positions"]).astype(np.float32)[0]  # (N, 2)
        at_goal = np.linalg.norm(drone_pos - prev_goal, axis=-1) < _LOCK_AT_GOAL_EPS
        for i in range(n):
            if not at_goal[i]:
                taken = np.linalg.norm(fire_pos - prev_goal[i], axis=-1) < _LOCK_MATCH_EPS
                valid &= ~taken

        threshold = float(getattr(self, "water_refuel_threshold", 0.0))
        needs_gs = np.zeros(n, dtype=bool)
        if threshold > 0.0 and "drone_water" in state:
            water = _last(state["drone_water"]).astype(np.float32)
            if water.ndim == 3:
                water = water[..., 0]
            needs_gs = water.reshape(-1) < threshold

        actions = np.tile(gs, (n, 1)).astype(np.float32)  # default: groundstation
        free = [i for i in range(n) if not needs_gs[i]]
        fires = list(np.flatnonzero(valid))
        return drone_pos, fire_pos, free, fires, actions

    def greedy_actions(self, state):
        """Heuristic baseline: greedy nearest-pair goal assignment (eval-only).

        Same machinery as the learned planner everywhere else — FlyAgent network,
        commitment lock, event replans, water release — only the assignment rule
        differs: closest (free drone, unassigned fire) pairs are matched first;
        low-water drones and left-over drones go to the groundstation (index 0).

        Returns coords (1, N, 2) float32 — same contract as act_certain.
        """
        drone_pos, fire_pos, free, fires, actions = self._baseline_setup(state)
        if free and fires:
            d = np.linalg.norm(drone_pos[free][:, None, :] - fire_pos[fires][None, :, :], axis=-1)
            while free and fires:
                r, c = np.unravel_index(np.argmin(d), d.shape)
                actions[free[r]] = fire_pos[fires[c]]
                free.pop(r)
                fires.pop(c)
                d = np.delete(np.delete(d, r, axis=0), c, axis=1)
        return actions[None, ...]  # (1, N, 2)

    def hungarian_actions(self, state):
        """Heuristic baseline: minimum-total-distance assignment (eval-only).

        Same preamble and fallbacks as greedy_actions; the (free drone, fire)
        matching minimises the summed Euclidean distance with the Hungarian
        algorithm instead of nearest-pair peeling. Surplus drones (more drones
        than fires) stay at the groundstation, like greedy.
        """
        drone_pos, fire_pos, free, fires, actions = self._baseline_setup(state)
        if free and fires:
            d = np.linalg.norm(drone_pos[free][:, None, :] - fire_pos[fires][None, :, :], axis=-1)
            rows, cols = linear_sum_assignment(d)
            for r, c in zip(rows, cols):
                actions[free[r]] = fire_pos[fires[c]]
        return actions[None, ...]  # (1, N, 2)

    @staticmethod
    def get_network(algorithm : str):
        if algorithm == "PPO":
            return PointerActor, PointerCritic
        elif algorithm == "IQL":
            return PointerActor, PointerOffPolicyCritic, PointerValue
        # elif algorithm == "TD3":
        #     return DeterministicActor, PointerOffPolicyCritic
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def get_module_names(self, algorithm_name: str):
        if algorithm_name == "PPO":
            return "PointerActor", "PointerCritic"
        elif algorithm_name == "IQL":
            return "PointerActor", "PointerOffPolicyCritic", "PointerValue"
        elif algorithm_name == "TD3":
            return None, None
        elif algorithm_name == "no_algo":
            return None, None
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

    def initialize_rnd_model(self, vision_range, drone_count, map_size, time_steps, lr=1e-4, betas=(0.9, 0.999)):
        device = get_device()
        self.rnd_model = RNDModel(vision_range, drone_count, map_size, time_steps).to(device)
        self.optimizer = torch.optim.Adam(self.rnd_model.parameters(), lr=lr, betas=betas, eps=1e-5)

    def get_intrinsic_reward(self, obs):
        return self.rnd_model.get_intrinsic_reward(obs)

    def update_rnd_model(self, memory, horizon, mini_batch_size):
        t_dict = memory.to_tensor()
        states = t_dict['state']
        states = memory.rearrange_states(states)

        for index in BatchSampler(SubsetRandomSampler(range(horizon)), mini_batch_size, True):
            batch_states = tuple(state[index] for state in states)
            tgt_features = self.rnd_model.target(batch_states)
            pred_features = self.rnd_model.predictor(batch_states)
            rnd_loss = self.MSE_loss(pred_features, tgt_features)
            self.optimizer.zero_grad()
            rnd_loss.backward()
            self.optimizer.step()

    @staticmethod
    def get_action(actions):
        drone_actions = []
        for activation in actions:
            drone_actions.append(
                firesim.PlanAction(activation))
        return drone_actions

