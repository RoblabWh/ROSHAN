from networks.network_planner_attn import PointerActor, PointerCritic, PointerOffPolicyCritic, PointerValue
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
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
        # Per-drone goal-commitment state. False on the very first decision after
        # ResetEnvironment so the planner can actually assign initial goals (FlyAgent
        # spawns are independent of initial goal coordinates — see rl_handler.cpp:225-237).
        self._first_decision_done = False

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

        # Override locked drones' coords with prev_goal. Free drones keep sampled.
        overridden = np.where(locked_mask[..., None], prev_goal, sampled)
        return overridden.astype(np.float32, copy=False), locked_mask

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

