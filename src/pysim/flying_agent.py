from networks.network_fly import StochasticActor, CriticPPO, OffPolicyCritic, DeterministicActor, Value
import numpy as np
import firesim
from agent import Agent

class FlyAgent(Agent):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.short_name = "fly"
        self.hierarchy_level = "low"
        self.use_intrinsic_reward = False
        self.action_dim = 2

    @staticmethod
    def get_network(algorithm : str):
        if algorithm == "PPO":
            return StochasticActor, CriticPPO
        elif algorithm == "IQL":
            return StochasticActor, OffPolicyCritic, Value
        elif algorithm == "TD3":
            return DeterministicActor, OffPolicyCritic
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")


    @staticmethod
    def get_action(actions):
        drone_actions = []
        for activation in actions:
            drone_actions.append(
                firesim.DroneAction(activation[0], activation[1]))
        return drone_actions

    def restructure_data(self, observations_):
        obs = observations_[self.name]

        drone_state_groups = [
            [state for state in deque if isinstance(state, firesim.AgentState)]
            for deque in obs
        ]
        drone_state_groups = [group for group in drone_state_groups if group]

        if not drone_state_groups:
            raise ValueError(f"No AgentState data found in observations for {self.name}")

        # Single-pass extraction: build 8 lists of lists directly without intermediate tuples
        num_groups = len(drone_state_groups)
        velocities = [None] * num_groups
        delta_goals = [None] * num_groups
        cos_sin_to_goal = [None] * num_groups
        speed = [None] * num_groups
        distance_to_goal = [None] * num_groups
        distances_to_others = [None] * num_groups
        distances_mask = [None] * num_groups
        self_id = [None] * num_groups

        for g_idx, group in enumerate(drone_state_groups):
            g_len = len(group)
            g_vel = [None] * g_len
            g_dg = [None] * g_len
            g_cs = [None] * g_len
            g_sp = [None] * g_len
            g_dtg = [None] * g_len
            g_dto = [None] * g_len
            g_dm = [None] * g_len
            g_id = [None] * g_len
            for s_idx, state in enumerate(group):
                g_vel[s_idx] = state.GetVelocityNorm()
                g_dg[s_idx] = state.GetDeltaGoal()
                g_cs[s_idx] = state.GetCosSinToGoal()
                g_sp[s_idx] = state.GetSpeed()
                g_dtg[s_idx] = state.GetDistanceToGoal()
                g_dto[s_idx] = state.GetDistancesToOtherAgents()
                g_dm[s_idx] = state.GetDistancesMask()
                g_id[s_idx] = state.GetID()
            velocities[g_idx] = g_vel
            delta_goals[g_idx] = g_dg
            cos_sin_to_goal[g_idx] = g_cs
            speed[g_idx] = g_sp
            distance_to_goal[g_idx] = g_dtg
            distances_to_others[g_idx] = g_dto
            distances_mask[g_idx] = g_dm
            self_id[g_idx] = g_id

        all_velocities = np.stack(velocities)
        all_delta_goals = np.stack(delta_goals)
        all_cos_sin_to_goal = np.stack(cos_sin_to_goal)
        all_speed = np.stack(speed)
        all_distance_to_goal = np.stack(distance_to_goal)
        all_distances_to_others = np.stack(distances_to_others)
        all_distances_mask = np.stack(distances_mask)
        all_ids = np.stack(self_id)

        return all_ids, all_velocities, all_delta_goals, all_cos_sin_to_goal, all_speed, all_distance_to_goal, all_distances_to_others, all_distances_mask