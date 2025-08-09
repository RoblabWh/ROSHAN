from networks.network_fly import Actor, CriticPPO, OffPolicyCritic, DeterministicActor, Value
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
            return Actor, CriticPPO
        elif algorithm == "IQL":
            return Actor, OffPolicyCritic, Value
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

        velocities = [[state.GetVelocityNorm() for state in group] for group in drone_state_groups]
        delta_goals = [[state.GetDeltaGoal() for state in group] for group in drone_state_groups]

        all_velocities = np.stack(velocities)
        all_delta_goals = np.stack(delta_goals)

        return all_velocities, all_delta_goals