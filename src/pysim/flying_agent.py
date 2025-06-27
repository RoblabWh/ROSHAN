from networks.network_fly import Actor, CriticPPO, OffPolicyCritic, DeterministicActor, Value
import numpy as np
import firesim
from agent import Agent

class FlyAgent(Agent):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.hierachy_level = "low"
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
        all_velocities, all_delta_goals = [], []
        obs = observations_[self.name]
        for deque in obs:
            drone_states = np.array([state for state in deque if isinstance(state, firesim.AgentState)])
            if len(drone_states) == 0:
                continue

            velocities = np.array([state.GetVelocityNorm() for state in drone_states])
            delta_goal = np.array([state.GetDeltaGoal() for state in drone_states])

            all_velocities.append(velocities)
            all_delta_goals.append(delta_goal)

        # Expanding Dims necessary for scalar values
        return np.array(all_velocities), np.array(all_delta_goals)