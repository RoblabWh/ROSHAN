from networks.network_fly import Inputspace, Actor, Critic
import numpy as np
import firesim

class FlyAgent:
    def __init__(self):
        self.name = "FlyAgent"
        self.hierachy_level = "low"

    def get_hierachy_level(self):
        return self.hierachy_level

    @staticmethod
    def get_network():
        return Actor, Critic

    @staticmethod
    def get_action(actions):
        drone_actions = []
        for activation in actions:
            drone_actions.append(
                firesim.DroneAction(activation[0], activation[1]))
        return drone_actions

    @staticmethod
    def restructure_data(observations_):
        all_velocities, all_delta_goals = [], []

        for deque in observations_:
            drone_states = np.array([state for state in deque if isinstance(state, firesim.DroneState)])
            if len(drone_states) == 0:
                continue

            velocities = np.array([state.GetVelocityNorm() for state in drone_states])
            delta_goal = np.array([state.GetDeltaGoal() for state in drone_states])

            all_velocities.append(velocities)
            all_delta_goals.append(delta_goal)

        # Expanding Dims necessary for scalar values
        return np.array(all_velocities), np.array(all_delta_goals)