from networks.network_fly import ActorCritic
import numpy as np
import firesim

class FlyAgent:
    def __init__(self):
        self.name = "FlyAgent"

    def get_network(self):
        return ActorCritic

    def get_action(self, actions):
        drone_actions = []
        for activation in actions:
            drone_actions.append(
                firesim.DroneAction(activation[0], activation[1]))
        return drone_actions

    def restructure_data(self, observations_):
        all_velocities, all_delta_goals, all_outside_areas = [], [], []

        for deque in observations_:
            drone_states = np.array([state for state in deque if isinstance(state, firesim.DroneState)])
            if len(drone_states) == 0:
                continue

            velocities = np.array([state.GetVelocityNorm() for state in drone_states])
            delta_goal = np.array([state.GetDeltaGoal() for state in drone_states])
            outside_area = np.array([state.GetOutsideAreaCounter() for state in drone_states])

            all_velocities.append(velocities)
            all_delta_goals.append(delta_goal)
            all_outside_areas.append(np.array(outside_area))

        # Expanding Dims necessary for scalar values
        return np.array(all_velocities), np.array(all_delta_goals)