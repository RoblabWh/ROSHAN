from networks.network_explore import Inputspace, Actor, Critic
import numpy as np
import firesim

class ExploreAgent:
    def __init__(self):
        self.name = "ExploreAgent"
        self.hierachy_level = "medium"
        self.low_level_steps = 200

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
                firesim.ExploreAction(activation[0], activation[1]))
        return drone_actions

    @staticmethod
    def restructure_data(observations_):
        all_explore_maps, all_fire_maps, all_positions = [], [], []

        for deque in observations_:
            drone_states = np.array([state for state in deque if isinstance(state, firesim.DroneState)])
            if len(drone_states) == 0:
                continue

            exploration_map = np.array([state.GetExplorationMapNorm() for state in drone_states])
            fire_map = np.array([state.GetFireMap() for state in drone_states])
            position = np.array([state.GetGridPositionDoubleNorm() for state in drone_states])

            all_explore_maps.append(exploration_map)
            all_fire_maps.append(fire_map)
            all_positions.append(position)

        return np.array(all_explore_maps), np.array(all_fire_maps), np.array(all_positions)