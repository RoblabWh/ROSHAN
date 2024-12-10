from networks.network_explore import ActorCritic
import numpy as np
import firesim

class ExploreAgent:
    def __init__(self):
        self.name = "ExploreAgent"

    @staticmethod
    def get_network():
        return ActorCritic

    @staticmethod
    def get_action(actions):
        drone_actions = []
        for activation in actions:
            drone_actions.append(
                firesim.DroneAction(activation[0], activation[1]))
        return drone_actions

    @staticmethod
    def restructure_data(observations_):
        all_drone_views, all_velocities, all_maps, all_fires, all_positions, all_water_dispense = [], [], [], [], [], []

        for deque in observations_:
            drone_states = np.array([state for state in deque if isinstance(state, firesim.DroneState)])
            if len(drone_states) == 0:
                continue

            # drone_view = np.array([state.GetDroneViewNorm() for state in drone_states])
            drone_view = np.array([state.GetFireStatus() for state in drone_states])
            velocities = np.array([state.GetVelocityNorm() for state in drone_states])
            maps = np.array([state.GetExplorationMapNorm() for state in drone_states])
            fire_map = np.array([state.GetFireMap() for state in drone_states])
            positions = np.array([state.GetPositionNorm() for state in drone_states])
            water_dispense = np.array([state.GetWaterDispense() for state in drone_states])

            all_drone_views.append(drone_view)
            all_velocities.append(velocities)
            all_maps.append(maps)
            all_fires.append(fire_map)
            all_positions.append(positions)
            all_water_dispense.append(water_dispense)

        return np.array(all_drone_views), np.array(all_maps), np.array(all_velocities), np.array(
            all_positions), np.array(all_water_dispense), np.array(all_fires)

    def restructure_data1(self, observations_):
        all_velocities, all_positions, all_goals = [], [], []

        for deque in observations_:
            drone_states = np.array([state for state in deque if isinstance(state, firesim.DroneState)])
            if len(drone_states) == 0:
                continue

            velocities = np.array([state.GetVelocityNorm() for state in drone_states])
            positions = np.array([state.GetGridPositionDoubleNorm() for state in drone_states])
            goals = np.array([state.GetGoalPositionNorm() for state in drone_states])

            all_velocities.append(velocities)
            all_positions.append(positions)
            all_goals.append(goals)

        return np.array(all_velocities), np.array(all_positions), np.array(all_goals)