from networks.network_explore import Actor, CriticPPO, OffPolicyCritic, DeterministicActor, RNDModel, Value
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
import torch.nn as nn
import torch
import firesim

class ExploreAgent:
    def __init__(self):
        self.name = "ExploreAgent"
        self.hierachy_level = "medium"
        self.low_level_steps = 200
        self.use_intrinsic_reward = False
        self.rnd_model = None
        self.optimizer = None
        self.MSE_loss = nn.MSELoss()
        self.action_dim = 2

    def get_hierachy_level(self):
        return self.hierachy_level

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

    def initialize_rnd_model(self, vision_range, drone_count, map_size, time_steps, lr=1e-4, betas=(0.9, 0.999)):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
                firesim.ExploreAction(activation[0], activation[1]))
        return drone_actions

    @staticmethod
    def restructure_data(observations_):
        all_explore_maps, all_total_views = [], []

        obs = observations_["ExploreAgent"]
        for deque in obs:
            drone_states = np.array([state for state in deque if isinstance(state, firesim.DroneState)])
            if len(drone_states) == 0:
                continue

            multi_total_drone_view = np.array([state.GetMultipleTotalDroneView() for state in drone_states])
            exploration_map = np.array([state.GetExplorationMapNorm() for state in drone_states])

            all_explore_maps.append(exploration_map)
            all_total_views.append(multi_total_drone_view)

        aemaps = np.array(all_explore_maps)
        return np.transpose(np.array(all_total_views), (0,2,1,3,4)), aemaps