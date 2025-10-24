from networks.network_planner import AttentionActor, CriticPPO, OffPolicyCritic, Value
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
import torch.nn as nn
import torch
import firesim
from agent import Agent

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

    def get_num_agents(self, num_agents):
        return 1

    @staticmethod
    def get_network(algorithm : str):
        if algorithm == "PPO":
            return AttentionActor, CriticPPO
        elif algorithm == "IQL":
            return AttentionActor, OffPolicyCritic, Value
        # elif algorithm == "TD3":
        #     return DeterministicActor, OffPolicyCritic
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
                firesim.PlanAction(activation))
        return drone_actions

    @staticmethod
    def restructure_data(observations_):
        obs = observations_["planner_agent"]

        drone_state_groups = [
            [state for state in deque if isinstance(state, firesim.AgentState)]
            for deque in obs
        ]
        drone_state_groups = [group for group in drone_state_groups if group]

        if not drone_state_groups:
            raise ValueError("No AgentState data found in observations for planner_agent")

        drone_positions = [[s.GetDronePositions() for s in group] for group in drone_state_groups]
        goal_positions = [[s.GetGoalPositions() for s in group] for group in drone_state_groups]
        fire_positions = [[s.GetFirePositions() for s in group] for group in drone_state_groups]

        all_drone_states = np.stack(drone_positions)
        all_goal_positions = np.stack(goal_positions)
        all_fire_states = np.stack(fire_positions)

        return all_drone_states, all_goal_positions, all_fire_states
