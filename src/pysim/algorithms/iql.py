import torch
import torch.nn as nn
import os
import warnings
import numpy as np
from algorithms.actor_critic import ActorCriticIQL  # you will need to slightly adjust this
from algorithms.rl_algorithm import RLAlgorithm
from algorithms.rl_config import IQLConfig
from memory import SwarmMemory
from utils import RunningMeanStd
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#from https://github.com/Manchery/iql-pytorch?tab=readme-ov-file

class IQL(RLAlgorithm):
    def __init__(self, network, config: IQLConfig):

        super().__init__(config)
        for field in config.__dataclass_fields__:
            setattr(self, field, getattr(config, field))

        self.actor = network[0]
        self.critic = network[1]
        self.value = network[2]

        self.policy = ActorCriticIQL(Actor=self.actor, Critic=self.critic, Value=self.value, action_dim=self.action_dim,
                                  vision_range=self.vision_range, drone_count=self.drone_count,
                                  map_size=self.map_size, time_steps=self.time_steps)

        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=self.lr, betas=self.betas, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.policy.critic.parameters(), lr=self.lr, betas=self.betas, eps=1e-5)
        self.value_optimizer = torch.optim.Adam(self.policy.value.parameters(), lr=self.lr, betas=self.betas, eps=1e-5)

        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, T_max=int(1e6))


        self.MSE_loss = nn.MSELoss()
        self.device = device

        self.use_next_obs = True

        # Target Q network
        self.critic_target = copy.deepcopy(self.policy.critic)

        self.set_train()

    def set_train(self):
        self.policy.train()

    def set_eval(self):
        self.policy.eval()

    def save(self):
        torch.save(self.policy.state_dict(), f'{os.path.join(self.model_path, self.model_name)}')

    def load(self):
        path = os.path.join(self.model_path, self.model_name)
        try:
            self.policy.load_state_dict(torch.load(f'{path}', map_location=self.device))
            self.policy.to(self.device)
            return True
        except FileNotFoundError:
            warnings.warn(f"Could not load model from {path}. Falling back to train mode.")
            return False

    def select_action(self, observations):
        return self.policy.act(observations)

    def select_action_certain(self, observations):
        """
        Select action with certain policy.
        """
        return self.policy.act_certain(observations)

    def loss(self, diff, expectile=0.8):
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def update_v(self, states, actions, logger=None):
        with torch.no_grad():
            q1, q2 = self.critic_target(states, actions)
            q = torch.minimum(q1, q2).detach()

        v = self.policy.value(states)
        value_loss = self.loss(q - v, self.expectile).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # logger.log('train/value_loss', value_loss, self.total_it)
        # logger.log('train/v', v.mean(), self.total_it)

    def update_q(self, states, actions, rewards, next_states, not_dones, logger=None):
        with torch.no_grad():
            next_v = self.policy.value(next_states)
            target_q = (rewards + self.discount * not_dones * next_v).detach()

        q1, q2 = self.policy.critic(states, actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # logger.log('train/critic_loss', critic_loss, self.total_it)
        # logger.log('train/q1', q1.mean(), self.total_it)
        # logger.log('train/q2', q2.mean(), self.total_it)

    def update_target(self):
        for param, target_param in zip(self.policy.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_actor(self, states, actions, logger=None):
        with torch.no_grad():
            v = self.policy.value(states)
            q1, q2 = self.critic_target(states, actions)
            q = torch.minimum(q1, q2)
            exp_a = torch.exp((q - v) * self.temperature)
            exp_a = torch.clamp(exp_a, max=100.0).squeeze(-1).detach()

        mu, _ = self.policy.actor(states)
        actor_loss = (exp_a.unsqueeze(-1) * ((mu - actions)**2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_scheduler.step()

        # logger.log('train/actor_loss', actor_loss, self.total_it)
        # logger.log('train/adv', (q - v).mean(), self.total_it)

    def update(self, memory: SwarmMemory, horizon, batch_size, n_steps, next_obs, logger):

        for _ in range(self.k_epochs):
            batch = memory.sample_batch(batch_size)

            states = batch['state']
            actions = batch['action']
            rewards = batch['reward']
            next_states = batch['next_state']
            not_dones = batch['not_done']

            # If your sampled states are still tuples, rearrange
            states = memory.rearrange_states(states)
            next_states = memory.rearrange_states(next_states)
            actions = memory.rearrange_tensor(actions)
            rewards = memory.rearrange_tensor(rewards)
            not_dones = memory.rearrange_tensor(not_dones)

            rewards = torch.clip(rewards * 1000, -5000, 5000)

            # Update
            self.update_v(states, actions, logger)
            self.update_actor(states, actions, logger)
            self.update_q(states, actions, rewards, next_states, not_dones, logger)
            self.update_target()

        return ""

    # def update(self, memory: SwarmMemory, horizon, batch_size, n_steps, next_obs, logger):
    #     """
    #     Update step for IQL.
    #     """
    #     console = ""
    #     for _ in range(self.k_epochs):
    #         batch = memory.sample_batch(batch_size)
    #
    #         states = batch['state']
    #         actions = batch['action']
    #         rewards = batch['reward']
    #         next_states = batch['next_state']
    #
    #         # If your sampled states are still tuples, rearrange
    #         states = memory.rearrange_states(states)
    #         next_states = memory.rearrange_states(next_states)
    #         actions = memory.rearrange_tensor(actions)
    #         rewards = memory.rearrange_tensor(rewards)
    #
    #         # Critic Update
    #         with torch.no_grad():
    #             next_v = self.policy.value(next_states)
    #             q_target = rewards + self.gamma * next_v.squeeze()
    #
    #         q_values = self.policy.critic(states, actions).squeeze()
    #         critic_loss = self.MSE_loss(q_values, q_target)
    #         self.optimizer_critic.zero_grad()
    #         critic_loss.backward()
    #         self.optimizer_critic.step()
    #
    #         # Value Update
    #         with torch.no_grad():
    #             q_estimate = self.policy.critic(states, actions)
    #         v_values = self.policy.value(states)
    #         v_target = torch.min(q_estimate, rewards.unsqueeze(1))
    #         value_loss = self.MSE_loss(v_values, v_target)
    #         self.optimizer_value.zero_grad()
    #         value_loss.backward()
    #         self.optimizer_value.step()
    #
    #         # Policy Update
    #         with torch.no_grad():
    #             adv = q_estimate - v_values
    #
    #         logprobs = self.policy.get_logprobs(states, actions)
    #         weights = torch.exp(adv / self.temperature).clamp(max=100.0)
    #         policy_loss = -(weights * logprobs).mean()
    #         self.optimizer_actor.zero_grad()
    #         policy_loss.backward()
    #         self.optimizer_actor.step()
    #
    #         # Soft update target
    #         for param, target_param in zip(self.policy.critic.parameters(), self.policy_target.critic.parameters()):
    #             target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    #
    #     return console