import torch
import torch.nn as nn
import os
import warnings
import math
import numpy as np
from algorithms.actor_critic import ActorCriticIQL
from algorithms.rl_algorithm import RLAlgorithm
from algorithms.rl_config import IQLConfig
from memory import SwarmMemory
import copy

#from https://github.com/Manchery/iql-pytorch?tab=readme-ov-file

class IQL(RLAlgorithm):
    def __init__(self, network, config: IQLConfig):

        super().__init__(config)
        for field in config.__dataclass_fields__:
            setattr(self, field, getattr(config, field))

        self.actor_network = network[0]
        self.critic_network = network[1]
        self.value_network = network[2]

        self.policy = ActorCriticIQL(Actor=self.actor_network, Critic=self.critic_network, Value=self.value_network, action_dim=self.action_dim,
                                     vision_range=self.vision_range, drone_count=self.drone_count,
                                     map_size=self.map_size, time_steps=self.time_steps)

        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.policy.critic.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), eps=1e-5)
        self.value_optimizer = torch.optim.Adam(self.policy.value.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), eps=1e-5)

        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, T_max=int(1e6))

        # Offline Flags (for logging and saving networks)
        self.offline_start = True
        self.offline_end = False
        self.MSE_loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Target Q network
        self.critic_target = copy.deepcopy(self.policy.critic)

        self.set_train()

    def set_train(self):
        self.policy.train()

    def set_eval(self):
        self.policy.eval()

    def load(self):
        path: str = os.path.join(self.loading_path, self.loading_name).__str__()
        try:
            self.policy.load_state_dict(torch.load(f'{path}', map_location=self.device))
            self.policy.to(self.device)
            return True
        except FileNotFoundError:
            warnings.warn(f"Could not load model from {path}. Falling back to train mode.")
            return False
        except TypeError:
            self.logger.warn(f"TypeError while loading model.")
            return False
        except RuntimeError:
            self.logger.warn(f"RuntimeError while loading model. Possibly due to architecture mismatch.")
            return False

    def select_action(self, observations):
        return self.policy.act(observations)

    def select_action_certain(self, observations):
        """
        Select action with certain policy.
        """
        return self.policy.act_certain(observations)

    @staticmethod
    def asymmetric_l2_loss(diff, expectile=0.8):
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def update_v(self, states, actions, logger=None):
        with torch.no_grad():
            q1, q2 = self.critic_target(states, actions)
            q = torch.minimum(q1, q2).detach()

        v = self.policy.value(states)
        value_loss = self.asymmetric_l2_loss(q - v, self.expectile).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        logger.add_metric("train/value_loss", value_loss.detach().cpu().numpy())
        logger.add_metric("train/v", v.mean().detach().cpu().numpy())

    def update_q(self, states, actions, rewards, next_states, not_dones, logger=None):
        with torch.no_grad():
            next_v = self.policy.value(next_states)
            target_q = (rewards + self.discount * not_dones * next_v).detach()

        q1, q2 = self.policy.critic(states, actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        logger.add_metric("train/critic_loss", critic_loss.detach().cpu().numpy())
        logger.add_metric("train/q1", q1.mean().detach().cpu().numpy())
        logger.add_metric("train/q2", q2.mean().detach().cpu().numpy())

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

        action_log_prob = self.policy.get_logprobs(states, actions)
        actor_loss = -(exp_a * action_log_prob).mean()

        # This no use? Can use...
        # mu, _ = self.policy.actor(states)
        # actor_loss = (exp_a.unsqueeze(-1) * ((mu - actions)**2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_scheduler.step()

        logger.add_metric("train/actor_loss", actor_loss.detach().cpu().numpy())
        logger.add_metric("train/advantage", (q - v).mean().detach().cpu().numpy())

    def get_batches_and_epochs(self, N, batch_size):
        batches_per_epoch = math.ceil(N / batch_size) if not self.offline_end else 1
        epochs = 1 if not self.offline_end else self.k_epochs
        return batches_per_epoch, epochs

    def update(self, memory: SwarmMemory, batch_size, next_obs, logger):

        t_dict = memory.to_tensor()
        states = memory.rearrange_states(t_dict['state'])  # [N, ...]
        next_states = memory.rearrange_states(t_dict['next_obs'])  # [N, ...]
        actions = torch.cat(t_dict['action'])  # [N, ...]
        ext_rewards = t_dict['reward']  # [D, N]
        not_dones = torch.cat(t_dict['not_done'])  # [N]

        rewards, log_rewards_raw, log_rewards_scaled = self.prepare_rewards(ext_rewards, t_dict)
        rewards = torch.cat(rewards)

        if self.offline_start or self.offline_end:
            self.offline_start = False
            logger.add_metric("Rewards/Rewards_Raw", log_rewards_raw)
            logger.add_metric("Rewards/Rewards_norm", log_rewards_scaled)

        # Save current weights if mean reward or objective is higher than the best so far
        # (Save BEFORE training, so if the policy worsens we can still go back)
        self.save(logger)

        N = rewards.shape[0]
        batches_per_epoch, epochs = self.get_batches_and_epochs(N, batch_size)

        for epoch in range(epochs):
            perm = torch.randperm(N)
            for i in range(batches_per_epoch):
                batch_idx = perm[i*batch_size : (i+1)*batch_size]
                b_states = tuple(state[batch_idx] for state in states)
                b_actions = actions[batch_idx]
                b_rewards = rewards[batch_idx]
                b_next_states = tuple(state[batch_idx] for state in next_states)
                b_not_dones = not_dones[batch_idx]

                # Update
                self.update_v(b_states, b_actions, logger)
                self.update_q(b_states, b_actions, b_rewards, b_next_states, b_not_dones, logger)
                self.update_actor(b_states, b_actions, logger)
                self.update_target()