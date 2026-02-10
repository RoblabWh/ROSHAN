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

        self.policy = None
        self.critic_target = None
        self.initialize_policy()

        self.actor_optimizer = None
        self.critic_optimizer = None
        self.value_optimizer = None
        self.actor_scheduler = None
        self.critic_scheduler = None
        self.value_scheduler = None
        self.initialize_optimizers()

        # Offline Flags (for logging and saving networks)
        self.offline_start = True
        self.offline_end = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_noised_action = True
        self.raw_action = None

        self.set_train()

    def initialize_policy(self):
        """
        Initializes the policy with the actor and critic networks.
        This method is called when the algorithm is reset or loaded.
        """
        self.policy = ActorCriticIQL(actor_network=self.actor_network,
                                     critic_network=self.critic_network,
                                     value_network=self.value_network,
                                     action_dim=self.action_dim,
                                     vision_range=self.vision_range,
                                     drone_count=self.drone_count,
                                     map_size=self.map_size,
                                     time_steps=self.time_steps,
                                     share_encoder=self.share_encoder,
                                     use_tanh_dist=self.use_tanh_dist,
                                     collision=self.collision)
        if self.use_torch_compile:
            try:
                self.policy.actor = torch.compile(self.policy.actor, mode=self.compile_mode)
                self.policy.critic = torch.compile(self.policy.critic, mode=self.compile_mode)
                self.policy.value = torch.compile(self.policy.value, mode=self.compile_mode)
            except Exception as e:
                self.logger.warning(f"torch.compile failed, using eager mode: {e}")
        # Target Q network
        self.critic_target = copy.deepcopy(self.policy.critic)

    def initialize_optimizers(self):
        """
        Initializes the optimizers for the policy networks.
        This method is called when the algorithm is reset or loaded.
        """
        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.policy.critic.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), eps=1e-5)
        self.value_optimizer = torch.optim.Adam(self.policy.value.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), eps=1e-5)
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, T_max=int(1e6))
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.critic_optimizer, T_max=int(1e6))
        self.value_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.value_optimizer, T_max=int(1e6))
        self.offline_start = True
        self.offline_end = False

    def save_optimizers(self, path: str):
        path += "_iql_optimizers.pth"
        torch.save({
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'actor_scheduler_state_dict': self.actor_scheduler.state_dict(),
            'critic_scheduler_state_dict': self.critic_scheduler.state_dict(),
            'value_scheduler_state_dict': self.value_scheduler.state_dict(),
        }, path)

    def load_optimizers(self, path: str):
        path += "_iql_optimizers.pth"
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
            self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler_state_dict'])
            self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state_dict'])
            self.value_scheduler.load_state_dict(checkpoint['value_scheduler_state_dict'])
        except Exception as e:
            warnings.warn(f"Could not load IQL optimizers from {path}: {e}")

    def select_action(self, observations):
        noised_action, raw_action = self.policy.act(observations)
        self.raw_action = raw_action
        return noised_action, None

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
        self.value_scheduler.step()

        logger.add_metric("train/value_LR", self.value_scheduler.get_last_lr())
        return value_loss.detach(), v.mean().detach()

    def update_q(self, states, actions, rewards, next_states, not_dones, logger=None):
        with torch.no_grad():
            next_v = self.policy.value(next_states)
            target_q = (rewards + self.discount * not_dones * next_v).detach()

        q1, q2 = self.policy.critic(states, actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_scheduler.step()

        logger.add_metric("train/critic_LR", self.critic_scheduler.get_last_lr())
        return critic_loss.detach(), q1.mean().detach(), q2.mean().detach()

    @torch.no_grad()
    def update_target(self):
        for param, target_param in zip(self.policy.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_actor(self, states, actions, logger=None):
        with torch.no_grad():
            v = self.policy.value(states)
            q1, q2 = self.critic_target(states, actions)
            q = torch.minimum(q1, q2)
            adv = (q - v)
            exp_a = torch.exp(adv * self.temperature)
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

        logger.add_metric("train/actor_LR", self.actor_scheduler.get_last_lr())
        return actor_loss.detach(), adv.mean().detach()

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

        rewards, log_rewards_raw, log_rewards_scaled = self.prepare_rewards(ext_rewards, t_dict, already_fit=self.offline_start)
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

        _log_value_loss, _log_v = [], []
        _log_critic_loss, _log_q1, _log_q2 = [], [], []
        _log_actor_loss, _log_advantage = [], []

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
                vl, v = self.update_v(b_states, b_actions, logger)
                cl, q1, q2 = self.update_q(b_states, b_actions, b_rewards, b_next_states, b_not_dones, logger)
                al, adv = self.update_actor(b_states, b_actions, logger)
                self.update_target()

                _log_value_loss.append(vl)
                _log_v.append(v)
                _log_critic_loss.append(cl)
                _log_q1.append(q1)
                _log_q2.append(q2)
                _log_actor_loss.append(al)
                _log_advantage.append(adv)

        # Single GPU->CPU sync for all accumulated metrics
        logger.add_metric("train/value_loss", torch.stack(_log_value_loss).cpu().numpy())
        logger.add_metric("train/v", torch.stack(_log_v).cpu().numpy())
        logger.add_metric("train/critic_loss", torch.stack(_log_critic_loss).cpu().numpy())
        logger.add_metric("train/q1", torch.stack(_log_q1).cpu().numpy())
        logger.add_metric("train/q2", torch.stack(_log_q2).cpu().numpy())
        logger.add_metric("train/actor_loss", torch.stack(_log_actor_loss).cpu().numpy())
        logger.add_metric("train/advantage", torch.stack(_log_advantage).cpu().numpy())