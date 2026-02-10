import copy
import warnings
import torch
import torch.nn as nn
import os

from algorithms.actor_critic import DeterministicActorCritic
from algorithms.rl_algorithm import RLAlgorithm
from algorithms.rl_config import TD3Config
from memory import SwarmMemory


class TD3(RLAlgorithm):
    def __init__(self, network, config: TD3Config):
        super().__init__(config)

        for field in config.__dataclass_fields__:
            setattr(self, field, getattr(config, field))

        self.actor_network = network[0]
        self.critic_network = network[1]

        self.policy = None
        self.actor_target = None
        self.critic_target = None
        self.initialize_policy()

        self.actor_optimizer = None
        self.actor_scheduler = None
        self.critic_optimizer = None
        self.critic_scheduler = None
        self.initialize_optimizers()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.total_it = 0
        self.set_train()

    def initialize_policy(self):
        """
        Initializes the policy with the actor and critic networks.
        This method is called when the algorithm is reset or loaded.
        """
        self.total_it = 0
        self.policy = DeterministicActorCritic(
            actor_network=self.actor_network,
            critic_network=self.critic_network,
            action_dim=self.action_dim,
            vision_range=self.vision_range,
            drone_count=self.drone_count,
            map_size=self.map_size,
            time_steps=self.time_steps,
            share_encoder=self.share_encoder,
            exploration_noise=self.exploration_noise,
            collision=self.collision,
        )
        if self.use_torch_compile:
            try:
                self.policy.actor = torch.compile(self.policy.actor, mode=self.compile_mode)
                self.policy.critic = torch.compile(self.policy.critic, mode=self.compile_mode)
            except Exception as e:
                self.logger.warning(f"torch.compile failed, using eager mode: {e}")
        self.actor_target = copy.deepcopy(self.policy.actor)
        self.critic_target = copy.deepcopy(self.policy.critic)

    def load_optimizers(self, path: str):
        path += "_td3_optimizers.pth"
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler_state_dict'])
            self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state_dict'])
        except Exception as e:
            warnings.warn(f"Could not load TD3 optimizers from {path}: {e}")

    def save_optimizers(self, path: str):
        path += "_td3_optimizers.pth"
        torch.save({
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'actor_scheduler_state_dict': self.actor_scheduler.state_dict(),
            'critic_scheduler_state_dict': self.critic_scheduler.state_dict(),
        }, path)

    def copy_networks(self):
        self.critic_target = copy.deepcopy(self.policy.critic)
        self.actor_target = copy.deepcopy(self.policy.actor)

    def initialize_optimizers(self):
        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), eps=1e-5)
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, T_max=int(1e6))
        self.critic_optimizer = torch.optim.Adam(self.policy.critic.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), eps=1e-5)
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.critic_optimizer, T_max=int(1e6))

    def set_train(self):
        self.policy.train()
        self.actor_target.train()
        self.critic_target.train()

    def set_eval(self):
        self.policy.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def update(self, memory: SwarmMemory, batch_size, next_obs, logger):
        # TODO test faster batch sampling
        # t_dicter = memory._sample_batch(64)
        # t_dict = memory.to_tensor()
        # states = memory.rearrange_states(t_dict['state'])  # [N, ...]
        # next_states = memory.rearrange_states(t_dict['next_obs'])  # [N, ...]
        # actions = torch.cat(t_dict['action'])  # [N, ...]
        # ext_rewards = t_dict['reward']  # [D, N]
        # not_dones = torch.cat(t_dict['not_done'])  # [N]

        # rewards, log_rewards_raw, log_rewards_scaled = self.prepare_rewards(ext_rewards, t_dict)
        # logger.add_metric("Rewards/Rewards_Raw", log_rewards_raw)
        # logger.add_metric("Rewards/Rewards_norm", log_rewards_scaled)
        # rewards = torch.cat(rewards)

        # N = rewards.shape[0]

        # Save current weights if mean reward or objective is higher than the best so far
        # (Save BEFORE training, so if the policy worsens we can still go back)
        self.save(logger)

        _log_critic_loss = []
        _log_actor_loss = []

        for _ in range(self.k_epochs):
            # perm = torch.randperm(N)
            # batch_idx = perm[0:batch_size]
            batch = memory.sample_batch(batch_size)
            self.total_it += 1

            b_states = batch["state"]
            b_actions = batch["action"]
            b_rewards = batch["reward"]
            b_next_states = batch["next_obs"]
            b_not_dones = batch["not_done"]

            with torch.no_grad():
                noise = (torch.randn_like(b_actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = self.actor_target(b_next_states)
                noised_action = (next_action + noise).clamp(-1, 1)

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(b_next_states, noised_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = b_rewards.to(self.device) + b_not_dones.to(self.device) * self.discount * target_Q.squeeze(-1)

            # Get current Q estimates
            current_Q1, current_Q2 = self.policy.critic(b_states, b_actions)

            # Compute critic loss
            critic_loss = self.MSE_loss(current_Q1, target_Q) + self.MSE_loss(current_Q2, target_Q)
            _log_critic_loss.append(critic_loss.detach())
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            self.critic_scheduler.step()
            logger.add_metric("Training/LR_Critic", self.critic_scheduler.get_last_lr())

            # Delayed policy updates
            if self.total_it % self.policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.policy.critic.Q1(b_states, self.policy.actor(b_states)).mean()
                _log_actor_loss.append(actor_loss.detach())

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.actor_scheduler.step()
                logger.add_metric("Training/LR_Actor", self.actor_scheduler.get_last_lr())

                # Update the frozen target models
                with torch.no_grad():
                    for param, target_param in zip(self.policy.critic.parameters(), self.critic_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                    for param, target_param in zip(self.policy.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Single GPU->CPU sync for accumulated metrics
        logger.add_metric("Training/critic_loss", torch.stack(_log_critic_loss).cpu().numpy())
        if _log_actor_loss:
            logger.add_metric("Training/actor_loss", torch.stack(_log_actor_loss).cpu().numpy())
