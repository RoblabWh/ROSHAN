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

        self.policy = DeterministicActorCritic(
            Actor=self.actor_network,
            Critic=self.critic_network,
            action_dim=self.action_dim,
            vision_range=self.vision_range,
            drone_count=self.drone_count,
            map_size=self.map_size,
            time_steps=self.time_steps,
            exploration_noise=self.exploration_noise
        )

        self.actor_target = copy.deepcopy(self.policy.actor)
        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), eps=1e-5)

        self.critic_target = copy.deepcopy(self.policy.critic)
        self.critic_optimizer = torch.optim.Adam(self.policy.critic.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), eps=1e-5)

        self.MSE_loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.total_it = 0

    def set_train(self):
        self.policy.train()
        self.actor_target.train()
        self.critic_target.train()

    def set_eval(self):
        self.policy.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def save(self):
        torch.save(self.policy.state_dict(), f'{os.path.join(self.model_path, self.model_name)}')

    def select_action(self, observations):
        return self.policy.act(observations)

    def load(self):
        path = os.path.join(self.model_path, self.model_name)
        if os.path.exists(path):
            self.policy.load_state_dict(torch.load(path))
            self.policy.eval()
            return True
        else:
            warnings.warn(f"Model path {path} does not exist. Loading failed.")
            return False

    def select_action_certain(self, observations):
        """
        Select action with certain policy.
        """
        return self.policy.act_certain(observations)

    def update(self, memory: SwarmMemory, batch_size, next_obs, logger):
        for _ in range(self.k_epochs):
            self.total_it += 1

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

            with torch.no_grad():
                noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = self.actor_target(next_states)
                noised_action = (next_action + noise).clamp(-1, 1)

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_states, noised_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = rewards.to(self.device) + not_dones.to(self.device) * self.discount * target_Q.squeeze(-1)

            # Get current Q estimates
            current_Q1, current_Q2 = self.policy.critic(states, actions)

            # Compute critic loss
            critic_loss = self.MSE_loss(current_Q1, target_Q) + self.MSE_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if self.total_it % self.policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.policy.critic.Q1(states, self.policy.actor(states)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.policy.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.policy.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return ""

