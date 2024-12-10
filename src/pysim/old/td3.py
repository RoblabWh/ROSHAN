import copy
import warnings

from old.network import Inputspace
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    """
    A PyTorch Module that represents the actor network of a TD3 agent.
    """
    def __init__(self, vision_range, state_dim=256):
        super(Actor, self).__init__()
        self.Inputspace = Inputspace(vision_range)

        # TODO More Movement Layers
        self.movement1 = nn.Linear(in_features=state_dim, out_features=64)
        self.movement_action = nn.Linear(in_features=64, out_features=2)
        #initialize_output_weights(self.movement_action, 'actor')
        self.water1 = nn.Linear(in_features=state_dim, out_features=64)
        self.water_action = nn.Linear(in_features=64, out_features=1)
        #initialize_output_weights(self.water_action, 'actor')

    def forward(self, observations):
        x = self.Inputspace(observations)
        movement = torch.tanh(self.movement1(x))
        movement = torch.tanh(self.movement_action(movement))
        water = torch.relu(self.water1(x))
        water = torch.sigmoid(self.water_action(water))
        actions = torch.cat((movement, water), dim=1)

        return actions


class Critic(nn.Module):
    """
    A PyTorch Module that represents the critic network of a PPO agent.
    """
    def __init__(self, vision_range, state_dim=256, action_dim=3):
        super(Critic, self).__init__()
        self.Inputspace = self.Inputspace = Inputspace(vision_range)

        # Q1 architecture
        self.l1 = nn.Linear(in_features=state_dim + action_dim, out_features=256)
        self.l2 = nn.Linear(in_features=256, out_features=256)
        self.l3 = nn.Linear(in_features=256, out_features=1)

        # Q2 architecture
        self.l4 = nn.Linear(in_features=state_dim + action_dim, out_features=256)
        self.l5 = nn.Linear(in_features=256, out_features=256)
        self.l6 = nn.Linear(in_features=256, out_features=1)

    def forward(self, observations, action):
        x = self.Inputspace(observations)

        x = torch.cat([x, action], 1)

        q1 = F.relu(self.l1(x))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(x))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, observations, action):
        x = self.Inputspace(observations)

        x = torch.cat([x, action], 1)

        q1 = F.relu(self.l1(x))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1


class TD3:

    def __init__(self, action_dim, vision_range, lr=0.0003, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        self.action_dim = action_dim

        self.actor = Actor(vision_range).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(vision_range).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def random_action(self):
        actions = np.random.uniform(-1, 1, self.action_dim).astype(np.float32)
        for action in actions:
            action[2] = np.round(action[2])
        return actions, [0]

    def select_action(self, observations):
        return self.actor(observations).cpu().data.numpy(), [0]

    def update(self, memory, batch_size, logger):
        self.total_it += 1

        if batch_size > memory.size:
            warnings.warn("Batch size is larger than memory capacity. Setting batch size to memory capacity.")
            batch_size = memory.size

        state, action, reward, next_state, not_done = memory.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = self.actor_target(next_state)
            noised_action = (next_action + noise).clamp(-1, 1)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, noised_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward.to(device) + not_done.to(device) * self.discount * target_Q.squeeze(-1)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

