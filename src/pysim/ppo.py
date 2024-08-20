import torch
import torch.nn as nn
import os
import warnings
import numpy as np
from copy import deepcopy
from network import ActorCritic
from utils import RunningMeanStd
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPO:
    """
    This class represents the PPO Algorithm. It is used to train an actor-critic network.

    :param scan_size: The number of lidar scans in the input lidar scan.
    :param action_std: The standard deviation of the action distribution.
    :param input_style: The style of the input to the network.
    :param lr: The learning rate of the network.
    :param betas: The betas of the Adam optimizer.
    :param gamma: The discount factor.
    :param K_epochs: The number of epochs to train the network.
    :param eps_clip: The epsilon value for clipping.
    :param logger: The logger to log data to.
    :param restore: Whether to restore the network from a checkpoint.
    :param ckpt: The checkpoint to restore from.
    """

    def __init__(self, vision_range, time_steps, lr, betas, gamma, _lambda, K_epochs, eps_clip, model_path, model_name):

        # Algorithm parameters
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self._lambda = _lambda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_path = os.path.abspath(model_path)
        self.model_name = os.path.join(model_path, model_name)

        # Current Policy
        self.policy = ActorCritic(vision_range=vision_range, time_steps=time_steps)

        self.optimizer_a = torch.optim.Adam(self.policy.actor.parameters(), lr=lr, betas=betas, eps=1e-5)
        self.optimizer_c = torch.optim.Adam(self.policy.critic.parameters(), lr=lr, betas=betas, eps=1e-5)

        self.MSE_loss = nn.MSELoss()
        self.running_reward_std = RunningMeanStd()
        self.set_train()

    def set_paths(self, model_path, model_name):
        self.model_path = os.path.abspath(model_path)
        self.model_name = os.path.join(model_path, model_name)

    def set_eval(self):
        self.policy.eval()

    def set_train(self):
        self.policy.train()

    def load(self):
        try:
            path = os.path.join(self.model_path, self.model_name)
            self.policy.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
            return True
        except FileNotFoundError:
            warnings.warn(f"Could not load model from {path}. Falling back to train mode.")
            return False

    def select_action(self, observations):
        return self.policy.act(observations)

    def select_action_certain(self, observations):
        return self.policy.act_certain(observations)

    def save(self, logger):
        console = f"Saving best with reward {logger.reward_best:.2f} and mean burned {logger.get_objective():.2f}%\n"
        torch.save(self.policy.state_dict(), f'{self.model_name}')
        return console

    def get_advantages(self, values, masks, rewards):
        """
        Computes the advantages of the given rewards and values.

        :param values: The values of the states.
        :param masks: The masks of the states.
        :param rewards: The rewards of the states.
        :return: The advantages of the states.
        """
        advantages = []
        returns = []
        gae = 0

        for i in reversed(range(len(rewards))):
            delta = rewards[i] - values[i]
            if masks[i] == 1:
                delta = delta + self.gamma * values[i + 1]
            gae = delta + self.gamma * self._lambda * masks[i] * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

        advantages = torch.FloatTensor(advantages).to(device)
        norm_adv = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        returns = torch.FloatTensor(returns).to(device)
        #norm_returns = (returns - returns.mean()) / (returns.std() + 1e-10)

        return norm_adv, returns

    def update(self, memory, batch_size, mini_batch_size, next_obs, next_terminals, logger):
        """
        This function implements the update step of the Proximal Policy Optimization (PPO) algorithm for a swarm of
        robots. It takes in the memory buffer containing the experiences of the swarm, as well as the number of batches
        to divide the experiences into for training. The function first computes the discounted rewards for each robot
        in the swarm and normalizes them. It then flattens the rewards and masks and converts them to PyTorch tensors.
        Next, the function retrieves the observations, actions, and log probabilities from the memory buffer and divides
        them into minibatches for training. For each minibatch, it calculates the advantages using the generalized
        advantage estimator (GAE) and trains the policy for K epochs using the surrogate loss function. The function
        then updates the weights of the actor and critic networks using the optimizer.
        Finally, the function copies the updated weights to the old policy for future use in the next update step.

        :param memory: The memory to update the network with.
        :param batch_size: The size of batches.
        """

        # TODO REIMPLEMENT
        # if batch_size > memory.size:
        #     warnings.warn("Batch size is larger than memory capacity. Setting batch size to memory capacity.")
        #     batch_size = memory.size
        #
        # if batch_size == 1:
        #     raise ValueError("Batch size must be greater than 1.")

        states, actions, old_logprobs, rewards, masks = memory.to_tensor()

        # Logger
        log_values = []
        logger.add_reward(rewards.detach().cpu().numpy())

        # Normalize rewards by running reward
        self.running_reward_std.update(np.array(rewards.detach().cpu()))
        rewards = np.clip(np.array(rewards.detach().cpu()) / self.running_reward_std.get_std(), -10, 10)
        rewards = torch.tensor(rewards).type(torch.float32)

        # TODO Shifting causes Agent to suicide??
        #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-10)

        # Advantages
        with torch.no_grad():
            _, values_, _ = self.policy.evaluate(states, actions)
            if masks[-1] == 1:
                last_state = tuple(torch.FloatTensor(np.array(state)).to(self.device) for state in next_obs)
                bootstrapped_value = self.policy.critic(last_state).detach()
                values_ = torch.cat((values_, bootstrapped_value[0]), dim=0)
            advantages, returns = self.get_advantages(values_.detach(), masks, rewards)

        # Train policy for K epochs: sampling and updating
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, False):
                # Evaluate old actions and values using current policy
                # batch_states = (
                #     states[0][index], states[1][index], states[2][index], states[3][index], states[4][index])
                batch_states = (states[0][index], states[1][index], states[2][index], states[3][index])
                batch_actions = actions[index]
                logprobs, values, dist_entropy = self.policy.evaluate(batch_states, batch_actions)
                log_values.append(values.detach().cpu().numpy())
                # Importance ratio: p/q
                ratios = torch.exp(logprobs - old_logprobs[index].detach())

                # Actor loss using Surrogate loss
                surr1 = ratios * advantages[index]
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[index]
                entropy = 0.01 * dist_entropy
                actor_loss = ((-torch.min(surr1, surr2).type(torch.float32)) - entropy).mean()

                # TODO CLIP VALUE LOSS ? Probably not necessary as according to:
                # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
                critic_loss = self.MSE_loss(returns[index].squeeze(), values)
                # Total loss
                # loss = actor_loss + critic_loss
                # self.logger.add_loss(loss.detach().mean().item(), entropy=entropy.detach().mean().item(), critic_loss=critic_loss.detach().mean().item(), actor_loss=actor_loss.detach().mean().item())

                # Sanity checks
                if torch.isnan(critic_loss).any():
                    print(entropy.mean())
                    print(returns)
                    print(values)
                assert not torch.isnan(actor_loss).any()

                assert not torch.isinf(critic_loss).any()
                assert not torch.isinf(actor_loss).any()
                # Backward gradients
                self.optimizer_a.zero_grad()
                actor_loss.backward(retain_graph=True)
                # Global gradient norm clipping https://vitalab.github.io/article/2020/01/14/Implementation_Matters.html
                torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_norm=0.5)
                self.optimizer_a.step()

                self.optimizer_c.zero_grad()
                critic_loss.backward()
                # Global gradient norm clipping https://vitalab.github.io/article/2020/01/14/Implementation_Matters.html
                torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), max_norm=0.5)
                self.optimizer_c.step()
                # # Global gradient norm clipping https://vitalab.github.io/article/2020/01/14/Implementation_Matters.html
                # torch.nn.utils.clip_grad_norm_(self.policy.ac.parameters(), max_norm=0.5)


        # Save current weights if the mean reward is higher than the best reward so far
        logger.add_value(np.array(log_values))
        if logger.better_reward():
            console = self.save(logger)
        else:
            console = ""
        # Clear memory
        memory.clear_memory()

        return console
