import torch
import torch.nn as nn
import os
import warnings
import numpy as np
from memory import SwarmMemory
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

    def __init__(self, network, vision_range, time_steps, lr, betas, gamma, _lambda, K_epochs, eps_clip, model_path, model_name):

        # Algorithm parameters
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self._lambda = _lambda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.vision_range = vision_range
        self.time_steps = time_steps
        self.network = network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_path = os.path.abspath(model_path)
        self.model_name = model_name
        self.version = 1
        self.model_version = model_name.split(".")[0] + "_v" + str(self.version) + "." + model_name.split(".")[1]
        self.model_latest = model_name.split(".")[0] + "_latest." + model_name.split(".")[1]

        # Current Policy
        self.policy = self.network(vision_range=self.vision_range, time_steps=self.time_steps)

        self.actor_params = self.policy.actor.parameters()
        self.critic_params = self.policy.critic.parameters()

        self.optimizer_a = torch.optim.Adam(self.actor_params, lr=lr, betas=betas, eps=1e-5)
        self.optimizer_c = torch.optim.Adam(self.critic_params, lr=lr, betas=betas, eps=1e-5)

        self.MSE_loss = nn.MSELoss()
        self.running_reward_std = RunningMeanStd()
        self.set_train()

    def reset(self):
        self.version += 1
        self.model_version = self.model_name.split(".")[0] + "_v" + str(self.version) + "." + self.model_name.split(".")[1]
        self.policy = self.network(vision_range=self.vision_range, time_steps=self.time_steps)
        self.optimizer_a = torch.optim.Adam(self.policy.actor.parameters(), lr=self.lr, betas=self.betas, eps=1e-5)
        self.optimizer_c = torch.optim.Adam(self.policy.critic.parameters(), lr=self.lr, betas=self.betas, eps=1e-5)
        self.MSE_loss = nn.MSELoss()
        self.running_reward_std = RunningMeanStd()
        self.set_train()

    def set_paths(self, model_path, model_name):
        self.model_path = os.path.abspath(model_path)
        self.model_name = model_name
        self.model_version = model_name.split(".")[0] + "_v" + str(self.version) + "." + model_name.split(".")[1]
        self.model_latest = model_name.split(".")[0] + "_latest." + model_name.split(".")[1]

    def set_eval(self):
        self.policy.eval()

    def set_train(self):
        self.policy.train()

    def load(self):
        try:
            path: str = os.path.join(self.model_path, self.model_name).__str__()
            # self.policy.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
            self.policy.load_state_dict(torch.load(path, map_location=self.device))
            self.policy.to(self.device)
            return True
        except FileNotFoundError:
            warnings.warn(f"Could not load model from {path}. Falling back to train mode.")
            return False

    def select_action(self, observations):
        return self.policy.act(observations)

    def select_action_certain(self, observations):
        return self.policy.act_certain(observations)

    def save(self, logger):
        console = ""
        if logger.is_better_reward():
            console = f"Saving best with reward {logger.best_reward:.4f} and mean burned {logger.get_objective():.2f}%\n"
            torch.save(self.policy.state_dict(), f'{os.path.join(self.model_path, self.model_version)}')
        torch.save(self.policy.state_dict(), f'{os.path.join(self.model_path, self.model_latest)}')

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

        # Norm advantages
        advantages = torch.FloatTensor(advantages).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        return advantages, torch.FloatTensor(returns).to(device)

    def calculate_explained_variance(self, values, returns):
        """
        Calculates the explained variance of the prediction and target.

        interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
        """
        var_returns = returns.var()
        return 0 if var_returns == 0 else 1 - (returns - values).var() / var_returns

    def update(self, memory: SwarmMemory, horizon, mini_batch_size, n_steps, next_obs, logger):
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
        :param horizon: The size of all data gathered before training
        """

        states, actions, old_logprobs, rewards, masks = memory.to_tensor()

        # Logger
        logging_values = []
        console = ""
        logger.add_rewards(torch.cat(rewards).detach().cpu().numpy())

        # Clipping most likely is unnecessary
        # rewards = [np.clip(np.array(reward.detach().cpu()) / self.running_reward_std.get_std(), -10, 10) for reward in rewards]
        # Reward normalization
        self.running_reward_std.update(torch.cat(rewards).detach().cpu().numpy())
        # rewards = [reward / self.running_reward_std.get_std() for reward in rewards]
        # logger.add_rewards_scaled(torch.cat(rewards).detach().cpu().numpy())

        # Don't shift rewards like this, it will mess up your reward function. Only do scaling
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-10)

        # Advantages
        with (torch.no_grad()):
            advantages = []
            returns = []
            for i in range(memory.num_agents):
                # _, values_, _ = self.policy.evaluate(states[i], actions[i])
                # if masks[i][-1] == 1:
                #     last_state = tuple(torch.FloatTensor(np.array(state)).to(self.device) for state in memory.get_agent_state(next_obs, i))
                #     bootstrapped_value = self.policy.critic(last_state).detach()
                #     values_ = torch.cat((values_, bootstrapped_value[0]), dim=0)
                # adv, ret = self.get_advantages(values_.detach(), masks[i], rewards[i].detach())
                #
                # advantages.append(adv)
                # returns.append(ret)

                step_indices = list(np.arange(0, len(states[0][0]), n_steps))
                if step_indices[-1] != len(states[0][0]):
                    step_indices.append(len(states[0][0]))

                for idx in range(len(step_indices) - 1):
                    begin = step_indices[idx]
                    end = step_indices[idx + 1]
                    batch_states = tuple(state[begin:end] for state in states[i])
                    batch_actions = actions[i][begin:end]
                    batch_rewards = rewards[i][begin:end]
                    batch_masks = masks[i][begin:end]
                    _, values_, _ = self.policy.evaluate(batch_states, batch_actions)
                    if batch_masks[-1] == 1:
                        last_state = tuple(torch.FloatTensor(np.array(state)).to(self.device) for state in memory.get_agent_state(next_obs, i))
                        bootstrapped_value = self.policy.critic(last_state).detach()
                        values_ = torch.cat((values_, bootstrapped_value[0]), dim=0)
                    adv, ret = self.get_advantages(values_.detach(), batch_masks, batch_rewards.detach())
                    advantages.append(adv)
                    returns.append(ret)

        # Merge all agent states, actions, rewards etc.
        advantages = torch.cat(advantages)
        # # Norm advantages
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        returns = torch.cat(returns)
        # Log returns
        logger.add_returns(returns.detach().cpu().numpy())
        actions = torch.cat(actions)
        old_logprobs = torch.cat(old_logprobs)
        states_ = tuple()
        for i in range(len(states[0])):
            states_ += (torch.cat([states[k][i] for k in range(len(states))]),)
        states = states_

        # Train policy for K epochs: sampling and updating
        for _ in range(self.K_epochs):
            epoch_values = []
            epoch_returns = []
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(horizon)), mini_batch_size, True):
                # Evaluate old actions and values using current policy
                # batch_states = (
                #     states[0][index], states[1][index], states[2][index], states[3][index], states[4][index])
                # batch_states = (state[i][index] for state in states)
                #batch_states = (states[0][index], states[1][index], states[2][index], states[3][index], states[4][index], states[5][index])
                #batch_states = (states[0][index], states[1][index], states[2][index])
                batch_states = tuple(state[index] for state in states)
                batch_actions = actions[index]
                logprobs, values, dist_entropy = self.policy.evaluate(batch_states, batch_actions)

                # Importance ratio: p/q
                ratios = torch.exp(logprobs - old_logprobs[index].detach())

                # Approximate KL Divergence
                # approxkl = 0.5 * torch.mean(torch.square(old_logprobs[index].detach() - logprobs))
                # if approxkl > 0.02:
                #     console += (f"Approximate Kulback-Leibler Divergence: {approxkl}. A value above 0.02 is considered bad since the policy changes too quickly.\n"
                #                 f"This happend in epoch {_} and mini_batch {index}. Early stopping for this training peroid.\n")
                #     break

                actor_loss = -torch.mean(torch.min(ratios * advantages[index], torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[index]))

                # Actor loss using Surrogate loss
                # surr1 = ratios * advantages[index]
                # surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[index]
                # # entropy = torch.Tensor(0) #0.001 * dist_entropy
                # actor_loss = ((-torch.min(surr1, surr2).type(torch.float32))).mean()

                # TODO CLIP VALUE LOSS ? Probably not necessary as according to:
                # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
                critic_loss = self.MSE_loss(returns[index].squeeze(), values)

                # Log loss
                logger.add_losses(critic_loss=critic_loss.detach().cpu().item(), actor_loss=actor_loss.detach().cpu().item())
                logger.add_std(torch.exp(self.policy.actor.log_std).cpu().detach().numpy())
                logging_values.append(values.detach().cpu().numpy())
                epoch_values.append(values.detach().cpu().numpy())
                epoch_returns.append(returns[index].detach().cpu().numpy())

                # Sanity checks
                assert not torch.isnan(critic_loss).any(), f"Critic Loss is NaN, check returns, values or entroy"
                assert not torch.isnan(actor_loss).any()
                assert not torch.isinf(critic_loss).any()
                assert not torch.isinf(actor_loss).any()

                # Backward gradients
                self.optimizer_a.zero_grad()
                actor_loss.backward(retain_graph=True)
                # Global gradient norm clipping https://vitalab.github.io/article/2020/01/14/Implementation_Matters.html
                torch.nn.utils.clip_grad_norm_(self.actor_params, max_norm=0.5)
                self.optimizer_a.step()

                self.optimizer_c.zero_grad()
                critic_loss.backward()
                # Global gradient norm clipping https://vitalab.github.io/article/2020/01/14/Implementation_Matters.html
                torch.nn.utils.clip_grad_norm_(self.critic_params, max_norm=0.5)
                self.optimizer_c.step()

            # Compute explained variance over the epoch
            epoch_values = np.concatenate(epoch_values)
            epoch_returns = np.concatenate(epoch_returns)
            ev = self.calculate_explained_variance(epoch_values, epoch_returns)
            if ev <= 0:
                console += (f"Explained Variance for Epoch {_}: {ev}\n"
                            f"This is bad. The Critic might aswell have predicted zero or is even doing worse than that.\n")
            logger.add_explained_variance(ev)


        # Save current weights if the mean reward is higher than the best reward so far
        logger.add_values(np.concatenate(logging_values))
        console += self.save(logger)

        # Clear memory
        memory.clear_memory()

        return console
