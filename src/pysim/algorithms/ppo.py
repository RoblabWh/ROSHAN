import torch
import torch.nn as nn
import os
import warnings
import numpy as np
from algorithms.actor_critic import ActorCriticPPO, CategoricalActorCritic
from memory import SwarmMemory
from utils import RunningMeanStd, Logger
from algorithms.rl_algorithm import RLAlgorithm
from algorithms.rl_config import PPOConfig
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPO(RLAlgorithm):
    """
    This class represents the PPO Algorithm. It is used to train an actor-critic network.
    """

    def __init__(self, network, config : PPOConfig):
        super().__init__(config)

        for field in config.__dataclass_fields__:
            setattr(self, field, getattr(config, field))

        self.actor = network[0]
        self.critic = network[1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Current Policy
        self.policy = None
        self.actor_params = None
        self.critic_params = None
        self.initialize_policy()

        self.optimizer_a = None
        self.optimizer_c = None
        self.initialize_optimizers()

        self.MSE_loss = nn.MSELoss()
        self.reward_rms = RunningMeanStd()
        self.int_reward_rms = RunningMeanStd()
        self.set_train()

    def initialize_optimizers(self):
        """
        Initializes the optimizers for the actor and critic networks.
        This method is called after the policy is reset or loaded.
        """
        if self.separate_optimizers:
            self.optimizer_a = torch.optim.Adam(self.actor_params, lr=self.lr, betas=(self.beta1, self.beta2), eps=1e-5)
            self.optimizer_c = torch.optim.Adam(self.critic_params, lr=self.lr, betas=(self.beta1, self.beta2), eps=1e-5)
        else:
            params = list(self.actor_params) + list(self.critic_params)
            self.optimizer_a = torch.optim.Adam(params, lr=self.lr, betas=(self.beta1, self.beta2), eps=1e-5)
            self.optimizer_c = None

    def initialize_policy(self):
        """
        Initializes the policy with the actor and critic networks.
        This method is called when the algorithm is reset or loaded.
        """
        if self.use_categorical:
            self.policy = CategoricalActorCritic(Actor=self.actor, Critic=self.critic, vision_range=self.vision_range,
                                                 drone_count=self.drone_count, map_size=self.map_size,
                                                 time_steps=self.time_steps)
        else:
            self.policy = ActorCriticPPO(Actor=self.actor, Critic=self.critic, vision_range=self.vision_range,
                                         drone_count=self.drone_count, map_size=self.map_size,
                                         time_steps=self.time_steps)

        self.actor_params = self.policy.actor.parameters()
        self.critic_params = self.policy.critic.parameters()

    def reset_algorithm(self):
        self.initialize_policy()
        self.initialize_optimizers()
        self.MSE_loss = nn.MSELoss()
        self.reward_rms = RunningMeanStd()
        self.int_reward_rms = RunningMeanStd()
        self.set_train()

    def set_eval(self):
        self.policy.eval()

    def set_train(self):
        self.policy.train()

    def load(self):
        path: str = os.path.join(self.loading_path, self.loading_name).__str__()
        try:
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

    def save(self, logger: Logger):
        if logger.is_better_reward():
            self.logger.info(f"Saving Network at Episode {logger.episode}, best Reward: {logger.best_metrics['best_reward']:.2f}")
            torch.save(self.policy.state_dict(), f'{os.path.join(self.get_model_path(), self.get_model_name_reward())}')
        if logger.is_better_objective():
            self.logger.info(f"Saving Network at Episode {logger.episode}, best Objective {logger.best_metrics['best_objective']:.2f}")
            torch.save(self.policy.state_dict(), f'{os.path.join(self.get_model_path(), self.get_model_name_obj())}')
        torch.save(self.policy.state_dict(), f'{os.path.join(self.get_model_path(), self.get_model_name_latest())}')

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
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        return advantages, torch.FloatTensor(returns).to(self.device)

    @staticmethod
    def calculate_explained_variance(values, returns):
        """
        Calculates the explained variance of the prediction and target.

        interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
        """
        var_returns = returns.var()
        return 0 if var_returns == 0 else 1 - (returns - values).var() / var_returns

    def update(self, memory: SwarmMemory, mini_batch_size, next_obs, logger):
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
        """

        t_dict = memory.to_tensor()

        states = t_dict['state']
        variable_state_masks = t_dict['mask']
        actions = t_dict['action']
        old_logprobs = t_dict['logprobs']
        ext_rewards = t_dict['reward']
        masks = t_dict['not_done']

        # Check for intrinsic rewards
        intrinsic_rewards = None
        if 'intrinsic_reward' in t_dict.keys():
            intrinsic_rewards = t_dict['intrinsic_reward']
            self.int_reward_rms.update(torch.cat(intrinsic_rewards).detach().cpu().numpy())

        # Logger
        logging_values = []
        log_rewards = torch.cat(ext_rewards).detach().cpu().numpy() if intrinsic_rewards is None \
           else torch.cat(intrinsic_rewards).detach().cpu().numpy() + torch.cat(ext_rewards).detach().cpu().numpy()
        logger.add_metric("Rewards/Rewards_Raw", log_rewards)

        # Save current weights if mean reward or objective is higher than the best so far
        # (Save BEFORE training, so if the policy worsens we can still go back)
        self.save(logger)

        # Clipping most likely is unnecessary
        # rewards = [np.clip(np.array(reward.detach().cpu()) / self.running_reward_std.get_std(), -10, 10) for reward in rewards]
        # Reward normalization !!!!DON'T SHIFT THE REWARDS BECAUSE YOU F UP YOUR OBJECTIVE FUNCTION!!!!
        self.reward_rms.update(torch.cat(ext_rewards).detach().cpu().numpy())
        ext_rewards = [reward / self.reward_rms.get_std() for reward in ext_rewards]
        intrinsic_rewards = [reward / self.int_reward_rms.get_std() for reward in intrinsic_rewards] if intrinsic_rewards is not None else None

        rewards = ext_rewards if intrinsic_rewards is None \
            else [ext_reward + int_reward for ext_reward, int_reward in zip(ext_rewards, intrinsic_rewards)]

        log_rewards = torch.cat(rewards).detach().cpu().numpy()
        logger.add_metric("Rewards/Rewards_norm", log_rewards)

        # Advantages
        with (torch.no_grad()):
            advantages = []
            returns = []
            for i in range(memory.num_agents):
                _, values_, _ = self.policy.evaluate(states[i], actions[i], variable_state_masks[0] if self.use_variable_state_masks and len(variable_state_masks[0]) > 0 else None)
                if masks[i][-1] == 1:
                    last_state = tuple(torch.FloatTensor(np.array(state)).to(self.device) for state in memory.get_agent_state(next_obs, i))
                    bootstrapped_value = self.policy.critic(last_state).detach()
                    if bootstrapped_value.dim() != 1:
                        bootstrapped_value = bootstrapped_value.mean(dim=1)
                    if bootstrapped_value.dim() > 1:
                        bootstrapped_value = bootstrapped_value.squeeze(0)
                    values_ = torch.cat((values_, bootstrapped_value), dim=0)
                adv, ret = self.get_advantages(values_.detach(), masks[i], rewards[i].detach())

                advantages.append(adv)
                returns.append(ret)

        # Merge all agent states, actions, rewards etc.
        advantages = torch.cat(advantages)
        returns = torch.cat(returns)
        actions = torch.cat(actions)
        old_logprobs = torch.cat(old_logprobs)
        states = memory.rearrange_states(states)
        #v_masks = memory.rearrange_masks(variable_state_masks) if self.use_variable_state_masks else None

        # Logging before proceeding to batched training
        logger.add_metric("Rewards/Returns", returns.detach().cpu().numpy())
        logger.add_metric("Rewards/Advantages", advantages.detach().cpu().numpy())
        logger.add_metric("Sanitylogs/old_logprobs", old_logprobs.detach().cpu().numpy())

        # Train policy for K epochs: sampling and updating
        for _ in range(self.k_epochs):
            epoch_values = []
            epoch_returns = []
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.horizon)), mini_batch_size, False):
                # Evaluate old actions and values using current policy
                batch_states = tuple(state[index] for state in states)
                batch_actions = actions[index]
                batch_variable_masks = variable_state_masks[0][index] if self.use_variable_state_masks and len(variable_state_masks[0]) > 0 else None
                logprobs, values, dist_entropy = self.policy.evaluate(batch_states, batch_actions, batch_variable_masks)

                # Importance ratio: p/q
                ratios = torch.exp(logprobs - old_logprobs[index].detach())

                # Approximate KL Divergence
                # approxkl = 0.5 * torch.mean(torch.square(old_logprobs[index].detach() - logprobs))
                # if approxkl > 0.02:
                #     console += (f"Approximate Kulback-Leibler Divergence: {approxkl}. A value above 0.02 is considered bad since the policy changes too quickly.\n"
                #                 f"This happend in epoch {_} and mini_batch {index}. Early stopping for this training peroid.\n")
                #     break

                # Actor loss using Surrogate loss
                surr1 = ratios * advantages[index]
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[index]
                entropy = self.entropy_coeff * dist_entropy[0]
                actor_loss = (-torch.min(surr1, surr2).type(torch.float32)).mean() - entropy
                # actor_loss = -torch.mean(torch.min(ratios * advantages[index], torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[index]))

                # TODO CLIP VALUE LOSS ? Probably not necessary as according to:
                # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
                value_loss = self.MSE_loss(values, returns[index].squeeze())

                # Log loss
                logger.add_metric("Loss/critic_loss", value_loss.detach().cpu().numpy())
                logger.add_metric("Loss/actor_loss", actor_loss.detach().cpu().numpy())
                logger.add_metric("Loss/log_std", torch.exp(self.policy.actor.log_std).detach().cpu().numpy())
                logging_values.append(values.detach().cpu().numpy())
                epoch_values.append(values.detach().cpu().numpy())
                epoch_returns.append(returns[index].detach().cpu().numpy())

                # Sanity checks
                assert not torch.isnan(value_loss).any(), f"Critic Loss is NaN, check returns, values or entroy"
                assert not torch.isnan(actor_loss).any()
                assert not torch.isinf(value_loss).any()
                assert not torch.isinf(actor_loss).any()

                # Backward gradients
                if not self.separate_optimizers:
                    loss = actor_loss + value_loss * self.value_loss_coef
                    self.optimizer_a.zero_grad()
                    loss.backward()
                    # Global gradient norm clipping https://vitalab.github.io/article/2020/01/14/Implementation_Matters.html
                    torch.nn.utils.clip_grad_norm_(self.actor_params, max_norm=0.5)
                    torch.nn.utils.clip_grad_norm_(self.critic_params, max_norm=0.5)
                    self.optimizer_a.step()
                else:
                    self.optimizer_a.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    # Global gradient norm clipping https://vitalab.github.io/article/2020/01/14/Implementation_Matters.html
                    torch.nn.utils.clip_grad_norm_(self.actor_params, max_norm=0.5)
                    self.optimizer_a.step()

                    self.optimizer_c.zero_grad()
                    value_loss.backward()
                    # Global gradient norm clipping https://vitalab.github.io/article/2020/01/14/Implementation_Matters.html
                    torch.nn.utils.clip_grad_norm_(self.critic_params, max_norm=0.5)
                    self.optimizer_c.step()

            # Compute explained variance over the epoch
            epoch_values = np.concatenate(epoch_values)
            epoch_returns = np.concatenate(epoch_returns)
            ev = self.calculate_explained_variance(epoch_values, epoch_returns)
            # if ev <= 0:
            #     console += (f"Explained Variance for Epoch {_}: {ev}\n"
            #                 f"This is bad. The Critic might aswell have predicted zero or is even doing worse than that.\n")
            logger.add_metric("Sanitylogs/Explained_Variance", ev)


        logger.add_metric("Values", np.concatenate(logging_values).flatten())