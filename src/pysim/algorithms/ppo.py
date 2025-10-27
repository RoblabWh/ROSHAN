import torch
import torch.nn as nn
import os
import numpy as np
from algorithms.actor_critic import ActorCriticPPO, CategoricalActorCritic
from memory import SwarmMemory
from evaluation import TensorboardLogger
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

        self.actor_network = network[0]
        self.critic_network = network[1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Current Policy
        self.policy = None
        self.actor_params = None
        self.critic_params = None
        self.initialize_policy()

        self.optimizer_a = None
        self.optimizer_c = None
        self.scheduler_a = None
        self.scheduler_c = None
        self.initialize_optimizers()

        self.MSE_loss = nn.MSELoss()
        self.set_train()

    def initialize_optimizers(self):
        """
        Initializes the optimizers for the actor and critic networks.
        This method is called after the policy is reset or loaded.
        """
        # --- Scheduler initialization ---
        def linear_lr_schedule(current_step: int):
            """Linear decay from initial lr to 0 over total_steps."""
            progress = min(current_step / total_steps, 1.0)
            return 1.0 - progress

        if self.separate_optimizers:
            self.optimizer_a = torch.optim.Adam(self.actor_params, lr=self.lr, betas=(self.beta1, self.beta2), eps=1e-5)
            self.optimizer_c = torch.optim.Adam(self.critic_params, lr=self.lr, betas=(self.beta1, self.beta2), eps=1e-5)
            self.scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_a, T_max=int(1e6))
            self.scheduler_c = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_c, T_max=int(1e6))
        else:
            params = list(self.actor_params) + list(self.critic_params)
            self.optimizer_a = torch.optim.Adam(params, lr=self.lr, betas=(self.beta1, self.beta2), eps=1e-5)
            self.optimizer_c = None
            self.scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_a, T_max=int(1e6))
            self.scheduler_c = None

    def save_optimizers(self, path: str):
        path += "_ppo_optimizers.pth"
        if self.separate_optimizers:
            torch.save({
                'actor_optimizer_state_dict': self.optimizer_a.state_dict(),
                'critic_optimizer_state_dict': self.optimizer_c.state_dict(),
                'actor_scheduler_state_dict': self.scheduler_a.state_dict(),
                'critic_scheduler_state_dict': self.scheduler_c.state_dict(),
            }, path)
        else:
            torch.save({
                'optimizer_state_dict': self.optimizer_a.state_dict(),
                'scheduler_state_dict': self.scheduler_a.state_dict(),
            }, path)

    def load_optimizers(self, path: str):
        path += "_ppo_optimizers.pth"
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if self.separate_optimizers:
                self.optimizer_a.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                self.optimizer_c.load_state_dict(checkpoint['critic_optimizer_state_dict'])
                self.scheduler_a.load_state_dict(checkpoint['actor_scheduler_state_dict'])
                self.scheduler_c.load_state_dict(checkpoint['critic_scheduler_state_dict'])
            else:
                self.optimizer_a.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler_a.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            self.logger.warning(f"Error loading optimizers from {path}: {e}.")

    def initialize_policy(self):
        """
        Initializes the policy with the actor and critic networks.
        This method is called when the algorithm is reset or loaded.
        """
        if self.use_categorical:
            self.policy = CategoricalActorCritic(actor_network=self.actor_network,
                                                 critic_network=self.critic_network,
                                                 vision_range=self.vision_range,
                                                 drone_count=self.drone_count,
                                                 map_size=self.map_size,
                                                 time_steps=self.time_steps,
                                                 manual_decay=self.manual_decay,
                                                 share_encoder=self.share_encoder)
        else:
            self.policy = ActorCriticPPO(actor_network=self.actor_network,
                                         critic_network=self.critic_network,
                                         vision_range=self.vision_range,
                                         drone_count=self.drone_count,
                                         map_size=self.map_size,
                                         time_steps=self.time_steps,
                                         share_encoder=self.share_encoder,
                                         manual_decay=self.manual_decay,
                                         use_tanh_dist=self.use_tanh_dist,
                                         collision=self.collision)

        self.actor_params = self.policy.actor.parameters()
        self.critic_params = self.policy.critic.parameters()

    def apply_manual_decay(self, train_step: int):
        if self.manual_decay:
            if not self.use_logstep_decay:
                ## Trainstep Decay
                decay = -20 * (1 - (self.decay_rate ** train_step) ** 5)
            else:
                ## Logstep Decay
                decay = np.log(np.exp(self.policy.actor.log_std.data[0].cpu()) * self.decay_rate)
            self.policy.actor.log_std.data.fill_(decay)

    def reset_algorithm(self):
        self.initialize_policy()
        self.initialize_optimizers()
        self.MSE_loss = nn.MSELoss()
        self.reward_rms = RunningMeanStd()
        self.int_reward_rms = RunningMeanStd()
        self.set_train()

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
            advantages.append(gae)
            returns.append(gae + values[i])

        # Reverse to restore original order
        advantages = torch.FloatTensor(list(reversed(advantages))).to(self.device)
        returns = torch.FloatTensor(list(reversed(returns))).to(self.device)

        # Norm advantages (Norm the adv in mini_batch)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        return advantages, returns

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

        logging_values = []
        # Prepare rewards
        rewards, log_rewards_raw, log_rewards_scaled = self.prepare_rewards(ext_rewards, t_dict)

        logger.add_metric("Rewards/Rewards_Raw", log_rewards_raw)
        logger.add_metric("Rewards/Rewards_norm", log_rewards_scaled)
        # Save current weights if mean reward or objective is higher than the best so far
        # (Save BEFORE training, so if the policy worsens we can still go back)
        self.save(logger)

        # Advantages
        with (torch.no_grad()):
            advantages = []
            returns = []
            for i in range(memory.num_agents):
                _, values_, _ = self.policy.evaluate(states[i], actions[i], variable_state_masks[0] if self.use_variable_state_masks and len(variable_state_masks[0]) > 0 else None)
                if masks[i][-1] == 1:
                    last_state = tuple(torch.FloatTensor(np.array(state)).to(self.device) if np.array(state).dtype!=bool else torch.BoolTensor(np.array(state)).to(self.device) for state in memory.get_agent_state(next_obs, i))
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
        old_logprobs = torch.cat(old_logprobs).detach()
        states = memory.rearrange_states(states)
        #v_masks = memory.rearrange_masks(variable_state_masks) if self.use_variable_state_masks else None
        logger.add_metric("Rewards/Returns", returns.detach().cpu().numpy())
        logger.add_metric("Rewards/Advantages", advantages.detach().cpu().numpy())

        # Train policy for K epochs: sampling and updating
        for _ in range(self.k_epochs):
            epoch_values = []
            epoch_returns = []
            batch_update = 0
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.horizon)), mini_batch_size, False):
                batch_update += 1
                # Evaluate old actions and values using current policy
                batch_states = tuple(state[index] for state in states)
                batch_actions = actions[index]
                batch_adv = advantages[index]
                batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)
                batch_variable_masks = variable_state_masks[0][index] if self.use_variable_state_masks and len(variable_state_masks[0]) > 0 else None
                logprobs, values, dist_entropy = self.policy.evaluate(batch_states, batch_actions, batch_variable_masks)

                # Importance ratio: p/q
                # Clamp the ratios for stability (higher LRs can cause overflow, which results in NaNs)
                log_ratio = logprobs - old_logprobs[index]
                clamped_ratio = torch.clamp(log_ratio, -10, 10)  # avoid overflow
                ratios = torch.exp(clamped_ratio)

                # Actor loss using Surrogate loss
                surr1 = ratios * batch_adv
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_adv
                entropy_loss = -self.entropy_coeff * dist_entropy.mean()

                actor_loss = (-torch.min(surr1, surr2).type(torch.float32)).mean()

                # Value Loss Clipping is not helpful according to:
                # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
                value_loss = self.MSE_loss(values, returns[index].squeeze())

                # Log loss
                with torch.no_grad():
                    logging_values.append(values.detach().cpu().numpy())
                    epoch_values.append(values.detach().cpu().numpy())
                    epoch_returns.append(returns[index].detach().cpu().numpy())
                    clip_fraction = ((ratios < (1 - self.eps_clip)) | (ratios > (1 + self.eps_clip))).float().mean().cpu().numpy()
                    approx_kl = (old_logprobs[index] - logprobs).mean().detach().cpu().numpy()
                    if not self.use_kl_1:
                        approx_kl = ((ratios - 1 - log_ratio).mean()).detach().cpu().numpy()
                    # Stop the policy updates early when the new policy diverges too much from the old one
                    if self.kl_early_stop and approx_kl >= self.kl_target:
                        self.logger.warning(
                            f"approx_kl({approx_kl:.3f}) exceeded target kl({self.kl_target}) at update {logger.train_step} and K_Epoch {_ + 1}.")
                        break
                    logger.add_metric("Training/Approx_KL", approx_kl)
                    logger.add_metric("Training/Clip_Fraction", clip_fraction)
                    logger.add_metric("Training/value_loss", value_loss.detach().cpu().numpy())
                    logger.add_metric("Training/actor_loss", actor_loss.detach().cpu().numpy())
                    logger.add_metric("Training/entropy_loss", entropy_loss.detach().cpu().numpy())
                    logger.add_metric("Training/log_std", torch.exp(self.policy.actor.log_std).detach().cpu().numpy())

                # Sanity checks
                # torch.cuda.synchronize()  # catch deferred CUDA errors
                assert isinstance(value_loss, torch.Tensor), f"value_loss not a tensor: {type(value_loss)}"
                assert isinstance(actor_loss, torch.Tensor), f"actor_loss not a tensor: {type(actor_loss)}"
                assert not torch.isnan(value_loss).any(), f"Critic Loss is NaN, check returns, values"
                assert not torch.isnan(entropy_loss).any(), f"Entropy Loss is NaN somewhere"
                assert not torch.isnan(actor_loss).any(), f"Actor Loss is NaN, check advantages: {torch.isnan(batch_adv).any()} or logprobs: {torch.isnan(logprobs).any()}"
                assert not torch.isinf(value_loss).any(), f"Critic Loss is Inf somewhere"
                assert not torch.isinf(entropy_loss).any(), f"Entropy Loss is Inf somewhere"
                assert not torch.isinf(actor_loss).any(), f"Actor Loss is Inf, check advantages or logprobs"

                # Add Entropy to actor loss
                actor_loss = actor_loss + entropy_loss

                # Backward gradients
                if not self.separate_optimizers:
                    loss = actor_loss + value_loss * self.value_loss_coef
                    self.optimizer_a.zero_grad()
                    loss.backward()
                    # Global gradient norm clipping https://vitalab.github.io/article/2020/01/14/Implementation_Matters.html
                    torch.nn.utils.clip_grad_norm_(self.actor_params, max_norm=0.5)
                    torch.nn.utils.clip_grad_norm_(self.critic_params, max_norm=0.5)
                    self.optimizer_a.step()
                    self.scheduler_a.step()
                    logger.add_metric("Training/LR", self.scheduler_a.get_last_lr())
                else:
                    self.optimizer_a.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    # Global gradient norm clipping https://vitalab.github.io/article/2020/01/14/Implementation_Matters.html
                    torch.nn.utils.clip_grad_norm_(self.actor_params, max_norm=0.5)
                    self.optimizer_a.step()
                    self.scheduler_a.step()

                    self.optimizer_c.zero_grad()
                    value_loss.backward()
                    # Global gradient norm clipping https://vitalab.github.io/article/2020/01/14/Implementation_Matters.html
                    torch.nn.utils.clip_grad_norm_(self.critic_params, max_norm=0.5)
                    self.optimizer_c.step()
                    self.scheduler_c.step()
                    logger.add_metric("Training/LR", self.scheduler_a.get_last_lr() + self.scheduler_c.get_last_lr())

            # Compute explained variance over the epoch
            epoch_values = np.concatenate(epoch_values)
            epoch_returns = np.concatenate(epoch_returns)
            # The Explained Variance should not go below zero, going towards 1 means critic is improving
            # The EV is a measurement of how well the value function predicts the actual returns
            ev = self.calculate_explained_variance(epoch_values, epoch_returns)
            logger.add_metric("Sanitylogs/Explained_Variance", ev)

        # Logging after training
        logger.add_metric("Rewards/Values", np.concatenate(logging_values).flatten())