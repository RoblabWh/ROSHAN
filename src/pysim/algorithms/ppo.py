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

        if self.use_torch_compile:
            try:
                self.policy.actor = torch.compile(self.policy.actor, mode=self.compile_mode)
                self.policy.critic = torch.compile(self.policy.critic, mode=self.compile_mode)
            except Exception as e:
                self.logger.warning(f"torch.compile failed, using eager mode: {e}")

        self.actor_params = self.policy.actor.parameters()
        self.critic_params = self.policy.critic.value.parameters() if self.share_encoder else self.policy.critic.parameters()

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
        Computes the advantages using vectorized GAE (Generalized Advantage Estimation).

        :param values: The values of the states. May have len(rewards)+1 if bootstrapped.
        :param masks: The masks of the states (1 = not done, 0 = done).
        :param rewards: The rewards of the states.
        :return: The advantages and returns as tensors on self.device.
        """
        # Ensure tensors are on the correct device
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        elif rewards.device != self.device:
            rewards = rewards.to(self.device, non_blocking=True)
        if not isinstance(masks, torch.Tensor):
            masks = torch.as_tensor(masks, dtype=torch.float32, device=self.device)
        elif masks.device != self.device:
            masks = masks.to(self.device, non_blocking=True)
        if not isinstance(values, torch.Tensor):
            values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        elif values.device != self.device:
            values = values.to(self.device, non_blocking=True)

        T = len(rewards)
        # next_values: values[1:] when bootstrapped, else zeros for terminal
        next_values = values[1:T+1] if len(values) > T else torch.zeros_like(rewards)
        deltas = rewards + self.gamma * masks * next_values - values[:T]

        # Vectorized GAE: reverse scan with discount factor gamma * lambda
        advantages = torch.zeros_like(deltas)
        gae = torch.tensor(0.0, device=self.device)
        for t in reversed(range(T)):
            gae = deltas[t] + self.gamma * self._lambda * masks[t] * gae
            advantages[t] = gae

        returns = advantages + values[:T]
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

        # Advantages — single batched evaluate across all agents
        with (torch.no_grad()):
            all_states = memory.rearrange_states(states)
            all_actions = torch.cat(actions)
            all_mask_arg = (
                torch.cat(variable_state_masks)
                if self.use_variable_state_masks and len(variable_state_masks[0]) > 0
                else None
            )
            _, all_values, _ = self.policy.evaluate(all_states, all_actions, all_mask_arg)

            # Split values back per-agent for bootstrap + GAE
            agent_sizes = [actions[i].shape[0] for i in range(memory.num_agents)]
            values_per_agent = torch.split(all_values.squeeze(-1), agent_sizes)

            advantages = []
            returns = []
            for i in range(memory.num_agents):
                values_ = values_per_agent[i]
                if masks[i][-1] == 1:
                    last_state = tuple(torch.as_tensor(np.array(state), dtype=torch.float32).to(self.device, non_blocking=True) if np.array(state).dtype != bool else torch.as_tensor(np.array(state), dtype=torch.bool).to(self.device, non_blocking=True) for state in memory.get_agent_state(next_obs, i))
                    bootstrapped_value = self.policy.critic(last_state).detach()
                    if bootstrapped_value.dim() != 1:
                        bootstrapped_value = bootstrapped_value.mean(dim=1)
                    if bootstrapped_value.dim() > 1:
                        bootstrapped_value = bootstrapped_value.squeeze(0)
                    values_ = torch.cat((values_, bootstrapped_value), dim=0)
                adv, ret = self.get_advantages(values_.detach(), masks[i], rewards[i].detach())

                advantages.append(adv)
                returns.append(ret)

        # Merge advantages/returns; reuse already-merged states and actions
        advantages = torch.cat(advantages)
        returns = torch.cat(returns)
        old_logprobs = torch.cat(old_logprobs).detach()
        states = all_states
        actions = all_actions
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
                batch_old_logprobs = old_logprobs[index]
                # Joint log_probs for multi-discrete actions
                if batch_old_logprobs.dim() == 2:
                    batch_old_logprobs = batch_old_logprobs.sum(dim=1)
                logprobs, values, dist_entropy = self.policy.evaluate(batch_states, batch_actions, batch_variable_masks)

                # Importance ratio: p/q
                # Clamp the ratios for stability (higher LRs can cause overflow, which results in NaNs)
                log_ratio = logprobs - batch_old_logprobs
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

                # Log loss — single CPU transfer for values (used by both logging_values and epoch_values)
                with torch.no_grad():
                    values_np = values.detach().cpu().numpy()
                    logging_values.append(values_np)
                    epoch_values.append(values_np)
                    epoch_returns.append(returns[index].detach().cpu().numpy())
                    clip_fraction = ((ratios < (1 - self.eps_clip)) | (ratios > (1 + self.eps_clip))).float().mean().item()
                    approx_kl = (batch_old_logprobs - logprobs).mean().detach().item()
                    if not self.use_kl_1:
                        approx_kl = (ratios - 1 - log_ratio).mean().detach().item()
                    # Stop the policy updates early when the new policy diverges too much from the old one
                    if self.kl_early_stop and approx_kl >= self.kl_target:
                        self.logger.warning(
                            f"approx_kl({approx_kl:.3f}) exceeded target kl({self.kl_target}) at update {logger.train_step} and K_Epoch {_ + 1}.")
                        break
                    logger.add_metric("Training/Approx_KL", approx_kl)
                    logger.add_metric("Training/Clip_Fraction", clip_fraction)
                    logger.add_metric("Training/value_loss", value_loss.detach().item())
                    logger.add_metric("Training/actor_loss", actor_loss.detach().item())
                    logger.add_metric("Training/entropy_loss", entropy_loss.detach().item())
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
                    # Combine losses and backward once to avoid retain_graph=True,
                    # which would keep the entire computation graph in memory.
                    # Each optimizer still steps only its own parameter group.
                    self.optimizer_a.zero_grad()
                    self.optimizer_c.zero_grad()
                    total_loss = actor_loss + value_loss * self.value_loss_coef
                    total_loss.backward()
                    # Global gradient norm clipping https://vitalab.github.io/article/2020/01/14/Implementation_Matters.html
                    torch.nn.utils.clip_grad_norm_(self.actor_params, max_norm=0.5)
                    torch.nn.utils.clip_grad_norm_(self.critic_params, max_norm=0.5)
                    self.optimizer_a.step()
                    self.optimizer_c.step()
                    self.scheduler_a.step()
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