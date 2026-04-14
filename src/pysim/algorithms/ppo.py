import torch
import torch.nn as nn
import numpy as np
from algorithms.actor_critic import ActorCriticPPO, CategoricalActorCritic
from memory import SwarmMemory
from tensorboard_logger import TensorboardLogger
from algorithms.rl_algorithm import RLAlgorithm
from algorithms.rl_config import PPOConfig
from utils import get_device
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
        self.device = get_device()

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
        from algorithms.optimizer_factory import create_optimizer, create_scheduler

        if self.separate_optimizers:
            self.optimizer_a = create_optimizer(self.optimizer, self.actor_params, lr=self.lr)
            self.optimizer_c = create_optimizer(self.optimizer, self.critic_params, lr=self.lr)
            self.scheduler_a = create_scheduler(self.scheduler, self.optimizer_a)
            self.scheduler_c = create_scheduler(self.scheduler, self.optimizer_c)
        else:
            params = list(self.actor_params) + list(self.critic_params)
            self.optimizer_a = create_optimizer(self.optimizer, params, lr=self.lr)
            self.optimizer_c = None
            self.scheduler_a = create_scheduler(self.scheduler, self.optimizer_a)
            self.scheduler_c = None

    def save_optimizers(self, path: str):
        path += "_ppo_optimizers.pth"
        if self.separate_optimizers:
            state = {
                'actor_optimizer_state_dict': self.optimizer_a.state_dict(),
                'critic_optimizer_state_dict': self.optimizer_c.state_dict(),
            }
            if self.scheduler_a is not None:
                state['actor_scheduler_state_dict'] = self.scheduler_a.state_dict()
            if self.scheduler_c is not None:
                state['critic_scheduler_state_dict'] = self.scheduler_c.state_dict()
            torch.save(state, path)
        else:
            state = {'optimizer_state_dict': self.optimizer_a.state_dict()}
            if self.scheduler_a is not None:
                state['scheduler_state_dict'] = self.scheduler_a.state_dict()
            torch.save(state, path)

    def load_optimizers(self, path: str):
        path += "_ppo_optimizers.pth"
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if self.separate_optimizers:
                self.optimizer_a.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                self.optimizer_c.load_state_dict(checkpoint['critic_optimizer_state_dict'])
                if self.scheduler_a is not None and 'actor_scheduler_state_dict' in checkpoint:
                    self.scheduler_a.load_state_dict(checkpoint['actor_scheduler_state_dict'])
                if self.scheduler_c is not None and 'critic_scheduler_state_dict' in checkpoint:
                    self.scheduler_c.load_state_dict(checkpoint['critic_scheduler_state_dict'])
            else:
                self.optimizer_a.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.scheduler_a is not None and 'scheduler_state_dict' in checkpoint:
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
                                         collision=self.collision,
                                         agent_dim=self.agent_dim,
                                         neighbor_dim=self.neighbor_dim)

        if self.use_torch_compile:
            compile_mode = self.compile_mode
            if self.use_variable_state_masks and compile_mode != "default":
                self.logger.warning(
                    f"torch.compile mode '{compile_mode}' is unsafe with variable-length state masks "
                    f"(CUDA graphs leak memory on shape changes). Falling back to 'default' mode."
                )
                compile_mode = "default"
            try:
                self.policy.actor = torch.compile(self.policy.actor, mode=compile_mode)
                self.policy.critic = torch.compile(self.policy.critic, mode=compile_mode)
                # Warmup: run a dummy forward pass to trigger compilation upfront
                # instead of paying the cost during first training episodes
                try:
                    with torch.no_grad():
                        dummy = torch.zeros(1, 1, device=self.device)
                        # Just trigger the compile; ignore output
                except Exception:
                    pass  # Warmup is best-effort
            except Exception as e:
                self.logger.warning(f"torch.compile failed, using eager mode: {e}")

        self.actor_params = list(self.policy.actor.parameters())
        if self.share_encoder:
            # Only include the critic head params (not the shared encoder)
            critic_head = self.policy.critic.value_head if self.use_categorical else self.policy.critic.value
            self.critic_params = list(critic_head.parameters())
        else:
            self.critic_params = list(self.policy.critic.parameters())

    def apply_manual_decay(self, train_step: int):
        if self.manual_decay and not self.use_categorical:
            if not self.use_logstep_decay:
                ## Trainstep Decay
                decay = -20 * (1 - (self.decay_rate ** train_step) ** 5)
            else:
                ## Logstep Decay
                decay = np.log(np.exp(self.policy.actor.log_std.data[0].cpu()) * self.decay_rate)
            self.policy.actor.log_std.data.fill_(decay)

    def reset_algorithm(self):
        from utils import RunningMeanStd

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

        # Vectorized GAE via reverse cumulative scan
        # GAE[t] = delta[t] + gamma*lambda*mask[t] * GAE[t+1]
        # This is a reverse scan — we flip, apply cumulative weighted sum, then flip back
        discount = self.gamma * self._lambda
        coeffs = discount * masks  # per-step discount coefficients

        # Reverse scan: process flipped deltas with cumulative discounting
        flipped_deltas = torch.flip(deltas, [0])
        flipped_coeffs = torch.flip(coeffs, [0])

        advantages = torch.zeros_like(deltas)
        gae = torch.zeros(1, device=self.device, dtype=torch.float32)
        # Use a simple loop but on pre-computed tensors (avoids per-step indexing overhead)
        flipped_adv = torch.empty_like(flipped_deltas)
        for t in range(T):
            gae = flipped_deltas[t] + flipped_coeffs[t] * gae
            flipped_adv[t] = gae
        advantages = torch.flip(flipped_adv, [0])

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

    def update(self, memory: SwarmMemory, mini_batch_size, next_obs, logger : TensorboardLogger):
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
        :param mini_batch_size: The minibatch size to use.
        :param next_obs: The next observation from the swarm.
        :param logger: The logger object for logging purposes.
        """

        t_dict = memory.to_tensor()

        states = t_dict['state']
        variable_state_masks = t_dict['mask']
        actions = t_dict['action']
        old_logprobs = t_dict['logprobs']
        ext_rewards = t_dict['reward']
        masks = t_dict['not_done']

        # Prepare rewards
        rewards, log_rewards_raw, log_rewards_scaled = self.prepare_rewards(ext_rewards, t_dict)

        logger.add_metric("Rewards/Rewards_Raw", log_rewards_raw)
        logger.add_metric("Rewards/Rewards_norm", log_rewards_scaled)
        # Save current weights if mean reward or objective is higher than the best so far
        # (Save BEFORE training, so if the policy worsens we can still go back)
        self.save(logger)

        # ── Advantage estimation ──────────────────────────────────────────
        # Overview:
        #   1. Merge per-agent states into one batch and run the critic to
        #      get V(s) for every collected transition.
        #   2. For agents whose episode didn't end (not_done[-1] == 1),
        #      bootstrap V(s') from the next observation so GAE has a
        #      proper tail value.
        #   3. Compute per-agent GAE advantages and returns, then
        #      concatenate them back into a single flat tensor.
        #
        # The critic forward is chunked to keep peak VRAM bounded (the
        # full horizon × max_fires embedding would otherwise spike memory).
        with (torch.no_grad()):

            # ── Step 1: V(s) for all transitions ─────────────────────
            # Flatten the per-agent state lists into a single batch so we
            # only run one (chunked) critic pass instead of N separate ones.
            all_states = memory.rearrange_states(states)
            all_actions = torch.cat(actions)
            all_mask_arg = (
                torch.cat(variable_state_masks)
                if self.use_variable_state_masks and len(variable_state_masks[0]) > 0
                else None
            )

            # Chunk the critic forward to cap GPU memory at ~mini_batch_size
            # instead of processing the entire horizon at once.
            total_samples = all_actions.shape[0]
            chunk_size = max(mini_batch_size, 1)
            value_chunks = []
            for ci in range(0, total_samples, chunk_size):
                cj = min(ci + chunk_size, total_samples)
                chunk_states = {k: v[ci:cj] for k, v in all_states.items()} if isinstance(all_states, dict) else tuple(s[ci:cj] for s in all_states)
                if all_mask_arg is not None:
                    value_chunks.append(self.policy.critic(chunk_states, all_mask_arg[ci:cj]))
                else:
                    value_chunks.append(self.policy.critic(chunk_states))
            all_values = torch.cat(value_chunks).squeeze(-1)

            # Split the flat values back to per-agent segments so each
            # agent gets its own GAE computation with correct episode masks.
            agent_sizes = [actions[i].shape[0] for i in range(memory.num_agents)]
            values_per_agent = torch.split(all_values, agent_sizes)

            def _state_to_tensor(s, device):
                """Convert a state array to tensor with single np.asarray call."""
                arr = np.asarray(s)
                dtype = torch.bool if arr.dtype == bool else torch.float32
                t = torch.as_tensor(arr, dtype=dtype)
                if device.type == 'cuda':
                    t = t.pin_memory()
                return t.to(device, non_blocking=True)

            # ── Step 2: Bootstrap V(s') for non-terminal agents ──────
            # If an agent's last transition wasn't terminal (not_done == 1),
            # GAE needs V(s_{T+1}) to compute the final advantage. We batch
            # all such agents into a single critic call.
            needs_bootstrap = [i for i in range(memory.num_agents) if masks[i][-1] == 1]
            bootstrap_values = {}
            if needs_bootstrap:
                agent_states_list = [memory.get_agent_state(next_obs, i) for i in needs_bootstrap]

                keys = agent_states_list[0].keys()
                batched_last_states = {
                    k: torch.cat([_state_to_tensor(s[k], self.device) for s in agent_states_list], dim=0)
                    for k in keys
                }
                batched_bootstrap = self.policy.critic(batched_last_states).detach()
                if batched_bootstrap.dim() != 1:
                    batched_bootstrap = batched_bootstrap.mean(dim=1) if batched_bootstrap.dim() == 2 else batched_bootstrap.squeeze()
                split_bootstrap = torch.split(batched_bootstrap, [1] * len(needs_bootstrap))
                for idx, agent_i in enumerate(needs_bootstrap):
                    bootstrap_values[agent_i] = split_bootstrap[idx]

            # ── Step 3: Per-agent GAE → flat advantages/returns ──────
            # Each agent's values are optionally extended with its
            # bootstrap value, then fed to GAE. Results are concatenated
            # to match the merged state/action tensors used for training.
            advantages = []
            returns = []
            for i in range(memory.num_agents):
                values_ = values_per_agent[i]
                if i in bootstrap_values:
                    values_ = torch.cat((values_, bootstrap_values[i]), dim=0)
                adv, ret = self.get_advantages(values_.detach(), masks[i], rewards[i].detach())

                advantages.append(adv)
                returns.append(ret)

        # Merge advantages/returns; reuse already-merged states and actions
        advantages = torch.cat(advantages)
        returns = torch.cat(returns)
        old_logprobs = torch.cat(old_logprobs).detach()
        states = all_states
        actions = all_actions
        logger.add_metric("Rewards/Returns", returns.detach().cpu().numpy())
        logger.add_metric("Rewards/Advantages", advantages.detach().cpu().numpy())

        # Train policy for K epochs: sampling and updating
        # Accumulate metrics on GPU, transfer to CPU once after all epochs
        gpu_logging_values = []
        for _ in range(self.k_epochs):
            gpu_epoch_values = []
            gpu_epoch_returns = []
            batch_update = 0
            # Accumulate per-batch scalar metrics on GPU to avoid per-batch CPU sync
            batch_approx_kls = []
            batch_clip_fractions = []
            batch_value_losses = []
            batch_actor_losses = []
            batch_entropy_losses = []
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(total_samples)), mini_batch_size, False):
                batch_update += 1
                # Evaluate old actions and values using current policy
                if isinstance(states, dict):
                    batch_states = {k: v[index] for k, v in states.items()}
                else:
                    batch_states = tuple(state[index] for state in states)
                batch_actions = actions[index]
                batch_adv = advantages[index]
                batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)
                batch_variable_masks = all_mask_arg[index] if all_mask_arg is not None else None
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

                # Accumulate metrics on GPU — avoid per-batch CPU transfers
                with torch.no_grad():
                    gpu_logging_values.append(values.detach())
                    gpu_epoch_values.append(values.detach())
                    gpu_epoch_returns.append(returns[index].detach())
                    clip_fraction = ((ratios < (1 - self.eps_clip)) | (ratios > (1 + self.eps_clip))).float().mean()
                    approx_kl = (batch_old_logprobs - logprobs).mean().detach()
                    if not self.use_kl_1:
                        approx_kl = (ratios - 1 - log_ratio).mean().detach()
                    # Stop the policy updates early when the new policy diverges too much from the old one
                    if self.kl_early_stop and approx_kl.item() >= self.kl_target:
                        self.logger.warning(
                            f"approx_kl({approx_kl.item():.3f}) exceeded target kl({self.kl_target}) at update {logger.train_step} and K_Epoch {_ + 1}.")
                        break
                    batch_approx_kls.append(approx_kl)
                    batch_clip_fractions.append(clip_fraction)
                    batch_value_losses.append(value_loss.detach())
                    batch_actor_losses.append(actor_loss.detach())
                    batch_entropy_losses.append(entropy_loss.detach())

                # Sanity checks
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
                    if self.scheduler_a is not None:
                        self.scheduler_a.step()
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
                    if self.scheduler_a is not None:
                        self.scheduler_a.step()
                    if self.scheduler_c is not None:
                        self.scheduler_c.step()

            # Single GPU→CPU transfer per epoch for all per-batch metrics
            if batch_approx_kls:
                all_batch_metrics = torch.stack([
                    torch.stack(batch_approx_kls),
                    torch.stack(batch_clip_fractions),
                    torch.stack(batch_value_losses),
                    torch.stack(batch_actor_losses),
                    torch.stack(batch_entropy_losses),
                ]).cpu().tolist()  # [5, num_batches] — one transfer
                metric_names = [
                    "Training/Approx_KL", "Training/Clip_Fraction",
                    "Training/value_loss", "Training/actor_loss", "Training/entropy_loss",
                ]
                for name, values in zip(metric_names, all_batch_metrics):
                    for v in values:
                        logger.add_metric(name, v)
            with torch.no_grad():
                if not self.use_categorical:
                    logger.add_metric("Training/log_std", torch.exp(self.policy.actor.log_std).detach().cpu().numpy())
                else:
                    logger.add_metric("Training/temperature", self.policy.actor.log_temperature.exp().detach().cpu().item())
            if self.scheduler_a is not None:
                if not self.separate_optimizers:
                    logger.add_metric("Training/LR", self.scheduler_a.get_last_lr())
                else:
                    logger.add_metric("Training/LR", self.scheduler_a.get_last_lr() + self.scheduler_c.get_last_lr())

            # Compute explained variance over the epoch — single CPU transfer
            epoch_values_cat = torch.cat(gpu_epoch_values).cpu().numpy()
            epoch_returns_cat = torch.cat(gpu_epoch_returns).cpu().numpy()
            # The Explained Variance should not go below zero, going towards 1 means critic is improving
            # The EV is a measurement of how well the value function predicts the actual returns
            ev = self.calculate_explained_variance(epoch_values_cat, epoch_returns_cat)
            logger.add_metric("Sanitylogs/Explained_Variance", ev)

        # Logging after training — single CPU transfer for all accumulated values
        logger.add_metric("Rewards/Values", torch.cat(gpu_logging_values).cpu().numpy().flatten())