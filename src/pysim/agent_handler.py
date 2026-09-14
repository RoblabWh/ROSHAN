import os
import logging
import numpy as np
import firesim
from observation_dict import ObservationDict


class FrameSkipController:
    """Manages frame-skip action repetition state machine.

    ``cached_actions`` is what the engine executes; ``cached_sampled_actions`` is
    what the policy actually sampled. They differ only when goal-commitment is
    active (PlannerAgent.apply_commitment) — for everyone else they're the same
    object. Memory stores ``cached_sampled_actions`` so the importance ratio in
    PPO stays consistent with the policy at sample time.
    """
    __slots__ = ('frame_skips', 'counter', 'cached_actions', 'cached_sampled_actions',
                 'cached_logprobs', 'cached_locked_mask')

    def __init__(self, frame_skips):
        self.frame_skips = frame_skips
        self.reset()

    @property
    def at_decision_point(self) -> bool:
        return self.counter % self.frame_skips == 0

    def advance(self, env_reset) -> bool:
        """Increment counter, return True if this is an end-of-control boundary."""
        self.counter += 1
        return env_reset or (self.counter % self.frame_skips == 0)

    def cache(self, actions, logprobs=None, locked_mask=None, sampled_actions=None):
        self.cached_actions = actions
        # Default: sampled == executed (no commitment override).
        self.cached_sampled_actions = sampled_actions if sampled_actions is not None else actions
        self.cached_logprobs = logprobs
        self.cached_locked_mask = locked_mask

    def reset(self):
        self.counter = 0
        self.cached_actions = None
        self.cached_sampled_actions = None
        self.cached_logprobs = None
        self.cached_locked_mask = None


class AgentHandler:
    """Runtime coordinator for a single RL agent. Constructed via AgentBuilder."""

    def __init__(self, *, agent_type, agent_type_str, algorithm, algorithm_name,
                 memory, monitor, sim_bridge, fsc, hierarchy_level,
                 use_intrinsic_reward, is_sub_agent,
                 use_next_obs, save_replay_buffer, save_size,
                 root_model_path, rl_mode, resume, no_gui, num_agents, logger,
                 eval_sampling=False):
        # Injected dependencies
        self.agent_type = agent_type
        self.agent_type_str = agent_type_str
        self.algorithm = algorithm
        self.algorithm_name = algorithm_name
        self.memory = memory
        self.monitor = monitor          # None for sub-agents
        self.sim_bridge = sim_bridge
        self.fsc = fsc
        self.hierarchy_level = hierarchy_level
        self.use_intrinsic_reward = use_intrinsic_reward
        self.is_sub_agent = is_sub_agent
        self.use_next_obs = use_next_obs
        self.save_replay_buffer = save_replay_buffer
        self.save_size = save_size
        self.root_model_path = root_model_path
        self.rl_mode = rl_mode
        self.resume = resume
        self.no_gui = no_gui
        self.num_agents = num_agents
        self.logger = logger
        self.eval_sampling = eval_sampling

        # Runtime state
        self.current_obs = None
        self.env_step = 0
        self.env_reset = False
        self.hierarchy_steps = 0
        self.hierarchy_early_stop = False
        self.last_summary = None  # most recent StepResult.summary (for event-driven replan)
        # Deferred SMDP transition buffer (planner only). At each decision the reward returned
        # describes the window governed by the PREVIOUS action, so we hold (obs, action,
        # logprob, locked_mask) here and store the completed transition once the next
        # decision's reward + fresh next_obs + window duration arrive. None = nothing pending.
        self._pending = None

    def should_train(self):
        if self.algorithm_name == 'PPO':
            return len(self.memory) >= self.algorithm.horizon
        elif self.algorithm_name == 'IQL':
            # Train always during offline phase, during online phase check policy frequency
            return (self.env_step < self.algorithm.offline_updates) or (self.env_step % self.algorithm.policy_freq == 0)
        elif self.algorithm_name == 'TD3':
            return len(self.memory) >= self.algorithm.min_memory_size #TODO: Could do tests here after each env_reset(first tests didn't show improvements)
        else:
            raise NotImplementedError("Algorithm {} not implemented".format(self.algorithm_name))

    def load_model(self, change_status=False, new_rl_mode=None):
        # Load model if possible, return new rl_mode and possible console string
        log = ""
        probe_rl_mode = self.rl_mode if new_rl_mode is None else new_rl_mode
        train = True if probe_rl_mode == "train" else False
        if self.algorithm_name == 'no_algo':
            log += "No algorithm used, no model to load"
        elif not train:
            if self.algorithm.load():
                self.algorithm.set_eval()
                log += (f"Load model from checkpoint: {os.path.join(self.algorithm.loading_path, self.algorithm.loading_name)}"
                        f" - Model set to evaluation mode")
            else:
                # Never silently train in eval mode: a baseline/eval arm that trains from
                # scratch would write garbage into the run dir and go unnoticed.
                raise RuntimeError(
                    f"Eval mode but checkpoint could not be loaded from "
                    f"{os.path.join(self.algorithm.loading_path, self.algorithm.loading_name)} (see warning above)")
        elif self.resume:
            if self.algorithm.load():
                self.algorithm.set_train()
                log += f"Load model from checkpoint: {os.path.join(self.algorithm.loading_path, self.algorithm.loading_name)}" \
                       f" - Resume Training from checkpoint"
            else:
                self.algorithm.set_train()
                log += "No checkpoint found to resume training, start training from scratch"
        else:
            self.algorithm.set_train()
            log += "Training from scratch"

        self.logger.info(log)

        if change_status:
            self.sim_bridge.set("rl_mode", self.rl_mode)

    def update_status(self):
        new_rl_mode = self.sim_bridge.get("rl_mode")

        self.sim_bridge.set("obs_collected", len(self.memory))

        if new_rl_mode != self.rl_mode:
            # rl_mode changed — re-sync paths and mode
            self.logger.warning(f"RL Mode changed from {self.rl_mode} to {new_rl_mode}")
            model_path = self.sim_bridge.get("model_path")
            model_name = self.sim_bridge.get("model_name")
            self.algorithm.set_paths(model_path, model_name)
            self.fsc.reset()
            if hasattr(self.agent_type, "notify_episode_reset"):
                self.agent_type.notify_episode_reset()
            self.rl_mode = new_rl_mode
            if self.algorithm_name != 'no_algo':
                if self.rl_mode == "train":
                    self.algorithm.set_train()
                else:
                    self.algorithm.set_eval()

    def intrinsic_reward(self, terminals_vector, engine):
        intrinsic_reward = None
        if self.use_intrinsic_reward:
            intrinsic_reward = self.agent_type.get_intrinsic_reward(self.current_obs)
            # The environment has not been reset so we can send the intrinsic
            # reward to the model (only for displaying purposes)
            if not any(terminals_vector):
                intrinsic_reward = intrinsic_reward.detach().cpu().numpy().tolist()
                self.sim_bridge.set("intrinsic_reward", intrinsic_reward)
                engine.SendRLStatusToModel(self.sim_bridge.status)
                engine.UpdateReward()
        return intrinsic_reward

    def train_loop(self, engine):
        # Check IQL conditions
        skip_step = False if not self.algorithm_name == "IQL" else (self.env_step < self.algorithm.offline_updates)
        next_obs = None
        if not skip_step:
            if self.fsc.at_decision_point:
                if hasattr(self.agent_type, "apply_commitment"):
                    # Planner: refresh the observation at the decision point so the network
                    # decides on the CURRENT post-window state (its per-step UpdateStates is
                    # gated out in the C++ Step). The refreshed obs still carries the previous
                    # goals a_{i-1}, which apply_commitment relies on for the commitment lock.
                    engine.RefreshObservations(self.agent_type.name)
                    self.current_obs = self._get_obs(engine)
                elif self.current_obs is None:
                    self.current_obs = self._get_obs(engine)
                actions, action_logprobs = self.act(self.current_obs)

                # Goal-commitment override (planner-only). The sampled action goes
                # into memory unchanged so the importance ratio stays clean; the
                # overridden action is what actually gets executed in C++. Locked
                # drones' contributions are masked out of the PPO loss downstream.
                locked_mask = None
                executed_actions = actions
                if hasattr(self.agent_type, "apply_commitment"):
                    executed_actions, locked_mask = self.agent_type.apply_commitment(
                        self.current_obs, actions
                    )

                # Cache: executed actions → engine, sampled actions → memory. They
                # differ only for locked drones; for everyone else they're identical.
                # Storing sampled keeps the PPO importance ratio consistent with the
                # policy at sample time; the locked_mask zeros locked-drone gradient.
                self.fsc.cache(
                    executed_actions if not self.algorithm.use_noised_action else self.algorithm.raw_action,
                    action_logprobs,
                    locked_mask=locked_mask,
                    sampled_actions=actions if not self.algorithm.use_noised_action else self.algorithm.raw_action,
                )

            rewards, terminals_vector, terminal_result, percent_burned = self.step_agent(engine, self.fsc.cached_actions)

            if not self.fsc.advance(terminal_result.env_reset):
                return

            next_obs = self._get_obs(engine)

            # Intrinsic Reward Calculation (optional)
            intrinsic_reward = self.intrinsic_reward(terminals_vector, engine)

            # SMDP duration = env steps this planner decision spanned. self.hierarchy_steps
            # here is the just-elapsed window length (reset only AFTER train_loop returns, in
            # hierarchy_manager._reset_agent). Ignored by memory unless use_duration is on
            # (planner + smdp_gae).
            decision_duration = max(int(getattr(self, "hierarchy_steps", 1)), 1) \
                if hasattr(self.agent_type, "apply_commitment") else None

            if hasattr(self.agent_type, "apply_commitment"):
                # Deferred SMDP collection (planner). The reward/terminals just returned
                # describe the window governed by the PREVIOUS action, so attach them to the
                # pending transition (s_{i-1}, a_{i-1}): next_obs (fresh here) is its successor
                # s_i, and decision_duration is that window's length k_{i-1}. Reward, duration,
                # and bootstrap now all describe the same window. The current sampled action
                # becomes the new pending transition.
                terminal = bool(terminal_result.env_reset) or (
                    bool(any(terminals_vector)) if terminals_vector else False)
                # Timeout is a truncation, not a true terminal: the stored done=1 would
                # give the critic a target of r + 0 for exactly the endgame states where
                # the clock runs out. Fold gamma^k * V(s_T) into the reward so the target
                # is the correct truncation value — no memory/PPO plumbing needed.
                if terminal and terminal_result.reason == firesim.FailureReason.Timeout \
                        and self._pending is not None:
                    boot = self._truncation_bootstrap(next_obs, decision_duration)
                    rewards = [r + boot for r in rewards]
                if self._pending is not None:
                    self.memory.add(self._pending["obs"],
                                    self._pending["action"],
                                    self._pending["logprob"],
                                    rewards,
                                    terminals_vector,
                                    next_obs=next_obs if self.use_next_obs else None,
                                    intrinsic_reward=intrinsic_reward,
                                    locked_mask=self._pending["locked_mask"],
                                    duration=decision_duration)
                if terminal:
                    # Episode ended at s_i; the just-sampled action never governs a window.
                    self._pending = None
                else:
                    self._pending = dict(obs=self.current_obs,
                                         action=self.fsc.cached_sampled_actions,
                                         logprob=self.fsc.cached_logprobs,
                                         locked_mask=self.fsc.cached_locked_mask)
            else:
                # Non-planner: store the SAMPLED action (not executed) so PPO's importance
                # ratio is computed at the policy's actual sample point. Unchanged behavior.
                self.memory.add(self.current_obs,
                                self.fsc.cached_sampled_actions,
                                self.fsc.cached_logprobs,
                                rewards,
                                terminals_vector,
                                next_obs=next_obs if self.use_next_obs else None,
                                intrinsic_reward=intrinsic_reward,
                                locked_mask=self.fsc.cached_locked_mask,
                                duration=decision_duration)

            # Update the Logger before checking if we should train, so that the logger has the latest information
            # to calculate the objective percentage and best reward
            self.update_logging(terminal_result)

            # Only change the env_reset after actually taking a step
            self.env_reset = terminal_result.env_reset

        # Training
        if self.should_train():
            self.algorithm.apply_manual_decay(self.sim_bridge.get("train_step"))
            self.update(mini_batch_size=self.algorithm.batch_size, next_obs=next_obs)
            if self.use_intrinsic_reward and self.algorithm_name == 'PPO':
                self.agent_type.update_rnd_model(self.memory, self.algorithm.horizon, self.algorithm.batch_size)
            if self.algorithm.clear_memory:
                self.memory.clear_memory()

        # Advance to the next step
        self.env_step += 1
        self.current_obs = next_obs
        self.handle_env_reset()

    def eval_loop(self, engine, evaluate=False):

        if self.fsc.at_decision_point:
            if hasattr(self.agent_type, "apply_commitment"):
                # Planner: refresh the observation at the decision point so eval matches the
                # (now fresh-obs) training path. Mirrors train_loop.
                engine.RefreshObservations(self.agent_type.name)
                self.current_obs = self._get_obs(engine)
            elif self.current_obs is None:
                self.current_obs = self._get_obs(engine)
            if getattr(self.agent_type, "heuristic_goals", False):
                # Assignment-rule baseline (planner-only): greedy_actions or
                # hungarian_actions per heuristic_method (validated in agent_builder);
                # the loaded pointer network is bypassed, everything else stays identical.
                method = getattr(self.agent_type, "heuristic_method", "greedy")
                actions = getattr(self.agent_type, f"{method}_actions")(self.current_obs)
            elif self.eval_sampling and hasattr(self.agent_type, "apply_commitment"):
                # Diagnostic: evaluate the stochastic policy (sample like training
                # rollouts) instead of argmax. Planner-only; sub-agents stay certain.
                actions, _ = self.act(self.current_obs)
            else:
                actions = self.act_certain(self.current_obs)
            # Mirror training-time commitment in eval so the policy we evaluate
            # matches the one we trained — otherwise eval drones oscillate while
            # training drones commit, and the metrics aren't comparable.
            locked_mask = None
            if hasattr(self.agent_type, "apply_commitment"):
                actions, locked_mask = self.agent_type.apply_commitment(self.current_obs, actions)
            self.fsc.cache(actions, locked_mask=locked_mask)

        rewards, terminals_vector, terminal_result, percent_burned = self.step_agent(engine, self.fsc.cached_actions)
        # Expose this step's summary so the HierarchyManager can read replan_recommended
        # (event-driven planning). plan_low (PlannerFlyAgent) is stepped every env step.
        self.last_summary = terminal_result

        # Only do these extra steps when you SHOULD populate memory
        if self.save_replay_buffer:
            # Intrinsic Reward Calculation (optional)
            intrinsic_reward = self.intrinsic_reward(terminals_vector, engine)
            if self.env_step % 5000 == 0:
                self.logger.info(f"Replay Buffer size: {len(self.memory)}/{int(self.save_size)}")
            # Memory Adding (eval/replay-buffer path). Note: this path stores
            # the EXECUTED action since there's no PPO update consuming the
            # importance ratio — the buffer is for offline algorithms (IQL).
            self.memory.add(self.current_obs,
                            self.fsc.cached_actions,
                            None,
                            rewards,
                            terminals_vector,
                            next_obs=self._get_obs(engine) if self.use_next_obs else None,
                            intrinsic_reward=intrinsic_reward,
                            locked_mask=self.fsc.cached_locked_mask)
            if len(self.memory) >= self.save_size:
                mem_name = os.path.join(self.root_model_path, 'memory.pkl')
                self.memory.save(mem_name)
                self.logger.info(f'Replay Buffer saved at {mem_name}')
                self.sim_bridge.set("agent_online", False)

        if not self.fsc.advance(terminal_result.env_reset):
            return [False] * self.num_agents

        self.current_obs = self._get_obs(engine)
        if evaluate and not self.save_replay_buffer:
            # For the planner, report the ACTUAL env steps elapsed for this decision so TTE
            # is correct under event-driven (variable-cadence) replanning. hierarchy_steps is
            # the runtime window counter (reset each decision by HierarchyManager). Non-planner
            # handlers pass None → evaluator uses its fixed per-step value (unchanged).
            elapsed = self.hierarchy_steps if self.hierarchy_level == "high" else None
            flags = self.monitor.evaluate(rewards, terminal_result, percent_burned,
                                          elapsed_steps=elapsed)
            self.check_reset(flags)

        self.env_step += 1

        return terminals_vector

    def check_reset(self, flags):
        """
        Check if the environment should be reset based on evaluation flags.
        :param flags: Dictionary containing evaluation flags.
        """
        if flags.get("reset", False):
            if flags.get("auto_train", False) and flags.get("auto_train_continue", True):
                self.sim_bridge.set("rl_mode", "train")
                self.sim_bridge.set("agent_is_running", True)
                self.algorithm.reset()
                if self.monitor:
                    self.monitor.handle_auto_train_reset(self.algorithm)
                if self.algorithm_name != 'IQL':
                    self.memory.clear_memory()

            self.hierarchy_steps = 0
            self.fsc.reset()
            if hasattr(self.agent_type, "notify_episode_reset"):
                self.agent_type.notify_episode_reset()
            self.env_step = 0
            self.current_obs = None
            self.env_reset = True
            # Drop any half-collected SMDP transition — the new episode starts clean.
            self._pending = None

    def get_final_metric(self, metric_name: str):
        if self.monitor:
            return self.monitor.get_final_metric(metric_name)
        self.logger.warning("Sub agents do not have a TrainingMonitor, returning None")
        return None

    def step_agent(self, engine, actions):
        env_step = engine.Step(self.agent_type.name, self.get_action(actions))

        rewards = env_step.rewards
        percent_burned = env_step.percent_burned
        terminals = env_step.terminals
        terminal_result = env_step.summary
        all_terminals = [t.is_terminal for t in terminals if t is not None]

        # Per-component reward logging. Each agent that took a hierarchy action this step
        # contributes its own reward_components dict. Summing per tag over a window and letting
        # tensorboard_logger take the mean gives the average magnitude of each reward term.
        if self.monitor is not None:
            tb = self.monitor.tensorboard
            for comp in env_step.reward_components:
                for tag, value in comp.items():
                    tb.add_metric(f"Rewards/{tag}", float(value))

        return rewards, all_terminals, terminal_result, percent_burned

    def step_without_network(self, engine):
        env_step = engine.Step(self.agent_type.name, self.get_action([[0,0] for _ in range(self.num_agents)]))
        self.env_reset = env_step.summary.env_reset
        self.handle_env_reset()

    def handle_env_reset(self):
        if self.env_reset:
            self.sim_bridge.set("current_episode", self.sim_bridge.get("current_episode") + 1)
            self.fsc.reset()
            if hasattr(self.agent_type, "notify_episode_reset"):
                self.agent_type.notify_episode_reset()
            self.current_obs = None
            # Drop any half-collected SMDP transition — the new episode starts clean.
            self._pending = None

    def update(self, mini_batch_size, next_obs):
        tb = self.monitor.tensorboard if self.monitor else None
        try:
            # Under deferred SMDP collection every STORED planner transition already carries
            # its own forward-window duration, so no separate bootstrap_duration is threaded
            # through here anymore (PPO.get_advantages derives k from durations directly).
            self.algorithm.update(self.memory, mini_batch_size, next_obs, tb)
        except Exception as e:
            self.logger.error(f"Error during algorithm update: {e}")
            raise e

        if self.monitor:
            self.monitor.on_training_update(self.algorithm, len(self.memory))

            if self.monitor.on_update_check():
                # Need to inject and load BEST model here
                from agent_builder import resolve_model_name
                model_name, _ = resolve_model_name(path=str(self.algorithm.get_model_path()), model_string="best_obj",
                                                   agent_type=self.agent_type_str, algorithm_name=self.algorithm_name,
                                                   is_loading_name=True)
                self.algorithm.loading_path = self.algorithm.get_model_path()
                self.algorithm.loading_name = model_name
                self.load_model(new_rl_mode="eval")

            self.monitor.summarize(eval_mode=False)

    def act(self, observations):
        actions, action_logprobs = self.algorithm.select_action(observations)
        return actions, action_logprobs if action_logprobs is not None else None

    def _truncation_bootstrap(self, next_obs, duration):
        """gamma^k * V(s_T) for a timed-out (truncated) planner episode.

        Mirrors the SMDP discount get_advantages would apply to this transition
        (k normalized by the hierarchy cap when smdp_normalize_k is on; gamma^1
        when smdp_gae is off), so folding it into the reward reproduces the
        correct truncation target r + gamma^k V(s_T) under the stored done=1.
        """
        import torch
        algo = self.algorithm
        with torch.no_grad():
            state = self.memory.get_agent_state(next_obs, 0)
            # Preserve native dtypes — mask keys must stay bool (the network inverts them).
            state_t = {k: torch.as_tensor(np.asarray(v), device=algo.device)
                       for k, v in state.items()}
            value = float(algo.policy.critic(state_t).reshape(-1)[0].cpu())
        if getattr(algo, "smdp_gae", False):
            k = float(max(duration or 1, 1))
            if getattr(algo, "smdp_normalize_k", False):
                cap = float(max(getattr(algo, "max_low_level_steps", 1), 1))
                k = max(k / cap, 1.0 / cap)
        else:
            k = 1.0
        return (algo.gamma ** k) * value

    def get_action(self, actions):
        return self.agent_type.get_action(actions)

    def update_logging(self, terminal_result):
        if self.monitor:
            self.monitor.log_step(terminal_result)

    def act_certain(self, observations):
        return self.algorithm.select_action_certain(observations)

    def _get_obs(self, engine):
        """Get observations via schema-driven batch API."""
        raw = engine.GetBatchedObservations(self.agent_type.name)
        return ObservationDict(raw)

