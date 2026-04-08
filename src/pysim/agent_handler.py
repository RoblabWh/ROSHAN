import os
import logging
from observation_dict import ObservationDict


class FrameSkipController:
    """Manages frame-skip action repetition state machine."""
    __slots__ = ('frame_skips', 'counter', 'cached_actions', 'cached_logprobs')

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

    def cache(self, actions, logprobs=None):
        self.cached_actions = actions
        self.cached_logprobs = logprobs

    def reset(self):
        self.counter = 0
        self.cached_actions = None
        self.cached_logprobs = None


class AgentHandler:
    """Runtime coordinator for a single RL agent. Constructed via AgentBuilder."""

    def __init__(self, *, agent_type, agent_type_str, algorithm, algorithm_name,
                 memory, monitor, sim_bridge, fsc, hierarchy_level,
                 use_intrinsic_reward, is_sub_agent,
                 use_next_obs, save_replay_buffer, save_size,
                 root_model_path, rl_mode, resume, no_gui, num_agents, logger):
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

        # Runtime state
        self.current_obs = None
        self.env_step = 0
        self.env_reset = False
        self.hierarchy_steps = 0
        self.hierarchy_early_stop = False

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
                self.algorithm.set_train()
                self.rl_mode = "train"
                log +=  "No checkpoint found to evaluate model, start training from scratch"
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
                if self.current_obs is None:
                    self.current_obs = self._get_obs(engine)
                actions, action_logprobs = self.act(self.current_obs)
                self.fsc.cache(
                    actions if not self.algorithm.use_noised_action else self.algorithm.raw_action,
                    action_logprobs
                )

            rewards, terminals_vector, terminal_result, percent_burned = self.step_agent(engine, self.fsc.cached_actions)

            if not self.fsc.advance(terminal_result.env_reset):
                return

            next_obs = self._get_obs(engine)

            # Intrinsic Reward Calculation (optional)
            intrinsic_reward = self.intrinsic_reward(terminals_vector, engine)

            # Memory Adding
            self.memory.add(self.current_obs,
                            self.fsc.cached_actions,
                            self.fsc.cached_logprobs,
                            rewards,
                            terminals_vector,
                            next_obs=next_obs if self.use_next_obs else None,
                            intrinsic_reward=intrinsic_reward)

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
            if self.current_obs is None:
                self.current_obs = self._get_obs(engine)
            self.fsc.cache(self.act_certain(self.current_obs))

        rewards, terminals_vector, terminal_result, percent_burned = self.step_agent(engine, self.fsc.cached_actions)

        # Only do these extra steps when you SHOULD populate memory
        if self.save_replay_buffer:
            # Intrinsic Reward Calculation (optional)
            intrinsic_reward = self.intrinsic_reward(terminals_vector, engine)
            if self.env_step % 5000 == 0:
                self.logger.info(f"Replay Buffer size: {len(self.memory)}/{int(self.save_size)}")
            # Memory Adding
            self.memory.add(self.current_obs,
                            self.fsc.cached_actions,
                            None,
                            rewards,
                            terminals_vector,
                            next_obs=self._get_obs(engine) if self.use_next_obs else None,
                            intrinsic_reward=intrinsic_reward)
            if len(self.memory) >= self.save_size:
                mem_name = os.path.join(self.root_model_path, 'memory.pkl')
                self.memory.save(mem_name)
                self.logger.info(f'Replay Buffer saved at {mem_name}')
                self.sim_bridge.set("agent_online", False)

        if not self.fsc.advance(terminal_result.env_reset):
            return [False] * self.num_agents

        self.current_obs = self._get_obs(engine)
        if evaluate and not self.save_replay_buffer:
            flags = self.monitor.evaluate(rewards, terminal_result, percent_burned)
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
            self.env_step = 0
            self.current_obs = None
            self.env_reset = True

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

        return rewards, all_terminals, terminal_result, percent_burned

    def step_without_network(self, engine):
        env_step = engine.Step(self.agent_type.name, self.get_action([[0,0] for _ in range(self.num_agents)]))
        self.env_reset = env_step.summary.env_reset
        self.handle_env_reset()

    def handle_env_reset(self):
        if self.env_reset:
            self.sim_bridge.set("current_episode", self.sim_bridge.get("current_episode") + 1)
            self.fsc.reset()
            self.current_obs = None

    def update(self, mini_batch_size, next_obs):
        tb = self.monitor.tensorboard if self.monitor else None
        try:
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

