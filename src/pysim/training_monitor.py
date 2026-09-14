import os
import logging
import yaml
from evaluation import Evaluator, METRIC_REGISTRY, METRIC_REGISTRY_FLY_AGENT
from tensorboard_logger import TensorboardLogger
from utils import SimulationBridge


class TrainingMonitor:
    """Owns evaluation, TensorBoard, and FileHandler lifecycle for the main (non-sub) agent."""

    def __init__(self, logging_path, config, sim_bridge: SimulationBridge,
                 algorithm, algorithm_name, is_loading, resume, no_gui,
                 rl_mode, log_eval):
        self.sim_bridge = sim_bridge
        self.algorithm_name = algorithm_name
        self.logger = logging.getLogger("TrainingMonitor")

        self.tensorboard = TensorboardLogger(log_dir=logging_path, resume=is_loading)

        # Snapshot the effective config as plain, resolved YAML (handles DictConfig) into the
        # write dir — also for eval arms with paths.run_dir set, so every arm is reproducible.
        # Skipped only when loading in place (it would clobber the trained run's snapshot).
        in_place = is_loading and os.path.abspath(str(algorithm.model_path)) == os.path.abspath(str(algorithm.loading_path))
        if not in_place:
            from config_loader import dump_resolved
            dump_resolved(config, os.path.join(algorithm.model_path, "config.yaml"))
        if is_loading:
            self.sim_bridge.set("current_episode", self.tensorboard.episode)
            self.sim_bridge.set("current_objective", self.tensorboard.best_metrics["current_objective"])
            self.sim_bridge.set("best_objective", self.tensorboard.best_metrics["best_objective"])
            self.sim_bridge.set("train_step", self.tensorboard.train_step)
            self.sim_bridge.set("policy_updates", self.tensorboard.policy_updates)

        sim_bridge.set("model_path", algorithm.model_path)
        sim_bridge.set("model_name", algorithm.model_name)

        min_update = 0
        if algorithm_name == 'PPO':
            min_update = algorithm.horizon
        elif algorithm_name == 'IQL':
            min_update = 0  # Will be set after memory is loaded
        elif algorithm_name == 'TD3':
            min_update = algorithm.min_memory_size
        sim_bridge.set("min_update", min_update)

        # Auto-train lifecycle config
        at_dict = config["settings"]["auto_train"]
        self.use_auto_train = at_dict["use_auto_train"]
        self.no_gui = no_gui
        self.no_gui_start_eval = no_gui and (rl_mode == "eval")
        self.train_episodes = at_dict["train_episodes"] if self.use_auto_train else 1
        self.is_planner = config["settings"]["hierarchy_type"] == "planner_agent"
        # Planner objective is TTE-aware: with the water-aware step budget, plain rolling
        # success saturates at 1.0 (Wave G0: heuristic hits 100% even at fire 0.08) and the
        # best_obj latch goes blind. Fly/explore keep the binary objective.
        self.tensorboard.tte_aware_objective = self.is_planner

        max_train = at_dict["max_train"] if algorithm_name != 'IQL' else (
            config["algorithm"]["IQL"]["offline_updates"] + config["algorithm"]["IQL"]["online_updates"]
        )
        self.max_train = max_train

        # Determine metric registry
        hierarchy_type = sim_bridge.get("hierarchy_type")
        use_fly_registry = hierarchy_type == "fly_agent"
        registry = METRIC_REGISTRY_FLY_AGENT if use_fly_registry else METRIC_REGISTRY

        hierarchy_steps = 1 if not self.is_planner else config["environment"]["agent"]["planner_agent"]["hierarchy_timesteps"]

        self.log_eval = log_eval
        self.logging_path = logging_path

        self.evaluator = Evaluator(
            max_eval=at_dict["max_eval"],
            hierarchy_steps=hierarchy_steps,
            log_eval=log_eval,
            log_dir=logging_path,
            tb_logger=self.tensorboard,
            registry=registry,
        )

    def set_iql_min_update(self, memory_size):
        """Set min_update for IQL after memory is loaded."""
        self.sim_bridge.set("min_update", memory_size)

    def log_step(self, terminal_result):
        """Log a training step result."""
        self.tensorboard.log_step(terminal_result)
        self.tensorboard.calc_current_objective()
        self.sim_bridge.set("objective", self.tensorboard.best_metrics["current_objective"])
        self.sim_bridge.set("best_objective", self.tensorboard.get_best_objective()[1])

    def on_training_update(self, algorithm, memory_len):
        """Track metrics after an algorithm update."""
        self.sim_bridge.add_value("train_step", 1)
        self.tensorboard.train_step += 1

        if self.algorithm_name == 'PPO':
            policy_updates = algorithm.horizon // algorithm.batch_size * algorithm.k_epochs
            self.sim_bridge.add_value("policy_updates", policy_updates)
            self.tensorboard.policy_updates += policy_updates
        elif self.algorithm_name == 'IQL':
            batches_per_epoch, epochs = algorithm.get_batches_and_epochs(memory_len, algorithm.batch_size)
            self.sim_bridge.add_value("policy_updates", batches_per_epoch * epochs)
            self.tensorboard.policy_updates += algorithm.k_epochs
            if self.sim_bridge.get("train_step") == algorithm.offline_updates:
                self.logger.info("Finished Offline Updates.")
                algorithm.offline_end = True
                if not algorithm.online_updates > 0:
                    self.logger.info("No Online Updates specified, stopping training.")
                else:
                    self.logger.info("Starting Online Updates.")
        elif self.algorithm_name == 'TD3':
            self.sim_bridge.add_value("policy_updates", algorithm.k_epochs)
            self.tensorboard.policy_updates += algorithm.k_epochs

    def on_update_check(self):
        """Check if training is done and should switch to evaluation."""
        if self.sim_bridge.get("train_step") >= self.max_train and (self.use_auto_train or self.no_gui):
            self.sim_bridge.add_value("train_episode", 1)
            self.logger.info("Training finished after {} training steps, now starting Evaluation".format(
                self.sim_bridge.get("train_step")))
            self.sim_bridge.set("rl_mode", "eval")
            return True
        return False

    def evaluate(self, rewards, terminal_result, percent_burned, elapsed_steps=None):
        """Run evaluation and return flags."""
        result = self.evaluator.evaluate(rewards, terminal_result, percent_burned,
                                         is_planner=self.is_planner, elapsed_steps=elapsed_steps)
        flags = {"auto_train": self.use_auto_train, "reset": False}

        if result.get("done"):
            flags["reset"] = True
            # Handle auto-train cycling or termination
            if self._handle_eval_complete():
                flags["auto_train_continue"] = False
            self.sim_bridge.set("agent_is_running", False)

        return flags

    def _handle_eval_complete(self):
        """Handle evaluation completion. Returns True if all training is done."""
        self.evaluator.reset()
        train_episode = self.sim_bridge.get("train_episode")

        if self.use_auto_train:
            if train_episode >= self.train_episodes:
                self.logger.info("Auto-training finished after {} training episodes".format(train_episode))
                self.evaluator.load_history_from_csvs()
                self.evaluator.plot_metrics()
                self.sim_bridge.set("agent_online", False)
                return True
            else:
                self.logger.info("Resume with next training step {}/{}".format(train_episode + 1, self.train_episodes))
                # Compute next training directory using the episode counter (not string surgery)
                base_dir = os.path.dirname(os.path.dirname(self.logging_path))
                new_logging_path = os.path.join(base_dir, f"training_{train_episode + 1}", "logs")
                self.logging_path = new_logging_path
                self.evaluator.update_log_dir(new_logging_path)
                self.sim_bridge.set("train_step", 0)
                self.sim_bridge.set("policy_updates", 0)
                self.sim_bridge.set("current_episode", 0)
                return False
        elif self.no_gui:
            self.sim_bridge.set("agent_online", False)
            return True

        return False

    def handle_auto_train_reset(self, algorithm):
        """Re-create loggers and TensorBoard for a new auto-train cycle."""
        self.tensorboard.close()
        model_path = algorithm.get_model_path()
        logging_path = os.path.join(model_path, "logs")
        # Remove and add another FileHandler to avoid issues
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
                handler.close()
        if not os.path.exists(logging_path):
            os.makedirs(logging_path, exist_ok=True)
        logging_file = os.path.join(logging_path, "logging.log")
        file_handler = logging.FileHandler(logging_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        self.tensorboard = TensorboardLogger(log_dir=logging_path)
        self.evaluator.tb_logger = self.tensorboard

    def summarize(self, eval_mode):
        """Flush tensorboard metrics."""
        self.tensorboard.summarize(eval_mode=eval_mode)

    def get_final_metric(self, metric_name):
        return self.evaluator.final_metrics()[metric_name]
