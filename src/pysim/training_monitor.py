import os
import logging
import yaml
from evaluation import Evaluator, TensorboardLogger
from utils import SimulationBridge


class TrainingMonitor:
    """Owns evaluation, TensorBoard, and FileHandler lifecycle for the main (non-sub) agent."""

    def __init__(self, logging_path, config, sim_bridge: SimulationBridge,
                 algorithm, algorithm_name, is_loading, resume, no_gui,
                 rl_mode, log_eval):
        self.sim_bridge = sim_bridge
        self.algorithm_name = algorithm_name

        self.tensorboard = TensorboardLogger(log_dir=logging_path, resume=is_loading)

        if not is_loading:
            root_model_path = algorithm.model_path
            config_path = os.path.join(root_model_path, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(config, f, sort_keys=False, indent=4)
        else:
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

        at_dict = config["settings"]["auto_train"]
        max_train = at_dict["max_train"] if algorithm_name != 'IQL' else (
            config["algorithm"]["IQL"]["offline_updates"] + config["algorithm"]["IQL"]["online_updates"]
        )
        self.evaluator = Evaluator(log_dir=logging_path,
                                   config=config,
                                   max_train=max_train,
                                   no_gui=no_gui,
                                   start_eval=rl_mode == "eval",
                                   sim_bridge=sim_bridge,
                                   logger=self.tensorboard,
                                   log_eval=log_eval)

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
                logging.getLogger("TrainingMonitor").info("Finished Offline Updates.")
                algorithm.offline_end = True
                if not algorithm.online_updates > 0:
                    logging.getLogger("TrainingMonitor").info("No Online Updates specified, stopping training.")
                else:
                    logging.getLogger("TrainingMonitor").info("Starting Online Updates.")
        elif self.algorithm_name == 'TD3':
            self.sim_bridge.add_value("policy_updates", algorithm.k_epochs)
            self.tensorboard.policy_updates += algorithm.k_epochs

    def on_update_check(self):
        """Check if evaluator wants to switch from train to eval."""
        return self.evaluator.on_update()

    def evaluate(self, rewards, terminal_result, percent_burned):
        """Run evaluation and return flags."""
        return self.evaluator.evaluate(rewards, terminal_result, percent_burned)

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
