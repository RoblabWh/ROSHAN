from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict, deque
from typing import Union, Any
import numpy as np
from tabulate import tabulate
import logging, os, csv
from dataclasses import dataclass, fields

from utils import SimulationBridge

@dataclass
class EvaluationStats:
    """
    A dataclass to hold evaluation statistics.
    """
    reward: float = 0.0
    time: float = 0.0
    objective: float = 0.0
    perc_burned: float = 0.0
    died: int = 0
    reached_goal: int = 0

class TensorboardLogger:
    """
    Logger class for logging training and evaluation metrics using TensorBoard.
    :param log_dir: Directory where the logs will be saved.
    :param resume: If True, resume logging from the last checkpoint.
    """
    def __init__(self, log_dir, resume=False):
        self.log_dir_tensorboard = os.path.join(log_dir, "tensorboard_logs")
        self.writer = SummaryWriter(self.log_dir_tensorboard)
        self.metrics = defaultdict(list)
        self.histograms = defaultdict(list)
        self.episode_steps = 0 # Steps in the current episode
        self.objectives = deque(maxlen=100)
        self.logging_step = 0
        self.episode = 0 # Current episode number
        self.best_metrics = {"current_objective": 0.0, "best_objective": -np.inf,
                             "current_reward": 0.0, "best_reward": -np.inf}
        # Non-essential metrics that still need to be saved
        self.train_step = 0  # How often did you train?
        self.policy_updates = 0  # How often did you update the policy?
        self.episode_ended = False

        if resume:
            self._load_state()

    def summarize(self):
        # Log all scalar metrics (averaged over the episode)
        for tag, values in self.metrics.items():
            if values:
                self.log_scalar(tag, np.mean(values))
        # Log all histogram metrics
        for tag, values in self.histograms.items():
            if values:
                self.log_histogram(tag, values)
        # Log Episode Length, Objectives, etc.
        if self.episode_ended:
            self.log_scalar("Sanitylogs/Episode_Steps", self.episode_steps)
            self.episode_steps = 0
        if len(self.objectives) == self.objectives.maxlen:
            self.log_scalar("Objective", np.mean(self.objectives))

        # Clear for next summary
        self.metrics.clear()
        self.histograms.clear()
        self.episode_ended = False
        self.logging_step += 1
        self.save_state()

    def log_hparams(self, hparams: dict):
        """
        Logs hyperparameters to TensorBoard.
        :param hparams: Dictionary of hyperparameters.
        """
        if not isinstance(hparams, dict):
            raise ValueError("Hyperparameters must be a dictionary.")
        # Ensure all values are bool, str, float, int or None
        allowed_types = (bool, str, float, int, type(None))
        for key, value in hparams.items():
            if not isinstance(value, allowed_types):
                raise ValueError(f"Hyperparameter '{key}' must be of {allowed_types}, got {type(value)} instead.")

        self.writer.add_hparams(hparams, {"hparam/metric": 0.0})

    def log_step(self, terminal_result: dict):
        """
        Marks the end of an episode and logs the episode length and objective.
        :param terminal_result: Dictionary containing terminal results of the episode.
        """
        self.episode_steps += 1
        env_done = terminal_result["EnvReset"]

        if env_done:
            self.episode += 1
            objective_reached = not terminal_result["OneAgentDied"]
            self.objectives.append(1 if objective_reached else 0)
            self.episode_ended = True

    def add_metric(self, tag, value, hist=False):
        """Add a metric for logging.

        Parameters
        ----------
        tag : str
            Name of the metric.
        value : Any
            Value to log.
        hist : bool, optional
            If True, store the value for histogram logging. If False, the value
            will only be tracked as a scalar metric. Defaults to False.
        """
        if hist:
            self.histograms[tag].append(value)
        self.metrics[tag].append(value)

    def log_scalar(self, tag, value, step: int = None):
        t = self.logging_step if step is None else step
        self.writer.add_scalar(tag, value, t, new_style=True)

    def log_histogram(self, tag, values, step: int = None):
        t = self.logging_step if step is None else step
        if isinstance(values, list):
            values = np.array(values)
        self.writer.add_histogram(tag + "_hist", values, t, bins='auto')

    def log_text(self, tag, text, step: int = None):
        t = self.logging_step if step is None else step
        self.writer.add_text(tag, text, t)

    def _load_state(self):
        state_path = os.path.join(self.log_dir_tensorboard, 'logger_state.npy')
        if os.path.exists(state_path):
            state = np.load(state_path, allow_pickle=True).item()
            self.episode = state.get("episode", 0)
            self.logging_step = state.get("logging_step", 0)
            self.objectives = deque(state.get("objectives", []), maxlen=100)
            self.best_metrics = state.get("best_metrics", {})
            self.train_step = state.get("train_step", 0)
            self.policy_updates = state.get("policy_updates", 0)

    def save_state(self):
        state: dict[str, Union[int, list[Any], dict[str, float]]] = {
            "episode": self.episode,
            "logging_step": self.logging_step,
            "objectives": list(self.objectives),
            "best_metrics": self.best_metrics,
            "train_step": self.train_step,
            "policy_updates": self.policy_updates
        }
        state_path = os.path.join(self.log_dir_tensorboard, 'logger_state.npy')
        np.save(state_path, state, allow_pickle=True) # type: ignore[arg-type]

    def calc_current_objective(self):
        if len(self.objectives) == self.objectives.maxlen:
            self.best_metrics["current_objective"] = float(np.mean(self.objectives))

    def get_best_objective(self):
        if self.best_metrics["current_objective"] > self.best_metrics["best_objective"]:
            return True, self.best_metrics["current_objective"]
        else:
            return False, self.best_metrics["best_objective"]

    def is_better_objective(self):
        """Checks if the current objective is better than the best recorded objective.
           Should only be called when checking if the Model should be saved.
           Otherwise, use `calc_current_objective` to check for the best objective.
        """
        is_better, self.best_metrics["best_objective"] = self.get_best_objective()
        return is_better

    def is_better_reward(self):
        """Checks if the current reward is better than the best recorded reward.
           Should only be called when checking if the Model should be saved.
        """
        current_reward = np.mean(self.metrics["Rewards/Rewards_Raw"]) if "Rewards/Rewards_Raw" in self.metrics else 0.0
        if current_reward >= self.best_metrics["best_reward"]:
            self.best_metrics["best_reward"] = current_reward
            return True
        else:
            return False

    def close(self):
        self.writer.close()

class Evaluator:
    """
    A class to evaluate the performance of an agent in a simulation environment.
    It collects statistics such as reward, time taken, objective achieved, percentage burned,
    number of agents that died, and number of agents that reached the goal.
    """

    def __init__(self, log_dir: str, auto_train_dict: dict, sim_bridge: SimulationBridge, logger: Union[None, TensorboardLogger] = None):
        self.stats = [EvaluationStats()]
        # Python logger for evaluation messages
        self.logger = logging.getLogger("Evaluator")
        # Optional TensorBoard style logger for metrics
        self.tb_logger = logger
        self.log_dir = log_dir
        self.sim_bridge = sim_bridge
        self.eval_steps = 0
        self.use_auto_train = auto_train_dict["use_auto_train"]
        self.max_train = auto_train_dict["max_train"] # Maximum number of training steps before switching to evaluation
        self.max_eval = auto_train_dict["max_eval"] # Maximum number of evaluation steps
        self.train_episodes = auto_train_dict["train_episodes"] # Number of training episodes to run before stopping Auto Training
        self.current_episode = 0  # Current episode number

    def on_update(self):
        if self.sim_bridge.get("train_step") >= self.max_train and self.use_auto_train:
            self.sim_bridge.add_value("train_episode", 1)
            self.logger.info("Training finished, after {} training steps, now starting Evaluation".format(self.sim_bridge.get("train_step")))
            self.sim_bridge.set("rl_mode", "eval")

    def reset(self):
        """
        Resets the agent and its algorithm to the initial state.
        """
        self.eval_steps = 0
        self.current_episode = 0
        self.stats = [EvaluationStats()]  # Reset the stats for the new evaluation

        if self.use_auto_train:
            train_episode = self.sim_bridge.get("train_episode")
            # Checks if the auto_training is finished
            if train_episode == self.train_episodes:
                self.sim_bridge.set("agent_online", False)
                self.logger.info("Training finished, after {} training episodes".format(train_episode))
                self.logger.info("Agent is now offline. Auto Training is finished")
                return
            else:
                self.logger.info("Resume with next training step {}/{}".format(train_episode + 1, self.train_episodes))
                self.sim_bridge.set("train_step", 0)

    def save_to_csv(self, path):
        """Save collected evaluation statistics to a CSV file."""
        field_names = [f.name for f in fields(EvaluationStats)] # type: ignore[arg-type]
        with open(path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names) # type: ignore[arg-type]
            writer.writeheader()
            for stat in self.stats:
                writer.writerow({name: getattr(stat, name) for name in field_names})


    def evaluate(self, rewards, terminals, terminal_result, percent_burned):
        metrics = self.update_evaluation_metrics(rewards, terminals, terminal_result, percent_burned)
        flag_dict = {"auto_train": self.use_auto_train, "reset": False}

        if metrics.get("episode_over"):
            self.eval_steps += 1
            if self.eval_steps >= self.max_eval:
                avg_reward = np.mean([s.reward for s in self.stats])
                self.logger.info(
                    "Evaluation finished, after {} evaluation steps with average reward: {}".format(self.eval_steps,
                                                                                                    avg_reward))
                csv_path = os.path.join(self.log_dir, "evaluation_stats.csv")
                self.save_to_csv(csv_path)
                self.reset()
                self.sim_bridge.set("agent_is_running", False)
                flag_dict.__setitem__("reset", True)

        return flag_dict

    def update_evaluation_metrics(self, rewards, terminals, terminal_result, percent_burned):
        stats = self.stats[self.current_episode]
        stats.reward += float(rewards[0])
        stats.time += 1

        metrics = {
            "episode": self.current_episode + 1,
            "reward": float(stats.reward),
            "time": stats.time,
            "perc_burned": float(stats.perc_burned),
            "died": stats.died,
            "reached": stats.reached_goal,
            "episode_over": False,
        }

        if terminal_result['EnvReset']:
            if terminal_result['OneAgentDied']:
                stats.died += 1
            else:
                stats.reached_goal += 1

            stats.perc_burned = float(percent_burned)
            completed_stats = self.stats
            avg_time = float(np.mean([s.time for s in completed_stats]))
            avg_perc_burned = float(np.mean([s.perc_burned for s in completed_stats]))
            avg_reward = float(np.mean([s.reward for s in completed_stats]))
            total_died = int(sum(s.died for s in completed_stats))
            total_reached = int(sum(s.reached_goal for s in completed_stats))
            success_rate = (
                total_reached / (total_died + total_reached)
                if (total_died + total_reached) > 0
                else 0.0
            )
            metrics.update({
                "perc_burned": stats.perc_burned,
                "died": stats.died,
                "reached": stats.reached_goal,
                "avg_time": avg_time,
                "avg_perc_burned": avg_perc_burned,
                "avg_reward": avg_reward,
                "total_died": total_died,
                "total_reached": total_reached,
                "success_rate": float(success_rate),
            })
            metrics["episode_over"] = True

            self.clean_print(completed_stats, summary =(self.eval_steps >= self.max_eval - 1))

            self.current_episode += 1
            self.stats.append(EvaluationStats())

            # Log metrics via the optional TensorBoard logger
            if self.tb_logger is not None:
                total = metrics["total_died"] + metrics["total_reached"]
                success_rate = metrics["total_reached"] / total if total > 0 else 0.0

                self.tb_logger.add_metric("Evaluation/Reward", metrics["reward"])
                self.tb_logger.add_metric("Evaluation/Success_Rate", success_rate)
                self.tb_logger.add_metric("Evaluation/Percent_Burned", metrics["perc_burned"])
                self.tb_logger.add_metric("Evaluation/Average_Reward", metrics["avg_reward"])
                self.tb_logger.add_metric("Evaluation/Average_Percent_Burned", metrics["avg_perc_burned"])
                self.tb_logger.add_metric("Evaluation/Average_Time", metrics["avg_time"])
                # Flush metrics to disk after each evaluation episode
                self.tb_logger.summarize()

            # self.logger.info(json.dumps(metrics))

        return metrics

    def clean_print(self, completed_stats, summary:bool = False):
        headers = ["Episode", "Reward", "Time", "Perc Burn", "Died", "Reached", "Success Rate"]

        if not summary: completed_stats = [completed_stats[self.current_episode]]

        rows = [
            [
                idx + 1 if summary else self.current_episode,
                f"{s.reward:.2f}",
                s.time,
                f"{s.perc_burned:.2f}",
                s.died,
                s.reached_goal,
                "",
            ]
            for idx, s in enumerate(completed_stats)
        ]
        avg_time = float(np.mean([s.time for s in completed_stats]))
        avg_perc_burned = float(np.mean([s.perc_burned for s in completed_stats]))
        avg_reward = float(np.mean([s.reward for s in completed_stats]))
        total_died = int(sum(s.died for s in completed_stats))
        total_reached = int(sum(s.reached_goal for s in completed_stats))
        success_rate = (
            total_reached / (total_died + total_reached)
            if (total_died + total_reached) > 0
            else 0.0
        )
        footer = [
            f"Averages",
            f"{avg_reward:.2f}",
            f"{avg_time:.2f}",
            f"{avg_perc_burned:.2f}",
            total_died,
            total_reached,
            f"{success_rate:.2%}",
        ]
        table = tabulate(rows + [footer], headers=headers, tablefmt="github")
        self.logger.info("\n%s", table)

