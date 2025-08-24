from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict, deque
from typing import Union, Any, Dict, List, Type
import numpy as np
from tabulate import tabulate
import logging, os, csv
import scipy.stats as stats
from dataclasses import dataclass, fields
from utils import SimulationBridge
from metrics import (
    Metric,
    RewardMetric,
    TimeMetric,
    PercentBurnedMetric,
    SuccessMetric,
    FailureReason,
)

# Registry of all metrics used during evaluation
METRIC_REGISTRY: List[Type[Metric]] = [
    RewardMetric,
    TimeMetric,
    PercentBurnedMetric,
    SuccessMetric,
]

METRIC_REGISTRY_FLY_AGENT: List[Type[Metric]] = [
    RewardMetric,
    TimeMetric,
    SuccessMetric,
    FailureReason,
]

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

    def log_step(self, terminal_result):
        """
        Marks the end of an episode and logs the episode length and objective.
        :param terminal_result: Dictionary containing terminal results of the episode.
        """
        self.episode_steps += 1

        if terminal_result.env_reset:
            self.episode += 1
            self.objectives.append(1 if terminal_result.any_succeeded else 0)
            self.episode_ended = True

    def add_metric(self, tag, value=None, hist=False):
        """Add metric(s) for logging.

        ``tag`` can either be the name of a single metric or a dictionary of
        metric name/value pairs. This allows evaluation code to simply pass a
        dictionary generated from the metric registry.
        """
        if isinstance(tag, dict):
            for t, v in tag.items():
                self.add_metric("Evaluation/"+t, v, hist=hist)
            return

        if value is None:
            return

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
            try:
                state = np.load(state_path, allow_pickle=True).item()
            except Exception as e:
                logging.warning(
                    "Failed to load logger state from %s: %s. Continuing with default state.",
                    state_path,
                    e,
                )
                return
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
        try:
            np.save(state_path, state, allow_pickle=True)  # type: ignore[arg-type]
        except Exception:
            logging.exception("Failed to save logger state to %s", state_path)

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
        # self.stats = [EvaluationStats()]
        # Python logger for evaluation messages
        self.logger = logging.getLogger("Evaluator")
        # Optional TensorBoard style logger for metrics
        self.tb_logger = logger
        self.log_dir = log_dir
        self.sim_bridge = sim_bridge
        self.eval_steps = 0
        self.use_auto_train = auto_train_dict["use_auto_train"]
        self.max_train = auto_train_dict["max_train"]  # Maximum number of training steps before switching to evaluation
        self.max_eval = auto_train_dict["max_eval"]  # Maximum number of evaluation steps
        self.train_episodes = auto_train_dict["train_episodes"]  # Number of training episodes to run before stopping Auto Training
        self.current_episode = 0  # Current episode number
        # History of per-episode metric values
        self.history: List[Dict[str, float]] = []
        registry = METRIC_REGISTRY_FLY_AGENT if self.sim_bridge.get("hierarchy_type") == "fly_agent" else METRIC_REGISTRY
        self.metrics = [m() for m in registry]  # Initialize metrics from the registry

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
        self.history = []
        for metric in self.metrics:
            metric.reset()

        if self.use_auto_train:
            train_episode = self.sim_bridge.get("train_episode")
            # Checks if the auto_training is finished
            if train_episode == self.train_episodes:
                self.sim_bridge.set("agent_online", False)
                self.logger.info("Training finished, after {} training episodes".format(train_episode))
                self.logger.info("Agent is now offline. Auto Training is finished")
                return True
            else:
                self.logger.info("Resume with next training step {}/{}".format(train_episode + 1, self.train_episodes))
                self.sim_bridge.set("train_step", 0)
        return False

    def save_to_csv(self, path):
        """Save collected evaluation statistics to a CSV file."""
        field_names = [m.name for m in self.metrics]
        with open(path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names) # type: ignore[arg-type]
            writer.writeheader()
            for stat in self.history:
                writer.writerow({name: stat.get(name, 0.0) for name in field_names})

    def plot_metrics(self, output_dir: str):
        """Create boxplots and additional visualizations for collected metrics.

        For each metric, a boxplot, line plot, and histogram are generated from
        the per-episode values stored in ``self.history``.  Figures are saved in
        ``output_dir`` with descriptive filenames.

        Parameters
        ----------
        output_dir: str
            Directory where the plots will be stored. The directory will be
            created if it does not already exist.
        """

        if not self.history:
            return

        try:
            mpl_logger = logging.getLogger("matplotlib")
            mpl_logger.setLevel(logging.WARNING)  # Suppress matplotlib warnings
            pil_logger = logging.getLogger("PIL")
            pil_logger.setLevel(logging.WARNING)  # Suppress PIL warnings
            import matplotlib.pyplot as plt  # type: ignore[import]
        except Exception as e:
            self.logger.warning("matplotlib is required for plotting metrics: %s", e)
            return

        os.makedirs(output_dir, exist_ok=True)

        for metric in self.metrics:
            values = [h[metric.name] for h in self.history if metric.name in h]
            if not values:
                continue

            # Boxplot
            plt.figure()
            plt.boxplot(values)
            plt.title(f"{metric.name} Distribution")
            plt.ylabel(metric.name)
            boxplot_path = os.path.join(output_dir, f"{metric.name}_boxplot.png")
            plt.savefig(boxplot_path)
            plt.close()

            # Line plot over episodes
            plt.figure()
            plt.plot(range(1, len(values) + 1), values, marker="o")
            plt.title(f"{metric.name} per Episode")
            plt.xlabel("Episode")
            plt.ylabel(metric.name)
            lineplot_path = os.path.join(output_dir, f"{metric.name}_lineplot.png")
            plt.savefig(lineplot_path)
            plt.close()

            # Histogram of values
            plt.figure()
            plt.hist(values, bins="auto")
            plt.title(f"{metric.name} Histogram")
            plt.xlabel(metric.name)
            plt.ylabel("Frequency")
            hist_path = os.path.join(output_dir, f"{metric.name}_hist.png")
            plt.savefig(hist_path)
            plt.close()


    def evaluate(self, rewards, terminal_result, percent_burned):
        metrics = self.update_evaluation_metrics(rewards, terminal_result, percent_burned)
        flag_dict = {"auto_train": self.use_auto_train, "reset": False}

        if metrics.get("episode_over"):
            self.eval_steps += 1
            if self.eval_steps >= self.max_eval:
                avg_reward = float(np.mean([s["Reward"] for s in self.history])) if self.history else 0.0
                self.logger.info(f"Evaluation finished, after {self.eval_steps} evaluation "
                                 f"steps with average reward: {avg_reward}")
                self.save_to_csv(os.path.join(self.log_dir, "evaluation_stats.csv"))
                self.plot_metrics(self.log_dir)
                if self.reset():
                    flag_dict.__setitem__("auto_train_not_finished", False)
                self.sim_bridge.set("agent_is_running", False)
                flag_dict.__setitem__("reset", True)

        return flag_dict

    def update_evaluation_metrics(self, rewards, terminal_result, percent_burned):
        step_stats = {
            "rewards": rewards,
            "terminal_result": terminal_result,
            "percent_burned": percent_burned,
        }

        metrics: Dict[str, Any] = {"episode": self.current_episode + 1, "episode_over": False}
        for metric in self.metrics:
            metric.update(step_stats)
            metrics[metric.name] = metric.value

        if terminal_result.env_reset:
            # Store final metrics for this episode
            episode_metrics = {m.name: m.value for m in self.metrics}
            self.history.append(episode_metrics)

            # Compute aggregated metrics
            for metric in self.metrics:
                metrics.update(metric.compute(self.history))
                metric.reset()

            metrics["episode_over"] = True

            self.clean_print()

            self.current_episode += 1

            # Log metrics via the optional TensorBoard logger
            if self.tb_logger is not None:
                tb_metrics = {k: v for k, v in metrics.items() if k not in ("episode", "episode_over", "Failure_Reason", "Failure_Reason_counts", "Failure_Reason_perc", "Failure_Reason_n")}
                self.tb_logger.add_metric(tb_metrics)
                # Flush metrics to disk after each evaluation episode
                self.tb_logger.summarize()

        return metrics

    def clean_print(self):
        last_episode = (self.eval_steps >= self.max_eval - 1)
        metric_headers = [m.name for m in self.metrics]
        headers = ["Episode"] + metric_headers + ["Success Rate"]

        local_history_ = [self.history[self.current_episode]] if not last_episode else self.history

        rows = [
            [
                idx + 1 if last_episode else (self.current_episode + 1),
                *[
                    f"{s[m.name]:.4f}" if m.dtype == "float"
                    else f"{s[m.name]:.2f}" if m.dtype == "percent"
                    else int(s[m.name]) if m.dtype == "int"
                    else f"{s[m.name]}" if m.dtype == "string"
                    else f"Undefined {m.name}"
                    for m in self.metrics
                ],
            ]
            for idx, s in enumerate(local_history_)
        ]

        tabular_data = rows

        if last_episode:
            end_footer = [
                f"Total(N={len(self.history)})",
                *[
                    m.get_compute_string(m.compute(self.history))
                    for m in self.metrics
                ],
            ]

            tabular_data += [end_footer]

        table = tabulate(tabular_data=tabular_data,
                         headers=headers,
                         tablefmt="rounded_grid")

        self.logger.info("\n%s", table)