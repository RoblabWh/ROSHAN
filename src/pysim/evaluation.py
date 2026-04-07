import numpy as np
from typing import Any, Dict, List, Type
from tabulate import tabulate
import logging, os, csv
from metrics import (
    Metric,
    RewardMetric,
    TimeMetric,
    PercentBurnedMetric,
    SuccessMetric,
    FailureReason,
)
from metrics_plotter import MetricsPlotter
from tensorboard_logger import TensorboardLogger

# Registry of all metrics used during evaluation
METRIC_REGISTRY: List[Type[Metric]] = [
    RewardMetric,
    TimeMetric,
    PercentBurnedMetric,
    SuccessMetric,
    FailureReason,
]

METRIC_REGISTRY_FLY_AGENT: List[Type[Metric]] = [
    RewardMetric,
    TimeMetric,
    SuccessMetric,
    FailureReason,
]


class Evaluator:
    """
    Collects per-episode metrics during evaluation and produces CSV/plot outputs.
    Does NOT own auto-train lifecycle — that belongs in TrainingMonitor.
    """

    def __init__(self, *, max_eval: int, hierarchy_steps: int, log_eval: bool,
                 log_dir: str, tb_logger: TensorboardLogger = None,
                 registry: List[Type[Metric]] = None):
        self.logger = logging.getLogger("Evaluator")
        self.tb_logger = tb_logger
        self.log_dir = log_dir
        self.max_eval = max_eval
        self.hierarchy_steps = hierarchy_steps
        self.log_eval = log_eval
        self.eval_steps = 0
        self.current_episode = 0
        self.avg_reward = -np.inf
        self.avg_objective = 0
        self.avg_tte = +np.inf
        self.history: List[Dict[str, float]] = []
        self.terminal_episode_dict = {}
        registry = registry or METRIC_REGISTRY
        self.metrics = [m() for m in registry]

    def evaluate(self, rewards, terminal_result, percent_burned, is_planner=False):
        """Process one step. Returns a flag dict with 'done' set when all episodes are complete."""
        metrics = self._update_metrics(rewards, terminal_result, percent_burned)
        result = {"done": False}

        if metrics.get("episode_over"):
            self.eval_steps += 1
            self.avg_reward = float(np.mean([s["Reward"] for s in self.history])) if self.history else 0.0

            if self.eval_steps >= self.max_eval:
                self.avg_objective = float(np.mean([s["Success"] for s in self.history])) if self.history else 0.0
                self.avg_tte = float(np.mean([s["Time"] for s in self.history])) if self.history and is_planner else 0.0
                self.logger.info(f"Evaluation finished after {self.eval_steps} episodes — "
                                 f"avg reward: {self.avg_reward:.3f}, avg objective: {self.avg_objective:.3f}")
                self.save_to_csv(os.path.join(self.log_dir, "evaluation_stats.csv"))
                self.plot_metrics()
                result["done"] = True

        return result

    def final_metrics(self):
        return {"reward": self.avg_reward, "objective": self.avg_objective, "time_to_end": self.avg_tte}

    def reset(self):
        """Clear all metric state for a fresh evaluation run."""
        self.eval_steps = 0
        self.current_episode = 0
        self.history = []
        self.terminal_episode_dict = {}
        for metric in self.metrics:
            metric.reset()

    def update_log_dir(self, new_log_dir):
        self.log_dir = new_log_dir

    def save_to_csv(self, path):
        if not self.log_eval:
            return
        field_names = [m.name for m in self.metrics]
        with open(path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
            for stat in self.history:
                writer.writerow({name: stat.get(name, 0.0) for name in field_names})

    def load_history_from_csvs(self):
        """Load evaluation statistics from CSV files in the log directory (for multi-run plotting)."""
        par_dir = os.path.dirname(self.log_dir)

        def _is_auto_train_dir(name: str) -> bool:
            parts = name.split("_")
            return len(parts) == 2 and parts[0] == "training" and parts[1].isdigit()

        self_dir = os.path.split(par_dir)[-1]
        if not _is_auto_train_dir(self_dir):
            self.logger.warning("Log directory %s does not appear to be from an auto-training run", self.log_dir)
            return
        par_dir = os.path.dirname(par_dir)

        auto_train_log_dirs = [os.path.join(os.path.join(par_dir, d), "logs")
                               for d in os.listdir(par_dir) if _is_auto_train_dir(d)]

        self.history = []
        for log_dir in auto_train_log_dirs:
            csv_path = os.path.join(log_dir, "evaluation_stats.csv")
            if not os.path.exists(csv_path):
                self.logger.warning("No evaluation_stats.csv found in %s", log_dir)
                continue
            with open(csv_path, "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                self.history.append(list(reader))

    def plot_metrics(self):
        if self.log_eval:
            plotter = MetricsPlotter(history=self.history,
                                     log_dir=self.log_dir,
                                     logger=self.logger)
            plotter.plot_all()

    def _update_metrics(self, rewards, terminal_result, percent_burned):
        step_stats = {
            "rewards": rewards,
            "terminal_result": terminal_result,
            "percent_burned": percent_burned,
        }

        metrics: Dict[str, Any] = {"episode": self.current_episode + 1, "episode_over": False}
        for metric in self.metrics:
            metric.update(step_stats, self.hierarchy_steps)
            metrics[metric.name] = metric.value

        if terminal_result.env_reset:
            episode_metrics = {m.name: m.value for m in self.metrics}
            self.history.append(episode_metrics)

            for metric in self.metrics:
                metrics.update(metric.compute(self.history))
                metric.reset()

            metrics["episode_over"] = True
            self._clean_print()
            self.current_episode += 1

            if self.tb_logger is not None and self.log_eval:
                tb_metrics = {k: v for k, v in metrics.items()
                              if k not in ("episode", "episode_over", "Failure_Reason",
                                           "Failure_Reason_counts", "Failure_Reason_perc", "Failure_Reason_n")}
                self.tb_logger.add_metric(tb_metrics)
                self.tb_logger.summarize(eval_mode=True)

        return metrics

    def _clean_print(self):
        last_episode = (self.eval_steps >= self.max_eval - 1)
        if last_episode:
            metric_headers = [m.name for m in self.metrics]
            headers = ["Episode"] + metric_headers + ["Success Rate"]

            rows = [
                [
                    idx + 1,
                    *[
                        f"{s[m.name]:.4f}" if m.dtype == "float"
                        else f"{s[m.name]:.2f}" if m.dtype == "percent"
                        else int(s[m.name]) if m.dtype == "int"
                        else f"{s[m.name]}" if m.dtype == "string"
                        else f"Undefined {m.name}"
                        for m in self.metrics
                    ],
                ]
                for idx, s in enumerate(self.history)
            ]

            end_footer = [
                f"Total(N={len(self.history)})",
                *[
                    m.get_compute_string(m.compute(self.history))
                    for m in self.metrics
                ],
            ]

            rows.append(end_footer)

            table = tabulate(tabular_data=rows,
                             headers=headers,
                             tablefmt="rounded_grid")
            print()
            self.logger.info("\n%s", table)
            self.terminal_episode_dict = {}
        else:
            terminal_reason = self.history[self.current_episode]["Failure_Reason"]
            self.terminal_episode_dict[terminal_reason] = self.terminal_episode_dict.get(terminal_reason, 0) + 1
            items = sorted(self.terminal_episode_dict.items())
            progress = "; ".join(f"{k}: {v}" for k, v in items)
            print(f"\rEvaluation running... {progress}", end='')
