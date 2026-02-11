import copy

from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict, deque, Counter
from typing import Union, Any, Dict, List, Type, Tuple
import numpy as np
from tabulate import tabulate
import logging, os, csv
import scipy.stats as stats
from math import sqrt
from dataclasses import dataclass
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
    FailureReason,
]

METRIC_REGISTRY_FLY_AGENT: List[Type[Metric]] = [
    RewardMetric,
    TimeMetric,
    SuccessMetric,
    FailureReason,
]

@dataclass
class MetricsPlotter:
    history: Any                    # List[Dict] (single) or List[List[Dict]] (multi)
    log_dir: str
    logger: logging.Logger

    # Call this to generate all plots
    def plot_all(self) -> None:
        if not self.history:
            return

        try:
            logging.getLogger("matplotlib").setLevel(logging.WARNING)
            logging.getLogger("PIL").setLevel(logging.WARNING)
            import matplotlib.pyplot as plt  # type: ignore[import]
        except Exception as e:
            self.logger.warning("matplotlib is required for plotting metrics: %s", e)
            return

        if isinstance(self.history, list) and self.history and isinstance(self.history[0], dict):
            self._plot_single_run(self.history)
        else:
            self._plot_multi_run(self.history)

    # ----------------------------- helpers (stats) -----------------------------
    @staticmethod
    def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0: return arr
        window = max(1, int(window))
        return np.convolve(arr, np.ones(window) / window, mode="same")

    @staticmethod
    def ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return np.array([]), np.array([])
        xs = np.sort(x)
        ys = np.arange(1, xs.size + 1) / xs.size
        return xs, ys

    @staticmethod
    def summary_stats(vals: np.ndarray, ci: float = 0.95) -> Tuple[float, float, float, float, float]:
        vals = np.asarray(vals, dtype=float)
        n = len(vals)
        if n == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        mean = float(np.nanmean(vals))
        median = float(np.nanmedian(vals))
        if n > 1:
            std = float(np.nanstd(vals, ddof=1))
            sem = std / np.sqrt(n)
            tcrit = stats.t.ppf((1 + ci) / 2, n - 1)
            half = tcrit * sem
            lo, hi = mean - half, mean + half
        else:
            std = 0.0
            lo = hi = mean
        return mean, std, median, lo, hi

    @staticmethod
    def mean_and_ci(mat: np.ndarray, axis=0, ci=0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mat = np.asarray(mat, dtype=float)
        n = np.sum(~np.isnan(mat), axis=axis)
        mean = np.nanmean(mat, axis=axis)
        std = np.nanstd(mat, axis=axis, ddof=1)
        sem = np.divide(std, np.sqrt(np.maximum(n, 1)), where=(n > 0))
        df = np.maximum(n - 1, 1)
        tcrit = stats.t.ppf((1 + ci) / 2, df)
        half = tcrit * sem
        lo = mean - half
        hi = mean + half
        mask = n <= 1
        if np.any(mask):
            lo = np.where(mask, mean, lo)
            hi = np.where(mask, mean, hi)
        return mean, lo, hi

    @staticmethod
    def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
        if n == 0: return 0.0, 0.0
        phat = k / n
        denom = 1 + z**2 / n
        center = (phat + z*z/(2*n)) / denom
        half = z * sqrt((phat*(1-phat) + z*z/(4*n)) / n) / denom
        return center - half, center + half

    def binned_mean_ci_over_x(self, x: np.ndarray, y: np.ndarray, nbins=20, ci=0.95):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        bins = np.linspace(np.nanmin(x), np.nanmax(x), nbins + 1)
        xc = 0.5 * (bins[:-1] + bins[1:])
        means = np.full(nbins, np.nan)
        los = np.full(nbins, np.nan)
        his = np.full(nbins, np.nan)
        for i in range(nbins):
            mask = (x >= bins[i]) & (x < bins[i+1] if i < nbins - 1 else x <= bins[i+1])
            vals = y[mask]
            if vals.size:
                m, _, _, lo, hi = self.summary_stats(vals, ci=ci)
                means[i], los[i], his[i] = m, lo, hi
        return xc, means, los, his

    # ----------------------------- helpers (plotting) -----------------------------
    @staticmethod
    def bar_counts(ax, labels, counts, title, y_label="Count", rotation=45):
        x = np.arange(len(labels))
        ax.bar(x, counts)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=rotation, ha="right")
        ax.set_title(title)
        ax.set_ylabel(y_label)

    @staticmethod
    def stacked_bars(ax, values_dict, x_positions, title, x_label=None, y_label="Count", legend=True):
        bottom = np.zeros(len(x_positions))
        for key, vals in values_dict.items():
            vals = np.asarray(vals, dtype=float)
            ax.bar(x_positions, vals, bottom=bottom, label=key)
            bottom += vals
        if x_label:
            ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        if legend:
            ax.legend()

    @staticmethod
    def line_with_band(ax, x, mean, lo, hi, label, title, y_label):
        ax.plot(x, mean, label=label)
        ax.fill_between(x, lo, hi, alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(y_label)
        ax.legend()

    def plot_ecdf_on(self, ax, arr, label=None, alpha=1.0, linewidth=None):
        x, y = self.ecdf(arr)
        if x.size:
            ax.step(x, y, where="post", alpha=alpha, linewidth=linewidth, label=label)

    @staticmethod
    def save_fig(fig, output_dir: str, filename: str):
        import matplotlib.pyplot as plt
        os.makedirs(output_dir, exist_ok=True)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close(fig)

    # ----------------------------- single run -----------------------------
    def _plot_single_run(self, history: List[Dict[str, Any]]) -> None:
        import matplotlib.pyplot as plt

        output_dir = copy.deepcopy(self.log_dir)
        os.makedirs(output_dir, exist_ok=True)

        episodes = np.arange(1, len(history) + 1)
        rewards = np.asarray([h.get("Reward", 0.0) for h in history], dtype=float)
        success = np.asarray([h.get("Success", 0.0) for h in history], dtype=float)
        times   = np.asarray([h.get("Time",   0.0) for h in history], dtype=float)
        failures = [h.get("Failure_Reason", "None") for h in history]

        # Reward vs episode
        fig, ax = plt.subplots()
        ax.plot(episodes, rewards, alpha=0.35, label="Reward")
        ax.set_xlabel("Episode"); ax.set_ylabel("Reward"); ax.set_title("Reward vs Episode")
        self.save_fig(fig, output_dir, "reward_vs_episode.png")

        # Time hist + ECDF
        fig, ax = plt.subplots()
        ax.hist(times, bins="auto")
        ax.set_xlabel("Time to end"); ax.set_ylabel("Frequency"); ax.set_title("Time-to-end Histogram")
        self.save_fig(fig, output_dir, "time_to_end_hist.png")

        fig, ax = plt.subplots()
        self.plot_ecdf_on(ax, times, label="Run")
        ax.set_xlabel("Time to end"); ax.set_ylabel("ECDF"); ax.set_title("Time-to-end ECDF")
        ax.legend()
        self.save_fig(fig, output_dir, "time_to_end_ecdf.png")

        # Failure reasons (map "None" -> "Success")
        counts = Counter("Success" if f == "None" else f for f in failures)
        labels = sorted(counts, key=counts.get, reverse=True)
        fig, ax = plt.subplots()
        self.bar_counts(ax, labels, [counts[k] for k in labels], "Failure Reasons") # type: ignore
        self.save_fig(fig, output_dir, "failure_reasons_bar.png")

        # Overall success bar
        fig, ax = plt.subplots()
        n_success = int(np.sum(success > 0.5))
        n_failure = len(success) - n_success
        self.bar_counts(ax, ["Success", "Failure"], [n_success, n_failure], "Overall Success Rate") # type: ignore
        self.save_fig(fig, output_dir, "overall_success_bar.png")

        # Reward/Success vs Time
        fig, ax = plt.subplots()
        ax.scatter(times, rewards)
        ax.set_xlabel("Time"); ax.set_ylabel("Reward"); ax.set_title("Reward vs Time")
        # ax.legend(loc="best", frameon=False, title="Curves")
        self.save_fig(fig, output_dir, "reward_vs_time.png")

        fig, ax = plt.subplots()
        ax.scatter(times, success)
        ax.set_xlabel("Time"); ax.set_ylabel("Success"); ax.set_title("Success vs Time")
        self.save_fig(fig, output_dir, "success_vs_time.png")

    # ----------------------------- multi run -----------------------------
    def _plot_multi_run(self, runs: List[List[Dict[str, Any]]]) -> None:
        import matplotlib.pyplot as plt

        # infer logs path like your original structure
        output_dir = copy.deepcopy(self.log_dir)
        par_dir = os.path.dirname(output_dir)
        at_dir = os.path.split(par_dir)[-1]
        parts = at_dir.split("_")
        auto_train_dir = len(parts) == 2 and parts[0] == "training" and parts[1].isdigit()
        if not auto_train_dir:
            self.logger.warning("Log directory %s does not appear to be from an auto-training run", par_dir)
            return
        par_dir = os.path.dirname(par_dir)
        output_dir = os.path.join(par_dir, "logs")
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info("Plotting summarized metrics to %s", output_dir)

        run_rewards = [np.asarray([h.get("Reward", 0.0) for h in run], dtype=float) for run in runs]
        run_success = [np.asarray([h.get("Success", 0.0) for h in run], dtype=float) for run in runs]
        run_times   = [np.asarray([h.get("Time",   0.0) for h in run], dtype=float) for run in runs]
        run_failures = [[h.get("Failure_Reason", "None") for h in run] for run in runs]

        min_len = min(len(r) for r in run_rewards) if run_rewards else 0
        episodes = np.arange(1, min_len + 1)

        # Reward mean ± CI vs episode
        if min_len > 0:
            rewards_mat = np.stack([r[:min_len] for r in run_rewards])  # runs × episodes
            mean, lo, hi = self.mean_and_ci(rewards_mat, axis=0, ci=0.95)
            fig, ax = plt.subplots()
            self.line_with_band(ax, episodes, mean, lo, hi, # type: ignore
                                label="Mean Reward",
                                title="Reward vs Episode (Mean ±95% CI)",
                                y_label="Reward")
            ax.grid(alpha=0.25, linewidth=0.5)
            self.save_fig(fig, output_dir, "reward_mean_ci.png")

        # Reward vs Time: scatter per run + pooled binned mean ± CI
        fig, ax = plt.subplots()
        for t, r in zip(run_times, run_rewards):
            ax.scatter(t, r, alpha=0.2, s=10)
        all_t = np.concatenate(run_times).astype(float) if run_times else np.array([])
        all_r = np.concatenate(run_rewards).astype(float) if run_rewards else np.array([])
        xc, mean_rt, lo_rt, hi_rt = self.binned_mean_ci_over_x(all_t, all_r, nbins=20, ci=0.95)
        valid = ~np.isnan(mean_rt)
        if valid.any():
            ax.plot(xc[valid], mean_rt[valid], label="Mean (binned)")
            ax.fill_between(xc[valid], lo_rt[valid], hi_rt[valid], alpha=0.3, label="95% CI")
        ax.set_xlabel("Time"); ax.set_ylabel("Reward"); ax.set_title("Reward vs Time (all runs)")
        ax.legend(loc="best", frameon=False, title="Curves")
        ax.grid(alpha=0.25, linewidth=0.5)
        self.save_fig(fig, output_dir, "reward_vs_time_runs.png")

        # Reward vs Time (successes only)
        fig, ax = plt.subplots()
        succ_all_t = np.concatenate([t[s > 0.5] for t, s in zip(run_times, run_success)]).astype(float)
        succ_all_r = np.concatenate([r[s > 0.5] for r, s in zip(run_rewards, run_success)]).astype(float)
        ax.scatter(succ_all_t, succ_all_r, alpha=0.2, s=10, label="success episodes")

        xc, mean_rt, lo_rt, hi_rt = self.binned_mean_ci_over_x(succ_all_t, succ_all_r, nbins=20, ci=0.95)
        valid = ~np.isnan(mean_rt)
        if valid.any():
            ax.plot(xc[valid], mean_rt[valid], label="Mean (binned)")
            ax.fill_between(xc[valid], lo_rt[valid], hi_rt[valid], alpha=0.3, label="95% CI")

        ax.set_xlabel("Time"); ax.set_ylabel("Reward"); ax.set_title("Reward vs Time (successes only)")
        ax.legend(loc="best", frameon=False)
        ax.grid(alpha=0.25, linewidth=0.5)
        self.save_fig(fig, output_dir, "reward_vs_time_success_only.png")

        # Per-run success rate with Wilson 95% CI
        fig, ax = plt.subplots()
        xs = np.arange(1, len(run_success) + 1)
        p, lo_ci, hi_ci = [], [], []
        for s in run_success:
            k, n = int(np.sum(s > 0.5)), len(s)
            p.append(k / n if n else 0.0)
            l, h = self.wilson_ci(k, n)
            lo_ci.append(l); hi_ci.append(h)
        if xs.size:
            ax.errorbar(xs, p,
                        yerr=[np.array(p) - np.array(lo_ci), np.array(hi_ci) - np.array(p)],
                        fmt='o', capsize=3)
        ax.set_xticks(xs)
        ax.set_xlabel("Run"); ax.set_ylabel("Success rate"); ax.set_title("Per-run success rate (95% CI, Wilson)")
        ax.grid(alpha=0.25, linewidth=0.5)
        pooled_p = float(np.mean(np.concatenate(run_success) > 0.5))
        ax.axhline(pooled_p, linestyle="--", linewidth=1, alpha=0.5, label=f"Pooled={pooled_p:.2f}")
        ax.legend()
        self.save_fig(fig, output_dir, "success_rate_wilson_per_run.png")

        # Success mean ± CI across runs (moving-avg per episode)
        if min_len > 0:
            window = max(1, min_len // 20)
            success_ma = np.stack([self.rolling_mean(s[:min_len], window) for s in run_success])
            mean, lo, hi = self.mean_and_ci(success_ma, axis=0, ci=0.95)
            fig, ax = plt.subplots()
            self.line_with_band(ax, episodes, mean, lo, hi, # type: ignore
                                label="Mean Success",
                                title="Success Rate vs Episode (Mean ±95% CI)",
                                y_label="Success Rate")
            ax.grid(alpha=0.25, linewidth=0.5)
            self.save_fig(fig, output_dir, "success_mean_ci.png")

        # Reward ECDFs per run + pooled
        fig, ax = plt.subplots()
        all_rewards = []
        for i, r in enumerate(run_rewards):
            x, y = self.ecdf(r)
            ax.step(x, y, where="post", alpha=0.3, label=f"Run {i+1}")
            all_rewards.extend(r.tolist())
        x, y = self.ecdf(np.asarray(all_rewards))
        ax.step(x, y, where="post", color="black", linewidth=2, label="All runs")
        ax.set_xlabel("Reward"); ax.set_ylabel("ECDF"); ax.set_title("Reward ECDFs per Run")
        ax.legend()
        self.save_fig(fig, output_dir, "reward_ecdf_runs.png")

        # Time ECDFs: success vs failure (pooled)
        succ_t = np.concatenate([t[s > 0.5] for t, s in zip(run_times, run_success)]).astype(float) if run_times else np.array([])
        fail_t = np.concatenate([t[s <= 0.5] for t, s in zip(run_times, run_success)]).astype(float) if run_times else np.array([])
        fig, ax = plt.subplots()
        self.plot_ecdf_on(ax, succ_t, label="Success only", linewidth=2)
        self.plot_ecdf_on(ax, fail_t, label="Failure only", alpha=0.6)
        ax.set_xlabel("Time to end"); ax.set_ylabel("ECDF"); ax.set_title("Time ECDF by Outcome")
        ax.legend(loc="lower right", frameon=False)
        self.save_fig(fig, output_dir, "time_ecdf_by_outcome.png")

        # Reward ECDFs: success vs failure (pooled)
        succ_r = np.concatenate([r[s > 0.5] for r, s in zip(run_rewards, run_success)]).astype(float) if run_rewards else np.array([])
        fail_r = np.concatenate([r[s <= 0.5] for r, s in zip(run_rewards, run_success)]).astype(float) if run_rewards else np.array([])
        fig, ax = plt.subplots()
        self.plot_ecdf_on(ax, succ_r, label="Success only", linewidth=2)
        self.plot_ecdf_on(ax, fail_r, label="Failure only", alpha=0.6)
        ax.set_xlabel("Reward"); ax.set_ylabel("ECDF"); ax.set_title("Reward ECDF by Outcome")
        ax.legend(loc="lower right", frameon=False)
        self.save_fig(fig, output_dir, "reward_ecdf_by_outcome.png")

        # Terminal reason mix per run (proportions)
        run_failures_mapped = [
            ["Success" if r == "None" else r for r in run]
            for run in run_failures
        ]
        reason_labels = sorted({r for run in run_failures_mapped for r in run})
        run_counts = [Counter(run) for run in run_failures_mapped]
        totals = np.asarray([sum(c.values()) for c in run_counts], dtype=float)
        fig, ax = plt.subplots()
        x_pos = np.arange(1, len(runs) + 1)
        stacked = {
            reason: [c.get(reason, 0) / t if t > 0 else 0.0 for c, t in zip(run_counts, totals)]
            for reason in reason_labels
        }
        self.stacked_bars(ax, stacked, x_pos, title="Terminal Reason per Run", # type: ignore
                          x_label="Run", y_label="Proportion", legend=True)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(i) for i in x_pos])
        ax.legend(loc="upper left", ncol=1, frameon=True)
        self.save_fig(fig, output_dir, "terminal_comparison.png")

        # Reward distribution per run (boxplot)
        fig, ax = plt.subplots()
        ax.boxplot(run_rewards, showfliers=False)
        # Create proxy artists for legend
        legend_elems = [
            plt.Line2D([0], [0], color='black', linewidth=2, label='Median'),
            plt.Line2D([0], [0], color='black', linestyle='--', label='Whiskers (IQR × 1.5)'),
            plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='black', label='IQR (25–75%)')
        ]

        ax.legend(handles=legend_elems, loc="upper right", frameon=False, title="Boxplot elements")
        ax.set_xlabel("Run"); ax.set_ylabel("Reward"); ax.set_title("Reward distribution per Run")
        self.save_fig(fig, output_dir, "reward_boxplot_runs.png")

        # Finish-within-T curve (pooled)
        # def finish_within_curve(times):
        #     t_sorted = np.sort(times)
        #     y = np.arange(1, len(t_sorted) + 1) / len(t_sorted)
        #     return t_sorted, y
        #
        # all_t = np.concatenate(run_times).astype(float) if run_times else np.array([])
        # fig, ax = plt.subplots()
        # x, y = finish_within_curve(all_t)
        # if x.size:
        #     ax.plot(x, y)
        # ax.set_xlabel("Time budget T")
        # ax.set_ylabel("P(finish ≤ T)")
        # ax.set_title("Finish-within-T (pooled)")
        # self.save_fig(fig, output_dir, "finish_within_T.png")

        # pooled terminal Pareto (multi-run)
        pooled_fail = Counter(r for run in run_failures for r in run)
        if "None" in pooled_fail:
            pooled_fail["Success"] = pooled_fail.pop("None")
        labels = [k for k, _ in pooled_fail.most_common()]
        counts = [pooled_fail[k] for k in labels]
        fig, ax = plt.subplots()
        self.bar_counts(ax, labels, counts, "Terminal Reasons (pooled)", y_label="Count", rotation=30) # type: ignore
        self.save_fig(fig, output_dir, "terminal_reasons_pooled.png")

        # Per-Run violin plots for reward
        fig, ax = plt.subplots()
        ax.violinplot(run_rewards, showmeans=True, showextrema=False)
        ax.set_xlabel("Run"); ax.set_ylabel("Reward"); ax.set_title("Reward distribution per run (violin)")
        self.save_fig(fig, output_dir, "reward_violin_runs.png")

        # Time-to-end ECDF per run
        fig, ax = plt.subplots()
        all_times: List[float] = []
        for t in run_times:
            x, y = self.ecdf(t)
            ax.step(x, y, where="post", alpha=0.3)
            all_times.extend(list(t))

        if all_times:
            x, y = self.ecdf(np.asarray(all_times))
            ax.step(x, y, where="post", color="black", linewidth=2, label="Mean ECDF")
        ax.set_xlabel("Time to end"); ax.set_ylabel("ECDF"); ax.set_title("Time-to-end ECDF per Run")
        if len(run_rewards) <= 8:
            ax.legend()
        else:
            ax.legend(loc="upper left", ncol=2, frameon=False, title="ECDFs")
        self.save_fig(fig, output_dir, "time_to_end_ecdf_runs.png")
        plt.close()

        # Metric correlation (pooled episodes)
        rows = []
        for run in runs:
            for h in run:
                rows.append([
                    h.get("Reward", np.nan),
                    h.get("Time", np.nan),
                    h.get("Success", np.nan),
                    h.get("Percent_Burned", np.nan),
                ])
        X = np.asarray(rows, dtype=float)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        if X.size:
            corr = np.corrcoef(X.T)
            fig, ax = plt.subplots()
            im = ax.imshow(corr, vmin=-1, vmax=1)
            ax.set_xticks(range(X.shape[1])); ax.set_yticks(range(X.shape[1]))
            ax.set_xticklabels(["Reward","Time","Success","%Burn"], rotation=45, ha="right")
            ax.set_yticklabels(["Reward","Time","Success","%Burn"])
            fig.colorbar(im, ax=ax)
            ax.set_title("Metric correlation (pooled episodes)")
            self.save_fig(fig, output_dir, "metric_correlation.png")

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

    def summarize(self, eval_mode: bool):
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
        if len(self.objectives) == self.objectives.maxlen and not eval_mode:
            self.log_scalar("Training/Objective", np.mean(self.objectives))

        # Clear for next summary
        self.metrics.clear()
        self.histograms.clear()
        self.episode_ended = False
        self.logging_step += 1
        self.writer.flush()
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
        if self.best_metrics["current_objective"] >= self.best_metrics["best_objective"]:
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
        try:
            current_reward = np.mean(self.metrics["Rewards/Rewards_Raw"][-1]) if "Rewards/Rewards_Raw" in self.metrics else 0.0
        except Exception as e:
            logging.warning("Failed to compute current reward: %s", e)
            current_reward = 0.0
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

    def __init__(self, log_dir: str, config: dict, max_train: int, sim_bridge: SimulationBridge, no_gui: bool, start_eval: bool, log_eval: bool, logger: Union[None, TensorboardLogger] = None):
        # self.stats = [EvaluationStats()]
        # Python logger for evaluation messages
        self.logger = logging.getLogger("Evaluator")
        # Optional TensorBoard style logger for metrics
        self.tb_logger = logger
        self.log_dir = log_dir
        self.sim_bridge = sim_bridge
        self.eval_steps = 0
        auto_train_dict = config["settings"]["auto_train"]
        self.hierarchy_steps = 1 if not config["settings"]["hierarchy_type"] == "planner_agent" else config["environment"]["agent"]["planner_agent"]["hierarchy_timesteps"]
        self.use_auto_train = auto_train_dict["use_auto_train"]
        self.no_gui = no_gui
        self.no_gui_eval = no_gui and start_eval # We stop the agent after evaluation because we are in NoGui Mode
        self.max_train = max_train  # Maximum number of training steps before switching to evaluation
        self.max_eval = auto_train_dict["max_eval"]  # Maximum number of evaluation steps
        self.train_episodes = auto_train_dict["train_episodes"] if auto_train_dict["use_auto_train"] else 1  # Number of training episodes to run before stopping Auto Training
        self.current_episode = 0  # Current episode number
        self.avg_reward = -np.inf
        self.avg_objective = 0
        self.avg_tte = +np.inf
        self.tte = np.inf
        # History of per-episode metric values
        self.history: List[Dict[str, float]] = []
        registry = METRIC_REGISTRY if not (self.sim_bridge.get("hierarchy_type") == "fly_agent" and not config["settings"]["eval_fly_policy"]) else METRIC_REGISTRY_FLY_AGENT
        self.metrics = [m() for m in registry]  # Initialize metrics from the registry
        self.terminal_episode_dict = {}
        self.log_eval = log_eval

    def on_update(self):
        if self.sim_bridge.get("train_step") >= self.max_train and (self.use_auto_train or self.no_gui):
            self.sim_bridge.add_value("train_episode", 1)
            self.logger.info("Training finished, after {} training steps, now starting Evaluation".format(self.sim_bridge.get("train_step")))
            self.sim_bridge.set("rl_mode", "eval")
            return True
        return False

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
                self.load_history_from_csvs()
                self.plot_metrics()
                self.logger.info("Agent is now offline. Auto Training is finished")
                return True
            else:
                self.logger.info("Resume with next training step {}/{}".format(train_episode + 1, self.train_episodes))
                self.log_dir = os.path.join(os.path.dirname(os.path.dirname(self.log_dir)),
                                       os.path.split(os.path.dirname(self.log_dir))[-1][:-1]
                                       + str(int(os.path.split(os.path.dirname(self.log_dir))[-1][-1]) + 1), "logs")
                self.sim_bridge.set("train_step", 0)
                self.sim_bridge.set("policy_updates", 0)
                self.sim_bridge.set("current_episode", 0)
        elif self.no_gui:
            self.sim_bridge.set("agent_online", False)
            return True
        return False

    def save_to_csv(self, path):
        """Save collected evaluation statistics to a CSV file."""
        if self.log_eval:
            field_names = [m.name for m in self.metrics]
            with open(path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=field_names) # type: ignore[arg-type]
                writer.writeheader()
                for stat in self.history:
                    writer.writerow({name: stat.get(name, 0.0) for name in field_names})

    def load_history_from_csvs(self):
        """Load evaluation statistics from CSV files in the log directory."""
        par_dir = os.path.dirname(self.log_dir)

        def _is_auto_train_dir(name: str) -> bool:
            parts = name.split("_")
            return len(parts) == 2 and parts[0] == "training" and parts[1].isdigit()

        self_dir = os.path.split(par_dir)[-1]
        is_auto_train_dict = _is_auto_train_dir(self_dir)
        if not is_auto_train_dict:
            self.logger.warning("Log directory %s does not appear to be from an auto-training run", self.log_dir)
            return
        par_dir = os.path.dirname(par_dir)

        auto_train_log_dirs = [os.path.join(os.path.join(par_dir, d), "logs") for d in os.listdir(par_dir) if _is_auto_train_dir(d)]

        self.history = []
        for log_dir in auto_train_log_dirs:
            if not os.path.exists(log_dir):
                self.logger.warning("Auto-training log directory %s does not exist", log_dir)
                continue
            # Find the CSV file in the log directory
            csv_path = os.path.join(log_dir, "evaluation_stats.csv")
            if not os.path.exists(csv_path):
                self.logger.warning("No evaluation_stats.csv found in %s", log_dir)
                continue
            with open(csv_path, "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                run_history = []
                for row in reader:
                    run_history.append(row)
                self.history.append(run_history)

    def plot_metrics(self):
        """Plot collected metrics for a single run or for multiple runs.

        The behaviour depends on the structure of ``self.history``:

        * ``List[Dict]``  – single run.  Produces a learning curve with a
          rolling mean, success rate moving average, time-to-end distribution,
          failure-mode visualisations, and a reward vs. time scatter plot.
        * ``List[List[Dict]]`` – multiple runs (e.g., several seeds).  Creates
          mean \u00b1 confidence-band plots, milestone distributions and a set of
          comparative visualisations across runs.

        If ``matplotlib`` is not available, the function logs a warning and returns silently.
        """
        if self.log_eval:
            plotter = MetricsPlotter(history=self.history,
                                     log_dir=self.log_dir,
                                     logger=self.logger)
            plotter.plot_all()

    def evaluate(self, rewards, terminal_result, percent_burned):
        metrics = self.update_evaluation_metrics(rewards, terminal_result, percent_burned)
        flag_dict = {"auto_train": self.use_auto_train, "reset": False}

        if metrics.get("episode_over"):
            self.eval_steps += 1
            if self.eval_steps >= self.max_eval:
                self.avg_reward = float(np.mean([s["Reward"] for s in self.history])) if self.history else 0.0
                self.avg_objective = float(np.mean([s["Success"] for s in self.history])) if self.history else 0.0
                self.avg_tte = float(
                    np.mean([s["Time"] for s in self.history])) if self.history and self.sim_bridge.get(
                    "hierarchy_type") == "planner_agent" else 0.0
                self.logger.info(f"Evaluation finished, after {self.eval_steps} evaluation "
                                 f"steps with average reward: {self.avg_reward} and average objective: {self.avg_objective}")
                self.save_to_csv(os.path.join(self.log_dir, "evaluation_stats.csv"))
                self.plot_metrics()
                if self.reset():
                    flag_dict.__setitem__("auto_train_not_finished", False)
                self.sim_bridge.set("agent_is_running", False)
                flag_dict.__setitem__("reset", True)
            else:
                self.avg_reward = float(np.mean([s["Reward"] for s in self.history])) if self.history else 0.0


        return flag_dict

    def final_metrics(self):
        """Returns the final evaluation metrics after all evaluation episodes are completed.
           Only used for Optuna Optimization and otherwise saved in CSV and Plots, as well as the logs
        """
        return {"reward": self.avg_reward, "objective": self.avg_objective, "time_to_end": self.avg_tte}

    def update_evaluation_metrics(self, rewards, terminal_result, percent_burned):
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
            # Store final metrics for this episode
            episode_metrics = {m.name: m.value for m in self.metrics} # type: ignore
            self.history.append(episode_metrics)

            # Compute aggregated metrics
            for metric in self.metrics:
                metrics.update(metric.compute(self.history))
                metric.reset()

            metrics["episode_over"] = True

            self.clean_print()

            self.current_episode += 1

            # Log metrics via the optional TensorBoard logger
            if self.tb_logger is not None and self.log_eval:
                tb_metrics = {k: v for k, v in metrics.items() if k not in ("episode", "episode_over", "Failure_Reason" , "Failure_Reason_counts", "Failure_Reason_perc", "Failure_Reason_n")}
                self.tb_logger.add_metric(tb_metrics)
                # Flush metrics to disk after each evaluation episode
                self.tb_logger.summarize(eval_mode=True)

        return metrics

    def clean_print(self):
        last_episode = (self.eval_steps >= self.max_eval - 1)
        if last_episode:
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
            print()
            self.logger.info("\n%s", table)
            self.terminal_episode_dict = {}
        else:
            def terminal_append(reason: str):
                self.terminal_episode_dict[reason] = self.terminal_episode_dict.get(reason, 0) + 1

            def get_terminal_string():
                items = self.terminal_episode_dict.items()
                items = sorted(items)
                return "; ".join(f"{k}: {v}" for k, v in items)

            terminal_reason = self.history[self.current_episode]["Failure_Reason"]
            terminal_append(terminal_reason)
            print(f"\rEvaluation running... {get_terminal_string()}", end='')
