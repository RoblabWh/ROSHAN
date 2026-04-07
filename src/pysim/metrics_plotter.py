import copy
import os
import logging
import numpy as np
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
from math import sqrt
import scipy.stats as stats
from metrics import Metric


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
                m, _, _, lo, hi = Metric.summary_stats(vals, ci=ci)
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
        from collections import Counter
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
        self.save_fig(fig, output_dir, "reward_vs_time.png")

        fig, ax = plt.subplots()
        ax.scatter(times, success)
        ax.set_xlabel("Time"); ax.set_ylabel("Success"); ax.set_title("Success vs Time")
        self.save_fig(fig, output_dir, "success_vs_time.png")

    # ----------------------------- multi run -----------------------------
    def _plot_multi_run(self, runs: List[List[Dict[str, Any]]]) -> None:
        import matplotlib.pyplot as plt
        from collections import Counter

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

        # Reward mean +/- CI vs episode
        if min_len > 0:
            rewards_mat = np.stack([r[:min_len] for r in run_rewards])  # runs x episodes
            mean, lo, hi = self.mean_and_ci(rewards_mat, axis=0, ci=0.95)
            fig, ax = plt.subplots()
            self.line_with_band(ax, episodes, mean, lo, hi, # type: ignore
                                label="Mean Reward",
                                title="Reward vs Episode (Mean +/-95% CI)",
                                y_label="Reward")
            ax.grid(alpha=0.25, linewidth=0.5)
            self.save_fig(fig, output_dir, "reward_mean_ci.png")

        # Reward vs Time: scatter per run + pooled binned mean +/- CI
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

        # Success mean +/- CI across runs (moving-avg per episode)
        if min_len > 0:
            window = max(1, min_len // 20)
            success_ma = np.stack([self.rolling_mean(s[:min_len], window) for s in run_success])
            mean, lo, hi = self.mean_and_ci(success_ma, axis=0, ci=0.95)
            fig, ax = plt.subplots()
            self.line_with_band(ax, episodes, mean, lo, hi, # type: ignore
                                label="Mean Success",
                                title="Success Rate vs Episode (Mean +/-95% CI)",
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
            plt.Line2D([0], [0], color='black', linestyle='--', label='Whiskers (IQR x 1.5)'),
            plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='black', label='IQR (25-75%)')
        ]

        ax.legend(handles=legend_elems, loc="upper right", frameon=False, title="Boxplot elements")
        ax.set_xlabel("Run"); ax.set_ylabel("Reward"); ax.set_title("Reward distribution per Run")
        self.save_fig(fig, output_dir, "reward_boxplot_runs.png")

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
