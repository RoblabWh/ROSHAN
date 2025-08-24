from typing import Dict, Any, List, Optional
import numpy as np
from scipy import stats as st

class Metric:
    """Base class for evaluation metrics.

    A metric keeps track of a value during an episode via :meth:`update` and
    can compute aggregated statistics from the history of episodes via
    :meth:`compute`.
    - dtype: "float" | "int" | "percent" | "string"
    - agg:   "mean" | "sum" | "last" | "rate" | "distribution"
    """

    def __init__(self, name: str = "Metric", dtype: str = "float", agg: str = "mean") -> None:
        self.value = None
        self.name : str = name
        self.dtype = dtype
        self.agg = agg
        self.reset()

    def reset(self) -> None:
        """Reset metric value for a new episode."""
        self.value: float = 0.0

    def update(self, stats: Dict[str, Any]) -> None:  # pragma: no cover - interface
        """Update the metric using environment statistics."""
        raise NotImplementedError

    @staticmethod
    def summary_stats(vals, ci=0.95):
        vals = np.asarray(vals, dtype=float)
        n = len(vals)
        mean = float(np.mean(vals)) if n else 0.0
        median = float(np.median(vals)) if n else 0.0
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        if n > 1:
            sem = st.sem(vals)
            if sem == 0:
                lo = hi = mean
            else:
                lo, hi = st.t.interval(ci, n - 1, loc=mean, scale=sem)
        else:
            lo = hi = mean
        return mean, std, median, lo, hi

    def _collect(self, history):
        # pull this metrics values from episode history
        return [h[self.name] for h in history]

    def compute(self, history: List[Dict[str, Any]], ci=0.95) -> Dict[str, Any]:
        vals = self._collect(history)
        if self.agg in ("mean", "sum", "last"):
            if self.agg == "last":
                last = vals[-1] if vals else (0 if self.dtype != "string" else "None")
                return {f"{self.name}_last": last}
            if self.dtype in ("float", "int", "percent"):
                mean, std, median, lo, hi = self.summary_stats(vals, ci=ci)
                base = {
                    f"{self.name}_mean": mean,
                    f"{self.name}_std": std,
                    f"{self.name}_median": median,
                    f"{self.name}_ci_low": lo,
                    f"{self.name}_ci_high": hi,
                }
                if self.agg == "sum":
                    base[f"{self.name}_sum"] = float(np.sum(vals)) if len(vals) else 0.0
                return base
            # strings default to last
            return {f"{self.name}_last": vals[-1] if vals else "None"}

        elif self.agg == "rate":
            # Bernoulli/Wilson for Success-like metrics
            successes = np.sum(np.asarray(vals, dtype=int))
            n = len(vals)
            result = st.binomtest(successes, n, alternative='two-sided')
            lo, hi = result.proportion_ci(ci, method="wilson") if n else (0.0, 0.0)
            rate = float(successes / n) if n else 0.0
            return {
                f"{self.name}_rate": rate,
                f"{self.name}_ci_low": lo,
                f"{self.name}_ci_high": hi,
                f"{self.name}_n": n
            }

        elif self.agg == "distribution":
            # e.g., failure reasons
            counts = {}
            for v in vals:
                k = (v if v not in (None, "", "None") else "None")
                counts[k] = counts.get(k, 0) + 1
            total = sum(counts.values()) # TODO Track this
            perc = {k: c/total for k,c in counts.items()} if total else {}
            return {
                f"{self.name}_counts": counts,
                f"{self.name}_perc": perc,
                f"{self.name}_n": total
            }

        else:
            raise ValueError(f"Unknown agg: {self.agg}")

    def get_compute_string(self, compute_dict: Dict[str, Any]) -> Optional[str]:
        """
        Produces a string for a given dict that was created by compute()
        Takes care of the agg-specific formatting
        """
        if self.agg in ("mean", "sum", "last"):
            if self.agg == "last" or self.dtype == "string":
                return f"{compute_dict[f'{self.name}_last']}"
            else:
                mean = compute_dict[f"{self.name}_mean"]
                std = compute_dict[f"{self.name}_std"]
                median = compute_dict[f"{self.name}_median"]
                lo = compute_dict[f"{self.name}_ci_low"]
                hi = compute_dict[f"{self.name}_ci_high"]
                if self.agg == "sum":
                    total = compute_dict[f"{self.name}_sum"]
                    return (f"mean={mean:.3f}\nstd={std:.3f}\nmedian={median:.3f}\n"
                            f"sum={total:.3f}\n95% CI=({lo:.3f}, {hi:.3f})")
                else:
                    return (f"mean={mean:.3f}\nstd={std:.3f}\nmedian={median:.3f}\n"
                            f"95% CI=({lo:.3f}, {hi:.3f})")
        elif self.agg == "rate":
            rate = compute_dict[f"{self.name}_rate"]
            lo = compute_dict[f"{self.name}_ci_low"]
            hi = compute_dict[f"{self.name}_ci_high"]
            n = compute_dict[f"{self.name}_n"]
            return f"rate={rate:.3f}\n95% CI=({lo:.3f}, {hi:.3f})\nn={n}"
        elif self.agg == "distribution":
            counts = compute_dict[f"{self.name}_counts"]
            perc = compute_dict[f"{self.name}_perc"]
            items = [f"{k}: {v} ({perc[k]*100:.1f}%)" for k,v in counts.items()]
            items_str = "\n".join(items)
            return f"{items_str}"

class RewardMetric(Metric):
    def __init__(self) -> None:
        super().__init__("Reward", dtype="float", agg="mean")
    def reset(self) -> None: self.value = 0.0
    def update(self, stats: Dict[str, Any]) -> None:
        # Maybe I should not use the mean of all agents here and keep track of each agent's reward separately?
        # On the contrary, what does it even tell me if I do that?
        # All agents share the same network, so the difference in rewards must be attributed to the environment.
        # I don't TRACK observations yet and I have no intention to do so right now.
        self.value += float(np.mean(stats["rewards"]))

class TimeMetric(Metric):
    def __init__(self) -> None:
        super().__init__("Time", dtype="int", agg="mean")
    def reset(self) -> None: self.value = 0
    def update(self, stats: Dict[str, Any]) -> None:
        self.value += 1


class PercentBurnedMetric(Metric):
    def __init__(self) -> None:
        super().__init__("Percent_Burned", dtype="float", agg="mean")
    def reset(self) -> None: self.value = 0.0
    def update(self, stats: Dict[str, Any]) -> None:
        if stats["terminal_result"].env_reset:
            self.value = float(stats["percent_burned"])

class SuccessMetric(Metric):
    def __init__(self) -> None:
        super().__init__("Success", dtype="percent", agg="rate")
    def reset(self) -> None: self.value = 0.0
    def update(self, stats: Dict[str, Any]) -> None:
        if stats["terminal_result"].env_reset:
            self.value = 0.0 if stats["terminal_result"].any_failed else 1.0

class FailureReason(Metric):
    def __init__(self) -> None:
        super().__init__("Failure_Reason", dtype="string", agg="distribution")
    def reset(self) -> None: self.value = "None"
    def update(self, stats: Dict[str, Any]) -> None:
        if stats["terminal_result"].env_reset and stats["terminal_result"].any_failed:
            self.value = stats["terminal_result"].reason.name if stats["terminal_result"].reason else "Unknown"