from typing import Dict, Any, List
import numpy as np
from scipy import stats as st

class Metric:
    """Base class for evaluation metrics.

    A metric keeps track of a value during an episode via :meth:`update` and
    can compute aggregated statistics from the history of episodes via
    :meth:`compute`.
    """

    def __init__(self, name: str = "Metric") -> None:
        self.value = None
        self.int_like = False  # whether this metric is an integer-like value
        self.is_percentage = False  # whether this metric is a percentage
        self.name : str = name
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
        # pull this metric’s values from episode history
        return [h[self.name] for h in history]

    # Default: numeric summary for scalar metrics
    def compute(self, history: List[Dict[str, float]]) -> Dict[str, float]:
        mean, std, median, lo, hi = self.summary_stats(self._collect(history))
        base = {
            f"{self.name}_mean": mean,
            f"{self.name}_std": std,
            f"{self.name}_median": median,
            f"{self.name}_ci_low": lo,
            f"{self.name}_ci_high": hi,
        }
        return base

    def is_int_like(self):
        return self.int_like

# Scalar metrics inherit default compute method

class RewardMetric(Metric):
    def __init__(self) -> None:
        super().__init__("Reward")

    def update(self, stats: Dict[str, Any]) -> None:
        self.value += float(stats["rewards"][0])


class TimeMetric(Metric):
    def __init__(self) -> None:
        super().__init__("Time")
        self.int_like = True

    def update(self, stats: Dict[str, Any]) -> None:
        self.value += 1


class PercentBurnedMetric(Metric):
    def __init__(self) -> None:
        super().__init__("Percent_Burned")

    def update(self, stats: Dict[str, Any]) -> None:
        if stats["terminal_result"]["EnvReset"]:
            self.value = float(stats["percent_burned"])

# Override compute for Totals/Rates

class SuccessMetric(Metric):
    def __init__(self) -> None:
        super().__init__("Success")
        self.is_percentage = True  # no need for summary stats, just 0 or 1

    def update(self, stats: Dict[str, Any]) -> None:
        if stats["terminal_result"]["EnvReset"]:
            self.value = 0.0 if stats["terminal_result"]["OneAgentDied"] else 1.0

    def compute(self, history: List[Dict[str, float]]) -> Dict[str, float]:
        avg = float(np.mean([h[self.name] for h in history])) if history else 0.0
        return {"Success_Rate": avg}