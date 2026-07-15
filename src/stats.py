"""Small statistical utilities: bootstrap confidence intervals for backtest metrics."""
import numpy as np
from typing import Callable, Sequence


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_iterations: int = 5000,
    ci: float = 0.90,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap a confidence interval for a statistic (default: the mean) of `values`.

    Returns (point_estimate, lower_bound, upper_bound). Resamples `values` with
    replacement `n_iterations` times; the interval is the [alpha/2, 1-alpha/2]
    percentiles of the resampled statistic. With few observations (as in a
    small backtest sample), a wide interval is the honest result, not a bug.
    """
    values_arr = np.asarray(values, dtype=float)
    if len(values_arr) == 0:
        return 0.0, 0.0, 0.0

    point_estimate = float(statistic(values_arr))
    if len(values_arr) == 1:
        return point_estimate, point_estimate, point_estimate

    rng = np.random.default_rng(seed)
    resampled_stats = np.empty(n_iterations)
    n = len(values_arr)
    for i in range(n_iterations):
        sample = values_arr[rng.integers(0, n, size=n)]
        resampled_stats[i] = statistic(sample)

    alpha = 1 - ci
    lower = float(np.percentile(resampled_stats, 100 * alpha / 2))
    upper = float(np.percentile(resampled_stats, 100 * (1 - alpha / 2)))
    return point_estimate, lower, upper
