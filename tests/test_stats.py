import numpy as np
import pytest

from src.stats import bootstrap_ci


class TestBootstrapCi:
    def test_point_estimate_matches_statistic(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        point, lower, upper = bootstrap_ci(values, statistic=np.mean, n_iterations=2000)
        assert point == pytest.approx(3.0)

    def test_interval_contains_point_estimate(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        point, lower, upper = bootstrap_ci(values, n_iterations=2000)
        assert lower <= point <= upper

    def test_wider_ci_gives_wider_interval(self):
        values = list(range(1, 21))
        _, lower_90, upper_90 = bootstrap_ci(values, ci=0.90, n_iterations=3000, seed=1)
        _, lower_99, upper_99 = bootstrap_ci(values, ci=0.99, n_iterations=3000, seed=1)
        assert (upper_99 - lower_99) >= (upper_90 - lower_90)

    def test_more_data_gives_narrower_interval(self):
        rng = np.random.default_rng(0)
        small_sample = rng.normal(0, 1, size=10)
        large_sample = rng.normal(0, 1, size=1000)

        _, lower_small, upper_small = bootstrap_ci(small_sample, n_iterations=3000, seed=2)
        _, lower_large, upper_large = bootstrap_ci(large_sample, n_iterations=3000, seed=2)

        assert (upper_large - lower_large) < (upper_small - lower_small)

    def test_empty_input_returns_zeros(self):
        assert bootstrap_ci([]) == (0.0, 0.0, 0.0)

    def test_single_value_returns_degenerate_interval(self):
        assert bootstrap_ci([7.0]) == (7.0, 7.0, 7.0)
