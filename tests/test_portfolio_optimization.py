import numpy as np
import pytest

from src.portfolio_optimization import (
    bet_return_moments,
    build_covariance_matrix,
    multivariate_kelly_weights,
)


def make_bet(model_prob, decimal_odds, home="A", away="B"):
    return {"model_prob": model_prob, "decimal_odds": decimal_odds, "teams": (home, away)}


class TestBetReturnMoments:
    def test_known_values(self):
        # p=0.6, decimal_odds=2.0 -> b=1: mean = 0.6*1 - 0.4 = 0.2
        # second moment = 0.6*1 + 0.4 = 1.0 -> variance = 1.0 - 0.04 = 0.96
        mean, variance = bet_return_moments(0.6, 2.0)
        assert mean == pytest.approx(0.2)
        assert variance == pytest.approx(0.96)

    def test_higher_probability_gives_higher_mean(self):
        mean_low, _ = bet_return_moments(0.5, 2.0)
        mean_high, _ = bet_return_moments(0.7, 2.0)
        assert mean_high > mean_low


class TestBuildCovarianceMatrix:
    def test_diagonal_when_no_shared_teams(self):
        bets = [make_bet(0.6, 2.0, "A", "B"), make_bet(0.55, 1.9, "C", "D")]
        cov = build_covariance_matrix(bets)
        assert cov[0, 1] == pytest.approx(0.0)
        assert cov[1, 0] == pytest.approx(0.0)

    def test_nonzero_off_diagonal_when_teams_shared(self):
        # Second bet's home team (A) is the first bet's home team -> a same-day
        # doubleheader for team A, the correlation source this module models.
        bets = [make_bet(0.6, 2.0, "A", "B"), make_bet(0.55, 1.9, "A", "C")]
        cov = build_covariance_matrix(bets, same_team_correlation=0.5)

        _, var0 = bet_return_moments(0.6, 2.0)
        _, var1 = bet_return_moments(0.55, 1.9)
        expected_cov = 0.5 * np.sqrt(var0 * var1)
        assert cov[0, 1] == pytest.approx(expected_cov)
        assert cov[1, 0] == pytest.approx(expected_cov)


class TestMultivariateKellyWeights:
    def test_empty_slate_returns_empty(self):
        assert multivariate_kelly_weights([]) == []

    def test_independent_bets_decouple(self):
        # With no shared teams, Sigma is diagonal, so each weight should equal
        # its own mu_i / var_i (fractional-Kelly-scaled), independent of the
        # other bet entirely -- the key "independent bets sized independently"
        # property, checked against the hand-derived closed form, not just
        # "some number came out."
        bets = [make_bet(0.6, 2.0, "A", "B"), make_bet(0.55, 1.9, "C", "D")]
        weights = multivariate_kelly_weights(
            bets, kelly_fraction_mult=0.5, max_bet_pct=1.0, max_slate_pct=1.0
        )

        mean0, var0 = bet_return_moments(0.6, 2.0)
        mean1, var1 = bet_return_moments(0.55, 1.9)
        assert weights[0] == pytest.approx(0.5 * mean0 / var0)
        assert weights[1] == pytest.approx(0.5 * mean1 / var1)

    def test_correlated_identical_bets_get_less_combined_weight_than_independent(self):
        # Two bets on the same team (a doubleheader), identical edge and odds.
        # Hand-derived closed form for this symmetric case: each weight is
        # mean / (variance * (1 + rho)), vs. mean/variance each if treated as
        # independent -- so positive correlation must strictly shrink each
        # weight, and the combined pair must be less than 2x the independent
        # single-bet weight.
        rho = 0.5
        bets = [make_bet(0.6, 2.0, "A", "B"), make_bet(0.6, 2.0, "A", "C")]
        weights = multivariate_kelly_weights(
            bets, same_team_correlation=rho, kelly_fraction_mult=1.0, max_bet_pct=1.0, max_slate_pct=1.0
        )

        mean, var = bet_return_moments(0.6, 2.0)
        expected_each = mean / (var * (1 + rho))
        independent_each = mean / var

        assert weights[0] == pytest.approx(expected_each)
        assert weights[1] == pytest.approx(expected_each)
        assert weights[0] < independent_each
        assert sum(weights) < 2 * independent_each

    def test_near_perfect_correlation_collapses_combined_weight_to_one_bet(self):
        # rho -> 1: two "duplicate" bets should be treated in aggregate like
        # one bet's worth of exposure, not two independent ones.
        bets = [make_bet(0.6, 2.0, "A", "B"), make_bet(0.6, 2.0, "A", "C")]
        weights = multivariate_kelly_weights(
            bets, same_team_correlation=0.999, kelly_fraction_mult=1.0, max_bet_pct=1.0, max_slate_pct=1.0
        )

        mean, var = bet_return_moments(0.6, 2.0)
        single_bet_weight = mean / var
        assert sum(weights) == pytest.approx(single_bet_weight, rel=0.01)

    def test_negative_edge_floored_at_zero(self):
        bets = [make_bet(0.3, 2.0, "A", "B")]  # p < implied 0.5 fair price -> negative edge
        weights = multivariate_kelly_weights(bets, max_bet_pct=1.0, max_slate_pct=1.0)
        assert weights[0] == pytest.approx(0.0)

    def test_per_bet_cap_applies(self):
        bets = [make_bet(0.95, 3.0, "A", "B")]  # huge edge -> would want a large stake
        weights = multivariate_kelly_weights(bets, max_bet_pct=0.05, max_slate_pct=1.0)
        assert weights[0] == pytest.approx(0.05)

    def test_slate_cap_scales_down_proportionally(self):
        bets = [make_bet(0.9, 3.0, "A", "B"), make_bet(0.9, 3.0, "C", "D")]
        weights = multivariate_kelly_weights(bets, max_bet_pct=0.10, max_slate_pct=0.15)
        assert sum(weights) == pytest.approx(0.15)
        assert weights[0] == pytest.approx(weights[1])
