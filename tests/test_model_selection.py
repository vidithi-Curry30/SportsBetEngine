import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.model_selection import select_regularization_strength, standardized_coefficients


def make_games_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    feat_a = rng.normal(0, 1, n)
    feat_b = rng.normal(0, 5, n)  # much larger scale than feat_a
    # feat_a is the real signal; feat_b is pure noise on a bigger scale.
    prob = 1 / (1 + np.exp(-(1.5 * feat_a)))
    y = (rng.random(n) < prob).astype(int)
    return pd.DataFrame(
        {"date": dates.strftime("%Y-%m-%d"), "feat_a": feat_a, "feat_b": feat_b, "team_a_win": y}
    )


class TestSelectRegularizationStrength:
    def test_returns_a_c_from_the_grid(self):
        df = make_games_df(200)
        grid = (0.01, 0.1, 1.0, 10.0)
        best_C, results_df = select_regularization_strength(
            df, ["feat_a", "feat_b"], "team_a_win", C_grid=grid
        )
        assert best_C in grid
        assert len(results_df) == len(grid)

    def test_best_c_matches_argmin_of_reported_grid(self):
        # Internal consistency: whatever the search picks must be the row
        # with the lowest reported inner-validation log loss, not some other
        # criterion silently used instead.
        df = make_games_df(200)
        best_C, results_df = select_regularization_strength(df, ["feat_a", "feat_b"], "team_a_win")
        expected_best_C = results_df.loc[results_df["inner_val_log_loss"].idxmin(), "C"]
        assert best_C == pytest.approx(expected_best_C)

    def test_log_losses_are_finite_and_nonnegative(self):
        df = make_games_df(200)
        _, results_df = select_regularization_strength(df, ["feat_a", "feat_b"], "team_a_win")
        assert (results_df["inner_val_log_loss"] >= 0).all()
        assert np.isfinite(results_df["inner_val_log_loss"]).all()

    def test_too_small_train_df_raises(self):
        df = make_games_df(3)
        with pytest.raises(ValueError):
            select_regularization_strength(df, ["feat_a", "feat_b"], "team_a_win")


class TestStandardizedCoefficients:
    def test_formula_is_coefficient_times_feature_std(self):
        df = make_games_df(300)
        model = LogisticRegression()
        model.fit(df[["feat_a", "feat_b"]], df["team_a_win"])

        result = standardized_coefficients(model, ["feat_a", "feat_b"], df)

        for _, row in result.iterrows():
            raw_coef = model.coef_[0][["feat_a", "feat_b"].index(row["feature"])]
            expected_std = df[row["feature"]].std()
            assert row["raw_coefficient"] == pytest.approx(raw_coef)
            assert row["standardized_coefficient"] == pytest.approx(raw_coef * expected_std)

    def test_sorted_by_absolute_standardized_coefficient_descending(self):
        df = make_games_df(300)
        model = LogisticRegression()
        model.fit(df[["feat_a", "feat_b"]], df["team_a_win"])

        result = standardized_coefficients(model, ["feat_a", "feat_b"], df)
        abs_coefs = result["standardized_coefficient"].abs().tolist()
        assert abs_coefs == sorted(abs_coefs, reverse=True)

    def test_real_signal_outranks_larger_scale_noise_feature(self):
        # feat_a is the actual signal; feat_b is a bigger-scale pure-noise
        # feature. A *raw* coefficient comparison could easily be misled by
        # feat_b's larger scale -- standardization should rank feat_a first.
        df = make_games_df(500)
        model = LogisticRegression()
        model.fit(df[["feat_a", "feat_b"]], df["team_a_win"])

        result = standardized_coefficients(model, ["feat_a", "feat_b"], df)
        assert result.iloc[0]["feature"] == "feat_a"
