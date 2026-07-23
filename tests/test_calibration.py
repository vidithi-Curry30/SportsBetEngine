import numpy as np
import pandas as pd
import pytest

from src.calibration import IsotonicCalibrator, PlattCalibrator, reliability_curve


class TestReliabilityCurve:
    def test_perfectly_calibrated_predictions(self):
        # 100 predictions at 0.2 with 20% observed frequency, 100 at 0.8 with 80%
        rng = np.random.default_rng(0)
        y_prob = np.array([0.2] * 100 + [0.8] * 100)
        y_true = np.array(
            [1] * 20 + [0] * 80 + [1] * 80 + [0] * 20
        )

        curve = reliability_curve(y_true, y_prob, n_bins=10)

        for _, row in curve.iterrows():
            assert row["mean_predicted_prob"] == pytest.approx(row["observed_frequency"], abs=0.05)

    def test_bin_counts_sum_to_total(self):
        y_prob = np.linspace(0.05, 0.95, 50)
        y_true = np.zeros(50)
        curve = reliability_curve(y_true, y_prob, n_bins=10)
        assert curve["count"].sum() == 50

    def test_empty_bins_are_omitted(self):
        # All predictions clustered near 0.5 -> most bins should have zero count and be dropped
        y_prob = np.full(20, 0.5)
        y_true = np.zeros(20)
        curve = reliability_curve(y_true, y_prob, n_bins=10)
        assert len(curve) == 1


class DummyModel:
    """A model whose raw probabilities are a fixed, known miscalibration of the truth."""

    def predict_proba(self, X):
        n = len(X)
        # Systematically overconfident: pushes everything toward the extremes
        raw = np.clip(X["true_prob"].to_numpy() * 1.5 - 0.25, 0.01, 0.99)
        return np.column_stack([1 - raw, raw])


class TestIsotonicCalibrator:
    def test_calibration_improves_reliability(self):
        rng = np.random.default_rng(1)
        n = 2000
        true_prob = rng.uniform(0.1, 0.9, size=n)
        y = (rng.random(n) < true_prob).astype(int)
        df = pd.DataFrame({"true_prob": true_prob, "team_a_win": y})

        model = DummyModel()
        calibrator = IsotonicCalibrator(model, feature_columns=["true_prob"]).fit(df)

        raw_probs = model.predict_proba(df[["true_prob"]])[:, 1]
        calibrated_probs = calibrator.predict_proba(df)

        raw_error = np.mean((raw_probs - true_prob) ** 2)
        calibrated_error = np.mean((calibrated_probs - true_prob) ** 2)
        assert calibrated_error < raw_error

    def test_predict_win_probability_returns_scalar_in_range(self):
        df = pd.DataFrame({"true_prob": np.linspace(0.1, 0.9, 50), "team_a_win": [0, 1] * 25})
        model = DummyModel()
        calibrator = IsotonicCalibrator(model, feature_columns=["true_prob"]).fit(df)

        prob = calibrator.predict_win_probability({"true_prob": 0.6})
        assert 0.0 <= prob <= 1.0


class TestPlattCalibrator:
    def test_calibration_improves_reliability(self):
        rng = np.random.default_rng(1)
        n = 2000
        true_prob = rng.uniform(0.1, 0.9, size=n)
        y = (rng.random(n) < true_prob).astype(int)
        df = pd.DataFrame({"true_prob": true_prob, "team_a_win": y})

        model = DummyModel()
        calibrator = PlattCalibrator(model, feature_columns=["true_prob"]).fit(df)

        raw_probs = model.predict_proba(df[["true_prob"]])[:, 1]
        calibrated_probs = calibrator.predict_proba(df)

        raw_error = np.mean((raw_probs - true_prob) ** 2)
        calibrated_error = np.mean((calibrated_probs - true_prob) ** 2)
        assert calibrated_error < raw_error

    def test_predict_win_probability_returns_scalar_in_range(self):
        df = pd.DataFrame({"true_prob": np.linspace(0.1, 0.9, 50), "team_a_win": [0, 1] * 25})
        model = DummyModel()
        calibrator = PlattCalibrator(model, feature_columns=["true_prob"]).fit(df)

        prob = calibrator.predict_win_probability({"true_prob": 0.6})
        assert 0.0 <= prob <= 1.0

    def test_stable_with_small_calibration_set(self):
        # Platt's 2-parameter fit shouldn't blow up or go non-monotonic on a small sample,
        # unlike isotonic regression's unconstrained step function.
        rng = np.random.default_rng(3)
        n = 30
        true_prob = rng.uniform(0.1, 0.9, size=n)
        y = (rng.random(n) < true_prob).astype(int)
        df = pd.DataFrame({"true_prob": true_prob, "team_a_win": y})

        model = DummyModel()
        calibrator = PlattCalibrator(model, feature_columns=["true_prob"]).fit(df)

        probs = calibrator.predict_proba(df.sort_values("true_prob"))
        assert (np.diff(probs) >= -1e-9).all()  # monotonically non-decreasing
