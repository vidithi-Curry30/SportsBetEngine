"""Model calibration: is 'the model said 70%' actually right about 70% of the time?

Accuracy alone doesn't answer that -- a model can be directionally right (good
accuracy) while its probabilities are badly miscalibrated, which is exactly
what breaks a probability-driven edge-detection threshold like `model.has_value_edge`
and Kelly sizing, both of which trust the raw probability, not just its rank.
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def reliability_curve(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Bucket predictions into `n_bins` and compare mean predicted probability
    to observed win frequency in each bucket -- a perfectly calibrated model
    has mean_predicted_prob == observed_frequency in every bucket."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bin_edges) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin_lower": bin_edges[b],
                "bin_upper": bin_edges[b + 1],
                "mean_predicted_prob": float(y_prob[mask].mean()),
                "observed_frequency": float(y_true[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


class _BaseCalibrator:
    """Wraps a fitted model with a recalibration mapping from its raw
    probabilities to calibrated ones, fit on a calibration set that is
    neither the model's training data nor the final backtest/test period --
    otherwise the "fix" is just more overfitting. Subclasses supply the
    mapping itself (isotonic vs. Platt/sigmoid).
    """

    def __init__(self, model, feature_columns: list[str]):
        self.model = model
        self.feature_columns = feature_columns

    def fit(self, calibration_df: pd.DataFrame, target_column: str = "team_a_win") -> "_BaseCalibrator":
        raw_probs = self.model.predict_proba(calibration_df[self.feature_columns])[:, 1]
        self._fit_mapping(raw_probs, calibration_df[target_column].to_numpy())
        return self

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        raw_probs = self.model.predict_proba(features_df[self.feature_columns])[:, 1]
        return self._transform(raw_probs)

    def predict_win_probability(self, game_features) -> float:
        row = pd.DataFrame([dict(game_features)])[self.feature_columns]
        return float(self.predict_proba(row)[0])

    def _fit_mapping(self, raw_probs: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def _transform(self, raw_probs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class IsotonicCalibrator(_BaseCalibrator):
    """Nonparametric (monotonic step function) recalibration. Flexible, but
    per sklearn's own guidance needs a large calibration set (~1000+ samples)
    to avoid overfitting the mapping itself."""

    def __init__(self, model, feature_columns: list[str]):
        super().__init__(model, feature_columns)
        self._isotonic = IsotonicRegression(out_of_bounds="clip")

    def _fit_mapping(self, raw_probs: np.ndarray, y: np.ndarray) -> None:
        self._isotonic.fit(raw_probs, y)

    def _transform(self, raw_probs: np.ndarray) -> np.ndarray:
        return self._isotonic.predict(raw_probs)


class PlattCalibrator(_BaseCalibrator):
    """Platt (sigmoid) scaling: a 2-parameter logistic regression of the raw
    probability against the outcome. Far less data-hungry than isotonic
    regression, at the cost of only being able to express a monotonic
    sigmoid remapping rather than an arbitrary one."""

    def __init__(self, model, feature_columns: list[str]):
        super().__init__(model, feature_columns)
        self._platt = LogisticRegression()

    def _fit_mapping(self, raw_probs: np.ndarray, y: np.ndarray) -> None:
        self._platt.fit(raw_probs.reshape(-1, 1), y)

    def _transform(self, raw_probs: np.ndarray) -> np.ndarray:
        return self._platt.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
