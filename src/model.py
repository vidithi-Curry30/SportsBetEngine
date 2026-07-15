"""Logistic regression win-probability model."""
import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURE_COLUMNS = [
    "pace_diff",
    "off_rating_diff",
    "def_rating_diff",
    "recent_win_pct_diff",
    "rest_days_diff",
    "home_flag",
]
TARGET_COLUMN = "team_a_win"


def chronological_split(
    games_df: pd.DataFrame, train_frac: float = 0.8, date_column: str = "date"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split games into train/test by time, never shuffling.

    The first `train_frac` of games (by date) become the training set; the
    remaining held-out period becomes the test set. The test set must never
    be touched during training or feature selection.
    """
    sorted_df = games_df.sort_values(date_column).reset_index(drop=True)
    split_idx = int(len(sorted_df) * train_frac)
    train_df = sorted_df.iloc[:split_idx].copy()
    test_df = sorted_df.iloc[split_idx:].copy()
    return train_df, test_df


def chronological_split_train_calib_test(
    games_df: pd.DataFrame,
    train_frac: float = 0.8,
    calib_frac_of_train: float = 0.15,
    date_column: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three-way chronological split: fit-training / calibration / test.

    The test set is identical to what `chronological_split` would return (the
    last `1 - train_frac` of games), so results computed on it remain directly
    comparable to an uncalibrated run. The calibration set is carved out of the
    *training* period (its most recent `calib_frac_of_train`) so a calibrator
    fit on it has still never seen the test period.
    """
    train_df, test_df = chronological_split(games_df, train_frac, date_column)
    calib_cut = int(len(train_df) * (1 - calib_frac_of_train))
    fit_train_df = train_df.iloc[:calib_cut].copy()
    calib_df = train_df.iloc[calib_cut:].copy()
    return fit_train_df, calib_df, test_df


def train_model(
    training_df: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
) -> LogisticRegression:
    """Fit a logistic regression win-probability model on the training period only."""
    X = training_df[feature_columns]
    y = training_df[target_column]
    model = LogisticRegression()
    model.fit(X, y)
    return model


def predict_win_probability(
    model: LogisticRegression, game_features: dict | pd.Series, feature_columns: list[str] = FEATURE_COLUMNS
) -> float:
    """Predict team_a's win probability for a single game's feature set."""
    row = pd.DataFrame([dict(game_features)])[feature_columns]
    return float(model.predict_proba(row)[0, 1])


def has_value_edge(model_prob: float, market_no_vig_prob: float, threshold: float = 0.03) -> bool:
    """Flag a candidate bet when the model's probability beats the no-vig market by `threshold`."""
    return (model_prob - market_no_vig_prob) >= threshold
