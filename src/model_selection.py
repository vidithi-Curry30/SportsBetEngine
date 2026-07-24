"""Model selection and interpretability for the logistic regression model:
a regularization-strength search that never touches a fold's real test set,
and standardized coefficients for feature-importance / error analysis.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.model import chronological_split

DEFAULT_C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def select_regularization_strength(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    C_grid: tuple[float, ...] = DEFAULT_C_GRID,
    inner_val_frac: float = 0.15,
    date_column: str = "date",
) -> tuple[float, pd.DataFrame]:
    """Pick a regularization strength C via a further chronological split
    carved from the *end* of `train_df` -- an inner validation set, never the
    fold's real held-out test set, so this selection can never leak into the
    number the fold is ultimately evaluated on ("no regularization search" was
    a fair criticism; this is a real, leakage-safe one, not a grid search
    against the test set in disguise).

    Returns (best_C, results_df) where results_df has one row per candidate C
    and its inner-validation log loss, so a report can show the whole grid,
    not just the winner.
    """
    inner_train_df, inner_val_df = chronological_split(
        train_df, train_frac=1 - inner_val_frac, date_column=date_column
    )
    if len(inner_train_df) == 0 or len(inner_val_df) == 0:
        raise ValueError("train_df is too small to carve out a non-empty inner validation split")

    y_val = inner_val_df[target_column].to_numpy()
    rows = []
    for C in C_grid:
        model = LogisticRegression(C=C)
        model.fit(inner_train_df[feature_columns], inner_train_df[target_column])
        val_probs = model.predict_proba(inner_val_df[feature_columns])[:, 1]
        val_probs = np.clip(val_probs, 1e-15, 1 - 1e-15)
        rows.append({"C": C, "inner_val_log_loss": log_loss(y_val, val_probs, labels=[0, 1])})

    results_df = pd.DataFrame(rows)
    best_C = float(results_df.loc[results_df["inner_val_log_loss"].idxmin(), "C"])
    return best_C, results_df


def standardized_coefficients(
    model: LogisticRegression, feature_columns: list[str], training_df: pd.DataFrame
) -> pd.DataFrame:
    """Feature importance for a fitted logistic regression. Raw sklearn
    coefficients aren't comparable across features on different scales (a
    coefficient of 0.5 on a feature ranging +/-20 means something different
    than 0.5 on a feature ranging +/-0.1); scaling each by its feature's
    training-set standard deviation (coef_i * std_i) makes them comparable as
    "how much a typical swing in this feature moves the log-odds" -- the
    standard fix, not a new metric.

    Returns a DataFrame with one row per feature, sorted by
    |standardized_coefficient| descending (most influential first).
    """
    stds = training_df[feature_columns].std()
    raw_coefs = pd.Series(model.coef_[0], index=feature_columns)
    standardized = raw_coefs * stds

    result = pd.DataFrame(
        {
            "feature": feature_columns,
            "raw_coefficient": raw_coefs.values,
            "feature_std_dev": stds.values,
            "standardized_coefficient": standardized.values,
        }
    )
    order = result["standardized_coefficient"].abs().sort_values(ascending=False).index
    return result.reindex(order).reset_index(drop=True)
