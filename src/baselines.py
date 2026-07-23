"""Naive baseline predictors: what any real model has to beat.

The strongest baseline here isn't a coin flip -- it's "just trust the market's
own no-vig price." If a model can't beat that, it isn't finding a real edge;
it's overfitting noise. Comparing against it is a direct, small-scale test of
weak-form market efficiency.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from src.probability import american_to_probability, remove_vig


def predict_market_favorite(games_df: pd.DataFrame) -> np.ndarray:
    """The no-vig market implied probability itself, as the prediction."""
    probs = []
    for _, game in games_df.iterrows():
        prob_a = american_to_probability(game["market_odds_team_a"])
        prob_b = american_to_probability(game["market_odds_team_b"])
        no_vig_prob_a, _ = remove_vig(prob_a, prob_b)
        probs.append(no_vig_prob_a)
    return np.array(probs)


def predict_always_home(games_df: pd.DataFrame, home_win_rate: float) -> np.ndarray:
    """Constant prediction: the home team's empirical win rate (from training data),
    applied when team_a is home, and its complement when team_a is away."""
    return np.where(games_df["home_flag"] == 1, home_win_rate, 1 - home_win_rate)


def predict_coin_flip(games_df: pd.DataFrame) -> np.ndarray:
    return np.full(len(games_df), 0.5)


def compare_to_baselines(
    test_df: pd.DataFrame, model_probs: np.ndarray, home_win_rate: float
) -> pd.DataFrame:
    """Score the model against naive baselines on the same held-out period."""
    y_true = test_df["team_a_win"].to_numpy()

    predictors = {
        "Model (logistic regression)": np.asarray(model_probs, dtype=float),
        "Always bet market favorite (no-vig price)": predict_market_favorite(test_df),
        "Always bet home team": predict_always_home(test_df, home_win_rate),
        "Coin flip": predict_coin_flip(test_df),
    }

    rows = []
    for name, probs in predictors.items():
        preds = (probs >= 0.5).astype(int)
        rows.append(
            {
                "predictor": name,
                "accuracy": accuracy_score(y_true, preds),
                "auc": roc_auc_score(y_true, probs),
                "log_loss": log_loss(y_true, probs, labels=[0, 1]),
                "brier_score": brier_score_loss(y_true, probs),
            }
        )

    return pd.DataFrame(rows)
