import pandas as pd
import pytest

from src.model import (
    FEATURE_COLUMNS,
    chronological_split,
    has_value_edge,
    predict_win_probability,
    train_model,
)


def make_games_df(n=100):
    dates = pd.date_range("2024-10-22", periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "pace_diff": [i % 7 - 3 for i in range(n)],
            "off_rating_diff": [(i % 5 - 2) * 2.0 for i in range(n)],
            "def_rating_diff": [(i % 3 - 1) * 1.5 for i in range(n)],
            "recent_win_pct_diff": [((i % 10) - 5) / 10 for i in range(n)],
            "rest_days_diff": [i % 4 - 2 for i in range(n)],
            "home_flag": [i % 2 for i in range(n)],
            "team_a_win": [1 if (i % 5 - 2) * 2.0 + i % 2 > 0 else 0 for i in range(n)],
        }
    )


class TestChronologicalSplit:
    def test_no_leakage_between_train_and_test(self):
        df = make_games_df(100)
        train_df, test_df = chronological_split(df, train_frac=0.8)

        assert train_df["date"].max() <= test_df["date"].min()

    def test_split_sizes(self):
        df = make_games_df(100)
        train_df, test_df = chronological_split(df, train_frac=0.8)

        assert len(train_df) == 80
        assert len(test_df) == 20

    def test_shuffled_input_is_still_sorted_chronologically(self):
        df = make_games_df(50).sample(frac=1, random_state=1).reset_index(drop=True)
        train_df, _ = chronological_split(df, train_frac=0.8)

        assert list(train_df["date"]) == sorted(train_df["date"])


class TestTrainModel:
    def test_returns_fitted_model_with_expected_coefficients(self):
        df = make_games_df(200)
        train_df, _ = chronological_split(df)
        model = train_model(train_df)

        assert model.coef_.shape == (1, len(FEATURE_COLUMNS))

    def test_predict_win_probability_in_valid_range(self):
        df = make_games_df(200)
        train_df, test_df = chronological_split(df)
        model = train_model(train_df)

        prob = predict_win_probability(model, test_df.iloc[0])
        assert 0.0 <= prob <= 1.0


class TestHasValueEdge:
    def test_flags_when_model_beats_market_by_threshold(self):
        assert has_value_edge(model_prob=0.58, market_no_vig_prob=0.52, threshold=0.03) is True

    def test_does_not_flag_below_threshold(self):
        assert has_value_edge(model_prob=0.53, market_no_vig_prob=0.52, threshold=0.03) is False

    def test_does_not_flag_when_market_favors_model(self):
        assert has_value_edge(model_prob=0.45, market_no_vig_prob=0.52, threshold=0.03) is False
