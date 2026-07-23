import pandas as pd
import pytest

from src.baselines import (
    compare_to_baselines,
    predict_always_home,
    predict_coin_flip,
    predict_market_favorite,
)


def make_games_df():
    return pd.DataFrame(
        {
            "home_flag": [1, 0, 1, 0],
            "market_odds_team_a": [-150, 130, -110, -200],
            "market_odds_team_b": [130, -150, -110, 170],
            "team_a_win": [1, 0, 1, 1],
        }
    )


class TestPredictMarketFavorite:
    def test_returns_no_vig_probability(self):
        df = make_games_df()
        probs = predict_market_favorite(df)

        assert len(probs) == 4
        # Row 0: -150/+130 market, team_a is the favorite -> no-vig prob > 0.5
        assert probs[0] > 0.5
        # Row 1: +130/-150 market, team_a is the underdog -> no-vig prob < 0.5
        assert probs[1] < 0.5

    def test_no_vig_pair_sums_to_one(self):
        # -110/-110 has vig; no-vig probs for both sides should sum to 1
        df = pd.DataFrame({"market_odds_team_a": [-110], "market_odds_team_b": [-110]})
        prob_a = predict_market_favorite(df)[0]
        assert prob_a == pytest.approx(0.5)


class TestPredictAlwaysHome:
    def test_uses_home_win_rate_when_home_else_complement(self):
        df = make_games_df()
        probs = predict_always_home(df, home_win_rate=0.58)

        assert probs[0] == pytest.approx(0.58)  # home_flag=1
        assert probs[1] == pytest.approx(0.42)  # home_flag=0
        assert probs[2] == pytest.approx(0.58)
        assert probs[3] == pytest.approx(0.42)


class TestPredictCoinFlip:
    def test_all_predictions_are_half(self):
        df = make_games_df()
        probs = predict_coin_flip(df)
        assert (probs == 0.5).all()


class TestCompareToBaselines:
    def test_returns_row_per_predictor_with_expected_columns(self):
        df = make_games_df()
        model_probs = [0.65, 0.30, 0.55, 0.60]

        result = compare_to_baselines(df, model_probs, home_win_rate=0.55)

        assert set(result["predictor"]) == {
            "Model (logistic regression)",
            "Always bet market favorite (no-vig price)",
            "Always bet home team",
            "Coin flip",
        }
        assert {"accuracy", "auc", "log_loss", "brier_score"}.issubset(result.columns)
        assert (result["accuracy"] >= 0).all() and (result["accuracy"] <= 1).all()
        assert (result["auc"] >= 0).all() and (result["auc"] <= 1).all()

    def test_coin_flip_brier_score_is_a_quarter(self):
        df = make_games_df()
        result = compare_to_baselines(df, model_probs=[0.5, 0.5, 0.5, 0.5], home_win_rate=0.5)
        coin_flip_row = result[result["predictor"] == "Coin flip"].iloc[0]
        assert coin_flip_row["brier_score"] == pytest.approx(0.25)

    def test_coin_flip_auc_is_one_half(self):
        # A constant prediction has no ranking power -> AUC of exactly 0.5
        df = make_games_df()
        result = compare_to_baselines(df, model_probs=[0.5, 0.5, 0.5, 0.5], home_win_rate=0.5)
        coin_flip_row = result[result["predictor"] == "Coin flip"].iloc[0]
        assert coin_flip_row["auc"] == pytest.approx(0.5)

    def test_perfect_ranking_gives_auc_of_one(self):
        df = make_games_df()
        # Model probs perfectly rank-order with team_a_win = [1, 0, 1, 1]
        result = compare_to_baselines(df, model_probs=[0.9, 0.1, 0.8, 0.7], home_win_rate=0.5)
        model_row = result[result["predictor"] == "Model (logistic regression)"].iloc[0]
        assert model_row["auc"] == pytest.approx(1.0)
