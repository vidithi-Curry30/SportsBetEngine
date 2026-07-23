import numpy as np
import pandas as pd
import pytest

from src.backtest import american_to_decimal, apply_slippage, run_backtest


class DummyModel:
    """Fixed-probability stub so backtest behavior is deterministic and testable
    without depending on a fitted sklearn model."""

    def __init__(self, prob: float):
        self.prob = prob

    def predict_proba(self, X):
        n = len(X)
        return np.array([[1 - self.prob, self.prob] for _ in range(n)])


class TestAmericanToDecimal:
    def test_positive_odds(self):
        assert american_to_decimal(150) == pytest.approx(2.5)

    def test_negative_odds(self):
        assert american_to_decimal(-150) == pytest.approx(1.6667, abs=1e-3)

    def test_even_money(self):
        assert american_to_decimal(100) == pytest.approx(2.0)
        assert american_to_decimal(-100) == pytest.approx(2.0)


class TestApplySlippage:
    def test_worsens_price_for_bettor(self):
        # Slippage nudges implied probability up, which is a worse price for the bettor
        original = american_to_decimal(120)
        slipped = american_to_decimal(apply_slippage(120, slippage_pct=0.02))
        assert slipped < original

    def test_zero_slippage_is_a_no_op(self):
        assert apply_slippage(-110, slippage_pct=0.0) == -110


def make_test_df(n=20, market_odds=-110, closing_odds=-120, win=1):
    dates = pd.date_range("2025-03-10", periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "pace_diff": [0.0] * n,
            "off_rating_diff": [0.0] * n,
            "def_rating_diff": [0.0] * n,
            "recent_win_pct_diff": [0.0] * n,
            "rest_days_diff": [0.0] * n,
            "home_flag": [0] * n,
            "team_a_win": [win] * n,
            "market_odds_team_a": [market_odds] * n,
            "market_odds_team_b": [market_odds] * n,
            "closing_odds_team_a": [closing_odds] * n,
        }
    )


class TestRunBacktest:
    def test_no_bets_placed_when_model_has_no_edge(self):
        # Model probability matches the market exactly -> no edge -> no bets
        df = make_test_df(market_odds=-110, closing_odds=-110)
        model = DummyModel(prob=100 / 210)  # matches -110's implied prob

        result = run_backtest(df, model, edge_threshold=0.03)

        assert result["num_bets"] == 0
        assert result["final_bankroll"] == result["starting_bankroll"]
        assert result["roi"] == pytest.approx(0.0)

    def test_bets_placed_and_bankroll_grows_on_all_wins(self):
        df = make_test_df(n=10, market_odds=100, closing_odds=100, win=1)
        model = DummyModel(prob=0.70)  # well above the ~0.5 no-vig market prob

        result = run_backtest(df, model, edge_threshold=0.03, starting_bankroll=10_000)

        assert result["num_bets"] == 10
        assert result["final_bankroll"] > result["starting_bankroll"]
        assert result["hit_rate"] == pytest.approx(1.0)
        assert result["roi"] > 0

    def test_bankroll_shrinks_on_all_losses(self):
        df = make_test_df(n=10, market_odds=100, closing_odds=100, win=0)
        model = DummyModel(prob=0.70)

        result = run_backtest(df, model, edge_threshold=0.03, starting_bankroll=10_000)

        assert result["final_bankroll"] < result["starting_bankroll"]
        assert result["hit_rate"] == pytest.approx(0.0)
        assert result["roi"] < 0

    def test_max_bet_pct_caps_stake_regardless_of_kelly_output(self):
        # A huge apparent edge would call for a huge Kelly stake -- the cap must bind
        df = make_test_df(n=1, market_odds=100, closing_odds=100, win=1)
        model = DummyModel(prob=0.95)

        result = run_backtest(df, model, edge_threshold=0.03, max_bet_pct=0.02, starting_bankroll=10_000)

        bet = result["bet_log"][0]
        assert bet["stake_fraction"] == pytest.approx(0.02)
        assert bet["stake"] == pytest.approx(200.0)

    def test_bankroll_series_length_matches_games_plus_one(self):
        df = make_test_df(n=15, market_odds=-110, closing_odds=-110)
        model = DummyModel(prob=100 / 210)

        result = run_backtest(df, model)

        assert len(result["bankroll_series"]) == 16
        assert result["num_games"] == 15

    def test_avg_clv_computed_for_placed_bets(self):
        # Bet at 100 (implied 0.5), closes at -120 (implied ~0.5455) -> positive CLV
        df = make_test_df(n=5, market_odds=100, closing_odds=-120, win=1)
        model = DummyModel(prob=0.70)

        result = run_backtest(df, model, edge_threshold=0.03, slippage_pct=0.0)

        assert result["avg_clv"] > 0

    def test_top_bet_pnl_share_flags_tail_dominated_profit(self):
        # One huge long-odds win plus several small losses: nearly all profit
        # comes from a single bet, which top_bet_pnl_share should surface.
        df = pd.concat(
            [
                make_test_df(n=1, market_odds=1000, closing_odds=1000, win=1),
                make_test_df(n=5, market_odds=100, closing_odds=100, win=0),
            ],
            ignore_index=True,
        )
        model = DummyModel(prob=0.70)

        result = run_backtest(df, model, edge_threshold=0.03, max_bet_pct=0.05)

        assert result["top_bet_pnl_share"] is not None
        assert result["top_bet_pnl_share"] > 1.0  # the one win outweighs the net losses

    def test_top_bet_pnl_share_is_none_when_no_bets_placed(self):
        df = make_test_df(market_odds=-110, closing_odds=-110)
        model = DummyModel(prob=100 / 210)

        result = run_backtest(df, model, edge_threshold=0.03)

        assert result["top_bet_pnl_share"] is None


class TestSlateExposureCap:
    def make_same_day_slate(self, n_games=4):
        # n games all on the same date -- a real slate, not one game per day
        return pd.DataFrame(
            {
                "date": ["2025-03-10"] * n_games,
                "pace_diff": [0.0] * n_games,
                "off_rating_diff": [0.0] * n_games,
                "def_rating_diff": [0.0] * n_games,
                "recent_win_pct_diff": [0.0] * n_games,
                "rest_days_diff": [0.0] * n_games,
                "home_flag": [0] * n_games,
                "team_a_win": [1] * n_games,
                "market_odds_team_a": [100] * n_games,
                "market_odds_team_b": [100] * n_games,
                "closing_odds_team_a": [100] * n_games,
            }
        )

    def test_slate_cap_binds_when_many_bets_flagged_same_day(self):
        # 4 games same day, each would want max_bet_pct=0.10 individually -> 40%
        # naive sum, well over a 15% slate cap.
        df = self.make_same_day_slate(n_games=4)
        model = DummyModel(prob=0.95)  # huge edge -> Kelly wants the full per-bet cap

        result = run_backtest(
            df, model, edge_threshold=0.03, max_bet_pct=0.10, max_slate_pct=0.15, starting_bankroll=10_000
        )

        assert result["num_bets"] == 4
        total_stake_fraction = sum(b["stake_fraction"] for b in result["bet_log"])
        assert total_stake_fraction == pytest.approx(0.15)
        for b in result["bet_log"]:
            assert b["stake_fraction"] == pytest.approx(0.15 / 4)

    def test_slate_cap_does_not_bind_when_under_limit(self):
        df = self.make_same_day_slate(n_games=2)
        model = DummyModel(prob=0.95)

        result = run_backtest(
            df, model, edge_threshold=0.03, max_bet_pct=0.05, max_slate_pct=0.50, starting_bankroll=10_000
        )

        for b in result["bet_log"]:
            assert b["stake_fraction"] == pytest.approx(0.05)

    def test_same_day_bets_settle_against_the_same_starting_bankroll(self):
        # All bets on the same slate should be sized off the day's *starting*
        # bankroll, not sequentially compounded against each other.
        df = self.make_same_day_slate(n_games=3)
        model = DummyModel(prob=0.95)

        result = run_backtest(
            df, model, edge_threshold=0.03, max_bet_pct=0.05, max_slate_pct=0.50, starting_bankroll=10_000
        )

        for b in result["bet_log"]:
            assert b["stake"] == pytest.approx(10_000 * 0.05)
