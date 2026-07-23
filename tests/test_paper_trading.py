import numpy as np
import pandas as pd
import pytest

from src.paper_trading import (
    append_new_paper_trades,
    build_paper_trade_rows,
    compute_live_edges,
    consensus_no_vig_prob,
    find_best_price,
    load_ledger,
    reconcile_paper_trades,
)


class DummyModel:
    def __init__(self, prob: float):
        self.prob = prob

    def predict_proba(self, X):
        return np.array([[1 - self.prob, self.prob] for _ in range(len(X))])


def make_book(title, home_team, away_team, home_price, away_price):
    return {
        "title": title,
        "markets": [
            {
                "key": "h2h",
                "outcomes": [
                    {"name": home_team, "price": home_price},
                    {"name": away_team, "price": away_price},
                ],
            }
        ],
    }


def make_game(home_team="Boston Red Sox", away_team="New York Yankees", commence_time="2026-07-23T23:00:00Z", books=None):
    if books is None:
        books = [
            make_book("BookA", home_team, away_team, -120, 105),
            make_book("BookB", home_team, away_team, -130, 115),
        ]
    return {
        "id": "g1",
        "home_team": home_team,
        "away_team": away_team,
        "commence_time": commence_time,
        "bookmakers": books,
    }


class TestFindBestPrice:
    def test_best_home_price_is_least_negative(self):
        # -120 implies a lower probability (better for the bettor) than -130
        game = make_game()
        book, price = find_best_price(game, "home")
        assert book == "BookA"
        assert price == -120

    def test_best_away_price_is_highest_positive(self):
        game = make_game()
        book, price = find_best_price(game, "away")
        assert book == "BookB"
        assert price == 115

    def test_none_when_no_books(self):
        game = make_game(books=[])
        assert find_best_price(game, "home") is None


class TestConsensusNoVigProb:
    def test_averages_across_books(self):
        game = make_game()
        home_prob, away_prob = consensus_no_vig_prob(game)
        assert home_prob + away_prob == pytest.approx(1.0)
        assert 0 < home_prob < 1

    def test_none_when_no_matching_books(self):
        game = make_game(books=[])
        assert consensus_no_vig_prob(game) is None


class TestComputeLiveEdges:
    def feature_lookup(self, home_team, away_team, game_date):
        return {"runs_scored_diff": 0.0}

    def test_flags_a_value_edge(self):
        game = make_game()
        model = DummyModel(prob=0.70)  # market's no-vig home prob is well below 0.70

        edges = compute_live_edges([game], self.feature_lookup, model, ["runs_scored_diff"], edge_threshold=0.03)

        assert len(edges) == 1
        assert edges.iloc[0]["has_value_edge"]
        assert edges.iloc[0]["best_book"] == "BookA"
        assert edges.iloc[0]["best_home_odds"] == -120

    def test_skips_game_with_no_features(self):
        game = make_game()
        model = DummyModel(prob=0.70)

        edges = compute_live_edges([game], lambda h, a, d: None, model, ["runs_scored_diff"])
        assert edges.empty

    def test_no_edge_when_model_matches_market(self):
        game = make_game(books=[make_book("BookA", "Boston Red Sox", "New York Yankees", -110, -110)])
        home_prob = 110 / 210
        model = DummyModel(prob=home_prob)

        edges = compute_live_edges([game], self.feature_lookup, model, ["runs_scored_diff"], edge_threshold=0.03)
        assert not edges.iloc[0]["has_value_edge"]


class TestBuildPaperTradeRows:
    def test_builds_rows_only_for_flagged_edges(self):
        edges_df = pd.DataFrame(
            [
                {
                    "game_date": "2026-07-23", "home_team": "A", "away_team": "B",
                    "model_prob": 0.65, "no_vig_market_prob": 0.55, "edge": 0.10,
                    "best_book": "BookA", "best_home_odds": -110, "has_value_edge": True,
                },
                {
                    "game_date": "2026-07-23", "home_team": "C", "away_team": "D",
                    "model_prob": 0.50, "no_vig_market_prob": 0.50, "edge": 0.0,
                    "best_book": "BookA", "best_home_odds": -110, "has_value_edge": False,
                },
            ]
        )
        rows = build_paper_trade_rows(edges_df, snapshot_time="2026-07-23T12:00:00Z")

        assert len(rows) == 1
        assert rows[0]["home_team"] == "A"
        assert rows[0]["status"] == "open"
        assert 0 < rows[0]["stake_fraction"] <= 0.05

    def make_flagged_edge(self, game_date, home, away, model_prob=0.90, odds=-110):
        return {
            "game_date": game_date, "home_team": home, "away_team": away,
            "model_prob": model_prob, "no_vig_market_prob": 0.50, "edge": model_prob - 0.50,
            "best_book": "BookA", "best_home_odds": odds, "has_value_edge": True,
        }

    def test_slate_cap_binds_across_same_day_flagged_edges(self):
        # 4 games same date, each independently capped at max_bet_pct=0.10 ->
        # 40% naive sum, well over a 15% slate cap -- same scenario backtest.py's
        # slate cap was built for, now exercised on the live paper-trading path.
        edges_df = pd.DataFrame(
            [self.make_flagged_edge("2026-07-23", f"Home{i}", f"Away{i}") for i in range(4)]
        )
        rows = build_paper_trade_rows(
            edges_df, snapshot_time="t0", max_bet_pct=0.10, max_slate_pct=0.15
        )

        assert len(rows) == 4
        assert sum(r["stake_fraction"] for r in rows) == pytest.approx(0.15)

    def test_slate_cap_does_not_cross_dates(self):
        # Two flagged games on different dates must each be capped independently,
        # not pooled together into one combined slate.
        edges_df = pd.DataFrame(
            [
                self.make_flagged_edge("2026-07-23", "A", "B"),
                self.make_flagged_edge("2026-07-24", "C", "D"),
            ]
        )
        rows = build_paper_trade_rows(edges_df, snapshot_time="t0", max_bet_pct=0.10, max_slate_pct=0.15)

        assert len(rows) == 2
        assert all(r["stake_fraction"] == pytest.approx(0.10) for r in rows)

    def test_unknown_sizing_strategy_raises(self):
        edges_df = pd.DataFrame([self.make_flagged_edge("2026-07-23", "A", "B")])
        with pytest.raises(ValueError):
            build_paper_trade_rows(edges_df, snapshot_time="t0", sizing_strategy="bogus")

    def test_correlation_aware_discounts_a_shared_team_doubleheader(self):
        # Team A plays twice the same day (a doubleheader); team C's game
        # shares no team with either and has identical model_prob/odds.
        edges_df = pd.DataFrame(
            [
                self.make_flagged_edge("2026-07-23", "A", "X"),
                self.make_flagged_edge("2026-07-23", "A", "Y"),
                self.make_flagged_edge("2026-07-23", "C", "D"),
            ]
        )
        rows = build_paper_trade_rows(
            edges_df, snapshot_time="t0", sizing_strategy="correlation_aware",
            max_bet_pct=1.0, max_slate_pct=1.0,
        )

        team_a_stakes = [r["stake_fraction"] for r in rows if r["home_team"] == "A"]
        team_c_stake = next(r["stake_fraction"] for r in rows if r["home_team"] == "C")
        assert len(team_a_stakes) == 2
        assert all(s < team_c_stake for s in team_a_stakes)
        assert "decimal_odds" not in rows[0]
        assert "teams" not in rows[0]


class TestLedgerRoundTrip:
    def test_load_missing_ledger_returns_empty_frame_with_columns(self, tmp_path):
        ledger = load_ledger(tmp_path / "does_not_exist.csv")
        assert ledger.empty
        assert "game_date" in ledger.columns

    def test_append_dedupes_by_game_key(self):
        ledger = load_ledger("/nonexistent/path.csv")
        rows = [{"game_date": "2026-07-23", "home_team": "A", "away_team": "B", "stake_fraction": 0.05}]

        ledger = append_new_paper_trades(rows, ledger)
        ledger = append_new_paper_trades(rows, ledger)  # same game logged again on a later run

        assert len(ledger) == 1

    def test_reconcile_settles_matching_open_rows(self):
        ledger = pd.DataFrame(
            [
                {
                    "game_date": "2026-07-23", "home_team": "A", "away_team": "B",
                    "snapshot_time": "t0", "model_prob": 0.6, "no_vig_market_prob": 0.5,
                    "edge": 0.1, "book": "BookA", "placed_odds": -110, "stake_fraction": 0.05,
                    "status": "open", "closing_odds": None, "result": None, "clv": None, "pnl": None,
                }
            ]
        )
        closing_odds = {("2026-07-23", "A", "B"): -130}
        results = {("2026-07-23", "A", "B"): 1}  # home team (A) won

        settled = reconcile_paper_trades(ledger, closing_odds, results)

        assert settled.iloc[0]["status"] == "settled"
        assert settled.iloc[0]["clv"] > 0  # bet at -110, closed at -130 -> line moved in your favor
        assert settled.iloc[0]["pnl"] > 0  # home team won

    def test_reconcile_leaves_unresolved_games_open(self):
        ledger = pd.DataFrame(
            [
                {
                    "game_date": "2026-07-23", "home_team": "A", "away_team": "B",
                    "snapshot_time": "t0", "model_prob": 0.6, "no_vig_market_prob": 0.5,
                    "edge": 0.1, "book": "BookA", "placed_odds": -110, "stake_fraction": 0.05,
                    "status": "open", "closing_odds": None, "result": None, "clv": None, "pnl": None,
                }
            ]
        )
        settled = reconcile_paper_trades(ledger, {}, {})
        assert settled.iloc[0]["status"] == "open"
