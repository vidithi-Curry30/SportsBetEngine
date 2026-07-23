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
