"""Integration test for scripts/collect_paper_trades.py: exercises the whole
wiring (fetch completed games -> train model -> pull live odds -> compute
edges -> log to ledger) with the two network calls mocked, rather than
relying only on the unit-level src/paper_trading.py tests -- this is the
gap a one-off manual smoke test left uncovered (see the module the script
was built alongside for why that matters)."""
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.collect_paper_trades as collect_paper_trades


def make_completed_game(date, home, away, home_score, away_score):
    return {
        "date": date, "game_pk": hash((date, home, away)) % 100_000, "home_team": home,
        "away_team": away, "home_score": home_score, "away_score": away_score,
        "home_win": int(home_score > away_score),
        "home_pitcher_id": None, "home_pitcher_name": None,
        "away_pitcher_id": None, "away_pitcher_name": None,
    }


def make_history(n_days=20):
    # Alternating results so the model has both classes to fit on (an all-wins
    # or all-losses history makes sklearn's LogisticRegression.fit raise).
    teams = ["Boston Red Sox", "New York Yankees", "Tampa Bay Rays", "Baltimore Orioles"]
    games = []
    for i in range(n_days):
        date = f"2026-04-{i + 1:02d}"
        home, away = teams[i % 4], teams[(i + 1) % 4]
        if i % 2 == 0:
            games.append(make_completed_game(date, home, away, 5, 3))
        else:
            games.append(make_completed_game(date, home, away, 2, 6))
    return games


def make_book(title, home_team, away_team, home_price, away_price):
    return {
        "title": title,
        "markets": [{"key": "h2h", "outcomes": [
            {"name": home_team, "price": home_price}, {"name": away_team, "price": away_price},
        ]}],
    }


def make_odds_game(home_team, away_team, home_price=-110, away_price=-110):
    return {
        "id": "g1", "home_team": home_team, "away_team": away_team,
        "commence_time": "2026-04-21T23:00:00Z",
        "bookmakers": [make_book("BookA", home_team, away_team, home_price, away_price)],
    }


class TestCollectPaperTradesScript:
    def test_logs_a_flagged_edge_to_the_ledger(self, tmp_path, capsys):
        history = make_history()
        # A heavily-favorable price for the home team relative to a coin-flip
        # market gives every reasonable model a clearable edge on this matchup.
        odds_game = make_odds_game("Boston Red Sox", "New York Yankees", home_price=105, away_price=105)
        ledger_path = tmp_path / "ledger.csv"

        with patch("scripts.collect_paper_trades.fetch_completed_games", return_value=history), \
             patch("scripts.collect_paper_trades.OddsAPIClient") as MockClient:
            MockClient.return_value.get_odds.return_value = [odds_game]
            sys.argv = ["collect_paper_trades.py", "--ledger", str(ledger_path)]
            collect_paper_trades.main()

        assert ledger_path.exists()
        ledger = pd.read_csv(ledger_path)
        assert len(ledger) == 1
        assert ledger.iloc[0]["home_team"] == "Boston Red Sox"
        assert ledger.iloc[0]["away_team"] == "New York Yankees"
        assert ledger.iloc[0]["status"] == "open"

        out = capsys.readouterr().out
        assert "VALUE" in out
        assert "Logged 1 new paper trade" in out

    def test_writes_header_only_ledger_when_no_edge_clears_threshold(self, tmp_path):
        history = make_history()
        odds_game = make_odds_game("Boston Red Sox", "New York Yankees", home_price=105, away_price=105)
        ledger_path = tmp_path / "ledger.csv"

        # An (almost) impossibly high threshold, rather than hand-tuning the
        # synthetic history to land the model exactly at the market's price --
        # deterministically exercises the "edges computed, none flagged" path.
        with patch("scripts.collect_paper_trades.fetch_completed_games", return_value=history), \
             patch("scripts.collect_paper_trades.OddsAPIClient") as MockClient:
            MockClient.return_value.get_odds.return_value = [odds_game]
            sys.argv = ["collect_paper_trades.py", "--ledger", str(ledger_path), "--edge-threshold", "0.99"]
            collect_paper_trades.main()

        assert ledger_path.exists()
        ledger = pd.read_csv(ledger_path)
        assert len(ledger) == 0

    def test_exits_early_with_no_completed_games(self, tmp_path, capsys):
        ledger_path = tmp_path / "ledger.csv"

        with patch("scripts.collect_paper_trades.fetch_completed_games", return_value=[]):
            sys.argv = ["collect_paper_trades.py", "--ledger", str(ledger_path)]
            collect_paper_trades.main()

        assert not ledger_path.exists()
        assert "No completed games available" in capsys.readouterr().out

    def test_dedupes_across_repeated_runs(self, tmp_path):
        history = make_history()
        odds_game = make_odds_game("Boston Red Sox", "New York Yankees", home_price=105, away_price=105)
        ledger_path = tmp_path / "ledger.csv"

        with patch("scripts.collect_paper_trades.fetch_completed_games", return_value=history), \
             patch("scripts.collect_paper_trades.OddsAPIClient") as MockClient:
            MockClient.return_value.get_odds.return_value = [odds_game]
            sys.argv = ["collect_paper_trades.py", "--ledger", str(ledger_path)]
            collect_paper_trades.main()  # first run: logs the edge
            collect_paper_trades.main()  # second run, same game: must not double-log

        ledger = pd.read_csv(ledger_path)
        assert len(ledger) == 1
