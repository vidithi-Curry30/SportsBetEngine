"""Integration test for scripts/train_mlb_model_walk_forward.py: exercises the
whole script (multi-season fetch -> per-season features -> walk-forward folds
-> regularization search -> feature importance -> error analysis -> report)
with the network call mocked, the same gap-closing this repo already closed
once for collect_paper_trades.py."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.train_mlb_model_walk_forward as walk_forward_script


def make_season_games(season_year, n_games=40):
    teams = [f"Team{season_year}A", f"Team{season_year}B", f"Team{season_year}C", f"Team{season_year}D"]
    games = []
    for i in range(n_games):
        date = f"{season_year}-{3 + i // 25:02d}-{1 + i % 25:02d}"
        home, away = teams[i % 4], teams[(i + 1) % 4]
        if i % 2 == 0:
            games.append(_game(date, home, away, 5, 3))
        else:
            games.append(_game(date, home, away, 2, 6))
    return games


def _game(date, home, away, hs, aws):
    return {
        "date": date, "game_pk": hash((date, home, away)) % 1_000_000, "home_team": home,
        "away_team": away, "home_score": hs, "away_score": aws, "home_win": int(hs > aws),
        "home_pitcher_id": None, "home_pitcher_name": None, "away_pitcher_id": None, "away_pitcher_name": None,
    }


class TestWalkForwardScript:
    def test_runs_end_to_end_and_writes_report_and_plots(self, tmp_path, monkeypatch, capsys):
        seasons_data = {
            "2023-01-01_2023-12-31": make_season_games("2023"),
            "2024-01-01_2024-12-31": make_season_games("2024"),
            "2025-01-01_2025-12-31": make_season_games("2025"),
        }

        def fake_fetch(start, end):
            return seasons_data.get(f"{start}_{end}", [])

        monkeypatch.setattr(
            walk_forward_script, "SEASON_RANGES",
            [("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
             ("2025", "2025-01-01", "2025-12-31")],
        )
        monkeypatch.setattr(walk_forward_script.config, "RESULTS_DIR", tmp_path)

        with patch("scripts.train_mlb_model_walk_forward.fetch_completed_games", side_effect=fake_fetch):
            sys.argv = ["train_mlb_model_walk_forward.py", "--n-splits", "2"]
            walk_forward_script.main()

        report_path = tmp_path / "mlb_walk_forward_report.md"
        assert report_path.exists()
        report_text = report_path.read_text()
        assert "Walk-Forward MLB Model Evaluation" in report_text
        assert "Feature importance" in report_text
        assert "Error analysis: by season" in report_text

        assert (tmp_path / "mlb_feature_importance.png").exists()
        assert (tmp_path / "mlb_walk_forward_stability.png").exists()

        out = capsys.readouterr().out
        assert "Real games by season" in out
