"""Point-in-time feature engineering for real MLB games.

Every feature for a given game is computed strictly from that team's games
*before* the current one -- a team's rolling stats are only updated after
its features have been read for the game being processed. Getting this
wrong (e.g. using season-end stats to "predict" a game from March) is the
single most common way a backtest silently cheats.
"""
from collections import defaultdict

import pandas as pd

RECENT_WINDOW = 10
MLB_FEATURE_COLUMNS = [
    "runs_scored_diff",
    "runs_allowed_diff",
    "season_run_diff_diff",
    "recent_win_pct_diff",
    "rest_days_diff",
]
MLB_TARGET_COLUMN = "home_win"


def build_features(games: list[dict]) -> pd.DataFrame:
    """games: dicts with date, home_team, away_team, home_score, away_score, home_win
    (see mlb_stats_client.fetch_completed_games). Returns one row per game with
    point-in-time features plus home_win, in chronological order. A team's first
    game of the season is dropped (no prior history to compute features from).

    There's no separate home-field feature: unlike the synthetic NBA dataset,
    which randomized which side was "home" so it could vary as a feature, real
    games always have a fixed home team -- home advantage is absorbed into the
    model's intercept instead.
    """
    games_sorted = sorted(games, key=lambda g: g["date"])

    runs_scored = defaultdict(list)
    runs_allowed = defaultdict(list)
    results = defaultdict(list)
    last_game_date: dict[str, str] = {}

    rows = []
    for game in games_sorted:
        home, away, date = game["home_team"], game["away_team"], game["date"]

        home_state = _team_state(home, runs_scored, runs_allowed, results, last_game_date, date)
        away_state = _team_state(away, runs_scored, runs_allowed, results, last_game_date, date)

        if home_state is not None and away_state is not None:
            rows.append(
                {
                    "date": date,
                    "game_pk": game.get("game_pk"),
                    "home_team": home,
                    "away_team": away,
                    "home_win": game["home_win"],
                    "runs_scored_diff": home_state["avg_runs_scored"] - away_state["avg_runs_scored"],
                    "runs_allowed_diff": away_state["avg_runs_allowed"] - home_state["avg_runs_allowed"],
                    "season_run_diff_diff": home_state["run_diff_per_game"] - away_state["run_diff_per_game"],
                    "recent_win_pct_diff": home_state["recent_win_pct"] - away_state["recent_win_pct"],
                    "rest_days_diff": home_state["rest_days"] - away_state["rest_days"],
                }
            )

        _update_state(home, game["home_score"], game["away_score"], game["home_win"], date,
                      runs_scored, runs_allowed, results, last_game_date)
        _update_state(away, game["away_score"], game["home_score"], 1 - game["home_win"], date,
                      runs_scored, runs_allowed, results, last_game_date)

    return pd.DataFrame(rows)


def _team_state(team, runs_scored, runs_allowed, results, last_game_date, current_date):
    if team not in last_game_date:
        return None

    scored, allowed, res = runs_scored[team], runs_allowed[team], results[team]
    n = len(scored)

    return {
        "avg_runs_scored": sum(scored) / n,
        "avg_runs_allowed": sum(allowed) / n,
        "run_diff_per_game": (sum(scored) - sum(allowed)) / n,
        "recent_win_pct": sum(res[-RECENT_WINDOW:]) / len(res[-RECENT_WINDOW:]),
        "rest_days": (pd.Timestamp(current_date) - pd.Timestamp(last_game_date[team])).days,
    }


def _update_state(team, scored, allowed, won, date, runs_scored, runs_allowed, results, last_game_date):
    runs_scored[team].append(scored)
    runs_allowed[team].append(allowed)
    results[team].append(won)
    last_game_date[team] = date
