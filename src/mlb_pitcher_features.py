"""Point-in-time starting-pitcher features, additive to src/mlb_features.py's
team-level features.

Why this exists: team-level rolling stats (runs scored/allowed) can't see
which pitcher is starting, but single-game MLB outcomes are driven heavily
by exactly that. Two starter-quality signals are used, deliberately not
just one:

- Rolling ERA: the classic outcome-based measure (earned runs allowed per 9
  innings), computed only from starts strictly before the game being
  predicted.
- Rolling K/9 (strikeouts per 9 innings): a "stuff"/skill measure. Per DIPS
  theory (defense-independent pitching statistics), strikeout rate is far
  less noisy over a small sample of starts than ERA, which is heavily
  influenced by batted-ball luck and defense behind the pitcher. Using both
  an outcome-based and a skill-based metric is deliberate, not redundant.

A pitcher with zero prior recorded starts this season (a rookie debut, or a
starter who hasn't started yet) has no point-in-time signal -- that game is
dropped, the same discipline src/mlb_features.py applies to a team's first
game of the season.
"""
import time
from collections import defaultdict

import pandas as pd
import requests

from src.mlb_stats_client import fetch_pitcher_game_log

PITCHER_FEATURE_COLUMNS = ["starting_pitcher_era_diff", "starting_pitcher_k9_diff"]


def fetch_all_starter_game_logs(
    games: list[dict], season: str, max_retries: int = 3, retry_backoff_seconds: float = 1.0
) -> dict[int, list[dict]]:
    """Fetch each unique probable starter's season game log once (not per-game).

    ~300-400 sequential requests over a real network occasionally hit a
    transient connection error -- retried a few times with backoff; a
    pitcher that still fails is skipped (logged, not fatal) rather than
    crashing the whole batch. Games involving a skipped pitcher simply drop
    out of the later inner join, same as a missing probable-pitcher game.
    """
    pitcher_ids = set()
    for game in games:
        if game.get("home_pitcher_id") is not None:
            pitcher_ids.add(game["home_pitcher_id"])
        if game.get("away_pitcher_id") is not None:
            pitcher_ids.add(game["away_pitcher_id"])

    logs = {}
    skipped = []
    for pitcher_id in pitcher_ids:
        for attempt in range(max_retries):
            try:
                logs[pitcher_id] = fetch_pitcher_game_log(pitcher_id, season)
                break
            except requests.exceptions.RequestException:
                if attempt + 1 == max_retries:
                    skipped.append(pitcher_id)
                else:
                    time.sleep(retry_backoff_seconds * (attempt + 1))

    if skipped:
        print(f"Warning: failed to fetch game logs for {len(skipped)} pitcher(s) after retries: {skipped}")
    return logs


def build_pitching_features(games: list[dict], pitcher_logs: dict[int, list[dict]]) -> pd.DataFrame:
    """Point-in-time rolling ERA/K9 diffs (home minus away) for each game's
    probable starters. Returns one row per game with a usable signal for both
    sides; games missing a probable pitcher or with a debuting starter are
    dropped. Columns: date, game_pk, starting_pitcher_era_diff,
    starting_pitcher_k9_diff.
    """
    rows = []
    for game in sorted(games, key=lambda g: g["date"]):
        home_id, away_id = game.get("home_pitcher_id"), game.get("away_pitcher_id")
        if home_id is None or away_id is None:
            continue

        home_stats = _rolling_stats_before(pitcher_logs.get(home_id, []), game["date"])
        away_stats = _rolling_stats_before(pitcher_logs.get(away_id, []), game["date"])
        if home_stats is None or away_stats is None:
            continue

        rows.append(
            {
                "date": game["date"],
                "game_pk": game["game_pk"],
                # Lower ERA is better, so away-minus-home so positive favors the home starter.
                "starting_pitcher_era_diff": away_stats["era"] - home_stats["era"],
                "starting_pitcher_k9_diff": home_stats["k9"] - away_stats["k9"],
            }
        )

    return pd.DataFrame(rows)


def _rolling_stats_before(starts: list[dict], as_of_date: str) -> dict | None:
    prior_starts = [s for s in starts if s["date"] < as_of_date]
    if not prior_starts:
        return None

    outs = sum(s["outs"] for s in prior_starts)
    if outs == 0:
        return None

    earned_runs = sum(s["earned_runs"] for s in prior_starts)
    strikeouts = sum(s["strikeouts"] for s in prior_starts)

    return {
        "era": 27 * earned_runs / outs,
        "k9": 27 * strikeouts / outs,
    }
