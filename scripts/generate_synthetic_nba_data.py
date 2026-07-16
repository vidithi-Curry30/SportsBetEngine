"""Generates data/processed/nba_games_synthetic.csv -- the synthetic NBA
season used by scripts/run_backtest.py, since no live nba_api/stats.nba.com
access and no historical Odds API access were available while building this
(see README). Committed here so the synthetic dataset is independently
reproducible from source, not just a CSV to take on faith.

Design: each team has a fixed hidden "true strength". Observed features
(pace, ratings, recent win %, rest days) are noisy proxies for that strength,
not the strength itself -- so a model trained on the observed features will
recover real signal but won't be perfectly accurate. Opening lines carry more
noise than closing lines, which is what gives a skilled model room to beat
the close.

Usage:
    python scripts/generate_synthetic_nba_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config

SEED = 42
TEAMS = [
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
    "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
    "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
    "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
    "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
    "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
    "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
    "Utah Jazz", "Washington Wizards",
]
HOME_COURT_LOGIT = 0.25  # roughly matches real NBA home-court advantage
N_GAMES = 1200
SEASON_START = pd.Timestamp("2024-10-22")
SEASON_END = pd.Timestamp("2025-04-13")


def prob_to_american(p: float) -> int:
    p = min(max(p, 0.01), 0.99)
    if p >= 0.5:
        return round(-100 * p / (1 - p))
    return round(100 * (1 - p) / p)


def vig_up(p: float, vig: float) -> float:
    return min(0.98, p * (1 + vig))


def main():
    rng = np.random.default_rng(SEED)
    true_strength = {team: rng.normal(0, 1) for team in TEAMS}

    game_dates = sorted(
        SEASON_START
        + pd.to_timedelta(rng.integers(0, (SEASON_END - SEASON_START).days, size=N_GAMES), unit="D")
    )

    rows = []
    for game_date in game_dates:
        team_a, team_b = rng.choice(TEAMS, size=2, replace=False)
        team_a_is_home = rng.random() < 0.5
        home_flag = 1 if team_a_is_home else 0

        strength_a, strength_b = true_strength[team_a], true_strength[team_b]

        # Noisy observed features: proxies for true strength, not the strength itself.
        pace_a = rng.normal(100, 4) + rng.normal(0, 1)
        pace_b = rng.normal(100, 4) + rng.normal(0, 1)
        off_rating_a = 112 + strength_a * 3 + rng.normal(0, 2.5)
        off_rating_b = 112 + strength_b * 3 + rng.normal(0, 2.5)
        def_rating_a = 112 - strength_a * 3 + rng.normal(0, 2.5)  # lower = better defense
        def_rating_b = 112 - strength_b * 3 + rng.normal(0, 2.5)
        recent_win_pct_a = np.clip(0.5 + strength_a * 0.12 + rng.normal(0, 0.12), 0.0, 1.0)
        recent_win_pct_b = np.clip(0.5 + strength_b * 0.12 + rng.normal(0, 0.12), 0.0, 1.0)
        rest_days_a = int(rng.integers(0, 5))
        rest_days_b = int(rng.integers(0, 5))

        # True win probability driven by hidden strength + home court, not the noisy features.
        true_logit = 0.8 * (strength_a - strength_b) + HOME_COURT_LOGIT * (1 if team_a_is_home else -1)
        true_prob_a = 1 / (1 + np.exp(-true_logit))
        team_a_win = int(rng.random() < true_prob_a)

        # Opening line: the market's own (imperfect) estimate of true_prob_a, plus vig.
        market_prob_a = np.clip(true_prob_a + rng.normal(0, 0.05), 0.03, 0.97)
        open_vig = rng.uniform(0.04, 0.06)
        market_odds_team_a = prob_to_american(vig_up(market_prob_a, open_vig))
        market_odds_team_b = prob_to_american(vig_up(1 - market_prob_a, open_vig))

        # Closing line: sharper (smaller noise) -- the market has absorbed information
        # by tip-off, which is why beating the closing line is a meaningful skill signal.
        closing_prob_a = np.clip(true_prob_a + rng.normal(0, 0.02), 0.03, 0.97)
        close_vig = rng.uniform(0.04, 0.06)
        closing_odds_team_a = prob_to_american(vig_up(closing_prob_a, close_vig))
        closing_odds_team_b = prob_to_american(vig_up(1 - closing_prob_a, close_vig))

        rows.append(
            {
                "date": game_date.strftime("%Y-%m-%d"),
                "team_a": team_a,
                "team_b": team_b,
                "home_flag": home_flag,
                "pace_diff": pace_a - pace_b,
                "off_rating_diff": off_rating_a - off_rating_b,
                "def_rating_diff": def_rating_b - def_rating_a,  # positive favors team_a
                "recent_win_pct_diff": recent_win_pct_a - recent_win_pct_b,
                "rest_days_diff": rest_days_a - rest_days_b,
                "team_a_win": team_a_win,
                "market_odds_team_a": market_odds_team_a,
                "market_odds_team_b": market_odds_team_b,
                "closing_odds_team_a": closing_odds_team_a,
                "closing_odds_team_b": closing_odds_team_b,
            }
        )

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out_path = config.DATA_PROCESSED_DIR / "nba_games_synthetic.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} synthetic games to {out_path}")
    print(f"Home team win rate: {df.loc[df['home_flag'] == 1, 'team_a_win'].mean():.3f} (team_a at home)")
    print(f"Overall team_a win rate: {df['team_a_win'].mean():.3f}")


if __name__ == "__main__":
    main()
