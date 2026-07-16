"""Generates data/raw/basketball_nba_synthetic_sample.json -- the synthetic
multi-book odds batch used to exercise the arbitrage scanner (notebooks/
exploration.ipynb) before a real Odds API key was available. Committed here
so that dataset is independently reproducible from source rather than a
JSON file to take on faith.

Usage:
    python scripts/generate_synthetic_arbitrage_sample.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

SEED = 7
GAMES = [
    ("Boston Celtics", "New York Knicks", 0.62),
    ("Golden State Warriors", "Los Angeles Lakers", 0.48),
    ("Denver Nuggets", "Oklahoma City Thunder", 0.55),
    ("Milwaukee Bucks", "Philadelphia 76ers", 0.58),
    ("Miami Heat", "Dallas Mavericks", 0.44),
    ("Phoenix Suns", "Minnesota Timberwolves", 0.51),
]
BOOKS = ["DraftKings", "FanDuel", "BetMGM", "Caesars"]


def prob_to_american(p: float) -> int:
    if p >= 0.5:
        return round(-100 * p / (1 - p))
    return round(100 * (1 - p) / p)


def vig_up(p: float, vig: float) -> float:
    return min(0.98, p * (1 + vig))


def main():
    random.seed(SEED)
    games = []

    for i, (home, away, true_home_prob) in enumerate(GAMES):
        bookmakers = []
        for book in BOOKS:
            # Each book independently prices off a slightly different power rating
            # (small noise) and applies its own realistic vig (4-6%, typical for
            # NBA moneylines) to both sides.
            noise = random.uniform(-0.01, 0.01)
            home_prob = min(max(true_home_prob + noise, 0.05), 0.95)
            away_prob = 1 - home_prob
            vig = random.uniform(0.04, 0.06)
            home_prob_vigged = vig_up(home_prob, vig)
            away_prob_vigged = vig_up(away_prob, vig)

            bookmakers.append(
                {
                    "key": book.lower().replace(" ", ""),
                    "title": book,
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": home, "price": prob_to_american(home_prob_vigged)},
                                {"name": away, "price": prob_to_american(away_prob_vigged)},
                            ],
                        }
                    ],
                }
            )

        games.append(
            {
                "id": f"synthetic_game_{i + 1}",
                "sport_key": "basketball_nba",
                "commence_time": f"2026-01-{15 + i:02d}T00:30:00Z",
                "home_team": home,
                "away_team": away,
                "bookmakers": bookmakers,
            }
        )

    out_path = config.DATA_RAW_DIR / "basketball_nba_synthetic_sample.json"
    out_path.write_text(json.dumps(games, indent=2))
    print(f"Wrote {len(games)} synthetic games to {out_path}")


if __name__ == "__main__":
    main()
