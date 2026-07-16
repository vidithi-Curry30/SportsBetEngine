"""CLI: pull current odds from The Odds API and save to data/raw/.

Usage:
    python scripts/fetch_odds.py --sport baseball_mlb
    python scripts/fetch_odds.py --sport basketball_nba   # once NBA is back in season
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.odds_client import OddsAPIClient, save_raw_pull


def main():
    parser = argparse.ArgumentParser(description="Fetch current odds from The Odds API")
    parser.add_argument("--sport", default=config.DEFAULT_SPORT, help="Sport key, e.g. basketball_nba")
    parser.add_argument("--regions", default=config.DEFAULT_REGIONS)
    parser.add_argument("--markets", default=config.DEFAULT_MARKETS)
    args = parser.parse_args()

    client = OddsAPIClient()
    data = client.get_odds(sport=args.sport, regions=args.regions, markets=args.markets)
    out_path = save_raw_pull(data, sport=args.sport)
    print(f"Saved {len(data)} games to {out_path}")


if __name__ == "__main__":
    main()
