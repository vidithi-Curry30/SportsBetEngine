"""CLI: pull today's live MLB odds, compute the model's edge against the
consensus no-vig market price (line-shopped across books), print a live edge
report, and log any flagged bet to the forward paper-trading ledger.

This is the forward-collection half of the fix for the backtest's biggest
honest limitation: The Odds API's historical endpoint is paywalled, so there's
no way to buy real historical CLV data. But nothing stops collecting it going
forward -- run this once a day (or a few times, closer to first pitch) during
the season, and scripts/reconcile_paper_trades.py once games finish, and
results/paper_trading_report.md accumulates real, out-of-sample CLV over time,
with leakage structurally impossible (the bet is logged before the outcome
exists).

Trains a team-only model fresh from all available completed games each run --
deliberately not the pitcher-feature model, to keep this fast enough to run
daily without a ~300-request pitcher-log fetch every time; see
src/mlb_pitcher_features.py for how that would extend to a live lookup if
that tradeoff stops being worth it.

Usage:
    python scripts/collect_paper_trades.py
    python scripts/collect_paper_trades.py --sport baseball_mlb --edge-threshold 0.03
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config
from src.mlb_features import MLB_FEATURE_COLUMNS, MLB_TARGET_COLUMN, build_features, compute_current_features
from src.mlb_stats_client import fetch_completed_games
from src.model import train_model
from src.odds_client import OddsAPIClient
from src.paper_trading import append_new_paper_trades, build_paper_trade_rows, compute_live_edges, load_ledger
from src.utils import utc_now_iso

DEFAULT_LEDGER_PATH = config.BASE_DIR / "data" / "paper_trades" / "mlb_paper_trades.csv"


def main():
    parser = argparse.ArgumentParser(description="Collect a live-odds snapshot and log flagged paper trades")
    parser.add_argument("--sport", default="baseball_mlb")
    parser.add_argument("--edge-threshold", type=float, default=0.03)
    parser.add_argument("--history-start-date", default="2026-03-01", help="Start of the history used to fit the model")
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    args = parser.parse_args()

    snapshot_time = utc_now_iso()

    print("Fetching completed games to fit the model and compute today's rolling features...")
    completed_games = fetch_completed_games(args.history_start_date, snapshot_time[:10])
    if not completed_games:
        print("No completed games available yet -- nothing to fit a model on. Exiting.")
        return

    features_df = build_features(completed_games)
    model = train_model(features_df, feature_columns=MLB_FEATURE_COLUMNS, target_column=MLB_TARGET_COLUMN)

    def feature_lookup(home_team, away_team, game_date):
        return compute_current_features(completed_games, home_team, away_team, game_date)

    print(f"Pulling live odds for {args.sport}...")
    odds_games = OddsAPIClient().get_odds(sport=args.sport)
    print(f"{len(odds_games)} games returned")

    edges = compute_live_edges(odds_games, feature_lookup, model, MLB_FEATURE_COLUMNS, edge_threshold=args.edge_threshold)
    if edges.empty:
        print("No games had both a usable model feature set and a consensus market price.")
        return

    _print_edge_report(edges)

    new_rows = build_paper_trade_rows(edges, snapshot_time=snapshot_time)
    ledger_path = Path(args.ledger)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    ledger = load_ledger(ledger_path)
    before = len(ledger)
    ledger = append_new_paper_trades(new_rows, ledger)
    ledger.to_csv(ledger_path, index=False)

    print(f"\nLogged {len(ledger) - before} new paper trade(s) to {ledger_path} ({len(ledger)} total)")


def _print_edge_report(edges) -> None:
    print("\nLive edge report (model vs. line-shopped best price):")
    print(f"{'Matchup':<45} {'Model':>7} {'Market':>7} {'Edge':>7}  {'Best price':>12}  Flag")
    for _, row in edges.iterrows():
        matchup = f"{row['away_team']} @ {row['home_team']}"
        if pd.notna(row["best_home_odds"]):
            price = f"{row['best_book']} {int(row['best_home_odds']):+d}"
        else:
            price = "n/a"
        flag = "VALUE" if row["has_value_edge"] else ""
        print(
            f"{matchup:<45} {row['model_prob']:>6.1%} {row['no_vig_market_prob']:>6.1%} "
            f"{row['edge']:>+6.1%}  {price:>12}  {flag}"
        )


if __name__ == "__main__":
    main()
