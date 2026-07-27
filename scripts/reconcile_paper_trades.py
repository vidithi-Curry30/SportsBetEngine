"""CLI: settle open paper trades against real results and write
results/paper_trading_report.md -- the real (not synthetic), forward-collected
CLV track record that scripts/collect_paper_trades.py has been logging.

Run this periodically (e.g. once a day) after games have had a chance to
finish. For each still-open ledger row whose game has a final result from
statsapi.mlb.com, this fills in the outcome and computes realized CLV/P&L
using that row's `latest_odds` as the closing-price proxy.

Closing-line caveat, stated plainly: "closing odds" means the last price
collect_paper_trades.py observed for that game before it started, not the
exact final tick before first pitch -- getting the literal closing tick would
need a snapshot timed to run right at game time for every game, which this
periodic collection doesn't attempt. This script used to try to grab a
closing price itself, with its own fresh odds pull at settle time -- that
reliably failed, because by settle time a game has already started or
finished and The Odds API no longer lists it (confirmed directly: the first
six real settled trades all showed exactly 0.00pp CLV, `latest_odds` never
having been refreshed after the initial log). Capturing the price is now
collect_paper_trades.py's job, done while the game is still on the board;
this script only reads what was already captured.

Usage:
    python scripts/reconcile_paper_trades.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config
from src.mlb_stats_client import fetch_completed_games
from src.paper_trading import load_ledger, reconcile_paper_trades
from src.stats import bootstrap_ci

DEFAULT_LEDGER_PATH = config.BASE_DIR / "data" / "paper_trades" / "mlb_paper_trades.csv"


def main():
    parser = argparse.ArgumentParser(description="Settle open paper trades and report realized CLV")
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    args = parser.parse_args()

    ledger_path = Path(args.ledger)
    ledger = load_ledger(ledger_path)
    if ledger.empty:
        print(f"No paper trades logged yet at {ledger_path}. Run scripts/collect_paper_trades.py first.")
        return

    open_rows = ledger[ledger["status"] == "open"]
    if open_rows.empty:
        print("No open paper trades to reconcile.")
    else:
        start_date = str(open_rows["game_date"].min())
        end_date = str(open_rows["game_date"].max())

        completed_games = fetch_completed_games(start_date, end_date)
        results_by_game = {
            (g["date"], g["home_team"], g["away_team"]): g["home_win"] for g in completed_games
        }

        newly_settled_before = (ledger["status"] == "settled").sum()
        ledger = reconcile_paper_trades(ledger, results_by_game)
        newly_settled = (ledger["status"] == "settled").sum() - newly_settled_before
        print(f"Settled {newly_settled} paper trade(s).")

    ledger.to_csv(ledger_path, index=False)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_report(ledger, config.RESULTS_DIR / "paper_trading_report.md")
    print(f"Report written to {config.RESULTS_DIR / 'paper_trading_report.md'}")


def _write_report(ledger, out_path: Path) -> None:
    settled = ledger[ledger["status"] == "settled"].copy()
    n_open = int((ledger["status"] == "open").sum())

    lines = [
        "# Paper Trading Report (Real, Forward-Collected CLV)",
        "",
        "Unlike results/backtest_report.md, every row here is a real bet logged at a "
        "real, live market price *before* the game was played -- not a synthetic "
        "backtest. Leakage is structurally impossible: the paper trade is written to "
        "the ledger by scripts/collect_paper_trades.py before the outcome exists. "
        "The tradeoff is sample size: this only grows by one slate at a time.",
        "",
        f"Total paper trades logged: {len(ledger)} ({len(settled)} settled, {n_open} still open)",
        "",
    ]

    if settled.empty:
        lines.append(
            "No settled paper trades yet -- run scripts/collect_paper_trades.py for a "
            "while, then scripts/reconcile_paper_trades.py once games finish."
        )
        out_path.write_text("\n".join(lines) + "\n")
        return

    clv_values = settled["clv"].astype(float).to_numpy()
    pnl_values = settled["pnl"].astype(float).to_numpy()
    hit_rate = float((settled["result"].astype(float) == 1).mean())

    clv_ci = bootstrap_ci(clv_values, statistic=np.mean, ci=0.90)
    pnl_ci = bootstrap_ci(pnl_values, statistic=np.mean, ci=0.90)

    lines += [
        "## Results",
        "",
        f"- Avg CLV: {clv_ci[0]:+.2f}pp, 90% CI [{clv_ci[1]:+.2f}, {clv_ci[2]:+.2f}]pp",
        f"- Avg per-bet return (in stake-fraction units): {pnl_ci[0]:+.4f}, "
        f"90% CI [{pnl_ci[1]:+.4f}, {pnl_ci[2]:+.4f}]",
        f"- Hit rate: {hit_rate:.1%}",
        f"- Settled bets: {len(settled)}",
        "",
        "(Wide or zero-crossing CIs on a small settled count are the honest result, "
        "not a bug -- see src/stats.py.)",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
