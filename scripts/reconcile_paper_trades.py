"""CLI: settle open paper trades against real results and write
results/paper_trading_report.md -- the real (not synthetic), forward-collected
CLV track record that scripts/collect_paper_trades.py has been logging.

Run this periodically (e.g. once a day) after games have had a chance to
finish. For each still-open ledger row whose game has a final result from
statsapi.mlb.com, this fills in the outcome and computes realized CLV/P&L.

Closing-line caveat, stated plainly: "closing odds" here means the last price
seen for that game in a fresh Odds API pull at the time this script runs, not
the exact final tick before first pitch -- getting the literal closing tick
would require a snapshot timed to run right at game time for every game,
which this single periodic script doesn't attempt. If a game has already
started or finished by the time this runs, no fresh price is available for
it and CLV falls back to 0 (closing == placed) for that row rather than a
fabricated number -- an honest gap, not a hidden one.

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
from src.odds_client import OddsAPIClient
from src.paper_trading import consensus_no_vig_prob, load_ledger, reconcile_paper_trades
from src.stats import bootstrap_ci

DEFAULT_LEDGER_PATH = config.BASE_DIR / "data" / "paper_trades" / "mlb_paper_trades.csv"


def main():
    parser = argparse.ArgumentParser(description="Settle open paper trades and report realized CLV")
    parser.add_argument("--sport", default="baseball_mlb")
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

        closing_odds_by_game = {}
        try:
            live_odds = OddsAPIClient().get_odds(sport=args.sport)
            for game in live_odds:
                consensus = consensus_no_vig_prob(game)
                if consensus is None:
                    continue
                game_date = game.get("commence_time", "")[:10]
                key = (game_date, game.get("home_team"), game.get("away_team"))
                # Approximate closing odds from whichever book's h2h price is quoted
                # first -- good enough for a "closer to close than placed_odds" proxy;
                # see the module docstring for the honest limitation here.
                for bookmaker in game.get("bookmakers", []):
                    h2h = next((m for m in bookmaker.get("markets", []) if m.get("key") == "h2h"), None)
                    if h2h is None:
                        continue
                    price = next(
                        (o["price"] for o in h2h["outcomes"] if o.get("name") == game.get("home_team")), None
                    )
                    if price is not None:
                        closing_odds_by_game[key] = price
                        break
        except Exception as exc:  # noqa: BLE001 -- reconciling results shouldn't fail just because odds are unavailable
            print(f"Warning: couldn't fetch a fresh odds pull for closing prices ({exc}); "
                  f"falling back to placed_odds (CLV=0) for newly-settled rows.")

        newly_settled_before = (ledger["status"] == "settled").sum()
        ledger = reconcile_paper_trades(ledger, closing_odds_by_game, results_by_game)
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
