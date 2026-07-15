"""CLI: train the model, run the backtest on the held-out period, and write
results/backtest_report.md + results/clv_plot.png.

Usage:
    python scripts/run_backtest.py --data data/processed/nba_games_synthetic.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from src.backtest import run_backtest
from src.model import chronological_split, train_model


def main():
    parser = argparse.ArgumentParser(description="Run the full backtest and generate the report")
    parser.add_argument("--data", default=str(config.DATA_PROCESSED_DIR / "nba_games_synthetic.csv"))
    parser.add_argument("--train-frac", type=float, default=0.8)
    args = parser.parse_args()

    games_df = pd.read_csv(args.data)
    train_df, test_df = chronological_split(games_df, train_frac=args.train_frac)
    model = train_model(train_df)
    result = run_backtest(test_df, model)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_report(result, train_df, test_df, config.RESULTS_DIR / "backtest_report.md")
    _write_clv_plot(result, config.RESULTS_DIR / "clv_plot.png")

    print(f"ROI: {result['roi']:.2%}  Avg CLV: {result['avg_clv']:+.2f}pp  Hit rate: {result['hit_rate']:.1%}")
    print(f"Report written to {config.RESULTS_DIR / 'backtest_report.md'}")


def _write_report(result: dict, train_df: pd.DataFrame, test_df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Backtest Report",
        "",
        f"Training period: {train_df['date'].min()} to {train_df['date'].max()} ({len(train_df)} games)",
        f"Held-out test period: {test_df['date'].min()} to {test_df['date'].max()} ({len(test_df)} games)",
        "",
        "## Results",
        "",
        f"- **Average CLV (the metric that matters more than raw ROI over a small "
        f"sample): {result['avg_clv']:+.2f} percentage points**",
        f"- ROI: {result['roi']:.2%}",
        f"- Sharpe-like ratio: {result['sharpe_like_ratio']:.3f}",
        f"- Max drawdown: {result['max_drawdown']:.2%}",
        f"- Hit rate: {result['hit_rate']:.1%}",
        f"- Bets placed: {result['num_bets']} / {result['num_games']} games scanned",
        f"- Final bankroll: ${result['final_bankroll']:,.2f} "
        f"(started at ${result['starting_bankroll']:,.2f})",
        "",
        "![Bankroll and CLV](clv_plot.png)",
        "",
    ]
    out_path.write_text("\n".join(lines))


def _write_clv_plot(result: dict, out_path: Path) -> None:
    bet_log = pd.DataFrame(result["bet_log"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(result["bankroll_series"], color="#2a6f97")
    axes[0].axhline(result["starting_bankroll"], color="gray", linestyle="--", linewidth=1)
    axes[0].set_title("Bankroll over the held-out test period")
    axes[0].set_xlabel("Game index")
    axes[0].set_ylabel("Bankroll ($)")

    if not bet_log.empty:
        colors = ["#2a9d8f" if c >= 0 else "#e76f51" for c in bet_log["clv"]]
        axes[1].bar(range(len(bet_log)), bet_log["clv"], color=colors)
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_title(f"CLV per bet (avg {result['avg_clv']:+.2f}pp)")
        axes[1].set_xlabel("Bet index")
        axes[1].set_ylabel("CLV (percentage points)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
