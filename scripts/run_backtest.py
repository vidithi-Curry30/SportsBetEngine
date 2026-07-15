"""CLI: train the model, run the backtest on the held-out period, and write
results/backtest_report.md, results/clv_plot.png, and results/calibration_plot.png.

Also runs three supporting analyses that don't change the primary backtest
number but stress-test it: a naive-baseline comparison (is the model actually
beating "just trust the market"?), bootstrap confidence intervals on ROI and
CLV (is 94 bets even enough to tell edge from noise?), and a before/after
calibration comparison -- both isotonic and Platt (sigmoid) scaling, since
they make different data-size tradeoffs and the report should show which one
actually held up on a ~150-game calibration set rather than assume either one
"fixed" anything.

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
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

import config
from src.backtest import run_backtest
from src.baselines import compare_to_baselines
from src.calibration import IsotonicCalibrator, PlattCalibrator, reliability_curve
from src.model import (
    FEATURE_COLUMNS,
    chronological_split,
    chronological_split_train_calib_test,
    predict_win_probability,
    train_model,
)
from src.stats import bootstrap_ci


def main():
    parser = argparse.ArgumentParser(description="Run the full backtest and generate the report")
    parser.add_argument("--data", default=str(config.DATA_PROCESSED_DIR / "nba_games_synthetic.csv"))
    parser.add_argument("--train-frac", type=float, default=0.8)
    args = parser.parse_args()

    games_df = pd.read_csv(args.data)

    # Primary backtest: identical to the original two-way split.
    train_df, test_df = chronological_split(games_df, train_frac=args.train_frac)
    model = train_model(train_df)
    result = run_backtest(test_df, model)

    home_win_rate = float(train_df.loc[train_df["home_flag"] == 1, "team_a_win"].mean())
    baseline_table = _run_baseline_comparison(model, test_df, home_win_rate)
    roi_ci, clv_ci = _run_bootstrap_ci(result)
    calibration = _run_calibration_analysis(games_df, args.train_frac)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_report(
        result, train_df, test_df, baseline_table, roi_ci, clv_ci, calibration,
        config.RESULTS_DIR / "backtest_report.md",
    )
    _write_clv_plot(result, config.RESULTS_DIR / "clv_plot.png")
    _write_calibration_plot(calibration, config.RESULTS_DIR / "calibration_plot.png")

    print(f"ROI: {result['roi']:.2%}  Avg CLV: {result['avg_clv']:+.2f}pp  Hit rate: {result['hit_rate']:.1%}")
    print(f"Bootstrap 90% CI -- avg CLV: [{clv_ci[1]:+.2f}, {clv_ci[2]:+.2f}]pp")
    print(f"Report written to {config.RESULTS_DIR / 'backtest_report.md'}")


def _run_baseline_comparison(model, test_df: pd.DataFrame, home_win_rate: float) -> pd.DataFrame:
    model_probs = np.array([predict_win_probability(model, row) for _, row in test_df.iterrows()])
    return compare_to_baselines(test_df, model_probs, home_win_rate)


def _run_bootstrap_ci(result: dict):
    bet_log = result["bet_log"]
    per_bet_returns = [b["pnl"] / (b["bankroll_after"] - b["pnl"]) for b in bet_log]
    clv_values = [b["clv"] for b in bet_log]

    roi_ci = bootstrap_ci(per_bet_returns, statistic=np.mean, ci=0.90)
    clv_ci = bootstrap_ci(clv_values, statistic=np.mean, ci=0.90)
    return roi_ci, clv_ci


def _run_calibration_analysis(games_df: pd.DataFrame, train_frac: float) -> dict:
    fit_train_df, calib_df, test_df = chronological_split_train_calib_test(games_df, train_frac=train_frac)
    fit_model = train_model(fit_train_df)

    y_true = test_df["team_a_win"].to_numpy()
    raw_test_probs = fit_model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]

    methods = {
        "isotonic": IsotonicCalibrator(fit_model, FEATURE_COLUMNS).fit(calib_df),
        "platt": PlattCalibrator(fit_model, FEATURE_COLUMNS).fit(calib_df),
    }

    results = {
        "calib_set_size": len(calib_df),
        "raw_brier": brier_score_loss(y_true, raw_test_probs),
        "raw_curve": reliability_curve(y_true, raw_test_probs),
        "raw_backtest": run_backtest(test_df, fit_model),
        "methods": {},
    }
    for name, calibrator in methods.items():
        calibrated_probs = calibrator.predict_proba(test_df)
        results["methods"][name] = {
            "brier": brier_score_loss(y_true, calibrated_probs),
            "curve": reliability_curve(y_true, calibrated_probs),
            "backtest": run_backtest(
                test_df, probability_fn=lambda game, c=calibrator: c.predict_win_probability(game)
            ),
        }
    return results


def _write_report(
    result: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    baseline_table: pd.DataFrame,
    roi_ci: tuple,
    clv_ci: tuple,
    calibration: dict,
    out_path: Path,
) -> None:
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
        f"- Top-bet P&L share: {_format_pnl_share(result['top_bet_pnl_share'])} "
        f"(share of total profit from the single best bet -- high values mean the "
        f"result rests on one outlier, not a repeatable edge)",
        "",
        "![Bankroll and CLV](clv_plot.png)",
        "",
        "## Is 94 bets enough to tell edge from noise? (bootstrap 90% CI)",
        "",
        f"- Avg CLV: {clv_ci[0]:+.2f}pp, 90% CI [{clv_ci[1]:+.2f}, {clv_ci[2]:+.2f}]pp",
        f"- Mean per-bet return: {roi_ci[0]:+.2%}, 90% CI [{roi_ci[1]:+.2%}, {roi_ci[2]:+.2%}]",
        "",
        "## Model vs. naive baselines (same held-out period)",
        "",
        _baseline_table_markdown(baseline_table),
        "",
        "## Calibration: does the model's raw probability mean what it says?",
        "",
        f"Calibration set: {calibration['calib_set_size']} games (carved out of the "
        "training period, never the test period). Isotonic regression is a flexible "
        "nonparametric fit that sklearn's own docs say needs ~1000+ calibration samples "
        "to avoid overfitting the mapping itself; Platt (sigmoid) scaling only fits 2 "
        "parameters and is far less data-hungry. Both are shown rather than assuming "
        "either one is the fix.",
        "",
        f"- Brier score, raw: {calibration['raw_brier']:.4f}",
        f"- Brier score, isotonic-calibrated: {calibration['methods']['isotonic']['brier']:.4f}",
        f"- Brier score, Platt-calibrated: {calibration['methods']['platt']['brier']:.4f}",
        "",
        "Backtest re-run on the identical test period for each:",
        "",
        _calibration_comparison_table(calibration),
        "",
        "![Reliability diagram](calibration_plot.png)",
        "",
    ]
    out_path.write_text("\n".join(lines))


def _format_pnl_share(share) -> str:
    return f"{share:.0%}" if share is not None else "n/a (no net profit)"


def _baseline_table_markdown(baseline_table: pd.DataFrame) -> str:
    lines = ["| Predictor | Accuracy | Log loss | Brier score |", "|---|---|---|---|"]
    for _, row in baseline_table.iterrows():
        lines.append(
            f"| {row['predictor']} | {row['accuracy']:.3f} | "
            f"{row['log_loss']:.4f} | {row['brier_score']:.4f} |"
        )
    return "\n".join(lines)


def _calibration_comparison_table(calibration: dict) -> str:
    raw = calibration["raw_backtest"]
    iso = calibration["methods"]["isotonic"]["backtest"]
    platt = calibration["methods"]["platt"]["backtest"]

    def fmt(bt):
        return (
            f"{bt['roi']:.2%}",
            f"{bt['avg_clv']:+.2f}",
            f"{bt['hit_rate']:.1%}",
            str(bt["num_bets"]),
            f"{bt['max_drawdown']:.2%}",
            _format_pnl_share(bt["top_bet_pnl_share"]),
        )

    metric_names = ["ROI", "Avg CLV (pp)", "Hit rate", "Bets placed", "Max drawdown", "Top-bet P&L share"]
    raw_vals, iso_vals, platt_vals = fmt(raw), fmt(iso), fmt(platt)

    lines = ["| Metric | Raw | Isotonic | Platt |", "|---|---|---|---|"]
    for name, r, i, p in zip(metric_names, raw_vals, iso_vals, platt_vals):
        lines.append(f"| {name} | {r} | {i} | {p} |")
    return "\n".join(lines)


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


def _write_calibration_plot(calibration: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Perfect calibration")

    raw_curve = calibration["raw_curve"]
    ax.plot(
        raw_curve["mean_predicted_prob"], raw_curve["observed_frequency"],
        marker="o", color="#e76f51", label=f"Raw (Brier {calibration['raw_brier']:.3f})",
    )

    iso_curve = calibration["methods"]["isotonic"]["curve"]
    ax.plot(
        iso_curve["mean_predicted_prob"], iso_curve["observed_frequency"],
        marker="o", color="#2a9d8f",
        label=f"Isotonic (Brier {calibration['methods']['isotonic']['brier']:.3f})",
    )

    platt_curve = calibration["methods"]["platt"]["curve"]
    ax.plot(
        platt_curve["mean_predicted_prob"], platt_curve["observed_frequency"],
        marker="o", color="#264653",
        label=f"Platt (Brier {calibration['methods']['platt']['brier']:.3f})",
    )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed win frequency")
    ax.set_title("Reliability diagram (held-out test period)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
