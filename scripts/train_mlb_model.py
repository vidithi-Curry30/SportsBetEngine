"""CLI: fetch real MLB game data, train the win-probability model on real
outcomes, and write results/mlb_model_report.md + results/mlb_reliability_plot.png.

Unlike scripts/run_backtest.py (which trains on the synthetic NBA dataset),
this trains and evaluates on real games from statsapi.mlb.com -- real dates,
real teams, real final scores, real point-in-time features. There is no
backtest/CLV section here: The Odds API's free tier doesn't include historical
odds, so there's no real historical market price to size bets against or
compute closing-line value from. This script answers a narrower but fully
real question: is the model's predicted probability, evaluated out-of-sample,
actually better than naive baselines and reasonably calibrated?

Usage:
    python scripts/train_mlb_model.py
    python scripts/train_mlb_model.py --start-date 2026-03-01 --end-date 2026-07-16
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
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

import config
from src.calibration import IsotonicCalibrator, PlattCalibrator, reliability_curve
from src.mlb_features import MLB_FEATURE_COLUMNS, MLB_TARGET_COLUMN, build_features
from src.mlb_stats_client import fetch_completed_games
from src.model import chronological_split, chronological_split_train_calib_test, train_model
from src.stats import bootstrap_ci, per_game_log_loss


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the model on real MLB data")
    parser.add_argument("--start-date", default="2026-03-01")
    parser.add_argument("--end-date", default="2026-07-16")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument(
        "--save-csv", default=str(config.DATA_PROCESSED_DIR / "mlb_games_real.csv"),
        help="Where to save the built feature dataset",
    )
    args = parser.parse_args()

    raw_games = fetch_completed_games(args.start_date, args.end_date)
    games_df = build_features(raw_games)
    games_df.to_csv(args.save_csv, index=False)
    print(f"Fetched {len(raw_games)} completed games -> {len(games_df)} feature rows -> {args.save_csv}")

    train_df, test_df = chronological_split(games_df, train_frac=args.train_frac)
    model = train_model(train_df, feature_columns=MLB_FEATURE_COLUMNS, target_column=MLB_TARGET_COLUMN)

    y_true = test_df[MLB_TARGET_COLUMN].to_numpy()
    model_probs = model.predict_proba(test_df[MLB_FEATURE_COLUMNS])[:, 1]
    home_win_rate = float(train_df[MLB_TARGET_COLUMN].mean())

    baseline_table = _score_baselines(y_true, model_probs, home_win_rate)
    accuracy_ci, log_loss_delta_ci = _run_bootstrap_ci(y_true, model_probs, home_win_rate)
    calibration = _run_calibration_analysis(games_df, args.train_frac)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_report(
        games_df, train_df, test_df, baseline_table, accuracy_ci, log_loss_delta_ci, calibration,
        config.RESULTS_DIR / "mlb_model_report.md",
    )
    _write_reliability_plot(y_true, model_probs, calibration, config.RESULTS_DIR / "mlb_reliability_plot.png")

    acc = accuracy_score(y_true, (model_probs >= 0.5).astype(int))
    print(f"Test accuracy: {acc:.3f}  Home-rate baseline: {home_win_rate:.3f}")
    print(f"Report written to {config.RESULTS_DIR / 'mlb_model_report.md'}")


def _score_baselines(y_true, model_probs, home_win_rate: float) -> pd.DataFrame:
    n = len(y_true)
    coin_flip_probs = np.full(n, 0.5)
    home_rate_probs = np.full(n, home_win_rate)

    rows = []
    for name, probs in [
        ("Model (logistic regression)", model_probs),
        ("Always predict home team (training home-win rate)", home_rate_probs),
        ("Coin flip", coin_flip_probs),
    ]:
        preds = (probs >= 0.5).astype(int)
        rows.append(
            {
                "predictor": name,
                "accuracy": accuracy_score(y_true, preds),
                "auc": roc_auc_score(y_true, probs),
                "log_loss": log_loss(y_true, probs, labels=[0, 1]),
                "brier_score": brier_score_loss(y_true, probs),
            }
        )
    return pd.DataFrame(rows)


def _run_bootstrap_ci(y_true, model_probs, home_win_rate: float):
    preds = (model_probs >= 0.5).astype(int)
    correct = (preds == y_true).astype(float)
    accuracy_ci = bootstrap_ci(correct, statistic=np.mean, ci=0.90)

    model_loss = per_game_log_loss(y_true, model_probs)
    baseline_loss = per_game_log_loss(y_true, np.full(len(y_true), home_win_rate))
    log_loss_delta = baseline_loss - model_loss  # positive means model beats the baseline
    log_loss_delta_ci = bootstrap_ci(log_loss_delta, statistic=np.mean, ci=0.90)

    return accuracy_ci, log_loss_delta_ci


def _run_calibration_analysis(games_df: pd.DataFrame, train_frac: float) -> dict:
    fit_train_df, calib_df, test_df = chronological_split_train_calib_test(games_df, train_frac=train_frac)
    fit_model = train_model(fit_train_df, feature_columns=MLB_FEATURE_COLUMNS, target_column=MLB_TARGET_COLUMN)

    y_true = test_df[MLB_TARGET_COLUMN].to_numpy()
    raw_probs = fit_model.predict_proba(test_df[MLB_FEATURE_COLUMNS])[:, 1]

    methods = {
        "isotonic": IsotonicCalibrator(fit_model, MLB_FEATURE_COLUMNS).fit(calib_df, MLB_TARGET_COLUMN),
        "platt": PlattCalibrator(fit_model, MLB_FEATURE_COLUMNS).fit(calib_df, MLB_TARGET_COLUMN),
    }

    results = {
        "calib_set_size": len(calib_df),
        "raw_brier": brier_score_loss(y_true, raw_probs),
        "raw_curve": reliability_curve(y_true, raw_probs),
        "methods": {},
    }
    for name, calibrator in methods.items():
        calibrated_probs = calibrator.predict_proba(test_df)
        results["methods"][name] = {
            "brier": brier_score_loss(y_true, calibrated_probs),
            "curve": reliability_curve(y_true, calibrated_probs),
        }
    return results


def _write_report(games_df, train_df, test_df, baseline_table, accuracy_ci, log_loss_delta_ci, calibration, out_path):
    lines = [
        "# Real MLB Model Report",
        "",
        "Trained and evaluated on real completed MLB games from statsapi.mlb.com "
        "(free, public, no key required) -- real teams, real dates, real final scores. "
        "No backtest/CLV section: The Odds API's free tier has no historical odds "
        "endpoint, so there's no real historical market price to size bets against.",
        "",
        f"Feature rows: {len(games_df)}, from {games_df['date'].min()} to "
        f"{games_df['date'].max()} (first game of the season for each team is "
        f"dropped -- no prior history to build features from)",
        f"Training period: {train_df['date'].min()} to {train_df['date'].max()} ({len(train_df)} games)",
        f"Held-out test period: {test_df['date'].min()} to {test_df['date'].max()} ({len(test_df)} games)",
        "",
        "## Model vs. naive baselines (real held-out games)",
        "",
        _baseline_table_markdown(baseline_table),
        "",
        "## Is this significant, or just this test period? (bootstrap 90% CI)",
        "",
        f"- Test accuracy: {accuracy_ci[0]:.3f}, 90% CI [{accuracy_ci[1]:.3f}, {accuracy_ci[2]:.3f}]",
        f"- Log-loss improvement over the home-rate baseline (per game, positive = model "
        f"better): {log_loss_delta_ci[0]:+.4f}, 90% CI [{log_loss_delta_ci[1]:+.4f}, "
        f"{log_loss_delta_ci[2]:+.4f}]",
        "",
        "## Calibration",
        "",
        f"Calibration set: {calibration['calib_set_size']} real games, carved out of the "
        "training period, never the test period.",
        "",
        f"- Brier score, raw: {calibration['raw_brier']:.4f}",
        f"- Brier score, isotonic-calibrated: {calibration['methods']['isotonic']['brier']:.4f}",
        f"- Brier score, Platt-calibrated: {calibration['methods']['platt']['brier']:.4f}",
        "",
        "![Reliability diagram](mlb_reliability_plot.png)",
        "",
    ]
    out_path.write_text("\n".join(lines))


def _baseline_table_markdown(baseline_table: pd.DataFrame) -> str:
    lines = ["| Predictor | Accuracy | AUC | Log loss | Brier score |", "|---|---|---|---|---|"]
    for _, row in baseline_table.iterrows():
        lines.append(
            f"| {row['predictor']} | {row['accuracy']:.3f} | {row['auc']:.3f} | "
            f"{row['log_loss']:.4f} | {row['brier_score']:.4f} |"
        )
    return "\n".join(lines)


def _write_reliability_plot(y_true, model_probs, calibration, out_path):
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
    ax.set_ylabel("Observed home-win frequency")
    ax.set_title("Reliability diagram -- real MLB held-out test period")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
