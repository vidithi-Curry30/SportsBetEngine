"""CLI: does adding starting-pitcher features improve on the team-only MLB
model? Fetches current data fresh, builds a team-only feature set and a
team+pitcher feature set on the identical games (an inner join on game_pk,
so both models are trained and evaluated on exactly the same train/test
split -- isolating the effect of the new features from any change in which
games are included), and reports accuracy/AUC/log-loss/Brier side by side
with a bootstrap significance test on the log-loss gap.

This deliberately does NOT reuse the held-out period already published in
results/mlb_model_report.md. Fetching fresh data naturally shifts the
window forward in time, so the "did pitcher features help" question is
answered on games that weren't already used to report the original null
result -- not tuned against a test set that's already been looked at.

Usage:
    python scripts/train_mlb_model_with_pitching.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

import config
from src.mlb_features import MLB_FEATURE_COLUMNS, MLB_TARGET_COLUMN, build_features
from src.mlb_pitcher_features import (
    PITCHER_FEATURE_COLUMNS,
    build_pitching_features,
    fetch_all_starter_game_logs,
)
from src.mlb_stats_client import fetch_completed_games
from src.model import chronological_split, train_model
from src.stats import bootstrap_ci, per_game_log_loss

EXTENDED_FEATURE_COLUMNS = MLB_FEATURE_COLUMNS + PITCHER_FEATURE_COLUMNS


def main():
    parser = argparse.ArgumentParser(description="Compare team-only vs. team+pitcher MLB models")
    parser.add_argument("--start-date", default="2026-03-01")
    parser.add_argument("--end-date", default="2026-12-31")
    parser.add_argument("--season", default="2026")
    parser.add_argument("--train-frac", type=float, default=0.8)
    args = parser.parse_args()

    raw_games = fetch_completed_games(args.start_date, args.end_date)
    print(f"Fetched {len(raw_games)} completed games")

    team_df = build_features(raw_games)

    print("Fetching starting-pitcher game logs (one request per unique starter)...")
    pitcher_logs = fetch_all_starter_game_logs(raw_games, season=args.season)
    print(f"Fetched logs for {len(pitcher_logs)} unique starters")

    pitching_df = build_pitching_features(raw_games, pitcher_logs)
    extended_df = team_df.merge(
        pitching_df[["game_pk"] + PITCHER_FEATURE_COLUMNS], on="game_pk", how="inner"
    )
    print(f"Team-only feature rows: {len(team_df)}; rows with pitcher features too: {len(extended_df)}")

    train_df, test_df = chronological_split(extended_df, train_frac=args.train_frac)
    team_only_model = train_model(train_df, feature_columns=MLB_FEATURE_COLUMNS, target_column=MLB_TARGET_COLUMN)
    extended_model = train_model(
        train_df, feature_columns=EXTENDED_FEATURE_COLUMNS, target_column=MLB_TARGET_COLUMN
    )

    y_true = test_df[MLB_TARGET_COLUMN].to_numpy()
    team_only_probs = team_only_model.predict_proba(test_df[MLB_FEATURE_COLUMNS])[:, 1]
    extended_probs = extended_model.predict_proba(test_df[EXTENDED_FEATURE_COLUMNS])[:, 1]

    comparison_table = _score_both(y_true, team_only_probs, extended_probs)
    significance_ci = _significance(y_true, team_only_probs, extended_probs)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "mlb_pitcher_features_report.md"
    _write_report(train_df, test_df, comparison_table, significance_ci, out_path)

    print(comparison_table.to_string(index=False))
    print(f"Log-loss improvement 90% CI: [{significance_ci[1]:+.4f}, {significance_ci[2]:+.4f}]")
    print(f"Report written to {out_path}")


def _score_both(y_true, team_only_probs, extended_probs) -> pd.DataFrame:
    rows = []
    for name, probs in [("Team-only", team_only_probs), ("Team + starting pitcher", extended_probs)]:
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


def _significance(y_true, team_only_probs, extended_probs):
    """Bootstrap CI on the per-game log-loss improvement from adding pitcher
    features. Positive delta means the extended model has lower loss."""
    team_only_loss = per_game_log_loss(y_true, team_only_probs)
    extended_loss = per_game_log_loss(y_true, extended_probs)
    delta = team_only_loss - extended_loss
    return bootstrap_ci(delta, statistic=np.mean, ci=0.90)


def _write_report(train_df, test_df, comparison_table, significance_ci, out_path: Path) -> None:
    lines = [
        "# Does Adding Starting-Pitcher Features Help?",
        "",
        "Fresh data pull, not the same held-out games as results/mlb_model_report.md. "
        "Both models below are trained and evaluated on the identical set of games "
        "(inner join on games where a probable starter with prior-start history exists "
        "for both teams) -- the only difference between them is whether "
        "starting_pitcher_era_diff and starting_pitcher_k9_diff are included as features.",
        "",
        f"Training period: {train_df['date'].min()} to {train_df['date'].max()} ({len(train_df)} games)",
        f"Held-out test period: {test_df['date'].min()} to {test_df['date'].max()} ({len(test_df)} games)",
        "",
        "| Predictor | Accuracy | AUC | Log loss | Brier score |",
        "|---|---|---|---|---|",
    ]
    for _, row in comparison_table.iterrows():
        lines.append(
            f"| {row['predictor']} | {row['accuracy']:.3f} | {row['auc']:.3f} | "
            f"{row['log_loss']:.4f} | {row['brier_score']:.4f} |"
        )
    lines += [
        "",
        f"Bootstrap 90% CI on the per-game log-loss improvement from adding pitcher "
        f"features: {significance_ci[0]:+.4f}, 90% CI [{significance_ci[1]:+.4f}, "
        f"{significance_ci[2]:+.4f}]"
        + (
            " -- excludes zero: the improvement is statistically distinguishable from noise."
            if significance_ci[1] > 0
            else " -- includes zero: not statistically distinguishable from no improvement "
            "on this sample size."
        ),
        "",
    ]
    out_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
