"""CLI: a more rigorous evaluation of the same real MLB win-probability model
(train_mlb_model.py's single 80/20 split, one season, no regularization
search, no feature-importance or error analysis) -- same six team-level
features, same logistic regression, evaluated properly instead of once:

- Multiple real MLB seasons (2023-2026), not one partial season -- several
  times the real data, still 100% real, not synthetic.
- Walk-forward (expanding-window) validation across multiple folds instead
  of a single static holdout, so results show whether the model's edge (or
  lack of one) is stable over time or just an artifact of which test window
  got picked.
- A regularization-strength search inside each fold's training data (an
  inner chronological validation split, never the fold's real test set).
- Feature importance via standardized coefficients, and error analysis by
  prediction confidence and by season, pooled across every fold's held-out
  predictions.

Usage:
    python scripts/train_mlb_model_walk_forward.py
"""
import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

import config
from src.mlb_features import MLB_FEATURE_COLUMNS, MLB_TARGET_COLUMN, build_features
from src.mlb_stats_client import fetch_completed_games
from src.model import walk_forward_splits
from src.model_selection import select_regularization_strength, standardized_coefficients
from src.stats import bootstrap_ci, per_game_log_loss

SEASON_RANGES = [
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
    ("2026", "2026-01-01", date.today().isoformat()),
]


def main():
    parser = argparse.ArgumentParser(description="Walk-forward evaluation of the real MLB model")
    parser.add_argument("--n-splits", type=int, default=4)
    args = parser.parse_args()

    features_df, season_counts = _build_multi_season_features()
    print(f"Real games by season: {season_counts}")
    print(f"Total feature rows: {len(features_df)}")

    splits = walk_forward_splits(features_df, n_splits=args.n_splits)
    fold_results, pooled = _run_folds(splits)

    fold_table = pd.DataFrame(fold_results)
    print(fold_table.to_string(index=False))

    agg = _aggregate_across_folds(fold_table)
    significance_ci = _pooled_significance(pooled)

    final_model, best_C_final, importance = _fit_final_model_for_importance(features_df)
    confidence_buckets = _error_analysis_by_confidence(pooled)
    season_buckets = _error_analysis_by_season(pooled)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_report(
        season_counts, len(features_df), fold_table, agg, significance_ci, importance,
        confidence_buckets, season_buckets, best_C_final,
        config.RESULTS_DIR / "mlb_walk_forward_report.md",
    )
    _write_importance_plot(importance, config.RESULTS_DIR / "mlb_feature_importance.png")
    _write_fold_stability_plot(fold_table, config.RESULTS_DIR / "mlb_walk_forward_stability.png")

    print(f"\nPooled log-loss improvement over home-rate baseline: {significance_ci[0]:+.4f}, "
          f"90% CI [{significance_ci[1]:+.4f}, {significance_ci[2]:+.4f}]")
    print(f"Report written to {config.RESULTS_DIR / 'mlb_walk_forward_report.md'}")


def _build_multi_season_features() -> tuple[pd.DataFrame, dict]:
    """Fetch and build features per season independently, then concatenate.
    Rolling team state (recent form, rest days) must reset at each season
    boundary -- carrying it across the off-season would be both unrealistic
    (rosters turn over) and would create huge, spurious rest_days_diff
    outliers from a ~150-day off-season gap. build_features already resets
    its internal state on every call, so calling it once per season and
    concatenating the results achieves this with no changes to that module.
    """
    season_frames = []
    season_counts = {}
    for season, start, end in SEASON_RANGES:
        raw_games = fetch_completed_games(start, end)
        season_counts[season] = len(raw_games)
        if not raw_games:
            continue
        season_features = build_features(raw_games)
        season_features["season"] = season
        season_frames.append(season_features)

    return pd.concat(season_frames, ignore_index=True), season_counts


def _run_folds(splits: list[tuple[pd.DataFrame, pd.DataFrame]]) -> tuple[list[dict], pd.DataFrame]:
    fold_rows = []
    pooled_rows = []

    for i, (train_df, test_df) in enumerate(splits):
        best_C, _ = select_regularization_strength(train_df, MLB_FEATURE_COLUMNS, MLB_TARGET_COLUMN)
        model = LogisticRegression(C=best_C)
        model.fit(train_df[MLB_FEATURE_COLUMNS], train_df[MLB_TARGET_COLUMN])

        y_true = test_df[MLB_TARGET_COLUMN].to_numpy()
        probs = model.predict_proba(test_df[MLB_FEATURE_COLUMNS])[:, 1]
        preds = (probs >= 0.5).astype(int)
        home_rate = float(train_df[MLB_TARGET_COLUMN].mean())

        fold_rows.append(
            {
                "fold": i,
                "train_start": train_df["date"].min(),
                "train_end": train_df["date"].max(),
                "test_start": test_df["date"].min(),
                "test_end": test_df["date"].max(),
                "train_size": len(train_df),
                "test_size": len(test_df),
                "best_C": best_C,
                "accuracy": accuracy_score(y_true, preds),
                "auc": roc_auc_score(y_true, probs),
                "log_loss": log_loss(y_true, probs, labels=[0, 1]),
                "brier_score": brier_score_loss(y_true, probs),
                "home_rate_baseline": home_rate,
            }
        )

        fold_pooled = test_df[["date", "season"]].copy()
        fold_pooled["fold"] = i
        fold_pooled["y_true"] = y_true
        fold_pooled["model_prob"] = probs
        fold_pooled["home_rate_baseline"] = home_rate
        pooled_rows.append(fold_pooled)

    return fold_rows, pd.concat(pooled_rows, ignore_index=True)


def _aggregate_across_folds(fold_table: pd.DataFrame) -> dict:
    metrics = ["accuracy", "auc", "log_loss", "brier_score"]
    return {m: (float(fold_table[m].mean()), float(fold_table[m].std())) for m in metrics}


def _pooled_significance(pooled: pd.DataFrame):
    """Bootstrap CI on the per-game log-loss improvement over each row's own
    fold-specific home-rate baseline, pooled across every fold's held-out
    predictions -- a much larger effective sample than any single fold's
    test set alone."""
    model_loss = per_game_log_loss(pooled["y_true"], pooled["model_prob"])
    baseline_loss = per_game_log_loss(pooled["y_true"], pooled["home_rate_baseline"])
    delta = baseline_loss - model_loss
    return bootstrap_ci(delta, statistic=np.mean, ci=0.90)


def _fit_final_model_for_importance(features_df: pd.DataFrame):
    """A model fit on *all* available data, for interpretability only -- never
    evaluated (that would leak), just used to report standardized coefficients."""
    best_C, _ = select_regularization_strength(features_df, MLB_FEATURE_COLUMNS, MLB_TARGET_COLUMN)
    model = LogisticRegression(C=best_C)
    model.fit(features_df[MLB_FEATURE_COLUMNS], features_df[MLB_TARGET_COLUMN])
    importance = standardized_coefficients(model, MLB_FEATURE_COLUMNS, features_df)
    return model, best_C, importance


def _error_analysis_by_confidence(pooled: pd.DataFrame) -> pd.DataFrame:
    """Accuracy/log-loss bucketed by how confident the model was (|prob - 0.5|)
    -- a model with real signal should do better on its more confident calls."""
    pooled = pooled.copy()
    pooled["confidence"] = (pooled["model_prob"] - 0.5).abs()
    pooled["confidence_bucket"] = pd.qcut(pooled["confidence"], q=5, duplicates="drop")

    rows = []
    for bucket, group in pooled.groupby("confidence_bucket", observed=True):
        preds = (group["model_prob"] >= 0.5).astype(int)
        clipped = np.clip(group["model_prob"], 1e-15, 1 - 1e-15)
        rows.append(
            {
                "confidence_bucket": str(bucket),
                "n": len(group),
                "accuracy": accuracy_score(group["y_true"], preds),
                "log_loss": log_loss(group["y_true"], clipped, labels=[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def _error_analysis_by_season(pooled: pd.DataFrame) -> pd.DataFrame:
    """Accuracy/log-loss per season, on held-out (never-trained-on-that-row)
    predictions only -- does performance hold up across different years, or
    was one season doing all the work?"""
    rows = []
    for season, group in pooled.groupby("season"):
        preds = (group["model_prob"] >= 0.5).astype(int)
        clipped = np.clip(group["model_prob"], 1e-15, 1 - 1e-15)
        rows.append(
            {
                "season": season,
                "n": len(group),
                "accuracy": accuracy_score(group["y_true"], preds),
                "log_loss": log_loss(group["y_true"], clipped, labels=[0, 1]),
            }
        )
    return pd.DataFrame(rows).sort_values("season")


def _write_report(
    season_counts, total_feature_rows, fold_table, agg, significance_ci, importance,
    confidence_buckets, season_buckets, best_C_final, out_path: Path,
) -> None:
    lines = [
        "# Walk-Forward MLB Model Evaluation",
        "",
        "A more rigorous evaluation of the same real MLB model in "
        "`results/mlb_model_report.md` (same six team-level features, same "
        "logistic regression) -- multiple real seasons instead of one, "
        "walk-forward folds instead of a single 80/20 split, a regularization "
        "search inside each fold's training data, feature importance, and "
        "error analysis pooled across every fold's held-out predictions.",
        "",
        "## Data",
        "",
        f"Real completed games by season (statsapi.mlb.com): "
        f"{', '.join(f'{s}: {n}' for s, n in season_counts.items())}",
        f"Total feature rows after point-in-time engineering: {total_feature_rows}",
        "",
        "Rolling team state resets at each season boundary (features are built "
        "per season, then concatenated) -- carrying rolling form across an "
        "off-season would be both unrealistic and would create spurious "
        "rest-days outliers from a ~150-day gap.",
        "",
        "## Walk-forward folds",
        "",
        "| Fold | Train period | Test period | Train N | Test N | Best C | Accuracy | AUC | Log loss | Brier |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in fold_table.iterrows():
        lines.append(
            f"| {int(row['fold'])} | {row['train_start']} to {row['train_end']} | "
            f"{row['test_start']} to {row['test_end']} | {int(row['train_size'])} | "
            f"{int(row['test_size'])} | {row['best_C']:g} | {row['accuracy']:.3f} | "
            f"{row['auc']:.3f} | {row['log_loss']:.4f} | {row['brier_score']:.4f} |"
        )

    lines += [
        "",
        "## Stability across folds (mean +/- std)",
        "",
        f"- Accuracy: {agg['accuracy'][0]:.3f} +/- {agg['accuracy'][1]:.3f}",
        f"- AUC: {agg['auc'][0]:.3f} +/- {agg['auc'][1]:.3f}",
        f"- Log loss: {agg['log_loss'][0]:.4f} +/- {agg['log_loss'][1]:.4f}",
        f"- Brier score: {agg['brier_score'][0]:.4f} +/- {agg['brier_score'][1]:.4f}",
        "",
        "![Fold stability](mlb_walk_forward_stability.png)",
        "",
        "## Is the pooled edge real? (bootstrap 90% CI, all folds' held-out predictions pooled)",
        "",
        f"Log-loss improvement over each fold's own home-rate baseline: "
        f"{significance_ci[0]:+.4f}, 90% CI [{significance_ci[1]:+.4f}, {significance_ci[2]:+.4f}]"
        + (
            " -- excludes zero: significant on the pooled sample."
            if significance_ci[1] > 0
            else " -- includes zero: still not statistically distinguishable from no edge, "
            "now on a much larger pooled sample across multiple real seasons."
        ),
        "",
        "## Feature importance (standardized coefficients)",
        "",
        f"From a final model fit on all {total_feature_rows} available rows "
        f"(C={best_C_final:g} via the same regularization search) -- used only "
        "for interpretability, never for evaluation.",
        "",
        "| Feature | Raw coefficient | Feature std dev | Standardized coefficient |",
        "|---|---|---|---|",
    ]
    for _, row in importance.iterrows():
        lines.append(
            f"| {row['feature']} | {row['raw_coefficient']:+.4f} | {row['feature_std_dev']:.3f} | "
            f"{row['standardized_coefficient']:+.4f} |"
        )
    lines += [
        "",
        "![Feature importance](mlb_feature_importance.png)",
        "",
        "## Error analysis: by prediction confidence",
        "",
        "Pooled held-out predictions bucketed by |model probability - 0.5| -- "
        "a model with real signal should do better on its more confident calls.",
        "",
        "| Confidence bucket | N | Accuracy | Log loss |",
        "|---|---|---|---|",
    ]
    for _, row in confidence_buckets.iterrows():
        lines.append(f"| {row['confidence_bucket']} | {int(row['n'])} | {row['accuracy']:.3f} | {row['log_loss']:.4f} |")

    lines += [
        "",
        "## Error analysis: by season",
        "",
        "Pooled held-out predictions grouped by season -- does performance hold "
        "up across different years, or was one season doing all the work?",
        "",
        "| Season | N | Accuracy | Log loss |",
        "|---|---|---|---|",
    ]
    for _, row in season_buckets.iterrows():
        lines.append(f"| {row['season']} | {int(row['n'])} | {row['accuracy']:.3f} | {row['log_loss']:.4f} |")
    lines.append("")

    out_path.write_text("\n".join(lines))


def _write_importance_plot(importance: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ordered = importance.iloc[::-1]
    colors = ["#2a9d8f" if c >= 0 else "#e76f51" for c in ordered["standardized_coefficient"]]
    ax.barh(ordered["feature"], ordered["standardized_coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardized coefficient (impact on log-odds per typical feature swing)")
    ax.set_title("Feature importance -- walk-forward MLB model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_fold_stability_plot(fold_table: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(fold_table["fold"], fold_table["accuracy"], marker="o", color="#2a6f97", label="Model accuracy")
    axes[0].plot(fold_table["fold"], fold_table["home_rate_baseline"].clip(lower=0.5), marker="o",
                 color="gray", linestyle="--", label="Home-rate baseline")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy per fold")
    axes[0].legend(fontsize=8)

    axes[1].plot(fold_table["fold"], fold_table["log_loss"], marker="o", color="#e76f51")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("Log loss")
    axes[1].set_title("Log loss per fold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
