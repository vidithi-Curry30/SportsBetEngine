# Does Adding Starting-Pitcher Features Help?

Fresh data pull, not the same held-out games as results/mlb_model_report.md. Both models below are trained and evaluated on the identical set of games (inner join on games where a probable starter with prior-start history exists for both teams) -- the only difference between them is whether starting_pitcher_era_diff and starting_pitcher_k9_diff are included as features.

Training period: 2026-03-31 to 2026-06-29 (1040 games)
Held-out test period: 2026-06-29 to 2026-07-22 (261 games)

| Predictor | Accuracy | AUC | Log loss | Brier score |
|---|---|---|---|---|
| Team-only | 0.529 | 0.569 | 0.6886 | 0.2477 |
| Team + starting pitcher | 0.533 | 0.574 | 0.6876 | 0.2472 |

Bootstrap 90% CI on the per-game log-loss improvement from adding pitcher features: +0.0010, 90% CI [-0.0021, +0.0041] -- includes zero: not statistically distinguishable from no improvement on this sample size.
