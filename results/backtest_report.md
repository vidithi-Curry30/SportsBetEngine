# Backtest Report

Training period: 2024-10-22 to 2025-03-09 (960 games)
Held-out test period: 2025-03-09 to 2025-04-12 (240 games)

## Results

- **Average CLV (the metric that matters more than raw ROI over a small sample): +2.64 percentage points**
- ROI: 6.32%
- Sharpe-like ratio: 0.042
- Max drawdown: -37.03%
- Hit rate: 29.8%
- Bets placed: 94 / 240 games scanned
- Final bankroll: $10,632.27 (started at $10,000.00)
- Top-bet P&L share: 634% (share of total profit from the single best bet -- high values mean the result rests on one outlier, not a repeatable edge)

![Bankroll and CLV](clv_plot.png)

## Is 94 bets enough to tell edge from noise? (bootstrap 90% CI)

- Avg CLV: +2.64pp, 90% CI [+1.78, +3.49]pp
- Mean per-bet return: +0.40%, 90% CI [-1.16%, +2.15%]

## Model vs. naive baselines (same held-out period)

| Predictor | Accuracy | AUC | Log loss | Brier score |
|---|---|---|---|---|
| Model (logistic regression) | 0.592 | 0.637 | 0.6614 | 0.2345 |
| Always bet market favorite (no-vig price) | 0.629 | 0.672 | 0.6637 | 0.2328 |
| Always bet home team | 0.492 | 0.490 | 0.6994 | 0.2531 |
| Coin flip | 0.442 | 0.500 | 0.6931 | 0.2500 |

**Is the model-vs-market gap real?** Bootstrap 90% CI on the per-game log-loss improvement over the market-favorite baseline: +0.0024, 90% CI [-0.0301, +0.0335] -- **includes zero**: not statistically distinguishable from no edge over the market's own price on this sample size.

## Calibration: does the model's raw probability mean what it says?

Calibration set: 144 games (carved out of the training period, never the test period). Isotonic regression is a flexible nonparametric fit that sklearn's own docs say needs ~1000+ calibration samples to avoid overfitting the mapping itself; Platt (sigmoid) scaling only fits 2 parameters and is far less data-hungry. Both are shown rather than assuming either one is the fix.

- Brier score, raw: 0.2353
- Brier score, isotonic-calibrated: 0.2424
- Brier score, Platt-calibrated: 0.2368

Backtest re-run on the identical test period for each:

| Metric | Raw | Isotonic | Platt |
|---|---|---|---|
| ROI | 9.00% | -0.06% | 46.00% |
| Avg CLV (pp) | +2.52 | +2.49 | +1.99 |
| Hit rate | 29.5% | 36.0% | 30.5% |
| Bets placed | 95 | 86 | 95 |
| Max drawdown | -37.26% | -45.09% | -31.64% |
| Top-bet P&L share | 447% | n/a (no net profit) | 101% |

![Reliability diagram](calibration_plot.png)
