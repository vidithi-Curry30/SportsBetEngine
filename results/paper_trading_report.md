# Paper Trading Report (Real, Forward-Collected CLV)

Unlike results/backtest_report.md, every row here is a real bet logged at a real, live market price *before* the game was played -- not a synthetic backtest. Leakage is structurally impossible: the paper trade is written to the ledger by scripts/collect_paper_trades.py before the outcome exists. The tradeoff is sample size: this only grows by one slate at a time.

Total paper trades logged: 6 (6 settled, 0 still open)

## Results

- Avg CLV: +0.00pp, 90% CI [+0.00, +0.00]pp
- Avg per-bet return (in stake-fraction units): -0.0251, 90% CI [-0.0500, -0.0002]
- Hit rate: 33.3%
- Settled bets: 6

(Wide or zero-crossing CIs on a small settled count are the honest result, not a bug -- see src/stats.py.)

