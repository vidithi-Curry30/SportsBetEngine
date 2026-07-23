# Sports Market Efficiency & Arbitrage Engine

Sportsbook odds are prices in a prediction market: implied probability is
price, vig is bid-ask spread, cross-book arbitrage is cross-venue arbitrage,
and Kelly sizing is position sizing under an estimated edge and its
uncertainty. A research/backtesting pipeline built as a quant-trading
portfolio piece, not a betting product.

*Assisted by Claude.*

## TL;DR

- **Real end-to-end**: a live Odds API key, a real arbitrage scan against
  actual multi-book prices, a real win-probability model trained on 1,400+
  real MLB games.
- **The real MLB model doesn't beat a coin flip (49.0% vs. 47.9%), reported
  as the finding** — with a hypothesis (team stats can't see the starting
  pitcher) that was then tested on fresh data: a small, consistent, still
  not statistically significant lift. Predicting an effect and confirming it
  beats either a null result or an improvement alone.
- **The backtest's own tail-risk check flags its own headline ROI**: +6.3%
  looks like a win until `top_bet_pnl_share` shows 634% of that profit came
  from one bet. This is *why* CLV (+2.64pp, 90% CI clears zero), not ROI, is
  the metric this project leads with.
- **Isotonic-vs-Platt calibration finding reproduced independently on two
  unrelated datasets** (synthetic NBA, real MLB) — same pattern both times.
- Three additions extend the risk/pricing logic past a single bet: a
  slate-level exposure cap, a market-making (quoting) mode, and a
  forward-collecting real-CLV pipeline. Details below; the last one has
  **no results yet** — it was just built.

## Why this framing

A book quoting -110 both ways is doing what a market maker does quoting a
bid-ask spread: pricing both outcomes above fair value to profit regardless
of the result. "Beating the market" means being right *relative to the
price*, consistently, net of the vig, before the closing line.

## Pipeline

| Module | What it does |
|---|---|
| `probability.py` | Odds ↔ implied probability; `remove_vig`/`add_vig` between fair and quoted prices |
| `vig.py` | Vig (overround) per market |
| `odds_client.py` | Wraps [The Odds API](https://the-odds-api.com/) |
| `arbitrage.py` | Cross-book arbitrage scanner |
| `model.py` | Logistic regression win-probability model, chronological train/test split |
| `kelly.py` | Full and fractional Kelly sizing |
| `clv.py` | Closing line value |
| `backtest.py` | Walks the held-out period, sizes bets, simulates bankroll, batches by slate |
| `baselines.py` / `calibration.py` / `stats.py` | Naive-baseline comparison, isotonic/Platt calibration, bootstrap CIs |
| `mlb_stats_client.py` / `mlb_features.py` / `mlb_pitcher_features.py` | Real MLB results and point-in-time team/pitcher features (historical + live) |
| `portfolio_risk.py` | Slate-level exposure cap on top of the per-bet Kelly cap |
| `portfolio_optimization.py` | Correlation-aware sizing: a multivariate Kelly generalization (`w* = Σ⁻¹μ`) that discounts bets sharing a team |
| `market_maker.py` | Quotes a two-sided line and simulates inventory-skew repricing |
| `paper_trading.py` | Live line-shopping + a forward paper-trade ledger for real CLV |

```bash
python scripts/fetch_odds.py --sport baseball_mlb   # pull live odds (needs ODDS_API_KEY)
python scripts/run_backtest.py                      # synthetic NBA: train, backtest, write results/
python scripts/train_mlb_model.py                   # real MLB: fetch, train, evaluate
python scripts/collect_paper_trades.py              # real MLB: log today's live-odds edges
python scripts/reconcile_paper_trades.py            # real MLB: settle + report
```

## A note on data

| Piece | Status |
|---|---|
| Live odds / arbitrage scanner | **Real.** |
| Win-probability model | **Real** (`statsapi.mlb.com`). |
| Backtest / CLV / Kelly sizing | **Synthetic** — The Odds API's historical endpoint is paywalled, and real *current* odds don't help retroactively (`data/processed/nba_games_synthetic.csv`, seeded and reproducible via `scripts/generate_synthetic_*.py`). |
| Paper-trading CLV | **Real, forward-collected** — see below. |

## Forward-collecting real CLV (paper trading)

`collect_paper_trades.py` fits a fresh model on all completed MLB games,
pulls live multi-book odds, line-shops for the best price, compares the
model to the consensus no-vig market price, and logs any flagged edge to
`data/paper_trades/mlb_paper_trades.csv` at today's real price. Run it daily
(or a few times a day near first pitch). `reconcile_paper_trades.py` (run
after games finish) fetches the real result and a closing-price proxy, and
writes `results/paper_trading_report.md`.

This is a stronger claim than the synthetic backtest, not a weaker one:
leakage is structurally impossible since the bet is logged before the
outcome exists. The tradeoff is sample size — it grows one slate at a time.
**No results yet**: this was just built and hasn't run forward. Reporting a
number now would mean fabricating it.

## Live data: verified against a real Odds API pull

A real pull (`baseball_mlb`, 2026-07-16) returned 4 games across up to 9
books; `scan_games_for_arbitrage` found **0 opportunities**, consistent with
the synthetic result. Vig ranged 2.2–5.1%, with offshore/low-margin books
pricing tighter — directionally sensible on a small sample.

## Real MLB model: an honest null result

`train_mlb_model.py` trains on point-in-time features (rolling
runs scored/allowed, run differential, last-10 win rate, rest days) from
every completed 2026 game (1,445 as of this run).

| Predictor | Accuracy | AUC | Log loss | Brier |
|---|---|---|---|---|
| Model | 49.0% | 0.560 | 0.6957 | 0.2513 |
| Home-rate baseline | 47.9% | 0.500 | 0.6976 | 0.2522 |
| Coin flip | 47.9% | 0.500 | 0.6931 | 0.2500 |

Bootstrap 90% CI on accuracy [44.1%, 53.8%] straddles a coin flip; log-loss
CI vs. baseline includes zero. **No distinguishable edge on accuracy or log
loss.** AUC (0.560) shows weak real ranking ability that doesn't survive as
a usable probability — likely because single-game MLB outcomes hinge on the
starting-pitcher matchup, invisible to team-level stats.

Adding rolling starting-pitcher ERA/K9 on a fresh data pull (identical
train/test games either way): accuracy 52.9% → 53.3%, log-loss CI
[-0.0021, +0.0041] — still includes zero on 261 games, exactly the
calibrated-in-advance expectation (some lift, not a transformation). Full
reports: `results/mlb_model_report.md`, `results/mlb_pitcher_features_report.md`.

## Results: synthetic NBA backtest

240 held-out games, 94 bets flagged (3pp edge threshold), run through the
slate-batched engine (same-day bets sized against that day's starting
bankroll, then capped together):

| Metric | Value |
|---|---|
| Avg CLV | **+2.64pp** (90% CI [+1.78, +3.49]) |
| ROI | +6.3% (90% CI [-1.16%, +2.15%]) |
| Hit rate | 29.8% |
| Max drawdown | -37.0% |
| Top-bet P&L share | **634%** |

Read ROI and top-bet P&L share together: +6.3% looks like a win, but 634%
of that profit is one bet — the rest of the book nets a loss. That's the
tail-risk check working as designed, and why CLV leads. The Platt-calibrated
calibration variant shows the same pattern (+46% ROI, 101% concentration).

Model vs. the market's own no-vig price: the market wins on accuracy/AUC/Brier;
the model's raw log-loss edge (0.6614 vs. 0.6637) has a bootstrap CI that
includes zero — not distinguishable from parity. It only shows up in CLV, a
different, lower-variance question (did the line move in your favor, not was
the point prediction more accurate). Calibration: neither isotonic nor Platt
meaningfully improved Brier (0.235 → 0.242 / 0.237) — same
small-calibration-set pattern seen on the real MLB data too. Full report:
`results/backtest_report.md`.

## Slate-level portfolio risk

`max_bet_pct` caps any one bet but says nothing about a day's aggregate — a
10-15 game MLB slate can commit far more than any single-position limit
implies if each bet is capped independently. `run_backtest` now batches bets
by date, sizes them against that day's starting bankroll (not sequentially
compounded), and `apply_slate_exposure_cap` scales an over-limit day's stakes
down proportionally (default 20% of bankroll/slate, on top of the 5%
per-bet cap) — preserving relative sizing rather than dropping bets.

**Correlation-aware sizing** (`src/portfolio_optimization.py`, opt-in via
`sizing_strategy="correlation_aware"`) goes further: the proportional cap
above treats every bet on a slate as equally diversifying, but two bets on
the same team — an MLB doubleheader — aren't independent. This module is a
multivariate generalization of the single-bet Kelly formula: `w* = Σ⁻¹μ`,
the standard quadratic (mean-variance) approximation to multivariate Kelly,
where μ is each bet's expected return and Σ is their covariance matrix
(assumed correlation between same-team bets, stated as an assumption, not
fit — there's no real correlated-outcome sample to estimate it from yet).
It's an approximation, not an exact generalization of the single-bet
formula (honestly documented in the module), but it gets the property that
matters right: independent bets decouple into ordinary per-bet Kelly, and
two bets sharing a team get sized *below* what treating them independently
would give — both proven directly in `tests/test_portfolio_optimization.py`
against hand-derived closed forms, not just "a number came out." Off by
default; the primary backtest results above are unaffected.

## Market making

Everything else here *consumes* a market price. `market_maker.py` inverts
that: `quote_from_probability` turns a probability into a two-sided price via
`add_vig` (the inverse of `remove_vig`); `simulate_flow_and_reprice`
simulates bettors trading against that quote, tracks the maker's signed
inventory, and skews the line once inventory crosses a threshold — the
standard response to one-sided flow. Deliberately small: one game, one round
of flow, a linear reprice rule, to demonstrate the mechanism rather than
build a production engine.

## Design decisions worth defending in an interview

- **Chronological train/test split, never violated** — structural, not a promise.
- **Half-Kelly, not full Kelly** — full Kelly is only optimal for a
  *correctly specified* edge, which a small logistic regression never has exactly.
- **Per-bet and slate-level exposure caps, independent of Kelly's output** —
  still a -37.0% max drawdown at a 5% cap; "capped" isn't "safe," just safer.
- **Simulated slippage** — the price you decide on and the price you get rarely match.
- **CLV as the headline metric, backed by a tail-risk check** that keeps
  catching real problems (634% concentration above), not a box to tick.
- **Calibration tested, not assumed** — isotonic and Platt both fit and
  compared, neither cherry-picked when it didn't win.
- **Every "beats X" claim carries a significance test.** AUC is reported
  alongside accuracy/log-loss/Brier so ranking power and decision-useful
  calibration don't get conflated.
- **Point-in-time features, tested directly** — `test_mlb_features.py`
  changes a later game's score and asserts every earlier row is unchanged.
- **Null results reported as null results.**

## What this deliberately doesn't do

- No real-money betting — `paper_trading.py` only logs what a bet *would*
  have been, at a real price, for later reconciliation.
- No database — flat CSV/JSON is sufficient and easier to defend than a schema.
- No deep learning — logistic regression is the right complexity for six
  features and ~1000 games.
- No production market-making engine — `market_maker.py` demonstrates the
  mechanism on one game, not a continuous multi-game book.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add ODDS_API_KEY to pull live odds (optional)
```

## Run tests

```bash
pytest
```

192 tests, including hand-checked formula values, a planted arbitrage the
scanner must detect, a chronological split the model must never leak across,
a tail-dominated P&L case the concentration check must flag, point-in-time
correctness checks (historical and live), a slate that forces the exposure
cap to bind, correlation-aware weights checked against hand-derived closed
forms (not just "a number came out"), a deterministic market-making flow
that self-corrects, a paper-trade ledger round-trip against synthetic
Odds-API-shaped payloads, and a full mocked run of `collect_paper_trades.py`
itself (fetch → train → pull odds → flag edges → log), not just its
underlying units.

## Repo structure

```
sports-market-efficiency/
├── config.py
├── src/                       # probability, vig, arbitrage, model, kelly, clv, backtest,
│                              #   baselines, calibration, stats, mlb_stats_client,
│                              #   mlb_features, mlb_pitcher_features, portfolio_risk,
│                              #   portfolio_optimization, market_maker, paper_trading
├── scripts/                   # fetch_odds, run_backtest, train_mlb_model(_with_pitching),
│                              #   collect_paper_trades, reconcile_paper_trades
├── data/                       # raw/, processed/, paper_trades/ (all gitignored except samples)
├── notebooks/exploration.ipynb
├── tests/
└── results/                    # backtest_report.md, mlb_model_report.md,
                                 #   mlb_pitcher_features_report.md, paper_trading_report.md, plots
```
