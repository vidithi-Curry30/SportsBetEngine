# Sports Market Efficiency & Arbitrage Engine

Sportsbook odds are prices in a prediction market. This project treats them
that way: **implied probability is price**, **vig is bid-ask spread**,
**cross-book arbitrage is cross-venue arbitrage**, and **Kelly sizing is
position sizing under an estimated edge and its uncertainty**. It's a
research/backtesting pipeline — data ingestion, a probability model, a
backtest, and risk-managed sizing — built as a quant-trading portfolio piece,
not a betting product.

## Why this framing

A sportsbook quoting -110 on both sides of a game is doing exactly what a
market maker does quoting a bid-ask spread: pricing both outcomes above their
fair value so it profits regardless of the result. "Beating the market" here
means the same thing it means in any market — not being right about outcomes
in isolation, but being right *relative to the price*, consistently, in a way
that survives transaction costs (the vig) and shows up before the market has
fully priced in the same information (i.e., before the closing line).

## Pipeline

| Module | What it does |
|---|---|
| `src/probability.py` | American odds → implied probability; strips the vig from a two-way market to get the "fair" no-vig probability |
| `src/vig.py` | Vig (overround) per market, as a percentage — the book's built-in edge |
| `src/odds_client.py` | Wraps [The Odds API](https://the-odds-api.com/); saves timestamped raw pulls to `data/raw/` |
| `src/utils.py` | Team name normalization and timestamp helpers, so odds from different books join on the same game |
| `src/arbitrage.py` | Scans every cross-book pair on a game for a guaranteed-profit mispricing (summed implied probability < 1) |
| `src/model.py` | Logistic regression win-probability model, fit only on a chronological training period, evaluated on a never-touched held-out test period |
| `src/kelly.py` | Full and fractional (half-Kelly) position sizing from model probability and offered odds |
| `src/clv.py` | Closing line value: implied probability at bet time vs. at market close |
| `src/backtest.py` | Walks the held-out test period, sizes flagged bets, simulates bankroll with realistic frictions, logs CLV per bet |
| `src/baselines.py` | Naive predictors (trust the market, always home, coin flip) the model has to actually beat |
| `src/calibration.py` | Reliability diagrams and isotonic/Platt recalibration — is the model's raw probability trustworthy enough to size bets with? |
| `src/stats.py` | Bootstrap confidence intervals for backtest metrics on a small sample |

Two commands run it end-to-end:

```bash
python scripts/fetch_odds.py --sport basketball_nba   # pull live odds (requires an API key)
python scripts/run_backtest.py                        # train, backtest, write results/
```

## A note on data

No Odds API key or `nba_api`/`stats.nba.com` access was available while
building this (`stats.nba.com` times out from this environment — a
notoriously bot-hostile endpoint even under normal conditions). Rather than
leave modules untested, each one is validated against **clearly labeled
synthetic data** with a realistic generative structure:

- **Arbitrage scanner**: `data/raw/basketball_nba_synthetic_sample.json` — 6
  games x 4 books, each book pricing independently with realistic 4-6% vig.
  Result: 0 arbitrage opportunities, the expected outcome (see
  `notebooks/exploration.ipynb`).
- **Model + backtest**: `data/processed/nba_games_synthetic.csv` — 1200
  games over a synthetic season. Each team has a fixed hidden "true
  strength"; observed features (pace, ratings, recent win %, rest days) are
  *noisy proxies* for that strength, not the strength itself, and the market's
  opening line is a noisier estimate of the true win probability than its
  closing line — so there's a real, learnable, but imperfectly-exploitable
  edge, by design, rather than a hand-planted one.

**This must be swapped for a real Odds API pull and real historical
game data (via `nba_api` or similar) before any number below is a genuine
market-efficiency finding.** Right now these numbers demonstrate that the
pipeline — ingestion → no-vig probability → model → chronological split →
Kelly sizing → backtest → CLV — is correctly wired and leak-free, which is
what the code needs to be defensible in an interview regardless of the
dataset behind it.

## Results

**Average CLV: +2.64 percentage points, 90% bootstrap CI [+1.78, +3.49]pp —
the metric that matters more than raw ROI over a small sample.**

Backtest on the synthetic held-out test period (240 games, 94 bets flagged
by a 3-point model-vs-market edge threshold):

| Metric | Value |
|---|---|
| Avg CLV | **+2.64 pp** (90% CI [+1.78, +3.49]) |
| ROI | -5.5% (90% CI on mean per-bet return: [-1.26%, +2.11%]) |
| Hit rate | 29.8% |
| Sharpe-like ratio | 0.03 |
| Max drawdown | -49.7% |

![Bankroll and CLV](results/clv_plot.png)

Raw ROI was *negative* over this sample, but average CLV was *consistently
positive* (see the right-hand panel above — mostly green), and its confidence
interval clears zero even at only 94 bets — ROI's doesn't. That's the whole
point of leading with CLV instead of ROI: CLV is measured bet-by-bet against
where the market itself ultimately settled, so it's a much lower-variance read
on whether an edge is real than a single bankroll trajectory dominated by a
7-game losing streak.

### Is this actually beating the market, or just beating a coin flip?

The real bar in sports prediction isn't 50% — it's the market's own no-vig
price, which is already a strong predictor. Scored against naive baselines on
the identical held-out period:

| Predictor | Accuracy | Log loss | Brier score |
|---|---|---|---|
| Model (logistic regression) | 59.2% | 0.6614 | 0.2345 |
| Always bet market favorite (no-vig price) | 62.9% | 0.6637 | 0.2328 |
| Always bet home team | 49.2% | 0.6994 | 0.2531 |
| Coin flip | 44.2% | 0.6931 | 0.2500 |

The model edges out "just trust the market" on log loss but loses to it on
accuracy and Brier score — a wash, not a rout. That's the honest, expected
result of a small-scale market-efficiency test: a six-feature logistic
regression finding a large, dominant edge over the market's own price would be
the red flag, not this. The genuine (if modest) finding here is in CLV, not in
beating the market's point predictions outright.

### Does the model's probability mean what it says? (calibration)

The flagged bets above skew toward underdogs (mean model probability 0.44 vs.
mean no-vig market probability 0.33) — a plausible sign of an imperfectly
calibrated model systematically "finding value" on long shots. The natural
next question: does recalibrating the model's probabilities fix that?

Fit a fresh model on 85% of the training period, held out the remaining 15%
(144 games) purely for calibration, then compared raw vs. recalibrated
probabilities on the same test period — both isotonic regression (flexible,
but sklearn's own docs say it needs ~1000+ calibration samples to avoid
overfitting the mapping itself) and Platt/sigmoid scaling (2 parameters, far
less data-hungry):

![Reliability diagram](results/calibration_plot.png)

Neither method meaningfully improved Brier score (raw 0.2353 vs. isotonic
0.2424 vs. Platt 0.2368) — the raw model was already reasonably calibrated,
and isotonic's curve is visibly noisier than Platt's, exactly what you'd
expect from fitting a flexible nonparametric mapping on only 144 calibration
games. **This is the honest result, not the one that was planned going in** —
the original hypothesis was that calibration would fix the underdog-skewed
bet selection; on this dataset it mostly didn't, and that's worth reporting
rather than quietly dropping.

What the calibration re-run *did* catch: the Platt-calibrated backtest posted
a flashy **+33% ROI** — until the pipeline's own tail-risk diagnostic
(`top_bet_pnl_share` in `backtest.py`) flagged that a single long-odds bet
contributed **136% of total profit**, meaning the strategy was net negative on
every other bet combined. One lucky longshot, not a validated edge. This is
exactly the failure mode CLV is meant to catch and ROI is not, and now the
backtest engine checks for it automatically on every run.

Full report (regenerated by `scripts/run_backtest.py`, including the baseline
and calibration tables above): `results/backtest_report.md`.

## Design decisions worth defending in an interview

- **Chronological train/test split, never violated.** `model.chronological_split`
  sorts by date and cuts a fixed fraction; `backtest.run_backtest` only ever
  sees the held-out period. This is the first thing any quant interviewer
  will probe, and the answer here is structural, not a promise.
- **Half-Kelly, not full Kelly.** Full Kelly is provably optimal for
  long-run log growth *given a correctly specified edge*, which a small
  logistic regression on noisy features will never have exactly. Half-Kelly
  trades some growth for a large reduction in variance and drawdown risk —
  a deliberate, statable risk-management choice, not an arbitrary default.
- **A hard bet-size cap independent of Kelly's output.** Kelly can size up
  arbitrarily large positions when it estimates a large edge (exactly what
  happened with the model's underdog-skewed picks, see Results); real books
  limit or cut off sharp bettors long before that, so the backtest enforces
  the same constraint rather than reporting a number that isn't achievable
  in practice. Max drawdown was still -49.7% at a 5% cap and a 29.8% hit
  rate — a reminder that "capped" isn't "safe," just safer.
- **Simulated slippage.** The odds used for sizing are nudged slightly
  against the bettor before the bet is "placed," since the price you decide
  on and the price you actually get rarely match exactly in a fast-moving
  market.
- **CLV as the headline metric, not ROI.** See Results above.
- **A tail-risk check the backtest engine runs automatically.**
  `top_bet_pnl_share` reports what fraction of total profit came from the
  single best bet. It's what caught the Platt-calibrated backtest's fake
  +33% ROI (see Results) — a strategy whose entire profit sits in one
  outlier hasn't demonstrated an edge, and a small backtest sample can hide
  that without a check like this.
- **Calibration was tested, not assumed.** Both isotonic and Platt scaling
  were tried and compared against the raw model with a reliability diagram
  and Brier score, on a calibration set kept separate from training and
  test. Neither clearly won here — reported as-is rather than picking
  whichever result looked better.

## What this deliberately doesn't do

- No live or real-money betting integration — this is a research/backtesting
  project, not a betting app.
- No database. Flat CSV/JSON under `data/` is sufficient and easier to
  explain in an interview than a schema; a Postgres/Timescale backend would
  be the natural next step if this needed to run continuously rather than
  as a research pipeline, but that's over-engineering for what this is today.
- No deep learning. Logistic regression is the right level of sophistication
  for six engineered features and ~1000 games — a simple, fully-understood
  model beats a complex one you can't defend when asked "why this
  architecture?"

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

92 tests across `probability`, `vig`, `arbitrage`, `kelly`, `clv`, `model`,
`backtest`, `odds_client`, `utils`, `baselines`, `calibration`, and `stats` —
including hand-checked example values for every formula in the spec, a
planted arbitrage the scanner must detect, a chronological split the model
must never leak across, and a tail-dominated P&L case the backtest's
concentration check must flag.

## Repo structure

```
sports-market-efficiency/
├── config.py                  # env vars, paths, API defaults
├── src/                       # probability, vig, arbitrage, model, kelly, clv, backtest,
│                              #   odds_client, utils, baselines, calibration, stats
├── scripts/
│   ├── fetch_odds.py          # pull live odds -> data/raw/
│   └── run_backtest.py        # train + backtest -> results/
├── data/
│   ├── raw/                   # unmodified API pulls (gitignored, except the synthetic sample)
│   └── processed/             # cleaned, joined datasets ready for analysis
├── notebooks/exploration.ipynb
├── tests/
└── results/
    ├── backtest_report.md
    ├── clv_plot.png
    └── calibration_plot.png
```
