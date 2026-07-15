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

**Average CLV: +2.64 percentage points — the metric that matters more than
raw ROI over a small sample.**

Backtest on the synthetic held-out test period (240 games, 94 bets flagged
by a 3-point model-vs-market edge threshold):

| Metric | Value |
|---|---|
| Avg CLV | **+2.64 pp** |
| ROI | -5.5% |
| Hit rate | 29.8% |
| Sharpe-like ratio | 0.03 |
| Max drawdown | -49.7% |

![Bankroll and CLV](results/clv_plot.png)

Raw ROI was *negative* over this sample, but average CLV was *consistently
positive* (see the right-hand panel above — mostly green). That's the whole
point of leading with CLV instead of ROI: 94 bets is nowhere near enough for
ROI to distinguish skill from variance — a 7-game losing streak alone can
swing a half-Kelly bankroll by double digits — but CLV is measured bet-by-bet
against where the market itself ultimately settled, so it's a much lower-
variance read on whether the model's edge is real. Here it says yes, modestly,
even though the realized bankroll says no over this particular stretch.

Two things worth flagging honestly rather than hiding:
1. **The flagged bets skew toward underdogs** (mean model probability 0.44 vs.
   mean no-vig market probability 0.33) — a known failure mode where an
   imperfectly calibrated model systematically "finds value" on long shots.
   A more mature version of this project would calibrate the model
   (e.g. Platt scaling) before using its raw probabilities for edge detection.
2. **Max drawdown (-49.7%) is large** for half-Kelly with a 5% bet cap. It's a
   direct consequence of the low hit rate above and a reminder of why the
   backtest caps bet size regardless of Kelly's raw output — real books also
   limit sharp bettors, and real bankrolls can't absorb what the math alone
   would size up to.

Full report (regenerated by `scripts/run_backtest.py`): `results/backtest_report.md`.

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
  happened with the underdog bias above); real books limit or cut off
  sharp bettors long before that, so the backtest enforces the same
  constraint rather than reporting a number that isn't achievable in practice.
- **Simulated slippage.** The odds used for sizing are nudged slightly
  against the bettor before the bet is "placed," since the price you decide
  on and the price you actually get rarely match exactly in a fast-moving
  market.
- **CLV as the headline metric, not ROI.** See Results above.

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

67 tests across `probability`, `vig`, `arbitrage`, `kelly`, `clv`, `model`,
`backtest`, `odds_client`, and `utils` — including hand-checked example
values for every formula in the spec, a planted arbitrage the scanner must
detect, and a chronological split the model must never leak across.

## Repo structure

```
sports-market-efficiency/
├── config.py                  # env vars, paths, API defaults
├── src/                       # probability, vig, arbitrage, model, kelly, clv, backtest, odds_client, utils
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
    └── clv_plot.png
```
