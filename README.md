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
| `src/mlb_stats_client.py` | Wraps the free, public MLB Stats API (`statsapi.mlb.com`) for real completed-game results |
| `src/mlb_features.py` | Point-in-time (no-lookahead) feature engineering on real MLB games — see "Real MLB model" below |

Three commands run it end-to-end:

```bash
python scripts/fetch_odds.py --sport baseball_mlb   # pull live odds (needs ODDS_API_KEY in .env)
python scripts/run_backtest.py                      # synthetic NBA: train, backtest, write results/
python scripts/train_mlb_model.py                   # real MLB: fetch, train, evaluate, write results/
```

## A note on data

**Two of the three real-data constraints from earlier are now resolved; one
is a genuine, structural limit of the free tier, not a gap in the code:**

| Piece | Status |
|---|---|
| Live odds (arbitrage scanner) | **Real.** The Odds API, verified with a real key (`.env` + `ODDS_API_KEY`). |
| Win-probability model | **Real.** Trained on real MLB games from `statsapi.mlb.com` — see "Real MLB model" below. |
| Backtest / CLV / Kelly sizing | **Still synthetic**, and will stay that way on the free tier — see why below. |

NBA is off-season as of this writing (the regular season doesn't resume
until October, so `basketball_nba` returns nothing from the odds endpoint
right now), and `stats.nba.com`/`nba_api` remain unreachable from this
environment regardless (a notoriously bot-hostile endpoint even normally).
MLB is in-season and both sides of it are reachable: The Odds API for real
odds, and the free, public `statsapi.mlb.com` (no key required) for real
team stats and outcomes. So the default sport switched to `baseball_mlb`,
and the model now trains on real MLB games rather than synthetic NBA ones.

**Why the backtest/CLV/Kelly sizing is still synthetic, and can't just be
"fixed" the same way:** those need real *historical* odds — the price at
the moment a bet would have been placed, and the closing price before first
pitch, for games that already happened — to compute real Kelly stakes and
real CLV. The Odds API's historical odds endpoint requires a paid plan
(confirmed directly: a real request against it returns
`HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN`). Real *current* odds don't help
here — you can't backtest against a price that hasn't closed yet. So
`data/processed/nba_games_synthetic.csv` (1200 games, hidden per-team "true
strength," noisy observed features, and an opening line noisier than the
closing line by design) remains the dataset behind `scripts/run_backtest.py`,
`results/backtest_report.md`, and the Results section below. The honest
path to a fully real backtest is either a paid odds plan, or collecting real
opening/closing lines forward in time from today rather than retroactively
— not something more code alone can solve on this tier.

## Live data: verified against a real Odds API pull

A real pull (`baseball_mlb`, `2026-07-16`) returned 4 games:

| Matchup | Books quoting |
|---|---|
| New York Mets @ Philadelphia Phillies | 9 |
| Pittsburgh Pirates @ Cleveland Guardians | 2 |
| Chicago White Sox @ Toronto Blue Jays | 2 |
| San Diego Padres @ Kansas City Royals | 1 |

Running `src.arbitrage.scan_games_for_arbitrage` on this real pull: **0
arbitrage opportunities**, consistent with the synthetic-data result — and
this time not a hand-tuned outcome, an actual live market. Vig by book on
this pull, low to high:

| Book | Avg vig |
|---|---|
| BetUS | 2.22% |
| BetOnline.ag / LowVig.ag | 2.44% |
| FanDuel | 3.98% |
| MyBookie.ag | 4.01% |
| BetRivers | 4.07% |
| Bovada | 4.47% |
| DraftKings | 4.62% |
| BetMGM | 5.07% |

Small sample (1-4 games per book), so read this as a snapshot, not a firm
ranking — but directionally it lines up with priors about this market: the
offshore/low-margin books (BetUS, BetOnline, LowVig — its name is not a
coincidence) price tighter than the mainstream US books. This is real data
freely pulled and analyzed with the same code that runs on the synthetic
sample above; the raw pull itself isn't committed to the repo (`data/raw/*`
stays gitignored except the synthetic sample) since redistributing a paid
data provider's live feed isn't something to put in a public repo, even on
the free tier — pull your own with `scripts/fetch_odds.py`.

## Real MLB model: an honest null result

`scripts/train_mlb_model.py` fetches every completed 2026 regular-season MLB
game to date from `statsapi.mlb.com` (1,444 games), builds features with
strict point-in-time discipline (`src/mlb_features.py` — a team's rolling
stats update *after* its features are read for the current game, never
before; unit-tested directly, including a test that a later game's outcome
can never change an earlier row), and trains the same kind of logistic
regression as the NBA model — but on entirely real teams, dates, and final
scores.

Home team: Team A always, since real games have a fixed home team (unlike
the synthetic dataset, which randomized it so `home_flag` could vary as an
explicit feature) — home advantage here is absorbed into the model's
intercept instead, which is the standard, correct way to handle it when the
data is real. Features: rolling runs scored/allowed, season run
differential, last-10-games win rate, and rest days, all as home-minus-away
diffs computed only from games strictly before the one being predicted.

**Result on 286 real held-out games (2026-06-22 to 2026-07-12):**

| Predictor | Accuracy | Log loss | Brier score |
|---|---|---|---|
| Model (logistic regression) | 49.0% | 0.6958 | 0.2513 |
| Always predict home team (training home-win rate) | 47.9% | 0.6977 | 0.2523 |
| Coin flip | 47.9% | 0.6931 | 0.2500 |

Bootstrap 90% CI on test accuracy: [44.1%, 53.5%] — straddles a coin flip.
90% CI on the model's per-game log-loss improvement over the home-rate
baseline: [-0.0040, +0.0075] — includes zero. **The model does not show a
statistically distinguishable edge over naive baselines on this real,
held-out sample.** This is the actual result, not a caveat to explain away.

The likely reason is a real, specific, and known one: single-game MLB
outcomes are driven heavily by the **starting pitcher matchup**, an effect
team-level rolling stats (runs scored/allowed, recent win rate) cannot
capture at all — two teams with identical recent form can have wildly
different win probabilities on a given day depending on who's on the mound.
NBA's five-man rotations dilute any single-player effect far more than
baseball's single starting pitcher does, which is a big part of why the
NBA-shaped feature set (team-level stats only) transfers poorly to MLB.
Starting-pitcher stats (ERA, FIP, recent starts) are the obvious next
feature to add and are available from the same free API — not built here,
to avoid tuning the model against the one held-out sample used to report
it, which would defeat the purpose of a held-out sample in the first place.

Full report: `results/mlb_model_report.md`, reliability diagram:
`results/mlb_reliability_plot.png` (isotonic calibration is visibly noisier
than Platt here too — 172 real calibration games, same small-sample pattern
as the synthetic NBA case, now confirmed on a second, independent dataset).

## Results: synthetic NBA backtest

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
- **Point-in-time feature engineering, tested directly.** `mlb_features.py`
  computes every feature from a team's games strictly before the one being
  predicted; `test_mlb_features.py` includes a test that literally changes
  a later game's score and asserts every earlier row is byte-identical. A
  model can look great in-sample and be useless in production if this is
  wrong, and "we tested it" is a much stronger claim than "we wrote it
  carefully."
- **A null result was reported as a null result.** The real MLB model
  doesn't beat naive baselines on held-out data (see above), and that's in
  the README, not quietly dropped in favor of the more flattering synthetic
  numbers. A portfolio project that only ever reports wins is a portfolio
  project that hasn't been checked by anyone yet.

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

102 tests across `probability`, `vig`, `arbitrage`, `kelly`, `clv`, `model`,
`backtest`, `odds_client`, `utils`, `baselines`, `calibration`, `stats`,
`mlb_stats_client`, and `mlb_features` — including hand-checked example
values for every formula in the spec, a planted arbitrage the scanner must
detect, a chronological split the model must never leak across, a
tail-dominated P&L case the backtest's concentration check must flag, and a
point-in-time correctness check that a later real game can never change an
earlier one's features.

## Repo structure

```
sports-market-efficiency/
├── config.py                  # env vars, paths, API defaults
├── src/                       # probability, vig, arbitrage, model, kelly, clv, backtest,
│                              #   odds_client, utils, baselines, calibration, stats,
│                              #   mlb_stats_client, mlb_features
├── scripts/
│   ├── fetch_odds.py          # pull live odds -> data/raw/
│   ├── run_backtest.py        # synthetic NBA: train + backtest -> results/
│   └── train_mlb_model.py     # real MLB: fetch + train + evaluate -> results/
├── data/
│   ├── raw/                   # unmodified API pulls (gitignored, except the synthetic sample)
│   └── processed/             # nba_games_synthetic.csv, mlb_games_real.csv
├── notebooks/exploration.ipynb
├── tests/
└── results/
    ├── backtest_report.md          # synthetic NBA backtest
    ├── clv_plot.png
    ├── calibration_plot.png
    ├── mlb_model_report.md         # real MLB model evaluation
    └── mlb_reliability_plot.png
```
