# Sports Market Efficiency & Arbitrage Engine

Sportsbook odds are prices in a prediction market. This project treats them
that way: **implied probability is price**, **vig is bid-ask spread**,
**cross-book arbitrage is cross-venue arbitrage**, and **Kelly sizing is
position sizing under an estimated edge and its uncertainty**. It's a
research/backtesting pipeline — data ingestion, a probability model, a
backtest, and risk-managed sizing — built as a quant-trading portfolio piece,
not a betting product.

*Built with Claude Code as a development tool: I directed scope and data
decisions, reviewed and approved each phase, and made the calls at every
decision point; Claude implemented most of the code. I can walk through and
defend every module.*

## TL;DR

- **The pipeline is real end-to-end**, not a toy: a live Odds API key, a real
  arbitrage scan against actual multi-book prices, and a real predictive
  model trained on 1,400+ real MLB games — not just a notebook that runs
  once on a CSV someone else cleaned.
- **The real model doesn't beat a coin flip on held-out data (49.0% vs.
  47.9%), reported as the finding, not tuned away** — with a specific,
  testable hypothesis for why (team-level stats can't see the
  starting-pitcher effect). That hypothesis was then actually tested by
  adding real pitcher features on a fresh, never-before-seen data pull: a
  small, consistent, positive lift across every metric, still not
  statistically significant on 261 games — exactly the calibrated
  expectation stated *before* running the experiment, not after. Predicting
  the size of an effect and then confirming it is a stronger signal than
  either the null result or the improvement alone.
- **A tail-risk check the backtest runs automatically flags its own headline
  number**: the primary backtest's ROI reads as a nominally positive +6.3%,
  but `top_bet_pnl_share` shows **634% of that profit came from a single
  bet** — not a repeatable edge, a lucky outlier. The Platt-calibrated
  variant shows the same pattern even more starkly (+46% ROI, 101%
  concentration). Built the check *because* it keeps catching this, not as
  a box to tick.
- **CLV, not ROI, is the headline metric precisely because ROI is this
  fragile** — on the synthetic backtest, average CLV is positive with a
  bootstrap confidence interval that clears zero (+2.64pp, 90% CI [+1.78,
  +3.49]) and, unlike ROI, doesn't hinge on any single bet's outcome. That
  gap *is* the thesis of the project, demonstrated with real numbers, not
  just asserted.
- **The isotonic-vs-Platt calibration finding reproduced independently on
  two unrelated datasets** (synthetic NBA, real MLB) — same pattern both
  times, which is what makes it a real finding about small calibration sets
  rather than a one-off fluke worth a footnote.
- **Slate-level portfolio risk, a market-making mode, and a forward-collecting
  real-CLV pipeline were added on top of the original backtest** — the first
  two extend the risk/pricing logic beyond a single bet in isolation; the
  third turns the synthetic-backtest limitation above into a real, growing
  dataset instead of just disclosing around it. All three are new
  infrastructure with test coverage, not new results — see "Forward-collecting
  real CLV," "Slate-level portfolio risk," and "Market making" below for what
  each actually does and, for the paper-trading pipeline, an explicit
  no-results-yet status.
- Full detail on every point above is below, in "Real MLB model" and
  "Results: synthetic NBA backtest."

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
| `src/mlb_stats_client.py` | Wraps the free, public MLB Stats API (`statsapi.mlb.com`) for real completed-game results and starting-pitcher game logs |
| `src/mlb_features.py` | Point-in-time (no-lookahead) team-level feature engineering on real MLB games — see "Real MLB model" below |
| `src/mlb_pitcher_features.py` | Point-in-time rolling ERA/K9 for starting pitchers, additive to the team features |
| `src/portfolio_risk.py` | Slate-level exposure cap — scales every bet on an over-limit day down proportionally, on top of the existing per-bet Kelly cap |
| `src/market_maker.py` | Quotes a two-sided line from a probability estimate (the sportsbook's side of the trade, not the bettor's) and simulates inventory-skew repricing against simulated flow |
| `src/paper_trading.py` | Forward-collecting pipeline: line-shops a live multi-book pull against the model, logs a paper bet at today's real price, and reconciles it against the real closing price and result once the game finishes |

Five commands run it end-to-end:

```bash
python scripts/fetch_odds.py --sport baseball_mlb   # pull live odds (needs ODDS_API_KEY in .env)
python scripts/run_backtest.py                      # synthetic NBA: train, backtest, write results/
python scripts/train_mlb_model.py                   # real MLB: fetch, train, evaluate, write results/
python scripts/collect_paper_trades.py              # real MLB: log today's live-odds edges as paper trades
python scripts/reconcile_paper_trades.py            # real MLB: settle paper trades, write results/paper_trading_report.md
```

## A note on data

| Piece | Status |
|---|---|
| Live odds (arbitrage scanner) | **Real.** The Odds API, live key. |
| Win-probability model | **Real.** Trained on real MLB games (`statsapi.mlb.com`). |
| Backtest / CLV / Kelly sizing (`scripts/run_backtest.py`) | **Synthetic**, structurally — see why. |
| Paper-trading CLV (`scripts/collect_paper_trades.py` + `reconcile_paper_trades.py`) | **Real, forward-collected**, growing one slate at a time — see below. |

NBA is off-season (nothing live until October) and `stats.nba.com`/`nba_api`
are unreachable from this environment anyway, so the default sport is
`baseball_mlb`: The Odds API covers real odds, and the free `statsapi.mlb.com`
covers real team stats and outcomes.

The backtest/CLV/Kelly sizing can't follow the same path *retroactively*: they need real
*historical* odds (the price at bet time and at close, for games already
played), and The Odds API's historical endpoint requires a paid plan
(confirmed directly — a real request returns
`HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN`). Real *current* odds don't help
retroactively. So `data/processed/nba_games_synthetic.csv` (hidden per-team
"true strength," an opening line noisier than the closing line, by design)
still backs `scripts/run_backtest.py` and the backtest results below. Both
synthetic datasets are generated by committed scripts
(`scripts/generate_synthetic_nba_data.py`,
`scripts/generate_synthetic_arbitrage_sample.py`, both seeded and
reproducing the committed files byte-for-byte) rather than taken on faith.

What *does* fix it without a paid plan: stop trying to buy history and collect it
forward instead. See "Forward-collecting real CLV" below.

## Forward-collecting real CLV (paper trading)

`src/paper_trading.py` + `scripts/collect_paper_trades.py` / `reconcile_paper_trades.py`
turn the synthetic-backtest limitation above into a real, growing dataset instead
of working around it with more synthetic data:

1. **`collect_paper_trades.py`** (run once a day, or a few times a day closer to
   first pitch) fits a fresh model on every completed MLB game so far, pulls a live
   multi-book odds snapshot, and for each of today's games: computes the model's
   probability from current point-in-time features
   (`mlb_features.compute_current_features` — the live counterpart to
   `build_features`, replaying the same rolling-state logic for a matchup that
   hasn't been played yet instead of one that's already a row), line-shops across
   every book for the best price (`paper_trading.find_best_price`), and compares
   the model to the **consensus no-vig price averaged across books**
   (`consensus_no_vig_prob`) rather than any single book's line. This prints a live
   edge-monitor report and logs any bet clearing `edge_threshold` to
   `data/paper_trades/mlb_paper_trades.csv` at today's real price — nothing is
   actually wagered, but the price, the model probability, and the size are all
   real and captured before the outcome exists.
2. **`reconcile_paper_trades.py`** (run after games finish) fetches the real result
   from `statsapi.mlb.com` and a fresh odds pull for a closing-price proxy, computes
   realized CLV and P&L per bet with `src/clv.calculate_clv`, and writes
   `results/paper_trading_report.md` with the same bootstrap-CI treatment as the
   other reports.

**Why this is a stronger claim than a backtest, not a weaker one:** leakage is
structurally impossible here — the bet is written to the ledger before the game is
played, not selected afterward from a dataset where the outcome was always
knowable. The honest tradeoff is sample size: it only grows by one slate a day, so
it takes weeks to accumulate a sample worth reporting a confidence interval on.

**Status: infrastructure only, no results yet.** This was just built and hasn't
run forward for any length of time — `results/paper_trading_report.md` doesn't
exist yet because there's nothing real to put in it. Reporting a number now would
mean fabricating it. Run `collect_paper_trades.py` daily for a few weeks, then
`reconcile_paper_trades.py`, and the report will reflect whatever actually happened,
including a null result if that's what it is — same discipline as the rest of this
project. The closing-odds proxy also has a stated limitation: it's the last price
observed on whatever run happens to catch a game shortly before first pitch, not
the literal final tick — see the docstring in `reconcile_paper_trades.py`.

## Live data: verified against a real Odds API pull

A real pull (`baseball_mlb`, `2026-07-16`) returned 4 games across up to 9
books each. `arbitrage.scan_games_for_arbitrage` found **0 opportunities** —
consistent with the synthetic result, this time on an actual live market.
Vig by book ranged 2.2% (BetUS) to 5.1% (BetMGM), with offshore/low-margin
books (BetUS, BetOnline, LowVig — no coincidence) pricing tighter than
mainstream US books — directionally sensible on a small sample (1-4 games
per book). The raw pull isn't committed (`data/raw/*` stays gitignored
except the synthetic sample) — redistributing a paid provider's live feed
doesn't belong in a public repo even on the free tier. Pull your own with
`scripts/fetch_odds.py`.

## Real MLB model: an honest null result

`scripts/train_mlb_model.py` fetches every completed 2026 game from
`statsapi.mlb.com` (1,445 games as of this run — it's a live pull, so the
count grows daily) and trains a logistic regression on
point-in-time features — rolling runs scored/allowed, season run
differential, last-10 win rate, and rest days, all home-minus-away diffs
computed only from games strictly before the one predicted
(`mlb_features.py`, tested directly: a later game's score can never change
an earlier row). Home team is fixed by the real data, so home advantage is
absorbed into the model's intercept rather than a separate feature.

**Result on 286 real held-out games:**

| Predictor | Accuracy | AUC | Log loss | Brier score |
|---|---|---|---|---|
| Model (logistic regression) | 49.0% | 0.560 | 0.6957 | 0.2513 |
| Always predict home team (training home-win rate) | 47.9% | 0.500 | 0.6976 | 0.2522 |
| Coin flip | 47.9% | 0.500 | 0.6931 | 0.2500 |

Bootstrap 90% CI on accuracy: [44.1%, 53.8%] — straddles a coin flip. CI on
log-loss improvement over the home-rate baseline: [-0.0038, +0.0075] —
includes zero. **No statistically distinguishable edge over naive baselines
on accuracy or log loss.** The one nuance: AUC (0.560 vs. 0.500) shows the
model isn't *pure* noise — it has some real, weak ability to rank games by
who's more likely to win — it just isn't strong enough to survive becoming
a threshold decision or a calibrated probability. That distinction (ranking
power vs. decision-useful probability) is worth being able to explain on
its own; a lot of "my model doesn't work" post-mortems stop at accuracy and
miss it. Likely reason for the ceiling: single-game MLB outcomes are driven
heavily by the **starting pitcher matchup**, which team-level rolling stats
can't see at all — unlike NBA, where a five-man rotation dilutes any one
player's effect. Starting-pitcher stats (ERA, FIP) are the obvious next
feature, available from the same free API — not added here, to avoid tuning
the model against the one held-out sample used to report it.

Full report: `results/mlb_model_report.md` (isotonic calibration is
visibly noisier than Platt here too, on an independent 172-game calibration
set — same small-sample pattern as the synthetic case, now seen twice).

### Does adding starting-pitcher features help?

Tested it (`scripts/train_mlb_model_with_pitching.py`, `src/mlb_pitcher_features.py`):
rolling ERA and K/9 (strikeouts per 9 innings — a "stuff" metric that per
DIPS theory is less noisy over a small sample than ERA, which is heavily
luck/defense-influenced) for both starting pitchers, point-in-time correct
the same way as the team features. Both models below are trained and
evaluated on the *identical* set of games, so the comparison isolates the
effect of the new features — and this uses a **fresh data pull**, not the
286 games already reported on above, so the "did it help" question isn't
answered against a test set that's already been looked at.

| Predictor | Accuracy | AUC | Log loss | Brier score |
|---|---|---|---|---|
| Team-only | 52.9% | 0.569 | 0.6886 | 0.2477 |
| Team + starting pitcher | 53.3% | 0.574 | 0.6876 | 0.2472 |

Small, consistent, positive lift across every metric — and a bootstrap 90%
CI on the log-loss improvement of [-0.0021, +0.0041] still includes zero.
Not statistically significant on 261 held-out games. This is exactly what
was predicted *before* running it: pitcher features should help some
(baseball's low events-per-game count means even a real signal is hard to
detect over one test period), but MLB has a structurally lower single-game
predictability ceiling than NBA regardless of feature set, so a small,
inconclusive-on-this-sample lift — not a transformation — is the correctly
calibrated expectation, and that's what showed up. Full report:
`results/mlb_pitcher_features_report.md`.

## Results: synthetic NBA backtest

Backtest on the synthetic held-out test period (240 games, 94 bets flagged
by a 3-point model-vs-market edge threshold), run through the slate-aware
engine described in "Slate-level portfolio risk" below (bets on the same day
are batched and capped together, not sized sequentially one game at a time):

| Metric | Value |
|---|---|
| Avg CLV | **+2.64 pp** (90% CI [+1.78, +3.49]) |
| ROI | +6.3% (90% CI on mean per-bet return: [-1.16%, +2.15%]) |
| Hit rate | 29.8% |
| Sharpe-like ratio | 0.042 |
| Max drawdown | -37.0% |
| Top-bet P&L share | **634%** |

![Bankroll and CLV](results/clv_plot.png)

**Read the ROI number and the top-bet P&L share together, not the ROI number
alone.** +6.3% looks like a win. `top_bet_pnl_share` says 634% of that profit
came from a single bet — the rest of the book is a net loser, and one long-odds
result happened to land. That's the tail-risk check working exactly as
designed: it doesn't just catch a *negative* result dressed up as positive,
it catches a *positive* result that isn't real either. Avg CLV doesn't have
this problem — no single bet's outcome can move a metric that's defined
independently of whether any individual bet won or lost, which is the whole
argument for leading with it instead of ROI.

**Model vs. naive baselines**, same held-out period — the real bar isn't
50%, it's the market's own no-vig price:

| Predictor | Accuracy | AUC | Log loss | Brier score |
|---|---|---|---|---|
| Model (logistic regression) | 59.2% | 0.637 | 0.6614 | 0.2345 |
| Always bet market favorite (no-vig price) | 62.9% | 0.672 | 0.6637 | 0.2328 |
| Always bet home team | 49.2% | 0.490 | 0.6994 | 0.2531 |
| Coin flip | 44.2% | 0.500 | 0.6931 | 0.2500 |

The market wins on accuracy, AUC, and Brier; the model edges it slightly on
raw log loss. Bootstrap 90% CI on that log-loss gap: +0.0024, 90% CI
[-0.0301, +0.0335] — **includes zero**. So even the one metric the model
nominally wins on isn't statistically distinguishable from parity with the
market on this sample. That's the honest reading, not "a wash, not a rout"
— the model doesn't demonstrably beat the market's own price on point
predictions at all. It only shows up in CLV (above), which is a different,
lower-variance question: not "was the model more accurate," but "did the
line move in the model's favor after betting." A six-feature model
dominating the market's own price on accuracy would be the red flag; not
beating it here is the expected result.

**Calibration.** The flagged bets skew toward underdogs (mean model
probability 0.44 vs. mean market 0.33) — a plausible sign of "finding
value" on long shots from imperfect calibration. Fit isotonic and Platt
recalibrators on a held-out 144-game calibration set and re-evaluated:

![Reliability diagram](results/calibration_plot.png)

Neither meaningfully improved Brier score (raw 0.2353, isotonic 0.2424,
Platt 0.2368) — the raw model was already reasonably calibrated. Not the
result the hypothesis predicted, reported anyway. What the re-run *did*
catch, again: the Platt-calibrated backtest posted a flashy **+46% ROI**,
until `top_bet_pnl_share` showed **101% of that profit came from one
long-odds bet** — one lucky longshot, not an edge, the same pattern the
primary backtest's own 634% already showed above. Full report:
`results/backtest_report.md`.

## Slate-level portfolio risk

`src/portfolio_risk.py` + `src/backtest.py` address a gap the original single-bet
Kelly sizing had: `max_bet_pct` caps any *one* bet, but says nothing about the sum
of everything flagged on the same day. A real MLB slate can have 10-15 games; if
each independently-capped bet gets sized as if it were the only bet that day, a
day with many flagged edges can commit a much larger fraction of the bankroll than
any single-position limit implies — exactly the kind of aggregate exposure a real
desk caps at the book/slate level, not just per position.

`run_backtest` now batches every game by date, sizes each flagged bet's stake
against that day's *starting* bankroll (all of a day's bets are decided and placed
before any of that day's results are known — they aren't sequentially compounded
against each other), and then `apply_slate_exposure_cap` scales every stake on an
over-limit day down proportionally, preserving the bigger-edge-gets-a-bigger-stake
ordering rather than dropping bets. Default cap: 20% of bankroll per slate, on top
of the existing 5%-per-bet cap. `tests/test_backtest.py::TestSlateExposureCap`
covers both the binding and non-binding cases directly.

## Market making

Every other module in this repo *consumes* a market's price — asks "does the
model disagree with what's quoted enough to bet." `src/market_maker.py` inverts
the question, because that's the other side of this trade and the one an actual
prop or sportsbook trading desk sits on: given a private probability estimate,
**set** a two-sided price.

- `quote_from_probability` turns a true win-probability into two-sided American
  odds embedding a target vig, via `probability.add_vig` — the literal inverse of
  `probability.remove_vig`, which every other module uses to strip vig back out.
- `simulate_flow_and_reprice` simulates a stream of bettors with noisy private
  beliefs trading against that quote, tracks the maker's signed inventory (net
  exposure to one side), and reprices — skews the line to make an overbet side
  less attractive — once inventory crosses a threshold, the standard market-making
  response to one-sided flow. A fully-informed, symmetric-belief population
  self-corrects back to the fair price after each reprice
  (`test_deterministic_symmetric_flow_self_corrects`, hand-verified with a fixed
  seed) rather than drifting — a real, if simplified, property of quoting from the
  same information your counterparties have, not an artifact of the toy setup.

Deliberately kept small: one game, one round of flow, a linear reprice rule. The
point is demonstrating the quote → inventory → skew mechanism directly, not
building a production market-making engine.

## Design decisions worth defending in an interview

- **Chronological train/test split, never violated.** Structural, not a
  promise — `chronological_split` sorts by date and cuts a fixed fraction;
  the backtest only ever sees what falls after that cut.
- **Half-Kelly, not full Kelly.** Full Kelly is optimal only for a
  *correctly specified* edge, which a small logistic regression never has
  exactly. Trading some growth for a lot less variance is a deliberate,
  statable choice.
- **A hard bet-size cap independent of Kelly's output.** Kelly sizes up
  arbitrarily on a large estimated edge (see the underdog skew above); real
  books cap sharp bettors long before that, so the backtest does too. Still
  a -37.0% max drawdown at a 5% per-bet cap — "capped" isn't "safe," just safer.
- **A slate-level exposure cap on top of the per-bet cap.** A per-bet cap alone
  says nothing about what a day with many flagged edges commits in aggregate;
  `apply_slate_exposure_cap` limits the whole day's exposure, not just any one
  bet in it — see "Slate-level portfolio risk" above.
- **Simulated slippage.** Odds are nudged against the bettor before a bet
  is "placed" — the price you decide on and the price you get rarely match.
- **CLV as the headline metric, not ROI, backed by a tail-risk check.** The
  check (`top_bet_pnl_share`) exists because it keeps catching something real —
  even the primary backtest's positive-looking +6.3% ROI turned out to be 634%
  concentrated in one bet (above) — not because it's a box to tick.
- **Calibration tested, not assumed** — isotonic and Platt both fit and
  compared against the raw model, neither cherry-picked when it didn't win.
- **Every "the model beats X" claim carries a significance test, not just a
  point estimate.** The NBA model's raw log-loss edge over the market
  favorite looked real (0.6614 vs. 0.6637) until a bootstrap CI on the gap
  turned out to include zero — a point estimate alone would have overstated
  it. AUC is reported alongside accuracy/log-loss/Brier for the same
  reason: a model can have real ranking signal (MLB: AUC 0.560) without it
  surviving as a usable accuracy or calibration edge, and reporting only
  accuracy would have hidden that distinction entirely.
- **Point-in-time feature engineering, tested directly.** `test_mlb_features.py`
  literally changes a later game's score and asserts every earlier row is
  unchanged — "we tested it" beats "we wrote it carefully."
- **A null result reported as a null result.** The MLB model's non-finding
  is in the README, not dropped for the more flattering synthetic numbers.

## What this deliberately doesn't do

- No real-money betting integration — `paper_trading.py` logs what a bet *would*
  have been, at a real price, for later reconciliation; nothing is ever actually
  wagered. This is a research/backtesting project, not a betting app.
- No database. Flat CSV/JSON under `data/` is sufficient and easier to
  explain in an interview than a schema; a Postgres/Timescale backend would
  be the natural next step if this needed to run continuously rather than
  as a research pipeline, but that's over-engineering for what this is today.
- No deep learning. Logistic regression is the right level of sophistication
  for six engineered features and ~1000 games — a simple, fully-understood
  model beats a complex one you can't defend when asked "why this
  architecture?"
- No production market-making engine. `market_maker.py` demonstrates the
  quote/inventory/reprice mechanism on one game and one round of flow; a real
  book runs this continuously across a whole slate with far more sophisticated
  flow modeling.

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

170 tests across `probability`, `vig`, `arbitrage`, `kelly`, `clv`, `model`,
`backtest`, `odds_client`, `utils`, `baselines`, `calibration`, `stats`,
`mlb_stats_client`, `mlb_features`, `mlb_pitcher_features`, `portfolio_risk`,
`market_maker`, and `paper_trading` — including hand-checked example values
for every formula in the spec, a planted arbitrage the scanner must detect,
a chronological split the model must never leak across, a tail-dominated
P&L case the backtest's concentration check must flag, a point-in-time
correctness check (for both team and pitcher features, live and historical)
that a later real game can never change an earlier one's, a slate that
forces the portfolio exposure cap to bind, a deterministic market-making
flow that self-corrects back to the fair price, and a paper-trade ledger
round-trip (log → dedupe → reconcile) against synthetic odds payloads shaped
like a real Odds API response.

## Repo structure

```
sports-market-efficiency/
├── config.py                  # env vars, paths, API defaults
├── src/                       # probability, vig, arbitrage, model, kelly, clv, backtest,
│                              #   odds_client, utils, baselines, calibration, stats,
│                              #   mlb_stats_client, mlb_features, mlb_pitcher_features,
│                              #   portfolio_risk, market_maker, paper_trading
├── scripts/
│   ├── fetch_odds.py                      # pull live odds -> data/raw/
│   ├── run_backtest.py                    # synthetic NBA: train + backtest -> results/
│   ├── train_mlb_model.py                 # real MLB: fetch + train + evaluate -> results/
│   ├── train_mlb_model_with_pitching.py   # does adding pitcher features help? -> results/
│   ├── collect_paper_trades.py            # real MLB: log today's live-odds edges
│   └── reconcile_paper_trades.py          # real MLB: settle + write paper_trading_report.md
├── data/
│   ├── raw/                   # unmodified API pulls (gitignored, except the synthetic sample)
│   ├── processed/             # nba_games_synthetic.csv, mlb_games_real.csv
│   └── paper_trades/          # forward-collected paper-trade ledger (gitignored, real data)
├── notebooks/exploration.ipynb
├── tests/
└── results/
    ├── backtest_report.md               # synthetic NBA backtest
    ├── clv_plot.png
    ├── calibration_plot.png
    ├── mlb_model_report.md              # real MLB model evaluation
    ├── mlb_reliability_plot.png
    ├── mlb_pitcher_features_report.md   # team-only vs. team+pitcher comparison
    └── paper_trading_report.md          # real forward-collected CLV (written once run)
```
