# Sports Market Efficiency & Arbitrage Engine

A research project that treats sportsbook odds as prices in a prediction market:
implied probability = price, vig = bid-ask spread, cross-book arbitrage =
cross-venue arbitrage, and Kelly sizing = position sizing under estimated edge.
Built as a quant-trading portfolio piece, not a betting tool.

Full domain framing and results will be written up in Phase 6, once the
pipeline (data ingestion, arbitrage scanning, predictive model, and backtest)
is complete.

## Status

- [x] Phase 1 — Core math: `probability.py`, `vig.py`, `kelly.py` (unit tested)
- [x] Phase 2 — Data pipeline (`odds_client.py`, `utils.py`)
- [x] Phase 3 — Arbitrage scanner
- [x] Phase 4 — Predictive model (chronological train/test split)
- [x] Phase 5 — CLV tracking + backtest
- [ ] Phase 6 — Results & full README

## Phase 1: core math

- `src/probability.py` — converts American odds to implied probability and
  strips the vig from a two-way market to get the "fair" no-vig probability.
- `src/vig.py` — computes the vig (overround) of a two-way market as a
  percentage, i.e. the sportsbook's built-in edge.
- `src/kelly.py` — full and fractional (half-Kelly by default) position sizing
  given a model's win probability and the offered decimal odds.

## Phase 2: data pipeline

- `src/odds_client.py` — thin wrapper around [The Odds API](https://the-odds-api.com/)
  (`get_sports`, `get_odds`), plus `save_raw_pull` which writes each pull to
  `data/raw/` as a timestamped, unmodified JSON file.
- `src/utils.py` — team name normalization (so odds from different books join
  on the same game) and ISO-8601 timestamp helpers.
- `scripts/fetch_odds.py` — CLI: `python scripts/fetch_odds.py --sport basketball_nba`.

To pull real data: get a free API key at the-odds-api.com, copy `.env.example`
to `.env`, and set `ODDS_API_KEY`. No key is checked into this repo. (This
project hasn't run a live pull yet — a key was not available at build time —
so `data/raw/` is currently empty; the client is tested against mocked HTTP
responses in `tests/test_odds_client.py`.)

## Phase 3: arbitrage scanner

- `src/arbitrage.py` — `find_arbitrage` checks every ordered book pair for a
  single game (best price on team A from book X vs. team B from book Y); if
  the summed implied probability is < 1, it's a guaranteed-profit arbitrage.
  `scan_games_for_arbitrage` runs this across a batch of games in The Odds
  API's response shape and tags each hit with the source game.

No live API key was available at build time (see Phase 2), so the scanner was
validated against a realistic **synthetic** batch of 6 games x 4 books
(`data/raw/basketball_nba_synthetic_sample.json`, ~4.6-4.9% vig per book, small
cross-book noise on the underlying win probability — see
`notebooks/exploration.ipynb` for the full run).

**Result: 0 arbitrage opportunities found**, which is itself the expected,
reportable outcome — arbitrage requires book disagreement large enough to
cross the combined vig on both sides, which is rare and closes quickly in
real markets. The scanner is confirmed to correctly return no opportunities
against a market with no exploitable mispricing (and unit tests in
`tests/test_arbitrage.py` confirm it correctly detects a planted arb). A real
pull against live odds is still pending an API key.

## Phase 4: predictive model

- `src/model.py` — `chronological_split` sorts games by date and cuts the
  first `train_frac` (default 80%) into training, the rest into a held-out
  test period the model never sees during fitting. `train_model` fits a
  `LogisticRegression` on `[pace_diff, off_rating_diff, def_rating_diff,
  recent_win_pct_diff, rest_days_diff, home_flag]`. `predict_win_probability`
  scores a single game. `has_value_edge` flags a candidate bet when the
  model's probability beats the no-vig market probability
  (`probability.remove_vig`) by a threshold (default 3 points).

No historical NBA stats were reachable from this environment (`nba_api` isn't
installed, and `stats.nba.com` times out — a known bot-hostile endpoint), so
the model was trained on a **synthetic** season (`data/processed/nba_games_synthetic.csv`,
1200 games, Oct 2024-Apr 2025). Each team has a fixed hidden "true strength";
observed features are noisy proxies for it (not the strength itself), and
game outcomes are Bernoulli-sampled from a logistic function of that hidden
strength plus a home-court term — so the relationship is realistic and
learnable, but not perfectly recoverable, by design.

**Held-out test results (240 games, chronologically after the 960-game
training period, never touched during fitting):**
- Accuracy: 59.2%
- Log loss: 0.661
- `home_flag` coefficient: +0.52 (positive, as expected — home-court advantage)
- `recent_win_pct_diff` is the strongest single feature, `pace_diff` carries
  almost no signal (by construction — pace was generated independent of team
  strength in the synthetic data)

This is deliberately in the realistic range for NBA moneyline models — real
models in this space typically land in the 60s, not the 90s, and a model
claiming 90%+ accuracy on game outcomes would be a red flag, not a strength,
in an interview. **This dataset is synthetic and must be replaced with real
historical data (via `nba_api` or another stats source) before any result
here is a genuine market-efficiency finding** — right now it validates that
the pipeline (chronological split → fit → held-out evaluation) is correctly
wired and leak-free.

## Phase 5: CLV tracking + backtest

- `src/clv.py` — `track_line_movement` stores opening/closing implied
  probability per game; `calculate_clv` compares the implied probability at
  bet time to the closing line. Positive CLV means you bet at a better price
  than where the market ultimately closed — the market moved toward you after
  you bet, independent of whether that individual bet won.
- `src/backtest.py` — walks the held-out test period from Phase 4
  chronologically. Each flagged bet (`model.has_value_edge`) is sized with
  half-Kelly (`kelly.fractional_kelly`), capped at a max % of bankroll
  regardless of Kelly output, with simulated slippage (the price moves
  slightly against you between decision and placement) applied before sizing.
  Outputs bankroll time series, ROI, a Sharpe-like ratio (mean bet return /
  std of bet returns), max drawdown, average CLV, and hit rate.

**Backtest results on the synthetic held-out test period (240 games, 94 bets
flagged):**
- ROI: **-5.5%**
- Avg CLV: **+2.64 percentage points**
- Hit rate: 29.8%
- Sharpe-like ratio: 0.03
- Max drawdown: -49.7%

This is the central result the project is built to produce, and it's exactly
the "CLV over ROI" story: **raw ROI was negative over this sample, but average
CLV was consistently positive** — the model was getting closing-line-beating
prices even though a modest, small-sample hit rate and a real losing streak
(7 straight) dragged down realized ROI. ROI over 94 bets is dominated by
variance; CLV, measured bet-by-bet against where the market ultimately
settled, is the more reliable signal of whether the edge is real. The flagged
bets also skew toward underdogs (mean model probability 0.44 vs. mean no-vig
market probability 0.33) — a known failure mode where an imperfectly
calibrated model systematically "finds value" on long shots, which is worth
flagging explicitly rather than laundering into an inflated backtest.

### Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Run tests

```bash
pytest
```
