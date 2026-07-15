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
- [ ] Phase 3 — Arbitrage scanner
- [ ] Phase 4 — Predictive model (chronological train/test split)
- [ ] Phase 5 — CLV tracking + backtest
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

### Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Run tests

```bash
pytest
```
