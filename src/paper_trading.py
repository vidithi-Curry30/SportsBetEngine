"""Forward-collecting paper-trading pipeline: line shopping across books (the live
edge monitor) plus a CLV ledger that grows into real, non-synthetic backtest data
over time.

Why this exists: results/backtest_report.md's CLV/ROI numbers are honestly disclosed
as synthetic, because The Odds API's historical-odds endpoint is paywalled -- there's
no way to buy real historical prices for games already played. But nothing stops
*forward* collection: pull live odds today, compute the model's edge, log a "paper
bet" if one is flagged, then once the game closes and finishes, reconcile against the
real closing price and the real result. Every row this produces is real market data
and genuinely out-of-sample (the bet was logged before the outcome existed) -- more
credible than a backtest, not less, precisely because leakage is structurally
impossible. The tradeoff: it only accumulates one slate at a time, so it takes weeks
to build a sample size worth reporting on. That's a real limitation, stated here
rather than worked around.

Ledger schema (one row per flagged bet):
    game_date, home_team, away_team, snapshot_time, model_prob, no_vig_market_prob,
    edge, book, placed_odds, stake_fraction, status ("open" | "settled"),
    closing_odds, result (1 = home team won), clv, pnl
"""
from typing import Callable, Optional

import pandas as pd

from src.clv import calculate_clv
from src.kelly import fractional_kelly, kelly_fraction
from src.model import has_value_edge
from src.portfolio_optimization import multivariate_kelly_weights
from src.portfolio_risk import apply_slate_exposure_cap
from src.probability import american_to_probability, remove_vig

LEDGER_COLUMNS = [
    "game_date", "home_team", "away_team", "snapshot_time", "model_prob",
    "no_vig_market_prob", "edge", "book", "placed_odds", "stake_fraction",
    "status", "closing_odds", "result", "clv", "pnl",
]


def find_best_price(game: dict, side: str) -> Optional[tuple[str, int]]:
    """Line shopping: across every book quoting this game's moneyline, find the
    book offering the best (lowest implied probability, i.e. most bettor-favorable)
    price for `side` ("home" or "away"). Returns (book_title, american_odds), or
    None if no book quotes an h2h price for that side."""
    team_name = game.get("home_team") if side == "home" else game.get("away_team")
    if team_name is None:
        return None

    best = None
    for bookmaker in game.get("bookmakers", []):
        h2h = next((m for m in bookmaker.get("markets", []) if m.get("key") == "h2h"), None)
        if h2h is None:
            continue
        price = next((o["price"] for o in h2h.get("outcomes", []) if o.get("name") == team_name), None)
        if price is None:
            continue

        implied_prob = american_to_probability(price)
        if best is None or implied_prob < best[2]:
            best = (bookmaker.get("title", bookmaker.get("key")), price, implied_prob)

    return (best[0], best[1]) if best else None


def consensus_no_vig_prob(game: dict) -> Optional[tuple[float, float]]:
    """Average no-vig (fair) home/away probability across every book quoting this
    game, as the "market" price to compare the model against -- less noisy than
    trusting any single book's line."""
    home_team, away_team = game.get("home_team"), game.get("away_team")
    if home_team is None or away_team is None:
        return None

    home_probs, away_probs = [], []
    for bookmaker in game.get("bookmakers", []):
        h2h = next((m for m in bookmaker.get("markets", []) if m.get("key") == "h2h"), None)
        if h2h is None:
            continue
        prices = {o["name"]: o["price"] for o in h2h.get("outcomes", [])}
        if home_team not in prices or away_team not in prices:
            continue

        no_vig_home, no_vig_away = remove_vig(
            american_to_probability(prices[home_team]), american_to_probability(prices[away_team])
        )
        home_probs.append(no_vig_home)
        away_probs.append(no_vig_away)

    if not home_probs:
        return None
    return sum(home_probs) / len(home_probs), sum(away_probs) / len(away_probs)


def compute_live_edges(
    odds_games: list[dict],
    feature_lookup: Callable[[str, str, str], Optional[dict]],
    model,
    feature_columns: list[str],
    edge_threshold: float = 0.03,
) -> pd.DataFrame:
    """The live cross-book edge monitor: for every game in a fresh Odds API pull,
    compute the model's home-team win probability, the consensus no-vig market
    price, the line-shopped best price, and whether the gap clears `edge_threshold`.

    `feature_lookup(home_team, away_team, game_date) -> dict | None` supplies the
    model's point-in-time features for an upcoming matchup -- see
    mlb_features.compute_current_features for the MLB implementation. Games the
    lookup can't build features for (no prior history yet) are skipped, same
    discipline as build_features dropping a team's first game of the season.

    Returns one row per game with a usable edge computation, sorted by edge
    descending (biggest model-vs-market disagreement first).
    """
    from src.model import predict_win_probability  # local import: avoid a hard sklearn dep at module load

    rows = []
    for game in odds_games:
        home_team, away_team = game.get("home_team"), game.get("away_team")
        game_date = game.get("commence_time", "")[:10]  # ISO date prefix of an ISO-8601 timestamp

        features = feature_lookup(home_team, away_team, game_date)
        if features is None:
            continue

        consensus = consensus_no_vig_prob(game)
        if consensus is None:
            continue
        no_vig_home, _ = consensus

        model_prob = predict_win_probability(model, features, feature_columns)
        best_home = find_best_price(game, "home")
        best_away = find_best_price(game, "away")

        rows.append(
            {
                "game_date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": game.get("commence_time"),
                "model_prob": model_prob,
                "no_vig_market_prob": no_vig_home,
                "edge": model_prob - no_vig_home,
                "best_book": best_home[0] if best_home else None,
                "best_home_odds": best_home[1] if best_home else None,
                "best_away_odds": best_away[1] if best_away else None,
                "has_value_edge": has_value_edge(model_prob, no_vig_home, threshold=edge_threshold),
            }
        )

    edges_df = pd.DataFrame(rows)
    if not edges_df.empty:
        edges_df = edges_df.sort_values("edge", ascending=False).reset_index(drop=True)
    return edges_df


def build_paper_trade_rows(
    edges_df: pd.DataFrame,
    snapshot_time: str,
    kelly_fraction_mult: float = 0.5,
    max_bet_pct: float = 0.05,
    max_slate_pct: float = 0.20,
    sizing_strategy: str = "proportional",
    same_team_correlation: float = 0.5,
) -> list[dict]:
    """Turn the flagged rows of a compute_live_edges() result into paper-trade
    ledger rows. `sizing_strategy` matches backtest.run_backtest:
      - "proportional" (default): independent per-bet Kelly, capped per bet
        and then per slate via portfolio_risk.apply_slate_exposure_cap.
      - "correlation_aware": portfolio_optimization.multivariate_kelly_weights
        sizes a day's bets jointly, discounting bets that share a team (e.g.
        an MLB doubleheader) relative to treating them as independent.
    A live MLB pull can flag several games on the same date, exactly the
    scenario both strategies exist for. Nothing is actually wagered; this
    only logs the bet that *would* have been placed, at today's real price,
    for later reconciliation."""
    if sizing_strategy not in ("proportional", "correlation_aware"):
        raise ValueError(f"Unknown sizing_strategy: {sizing_strategy!r}")

    candidates = []
    for _, edge in edges_df[edges_df["has_value_edge"]].iterrows():
        if pd.isna(edge["best_home_odds"]):
            continue

        best_home_odds = int(edge["best_home_odds"])
        decimal_odds = 1 + best_home_odds / 100 if best_home_odds > 0 else 1 + 100 / -best_home_odds

        candidates.append(
            {
                "game_date": edge["game_date"],
                "home_team": edge["home_team"],
                "away_team": edge["away_team"],
                "snapshot_time": snapshot_time,
                "model_prob": edge["model_prob"],
                "no_vig_market_prob": edge["no_vig_market_prob"],
                "edge": edge["edge"],
                "book": edge["best_book"],
                "placed_odds": best_home_odds,
                "decimal_odds": decimal_odds,
                "teams": (edge["home_team"], edge["away_team"]),
                "status": "open",
                "closing_odds": None,
                "result": None,
                "clv": None,
                "pnl": None,
            }
        )

    rows = []
    for game_date in sorted({c["game_date"] for c in candidates}):
        slate = [c for c in candidates if c["game_date"] == game_date]
        if sizing_strategy == "correlation_aware":
            weights = multivariate_kelly_weights(
                slate, same_team_correlation=same_team_correlation,
                kelly_fraction_mult=kelly_fraction_mult, max_bet_pct=max_bet_pct, max_slate_pct=max_slate_pct,
            )
            for c, w in zip(slate, weights):
                c["stake_fraction"] = w
        else:
            for c in slate:
                f_star = kelly_fraction(c["model_prob"], c["decimal_odds"])
                c["stake_fraction"] = max(0.0, min(fractional_kelly(f_star, kelly_fraction_mult), max_bet_pct))
            slate = apply_slate_exposure_cap(slate, max_slate_pct=max_slate_pct)
        rows.extend(slate)

    for row in rows:
        del row["decimal_odds"]
        del row["teams"]

    return rows
    return rows


def load_ledger(path) -> pd.DataFrame:
    """Load the paper-trade ledger, or an empty one with the right columns if it
    doesn't exist yet (the first run of scripts/collect_paper_trades.py)."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=LEDGER_COLUMNS)


def append_new_paper_trades(new_rows: list[dict], ledger: pd.DataFrame) -> pd.DataFrame:
    """Append newly-flagged bets to the ledger, skipping any (game_date, home_team,
    away_team) already logged -- the ledger keeps only the *first* snapshot's price
    for a game, since CLV is defined relative to the price you actually bet at, not
    the freshest one seen on a later run."""
    if ledger.empty:
        existing_keys = set()
    else:
        existing_keys = set(zip(ledger["game_date"], ledger["home_team"], ledger["away_team"]))

    rows_to_add = [
        row for row in new_rows
        if (row["game_date"], row["home_team"], row["away_team"]) not in existing_keys
    ]
    if not rows_to_add:
        return ledger

    return pd.concat([ledger, pd.DataFrame(rows_to_add)], ignore_index=True)


def reconcile_paper_trades(
    ledger: pd.DataFrame,
    closing_odds_by_game: dict[tuple[str, str, str], int],
    results_by_game: dict[tuple[str, str, str], int],
) -> pd.DataFrame:
    """Settle every still-"open" ledger row whose game has a known result: fill in
    closing_odds/result/clv/pnl and mark it "settled". `closing_odds_by_game` and
    `results_by_game` are keyed by (game_date, home_team, away_team) -- the caller
    builds these from the last odds snapshot observed before a game started and
    mlb_stats_client.fetch_completed_games, respectively. Rows for games with no
    known result yet are left untouched (still "open")."""
    ledger = ledger.copy()

    for idx, row in ledger.iterrows():
        if row["status"] == "settled":
            continue
        key = (row["game_date"], row["home_team"], row["away_team"])
        if key not in results_by_game:
            continue

        result = results_by_game[key]
        closing_odds = closing_odds_by_game.get(key, row["placed_odds"])
        clv = calculate_clv(int(row["placed_odds"]), int(closing_odds))

        decimal_odds = 1 + row["placed_odds"] / 100 if row["placed_odds"] > 0 else 1 + 100 / -row["placed_odds"]
        pnl = row["stake_fraction"] * (decimal_odds - 1) if result == 1 else -row["stake_fraction"]

        ledger.at[idx, "closing_odds"] = closing_odds
        ledger.at[idx, "result"] = result
        ledger.at[idx, "clv"] = clv
        ledger.at[idx, "pnl"] = pnl
        ledger.at[idx, "status"] = "settled"

    return ledger
