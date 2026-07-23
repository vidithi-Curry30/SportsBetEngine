"""Backtest engine: walk the held-out test period chronologically, size flagged
bets with fractional Kelly, simulate bankroll, and log CLV per bet.

Includes two realistic frictions: a hard cap on bet size regardless of Kelly
output (books limit sharp bettors in practice), and simulated slippage (the
odds offered move slightly against you between decision and placement).
"""
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.clv import calculate_clv
from src.kelly import fractional_kelly, kelly_fraction
from src.model import has_value_edge, predict_win_probability
from src.portfolio_optimization import multivariate_kelly_weights
from src.portfolio_risk import apply_slate_exposure_cap
from src.probability import american_to_probability, probability_to_american, remove_vig


def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / -odds


def apply_slippage(odds: int, slippage_pct: float) -> int:
    """Nudge the offered price against the bettor by `slippage_pct` of implied probability,
    simulating the line moving between the bet decision and the bet actually being placed."""
    prob = american_to_probability(odds)
    slipped_prob = min(0.99, prob + slippage_pct)
    return probability_to_american(slipped_prob)


def run_backtest(
    test_df: pd.DataFrame,
    model=None,
    starting_bankroll: float = 10_000.0,
    kelly_fraction_mult: float = 0.5,
    max_bet_pct: float = 0.05,
    edge_threshold: float = 0.03,
    slippage_pct: float = 0.005,
    max_slate_pct: float = 0.20,
    sizing_strategy: str = "proportional",
    same_team_correlation: float = 0.5,
    probability_fn: Optional[Callable] = None,
) -> dict:
    """Run the backtest over `test_df` (the held-out period from model.chronological_split
    -- never games the model was trained on).

    Expects columns: date, team_a, team_b, team_a_win, market_odds_team_a,
    market_odds_team_b, closing_odds_team_a, plus the model's feature columns.
    Bets are only ever placed on team_a for simplicity; a symmetric team_b leg
    is a natural extension.

    Pass either `model` (a fitted model.train_model() estimator, scored via
    model.predict_win_probability) or `probability_fn` (any callable taking a
    game row and returning a probability -- e.g. a calibration.IsotonicCalibrator)
    to control how win probability is estimated.

    Bets are batched by date: every flagged bet on the same date is sized against
    that day's starting bankroll (not sequentially compounded against each other,
    since a real slate is decided and placed simultaneously, not one game at a
    time as results trickle in). `sizing_strategy` controls how that day's bets
    are sized against each other:
      - "proportional" (default): independent per-bet Kelly, then
        `portfolio_risk.apply_slate_exposure_cap` scales every stake on an
        over-limit day down by the same factor.
      - "correlation_aware": `portfolio_optimization.multivariate_kelly_weights`
        replaces the independent per-bet sizing with a covariance-aware
        allocation -- two bets sharing a team (e.g. a doubleheader) get sized
        down relative to treating them as independent, not just capped
        the same as everything else on an over-limit day.
    """
    if sizing_strategy not in ("proportional", "correlation_aware"):
        raise ValueError(f"Unknown sizing_strategy: {sizing_strategy!r}")

    if probability_fn is None:
        if model is None:
            raise ValueError("run_backtest requires either model or probability_fn")
        probability_fn = lambda game: predict_win_probability(model, game)  # noqa: E731

    sorted_df = test_df.sort_values("date").reset_index(drop=True)
    num_games = len(sorted_df)

    bankroll = starting_bankroll
    bankroll_series = [bankroll]
    bet_log = []

    for _, day_games in sorted_df.groupby("date", sort=True):
        day_start_bankroll = bankroll
        candidates = []

        for _, game in day_games.iterrows():
            model_prob = probability_fn(game)
            market_prob_a = american_to_probability(game["market_odds_team_a"])
            market_prob_b = american_to_probability(game["market_odds_team_b"])
            no_vig_prob_a, _ = remove_vig(market_prob_a, market_prob_b)

            if not has_value_edge(model_prob, no_vig_prob_a, threshold=edge_threshold):
                continue

            placed_odds = apply_slippage(game["market_odds_team_a"], slippage_pct)
            decimal_odds = american_to_decimal(placed_odds)

            candidates.append(
                {
                    "game": game,
                    "model_prob": model_prob,
                    "no_vig_market_prob": no_vig_prob_a,
                    "placed_odds": placed_odds,
                    "decimal_odds": decimal_odds,
                    # Only needed by sizing_strategy="correlation_aware"; absent in
                    # test fixtures/datasets that don't carry team identity is fine
                    # for the default "proportional" strategy, which never reads it.
                    "teams": (game.get("team_a"), game.get("team_b")),
                }
            )

        if sizing_strategy == "correlation_aware":
            weights = multivariate_kelly_weights(
                candidates,
                same_team_correlation=same_team_correlation,
                kelly_fraction_mult=kelly_fraction_mult,
                max_bet_pct=max_bet_pct,
                max_slate_pct=max_slate_pct,
            )
            for c, w in zip(candidates, weights):
                c["stake_fraction"] = w
        else:
            for c in candidates:
                f_star = kelly_fraction(c["model_prob"], c["decimal_odds"])
                c["stake_fraction"] = max(0.0, min(fractional_kelly(f_star, kelly_fraction_mult), max_bet_pct))
            candidates = apply_slate_exposure_cap(candidates, max_slate_pct=max_slate_pct)

        day_bets = []
        for c in candidates:
            game = c["game"]
            stake = day_start_bankroll * c["stake_fraction"]
            won = bool(game["team_a_win"])
            pnl = stake * (c["decimal_odds"] - 1) if won else -stake

            day_bets.append(
                {
                    "date": game["date"],
                    "model_prob": c["model_prob"],
                    "no_vig_market_prob": c["no_vig_market_prob"],
                    "placed_odds": c["placed_odds"],
                    "stake": stake,
                    "stake_fraction": c["stake_fraction"],
                    "won": won,
                    "pnl": pnl,
                    "clv": calculate_clv(c["placed_odds"], game["closing_odds_team_a"]),
                    "bankroll_before": day_start_bankroll,
                }
            )

        # All bets on a slate are placed simultaneously (before any of that day's
        # results are known), so every bet's post-settlement bankroll is the same:
        # the bankroll at the end of the day, once the whole day's results are in.
        day_pnl = sum(b["pnl"] for b in day_bets)
        bankroll = day_start_bankroll + day_pnl
        for b in day_bets:
            b["bankroll_after"] = bankroll
        bet_log.extend(day_bets)

        bankroll_series.append(bankroll)

    return _summarize(bankroll_series, bet_log, starting_bankroll, num_games)


def _summarize(bankroll_series: list[float], bet_log: list[dict], starting_bankroll: float, num_games: int) -> dict:
    final_bankroll = bankroll_series[-1]
    roi = (final_bankroll - starting_bankroll) / starting_bankroll

    bankroll_arr = np.array(bankroll_series)
    running_max = np.maximum.accumulate(bankroll_arr)
    max_drawdown = float(((bankroll_arr - running_max) / running_max).min())

    if bet_log:
        per_bet_returns = [b["pnl"] / b["bankroll_before"] for b in bet_log]
        mean_return = float(np.mean(per_bet_returns))
        std_return = float(np.std(per_bet_returns))
        sharpe_like_ratio = mean_return / std_return if std_return > 0 else 0.0
        avg_clv = float(np.mean([b["clv"] for b in bet_log]))
        hit_rate = sum(1 for b in bet_log if b["won"]) / len(bet_log)
        total_pnl = sum(b["pnl"] for b in bet_log)
        max_single_bet_pnl = max(b["pnl"] for b in bet_log)
        # Share of total profit from the single best-performing bet -- a standard
        # tail-risk check: a strategy whose entire profit rests on one outlier bet
        # hasn't demonstrated a repeatable edge. None when there's no profit to share.
        top_bet_pnl_share = (max_single_bet_pnl / total_pnl) if total_pnl > 0 else None
    else:
        sharpe_like_ratio = avg_clv = hit_rate = 0.0
        top_bet_pnl_share = None

    return {
        "starting_bankroll": starting_bankroll,
        "final_bankroll": final_bankroll,
        "roi": roi,
        "sharpe_like_ratio": sharpe_like_ratio,
        "max_drawdown": max_drawdown,
        "avg_clv": avg_clv,
        "hit_rate": hit_rate,
        "top_bet_pnl_share": top_bet_pnl_share,
        "num_bets": len(bet_log),
        "num_games": num_games,
        "bankroll_series": bankroll_series,
        "bet_log": bet_log,
    }
