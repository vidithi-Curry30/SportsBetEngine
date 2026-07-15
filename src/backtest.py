"""Backtest engine: walk the held-out test period chronologically, size flagged
bets with fractional Kelly, simulate bankroll, and log CLV per bet.

Includes two realistic frictions: a hard cap on bet size regardless of Kelly
output (books limit sharp bettors in practice), and simulated slippage (the
odds offered move slightly against you between decision and placement).
"""
import numpy as np
import pandas as pd

from src.clv import calculate_clv
from src.kelly import fractional_kelly, kelly_fraction
from src.model import has_value_edge, predict_win_probability
from src.probability import american_to_probability, remove_vig


def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / -odds


def apply_slippage(odds: int, slippage_pct: float) -> int:
    """Nudge the offered price against the bettor by `slippage_pct` of implied probability,
    simulating the line moving between the bet decision and the bet actually being placed."""
    prob = american_to_probability(odds)
    slipped_prob = min(0.99, prob + slippage_pct)
    return _probability_to_american(slipped_prob)


def _probability_to_american(prob: float) -> int:
    if prob >= 0.5:
        return round(-100 * prob / (1 - prob))
    return round(100 * (1 - prob) / prob)


def run_backtest(
    test_df: pd.DataFrame,
    model,
    starting_bankroll: float = 10_000.0,
    kelly_fraction_mult: float = 0.5,
    max_bet_pct: float = 0.05,
    edge_threshold: float = 0.03,
    slippage_pct: float = 0.005,
) -> dict:
    """Run the backtest over `test_df` (the held-out period from model.chronological_split
    -- never games the model was trained on).

    Expects columns: date, team_a_win, market_odds_team_a, market_odds_team_b,
    closing_odds_team_a, plus the model's feature columns. Bets are only ever
    placed on team_a for simplicity; a symmetric team_b leg is a natural extension.
    """
    sorted_df = test_df.sort_values("date").reset_index(drop=True)

    bankroll = starting_bankroll
    bankroll_series = [bankroll]
    bet_log = []

    for _, game in sorted_df.iterrows():
        model_prob = predict_win_probability(model, game)
        market_prob_a = american_to_probability(game["market_odds_team_a"])
        market_prob_b = american_to_probability(game["market_odds_team_b"])
        no_vig_prob_a, _ = remove_vig(market_prob_a, market_prob_b)

        if not has_value_edge(model_prob, no_vig_prob_a, threshold=edge_threshold):
            bankroll_series.append(bankroll)
            continue

        placed_odds = apply_slippage(game["market_odds_team_a"], slippage_pct)
        decimal_odds = american_to_decimal(placed_odds)

        f_star = kelly_fraction(model_prob, decimal_odds)
        stake_fraction = max(0.0, min(fractional_kelly(f_star, kelly_fraction_mult), max_bet_pct))
        stake = bankroll * stake_fraction

        won = bool(game["team_a_win"])
        pnl = stake * (decimal_odds - 1) if won else -stake
        bankroll += pnl
        bankroll_series.append(bankroll)

        bet_log.append(
            {
                "date": game["date"],
                "model_prob": model_prob,
                "no_vig_market_prob": no_vig_prob_a,
                "placed_odds": placed_odds,
                "stake": stake,
                "stake_fraction": stake_fraction,
                "won": won,
                "pnl": pnl,
                "clv": calculate_clv(placed_odds, game["closing_odds_team_a"]),
                "bankroll_after": bankroll,
            }
        )

    return _summarize(bankroll_series, bet_log, starting_bankroll)


def _summarize(bankroll_series: list[float], bet_log: list[dict], starting_bankroll: float) -> dict:
    final_bankroll = bankroll_series[-1]
    roi = (final_bankroll - starting_bankroll) / starting_bankroll

    bankroll_arr = np.array(bankroll_series)
    running_max = np.maximum.accumulate(bankroll_arr)
    max_drawdown = float(((bankroll_arr - running_max) / running_max).min())

    if bet_log:
        per_bet_returns = [b["pnl"] / (b["bankroll_after"] - b["pnl"]) for b in bet_log]
        mean_return = float(np.mean(per_bet_returns))
        std_return = float(np.std(per_bet_returns))
        sharpe_like_ratio = mean_return / std_return if std_return > 0 else 0.0
        avg_clv = float(np.mean([b["clv"] for b in bet_log]))
        hit_rate = sum(1 for b in bet_log if b["won"]) / len(bet_log)
    else:
        sharpe_like_ratio = avg_clv = hit_rate = 0.0

    return {
        "starting_bankroll": starting_bankroll,
        "final_bankroll": final_bankroll,
        "roi": roi,
        "sharpe_like_ratio": sharpe_like_ratio,
        "max_drawdown": max_drawdown,
        "avg_clv": avg_clv,
        "hit_rate": hit_rate,
        "num_bets": len(bet_log),
        "num_games": len(bankroll_series) - 1,
        "bankroll_series": bankroll_series,
        "bet_log": bet_log,
    }
