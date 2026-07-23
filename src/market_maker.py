"""Toy market-making: quoting a two-sided line from a private probability
estimate, then managing inventory risk as simulated flow comes in.

Everything else in this repo takes the market's price as given and asks
"should I bet into it." This module inverts the question: given a private
"true" probability, *set* a two-sided price (the sportsbook's actual job),
then simulate a stream of bettors trading against that price and show the
classic market-maker response -- when one side gets overbet, skew the line
to make that side less attractive and the other side more attractive, to
balance inventory rather than carry a directional bet on the outcome.

Deliberately simple: one game, one round of flow, a linear reprice rule.
The point is to demonstrate the *mechanism* (quote -> inventory -> skew),
not to build a production-grade market-making engine.
"""
import numpy as np

from src.probability import add_vig, probability_to_american


def quote_from_probability(true_prob_a: float, target_vig_pct: float = 4.5) -> dict:
    """Turn a private true win-probability for team_a into a two-sided American-odds
    quote embedding `target_vig_pct` of overround, split proportionally to each side
    (the inverse of probability.remove_vig)."""
    if not 0 < true_prob_a < 1:
        raise ValueError("true_prob_a must be strictly between 0 and 1")

    quoted_prob_a, quoted_prob_b = add_vig(true_prob_a, 1 - true_prob_a, target_vig_pct)
    return {
        "team_a_odds": probability_to_american(quoted_prob_a),
        "team_b_odds": probability_to_american(quoted_prob_b),
        "team_a_prob_quoted": quoted_prob_a,
        "team_b_prob_quoted": quoted_prob_b,
    }


def simulate_flow_and_reprice(
    true_prob_a: float,
    n_bettors: int = 200,
    belief_noise_std: float = 0.08,
    bettor_edge_threshold: float = 0.02,
    reprice_inventory_threshold: float = 10.0,
    reprice_step: float = 0.01,
    target_vig_pct: float = 4.5,
    unit_size: float = 1.0,
    settle: bool = True,
    seed: int = 42,
) -> dict:
    """Simulate a stream of bettors trading against a market maker's quote, with the
    maker skewing its line in response to one-sided inventory -- the standard
    market-making response to flow, distinct from every other module in this repo
    (which only ever *consumes* a market's price rather than sets one).

    Each simulated bettor has a noisy private belief about team_a's true win
    probability (`true_prob_a` + Normal(0, belief_noise_std)); they bet team_a if
    their belief clears the maker's current quoted price by `bettor_edge_threshold`,
    team_b if the opposite, otherwise they pass. Net inventory is the maker's signed
    exposure: positive means the maker has taken on more team_a action (and so is
    short team_a -- owes more if team_a wins) and will reprice team_a *shorter*
    (worse odds for the bettor) to discourage further team_a action and attract
    team_b action, exactly the way a real book leans a line into one-sided flow.

    If `settle`, a single random outcome is drawn from true_prob_a (using the same
    seeded RNG) and the maker's realized P&L on the book of bets taken is computed.

    Returns: final_quote, inventory (final signed exposure, in units), n_bets_team_a,
    n_bets_team_b, n_reprices, quote_history (list of quotes after each reprice),
    and (if settle) outcome and maker_pnl.
    """
    if not 0 < true_prob_a < 1:
        raise ValueError("true_prob_a must be strictly between 0 and 1")
    if reprice_inventory_threshold <= 0:
        raise ValueError("reprice_inventory_threshold must be positive")

    rng = np.random.default_rng(seed)

    center_prob_a = true_prob_a
    quote = quote_from_probability(center_prob_a, target_vig_pct)
    quote_history = [quote]

    inventory = 0.0  # positive = net long team_a action taken (maker is short team_a)
    n_bets_a = n_bets_b = 0
    bets = []  # each: {"side": "a"|"b", "odds": american_odds, "stake": unit_size}

    for _ in range(n_bettors):
        belief_prob_a = float(np.clip(rng.normal(true_prob_a, belief_noise_std), 0.01, 0.99))

        if belief_prob_a - quote["team_a_prob_quoted"] >= bettor_edge_threshold:
            bets.append({"side": "a", "odds": quote["team_a_odds"], "stake": unit_size})
            inventory += unit_size
            n_bets_a += 1
        elif (1 - belief_prob_a) - quote["team_b_prob_quoted"] >= bettor_edge_threshold:
            bets.append({"side": "b", "odds": quote["team_b_odds"], "stake": unit_size})
            inventory -= unit_size
            n_bets_b += 1
        else:
            continue

        if abs(inventory) >= reprice_inventory_threshold:
            # Skew the center probability in the direction of the overbet side:
            # more team_a action taken -> raise the perceived probability of team_a
            # -> team_a's quoted price gets worse for the next bettor, team_b's gets
            # better -- exactly the lean a real book applies to balance a book.
            direction = 1 if inventory > 0 else -1
            center_prob_a = float(np.clip(center_prob_a + direction * reprice_step, 0.02, 0.98))
            quote = quote_from_probability(center_prob_a, target_vig_pct)
            quote_history.append(quote)
            inventory = 0.0  # inventory measured relative to the current quote

    result = {
        "final_quote": quote,
        "n_bets_team_a": n_bets_a,
        "n_bets_team_b": n_bets_b,
        "n_reprices": len(quote_history) - 1,
        "quote_history": quote_history,
    }

    if settle:
        outcome_a_wins = bool(rng.random() < true_prob_a)
        result["outcome_team_a_wins"] = outcome_a_wins
        result["maker_pnl"] = _settle_book(bets, outcome_a_wins)

    return result


def _settle_book(bets: list[dict], outcome_a_wins: bool) -> float:
    """Maker's P&L: collects every losing bettor's stake, pays out
    stake * (decimal_odds - 1) to every winning bettor."""
    pnl = 0.0
    for bet in bets:
        bettor_won = (bet["side"] == "a") == outcome_a_wins
        decimal_odds = 1 + bet["odds"] / 100 if bet["odds"] > 0 else 1 + 100 / -bet["odds"]
        if bettor_won:
            pnl -= bet["stake"] * (decimal_odds - 1)
        else:
            pnl += bet["stake"]
    return pnl
