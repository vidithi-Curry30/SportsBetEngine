"""American odds <-> implied probability conversions."""


def american_to_probability(odds: int) -> float:
    """Convert American odds to implied probability.

    Negative odds (favorite): -odds / (-odds + 100)
    Positive odds (underdog):  100 / (odds + 100)
    """
    if odds == 0:
        raise ValueError("American odds cannot be 0")

    if odds < 0:
        return -odds / (-odds + 100)
    return 100 / (odds + 100)


def remove_vig(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Normalize two implied probabilities so they sum to 1 (the "fair" no-vig price)."""
    total = prob_a + prob_b
    if total <= 0:
        raise ValueError("Probabilities must sum to a positive number")

    return prob_a / total, prob_b / total


def probability_to_american(prob: float) -> int:
    """Convert an implied probability back to American odds -- the inverse of
    american_to_probability. Used both for simulated slippage (nudge a probability,
    convert back to a postable price) and for market-making (quote a price from a
    model's probability estimate rather than only reading one off an existing market).
    """
    if not 0 < prob < 1:
        raise ValueError("prob must be strictly between 0 and 1")

    if prob >= 0.5:
        return round(-100 * prob / (1 - prob))
    return round(100 * (1 - prob) / prob)


def add_vig(prob_a: float, prob_b: float, vig_pct: float) -> tuple[float, float]:
    """Inflate a pair of fair (summing-to-1) probabilities to embed a target vig,
    splitting the added overround proportionally to each side's fair probability.
    The inverse operation of remove_vig -- turns a "true" probability into the kind
    of overround-priced market a sportsbook or market maker actually quotes.
    """
    total = prob_a + prob_b
    if total <= 0:
        raise ValueError("Probabilities must sum to a positive number")
    if vig_pct < 0:
        raise ValueError("vig_pct must be non-negative")

    scale = (1 + vig_pct / 100) / total
    return prob_a * scale, prob_b * scale
