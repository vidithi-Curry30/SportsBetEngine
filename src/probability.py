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
