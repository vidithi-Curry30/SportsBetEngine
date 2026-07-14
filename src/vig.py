"""Vig / overround calculation — the sportsbook's built-in edge, analogous to bid-ask spread."""


def calculate_vig(prob_a: float, prob_b: float) -> float:
    """Return the vig as a percentage: (prob_a + prob_b - 1) * 100.

    A two-way market with no edge to the book sums to exactly 1.0 (100%).
    Anything above that is the vig baked into the quoted prices.
    """
    return (prob_a + prob_b - 1) * 100
