"""Kelly criterion position sizing."""


def kelly_fraction(model_prob: float, decimal_odds: float) -> float:
    """Compute the full Kelly stake as a fraction of bankroll.

    f* = (b*p - q) / b, where b = decimal_odds - 1, p = model_prob, q = 1 - p.
    """
    if not 0 <= model_prob <= 1:
        raise ValueError("model_prob must be between 0 and 1")
    if decimal_odds <= 1:
        raise ValueError("decimal_odds must be greater than 1")

    b = decimal_odds - 1
    p = model_prob
    q = 1 - p
    return (b * p - q) / b


def fractional_kelly(f_star: float, fraction: float = 0.5) -> float:
    """Scale down a Kelly stake (default half-Kelly) to reduce variance."""
    return f_star * fraction
