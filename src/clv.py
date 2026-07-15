"""Closing line value (CLV) tracking -- the metric that isolates genuine
predictive skill from the noise of individual bet outcomes."""
from src.probability import american_to_probability


def track_line_movement(game_id: str, opening_odds: int, closing_odds: int) -> dict:
    """Store opening and closing implied probability for a game's line."""
    return {
        "game_id": game_id,
        "opening_odds": opening_odds,
        "closing_odds": closing_odds,
        "opening_implied_prob": american_to_probability(opening_odds),
        "closing_implied_prob": american_to_probability(closing_odds),
    }


def calculate_clv(bet_odds: int, closing_odds: int) -> float:
    """CLV as percentage points: (closing implied prob) - (bet implied prob).

    Positive CLV means you bet at a lower implied probability (a better price)
    than where the market closed -- the line moved in your favor after you bet,
    which is the strongest available signal of edge, independent of whether
    that particular bet won or lost.
    """
    bet_prob = american_to_probability(bet_odds)
    closing_prob = american_to_probability(closing_odds)
    return (closing_prob - bet_prob) * 100
