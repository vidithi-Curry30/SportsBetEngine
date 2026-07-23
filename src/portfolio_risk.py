"""Slate-level (portfolio) risk: an aggregate exposure cap across every bet placed
on the same date, independent of and in addition to the existing per-bet Kelly cap.

Why this exists: `backtest.max_bet_pct` caps any single bet, but on a day with many
flagged edges (a full MLB slate can have 10-15 games) each capped-but-independent
bet can still sum to a large fraction of the bankroll committed on one day -- exactly
the kind of aggregate exposure a real trading desk limits at the book/slate level,
not just per position. This module scales every bet on an over-limit slate down
proportionally rather than dropping any of them, so the *relative* sizing the model
assigned across bets (a bigger edge still gets a bigger stake) is preserved.
"""


def apply_slate_exposure_cap(bets: list[dict], max_slate_pct: float = 0.20) -> list[dict]:
    """Scale down `stake_fraction` proportionally across all bets in `bets` (assumed
    to be every bet flagged on the same slate/date) if their sum exceeds `max_slate_pct`
    of bankroll. Each bet dict must have a "stake_fraction" key; every other key is
    passed through unchanged. A no-op (returns `bets` as-is) when the slate is already
    within the cap or empty.
    """
    if max_slate_pct <= 0:
        raise ValueError("max_slate_pct must be positive")

    total = sum(bet["stake_fraction"] for bet in bets)
    if total <= max_slate_pct or total == 0:
        return bets

    scale = max_slate_pct / total
    return [{**bet, "stake_fraction": bet["stake_fraction"] * scale} for bet in bets]
