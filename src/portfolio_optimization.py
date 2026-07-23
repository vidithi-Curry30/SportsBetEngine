"""Correlation-aware position sizing: a multivariate generalization of the
single-bet Kelly formula in src/kelly.py to a slate of simultaneous bets
that may not be independent.

Why this exists: portfolio_risk.apply_slate_exposure_cap scales every bet on
an over-limit slate down by the same factor, treating every bet as equally
diversifying. Two bets sharing a team -- an MLB doubleheader, most concretely
-- are not independent: if the model's read on that team is wrong, both bets
lose together. A real desk sizes a correlated book differently from an
uncorrelated one; this module does the same.

The math, honestly stated: maximizing expected log-growth E[log(1 + w.R)]
for a *single* bet has the exact closed form in src/kelly.py
(f* = (bp - q) / b). For a *vector* of simultaneous bets there is no closed
form -- the standard tractable approximation (used throughout the
multi-asset Kelly / growth-optimal-portfolio literature) is a second-order
Taylor expansion of log(1 + w.R) around w=0, which turns the maximization
into a quadratic problem with the closed-form solution:

    w* = Sigma^-1 @ mu

where mu is the vector of each bet's expected per-unit-stake return and
Sigma is their covariance matrix. This is an *approximation* to multivariate
Kelly, not an exact generalization -- for a single independent bet it does
NOT reduce algebraically to kelly_fraction()'s exact formula (mu/var != the
exact (bp-q)/b in general). What it does get exactly right, and is used for
here: when Sigma is diagonal (bets independent), it decouples into each
bet's own mu_i/var_i with no cross-bet effect; when two bets are positively
correlated, it exactly discounts their combined weight relative to treating
them as independent, which is the property this module exists to add.
"""
import numpy as np


def bet_return_moments(model_prob: float, decimal_odds: float) -> tuple[float, float]:
    """Mean and variance of a single bet's per-unit-stake return: payout is
    (decimal_odds - 1) with probability model_prob, -1 otherwise."""
    b = decimal_odds - 1
    mean = model_prob * b - (1 - model_prob)
    second_moment = model_prob * b**2 + (1 - model_prob)
    variance = second_moment - mean**2
    return mean, variance


def _shares_a_team(bet_a: dict, bet_b: dict) -> bool:
    return bool(set(bet_a["teams"]) & set(bet_b["teams"]))


def build_covariance_matrix(bets: list[dict], same_team_correlation: float = 0.5) -> np.ndarray:
    """Covariance matrix for a slate of bets. Each bet dict needs "model_prob",
    "decimal_odds", and "teams" (a 2-item iterable of the two team names in
    that game -- callers adapt their own schema, e.g. (team_a, team_b) or
    (home_team, away_team), into this). Two bets sharing either team (e.g. a
    doubleheader) are treated as correlated at an assumed
    `same_team_correlation` -- a modeling assumption stated plainly, not
    empirically estimated, since a real correlated-outcome sample doesn't
    exist yet. Bets sharing no team are treated as independent (covariance 0).
    """
    n = len(bets)
    variances = np.array([bet_return_moments(b["model_prob"], b["decimal_odds"])[1] for b in bets])

    cov = np.diag(variances)
    for i in range(n):
        for j in range(i + 1, n):
            if _shares_a_team(bets[i], bets[j]):
                cov_ij = same_team_correlation * np.sqrt(variances[i] * variances[j])
                cov[i, j] = cov[j, i] = cov_ij
    return cov


def multivariate_kelly_weights(
    bets: list[dict],
    same_team_correlation: float = 0.5,
    kelly_fraction_mult: float = 0.5,
    max_bet_pct: float = 0.05,
    max_slate_pct: float = 0.20,
) -> list[float]:
    """Correlation-aware stake fractions for a slate of simultaneous bets:
    w* = Sigma^-1 @ mu (see module docstring), fractional-Kelly-scaled, then
    capped per bet and per slate the same way portfolio_risk.apply_slate_exposure_cap
    caps the naive independent-Kelly sizing -- this replaces the *base*
    allocation with a covariance-aware one, not the risk limits on top of it.
    """
    if not bets:
        return []

    mu = np.array([bet_return_moments(b["model_prob"], b["decimal_odds"])[0] for b in bets])
    cov = build_covariance_matrix(bets, same_team_correlation=same_team_correlation)

    # Pseudo-inverse: two bets on the same team at similar odds make Sigma
    # near-singular (close to perfectly collinear); a plain inverse can blow
    # up there, pinv degrades gracefully.
    raw_weights = np.linalg.pinv(cov) @ mu
    weights = np.clip(raw_weights, 0.0, None) * kelly_fraction_mult
    weights = np.minimum(weights, max_bet_pct)

    total = weights.sum()
    if total > max_slate_pct:
        weights = weights * (max_slate_pct / total)

    return weights.tolist()
