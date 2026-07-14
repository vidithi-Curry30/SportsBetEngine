import pytest

from src.probability import american_to_probability
from src.vig import calculate_vig


class TestCalculateVig:
    def test_known_overround(self):
        # 0.55 + 0.52 - 1 = 0.07 -> 7.0%
        assert calculate_vig(0.55, 0.52) == pytest.approx(7.0)

    def test_no_vig_market(self):
        assert calculate_vig(0.5, 0.5) == pytest.approx(0.0)

    def test_from_american_odds_pair(self):
        # Standard -110/-110 market: each side implies 110/210 ~= 0.52381
        prob_a = american_to_probability(-110)
        prob_b = american_to_probability(-110)
        assert calculate_vig(prob_a, prob_b) == pytest.approx(4.7619, abs=1e-3)

    def test_negative_vig_is_possible_mathematically(self):
        # Below-fair-value pricing (e.g. a promo) yields a negative "vig"
        assert calculate_vig(0.45, 0.45) == pytest.approx(-10.0)
