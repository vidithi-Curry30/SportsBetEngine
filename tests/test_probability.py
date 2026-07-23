import pytest

from src.probability import american_to_probability, remove_vig


class TestAmericanToProbability:
    def test_even_money_negative(self):
        # -100 is a coin flip priced at even money
        assert american_to_probability(-100) == pytest.approx(0.5)

    def test_even_money_positive(self):
        assert american_to_probability(100) == pytest.approx(0.5)

    def test_favorite(self):
        # -150 favorite: 150 / (150 + 100) = 0.6
        assert american_to_probability(-150) == pytest.approx(0.6)

    def test_heavy_favorite(self):
        # -300: 300 / 400 = 0.75
        assert american_to_probability(-300) == pytest.approx(0.75)

    def test_underdog(self):
        # +150: 100 / 250 = 0.4
        assert american_to_probability(150) == pytest.approx(0.4)

    def test_heavy_underdog(self):
        # +300: 100 / 400 = 0.25
        assert american_to_probability(300) == pytest.approx(0.25)

    def test_zero_odds_raises(self):
        with pytest.raises(ValueError):
            american_to_probability(0)


class TestRemoveVig:
    def test_normalizes_to_one(self):
        prob_a, prob_b = remove_vig(0.55, 0.52)
        assert prob_a + prob_b == pytest.approx(1.0)

    def test_known_values(self):
        # 0.55 / 1.07 and 0.52 / 1.07
        prob_a, prob_b = remove_vig(0.55, 0.52)
        assert prob_a == pytest.approx(0.55 / 1.07)
        assert prob_b == pytest.approx(0.52 / 1.07)

    def test_no_vig_market_unchanged(self):
        # A market that already sums to 1 should pass through unchanged
        prob_a, prob_b = remove_vig(0.5, 0.5)
        assert prob_a == pytest.approx(0.5)
        assert prob_b == pytest.approx(0.5)

    def test_zero_sum_raises(self):
        with pytest.raises(ValueError):
            remove_vig(0.0, 0.0)
