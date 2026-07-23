import pytest

from src.probability import add_vig, american_to_probability, probability_to_american, remove_vig


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


class TestProbabilityToAmerican:
    def test_round_trips_with_american_to_probability(self):
        # +100 and -100 both imply prob=0.5, so that boundary is inherently
        # non-invertible (probability_to_american(0.5) picks -100 by convention).
        for odds in [-300, -150, -110, 150, 300]:
            prob = american_to_probability(odds)
            assert probability_to_american(prob) == odds

    def test_favorite(self):
        assert probability_to_american(0.6) == -150

    def test_underdog(self):
        assert probability_to_american(0.4) == 150

    def test_even_money(self):
        assert probability_to_american(0.5) == -100

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            probability_to_american(0.0)
        with pytest.raises(ValueError):
            probability_to_american(1.0)


class TestAddVig:
    def test_inverse_of_remove_vig(self):
        prob_a, prob_b = remove_vig(0.55, 0.52)
        vigged_a, vigged_b = add_vig(prob_a, prob_b, vig_pct=(0.55 + 0.52 - 1) * 100)
        assert vigged_a == pytest.approx(0.55)
        assert vigged_b == pytest.approx(0.52)

    def test_zero_vig_preserves_fair_probabilities(self):
        vigged_a, vigged_b = add_vig(0.5, 0.5, vig_pct=0.0)
        assert vigged_a == pytest.approx(0.5)
        assert vigged_b == pytest.approx(0.5)

    def test_output_sums_to_more_than_one(self):
        vigged_a, vigged_b = add_vig(0.5, 0.5, vig_pct=4.5)
        assert vigged_a + vigged_b == pytest.approx(1.045)

    def test_negative_vig_raises(self):
        with pytest.raises(ValueError):
            add_vig(0.5, 0.5, vig_pct=-1.0)

    def test_zero_sum_raises(self):
        with pytest.raises(ValueError):
            add_vig(0.0, 0.0, vig_pct=4.5)
