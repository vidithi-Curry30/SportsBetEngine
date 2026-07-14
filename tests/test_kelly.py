import pytest

from src.kelly import fractional_kelly, kelly_fraction


class TestKellyFraction:
    def test_known_edge(self):
        # p=0.55, decimal_odds=2.0 (even money) -> b=1
        # f* = (1*0.55 - 0.45) / 1 = 0.10
        assert kelly_fraction(0.55, 2.0) == pytest.approx(0.10)

    def test_no_edge_gives_zero(self):
        # A fair coin at fair (even money) odds has zero edge
        assert kelly_fraction(0.5, 2.0) == pytest.approx(0.0)

    def test_negative_edge_gives_negative_fraction(self):
        # Betting into a price worse than your estimated probability is a losing bet
        assert kelly_fraction(0.4, 2.0) < 0

    def test_higher_probability_gives_higher_stake_at_fixed_odds(self):
        # At fixed odds, more confidence in the outcome should size up the stake
        f_low = kelly_fraction(0.55, 2.0)
        f_high = kelly_fraction(0.65, 2.0)
        assert f_high > f_low

    def test_invalid_probability_raises(self):
        with pytest.raises(ValueError):
            kelly_fraction(1.5, 2.0)
        with pytest.raises(ValueError):
            kelly_fraction(-0.1, 2.0)

    def test_invalid_decimal_odds_raises(self):
        with pytest.raises(ValueError):
            kelly_fraction(0.55, 1.0)
        with pytest.raises(ValueError):
            kelly_fraction(0.55, 0.5)


class TestFractionalKelly:
    def test_half_kelly_default(self):
        assert fractional_kelly(0.10) == pytest.approx(0.05)

    def test_custom_fraction(self):
        assert fractional_kelly(0.10, fraction=0.25) == pytest.approx(0.025)

    def test_full_kelly(self):
        assert fractional_kelly(0.10, fraction=1.0) == pytest.approx(0.10)

    def test_zero_fraction_gives_zero_stake(self):
        assert fractional_kelly(0.10, fraction=0.0) == pytest.approx(0.0)
