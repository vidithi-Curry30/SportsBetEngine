import pytest

from src.clv import calculate_clv, track_line_movement


class TestTrackLineMovement:
    def test_stores_implied_probabilities(self):
        result = track_line_movement("game1", opening_odds=120, closing_odds=-110)

        assert result["game_id"] == "game1"
        assert result["opening_odds"] == 120
        assert result["closing_odds"] == -110
        assert result["opening_implied_prob"] == pytest.approx(100 / 220)
        assert result["closing_implied_prob"] == pytest.approx(110 / 210)


class TestCalculateClv:
    def test_positive_clv_when_line_moves_in_your_favor(self):
        # Bet at +120 (implied 0.4545), closes at -110 (implied 0.5238) -- you
        # got a better price than the market ultimately settled on.
        clv = calculate_clv(bet_odds=120, closing_odds=-110)
        assert clv > 0
        assert clv == pytest.approx((110 / 210 - 100 / 220) * 100)

    def test_negative_clv_when_line_moves_against_you(self):
        # Bet at -110, closes at +120 -- you paid more than the closing price.
        clv = calculate_clv(bet_odds=-110, closing_odds=120)
        assert clv < 0

    def test_zero_clv_when_price_unchanged(self):
        assert calculate_clv(bet_odds=-110, closing_odds=-110) == pytest.approx(0.0)
