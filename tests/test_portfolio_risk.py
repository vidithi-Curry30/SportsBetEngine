import pytest

from src.portfolio_risk import apply_slate_exposure_cap


class TestApplySlateExposureCap:
    def test_no_op_when_under_cap(self):
        bets = [{"stake_fraction": 0.05}, {"stake_fraction": 0.05}]
        result = apply_slate_exposure_cap(bets, max_slate_pct=0.20)
        assert result == bets

    def test_no_op_when_bets_empty(self):
        assert apply_slate_exposure_cap([], max_slate_pct=0.20) == []

    def test_scales_down_proportionally_when_over_cap(self):
        # Three bets each wanting 10% -> 30% total, over a 20% slate cap
        bets = [{"stake_fraction": 0.10}, {"stake_fraction": 0.10}, {"stake_fraction": 0.10}]
        result = apply_slate_exposure_cap(bets, max_slate_pct=0.20)

        total_after = sum(b["stake_fraction"] for b in result)
        assert total_after == pytest.approx(0.20)
        for b in result:
            assert b["stake_fraction"] == pytest.approx(0.20 / 3)

    def test_preserves_relative_sizing(self):
        # A bigger edge should still get a bigger stake after scaling
        bets = [{"stake_fraction": 0.05}, {"stake_fraction": 0.15}, {"stake_fraction": 0.10}]
        result = apply_slate_exposure_cap(bets, max_slate_pct=0.15)

        assert result[1]["stake_fraction"] > result[2]["stake_fraction"] > result[0]["stake_fraction"]
        assert sum(b["stake_fraction"] for b in result) == pytest.approx(0.15)

    def test_passes_through_other_keys_unchanged(self):
        bets = [{"stake_fraction": 0.20, "game": "A"}, {"stake_fraction": 0.20, "game": "B"}]
        result = apply_slate_exposure_cap(bets, max_slate_pct=0.20)

        assert result[0]["game"] == "A"
        assert result[1]["game"] == "B"

    def test_invalid_max_slate_pct_raises(self):
        with pytest.raises(ValueError):
            apply_slate_exposure_cap([{"stake_fraction": 0.1}], max_slate_pct=0.0)
