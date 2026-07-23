import pytest

from src.market_maker import quote_from_probability, simulate_flow_and_reprice


class TestQuoteFromProbability:
    def test_embeds_target_vig(self):
        # Fair 50/50 with 4.5% vig: total quoted prob = 1.045, split evenly ->
        # 0.5225 each side -> American odds of -109 (hand-computed: -100*0.5225/0.4775).
        quote = quote_from_probability(0.5, target_vig_pct=4.5)
        assert quote["team_a_prob_quoted"] == pytest.approx(0.5225)
        assert quote["team_b_prob_quoted"] == pytest.approx(0.5225)
        assert quote["team_a_odds"] == -109
        assert quote["team_b_odds"] == -109

    def test_favors_the_favorite(self):
        quote = quote_from_probability(0.7, target_vig_pct=4.5)
        assert quote["team_a_prob_quoted"] > quote["team_b_prob_quoted"]
        assert quote["team_a_odds"] < 0  # favorite quoted as a negative price
        assert quote["team_b_odds"] > 0

    def test_zero_vig_gives_a_fair_two_sided_price(self):
        quote = quote_from_probability(0.6, target_vig_pct=0.0)
        assert quote["team_a_prob_quoted"] + quote["team_b_prob_quoted"] == pytest.approx(1.0)

    def test_out_of_range_probability_raises(self):
        with pytest.raises(ValueError):
            quote_from_probability(0.0)
        with pytest.raises(ValueError):
            quote_from_probability(1.0)


class TestSimulateFlowAndReprice:
    def test_no_reprice_when_inventory_never_crosses_threshold(self):
        result = simulate_flow_and_reprice(
            true_prob_a=0.5,
            n_bettors=200,
            belief_noise_std=0.06,
            bettor_edge_threshold=0.03,
            reprice_inventory_threshold=100.0,  # effectively unreachable
            target_vig_pct=0.0,
            seed=7,
            settle=False,
        )
        assert result["n_reprices"] == 0
        assert len(result["quote_history"]) == 1
        assert result["final_quote"] == result["quote_history"][0]

    def test_tight_threshold_forces_reprices(self):
        result = simulate_flow_and_reprice(
            true_prob_a=0.5,
            n_bettors=200,
            belief_noise_std=0.06,
            bettor_edge_threshold=0.03,
            reprice_inventory_threshold=5.0,
            target_vig_pct=0.0,
            seed=7,
            settle=False,
        )
        assert result["n_reprices"] > 0
        assert len(result["quote_history"]) == result["n_reprices"] + 1

    def test_deterministic_symmetric_flow_self_corrects(self):
        # belief_noise_std=0 and bettor_edge_threshold=0: every bettor's belief
        # equals the maker's own true_prob_a exactly, so the instant the maker
        # reprices away from that value, the same informed population finds
        # value on the *other* side and corrects it straight back -- a real
        # property of a market maker quoting from the same information its
        # counterparties have, not an artifact. Hand-verified with seed=1.
        result = simulate_flow_and_reprice(
            true_prob_a=0.5,
            n_bettors=50,
            belief_noise_std=0.0,
            bettor_edge_threshold=0.0,
            reprice_inventory_threshold=5.0,
            reprice_step=0.01,
            target_vig_pct=0.0,
            seed=1,
            settle=False,
        )
        assert result["n_reprices"] == 10
        assert result["n_bets_team_a"] == result["n_bets_team_b"] == 25
        quoted_probs = {round(q["team_a_prob_quoted"], 4) for q in result["quote_history"]}
        assert quoted_probs == {0.5, 0.51}

    def test_settle_true_adds_outcome_and_pnl(self):
        result = simulate_flow_and_reprice(true_prob_a=0.6, n_bettors=50, seed=3, settle=True)
        assert isinstance(result["outcome_team_a_wins"], bool)
        assert isinstance(result["maker_pnl"], float)

    def test_settle_false_omits_outcome_and_pnl(self):
        result = simulate_flow_and_reprice(true_prob_a=0.6, n_bettors=50, seed=3, settle=False)
        assert "outcome_team_a_wins" not in result
        assert "maker_pnl" not in result

    def test_out_of_range_probability_raises(self):
        with pytest.raises(ValueError):
            simulate_flow_and_reprice(true_prob_a=0.0)

    def test_invalid_reprice_threshold_raises(self):
        with pytest.raises(ValueError):
            simulate_flow_and_reprice(true_prob_a=0.5, reprice_inventory_threshold=0.0)
