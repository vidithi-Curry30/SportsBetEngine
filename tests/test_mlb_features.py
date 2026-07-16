import pytest

from src.mlb_features import build_features


def make_game(date, home, away, home_score, away_score):
    return {
        "date": date,
        "home_team": home,
        "away_team": away,
        "home_score": home_score,
        "away_score": away_score,
        "home_win": int(home_score > away_score),
    }


class TestBuildFeatures:
    def test_first_game_for_each_team_is_dropped(self):
        # Both A and B are playing their first game -- no prior history to build features from
        games = [make_game("2026-04-01", "A", "B", 5, 3)]
        df = build_features(games)
        assert len(df) == 0

    def test_row_dropped_until_both_teams_have_history(self):
        games = [
            make_game("2026-04-01", "A", "B", 5, 3),  # both first games -> dropped
            make_game("2026-04-02", "A", "C", 2, 4),  # C's first game -> dropped
            make_game("2026-04-04", "B", "A", 1, 6),  # both have history now -> kept
        ]
        df = build_features(games)
        assert len(df) == 1
        assert df.iloc[0]["home_team"] == "B"
        assert df.iloc[0]["away_team"] == "A"

    def test_features_computed_only_from_prior_games(self):
        # Hand-computed expected values for the third game (B home vs A away)
        games = [
            make_game("2026-04-01", "A", "B", 5, 3),
            make_game("2026-04-02", "A", "C", 2, 4),
            make_game("2026-04-04", "B", "A", 1, 6),
        ]
        df = build_features(games)
        row = df.iloc[0]

        # B (home): 1 prior game, scored 3, allowed 5, lost -> avg_scored=3, avg_allowed=5,
        #           run_diff_per_game=-2, recent_win_pct=0, last game 2026-04-01 -> rest=3 days
        # A (away): 2 prior games, scored [5,2]=avg 3.5, allowed [3,4]=avg 3.5,
        #           run_diff_per_game=0, recent_win_pct=0.5, last game 2026-04-02 -> rest=2 days
        assert row["runs_scored_diff"] == pytest.approx(3 - 3.5)
        assert row["runs_allowed_diff"] == pytest.approx(3.5 - 5)
        assert row["season_run_diff_diff"] == pytest.approx(-2 - 0)
        assert row["recent_win_pct_diff"] == pytest.approx(0 - 0.5)
        assert row["rest_days_diff"] == pytest.approx(3 - 2)
        assert row["home_win"] == 0  # B (home) scored 1, lost to A's 6

    def test_a_later_games_outcome_never_affects_an_earlier_row(self):
        # Construct two datasets identical except for the outcome of the LAST game;
        # every earlier row must be byte-identical regardless of what happens later.
        base_games = [
            make_game("2026-04-01", "A", "B", 5, 3),
            make_game("2026-04-02", "A", "C", 2, 4),
            make_game("2026-04-04", "B", "A", 1, 6),
            make_game("2026-04-06", "C", "B", 7, 2),
        ]
        blowout_variant = base_games[:-1] + [make_game("2026-04-06", "C", "B", 20, 0)]

        df_base = build_features(base_games)
        df_variant = build_features(blowout_variant)

        pd_equal_cols = [c for c in df_base.columns if c not in ("date",)]
        earlier_rows_base = df_base.iloc[:-1][pd_equal_cols].reset_index(drop=True)
        earlier_rows_variant = df_variant.iloc[:-1][pd_equal_cols].reset_index(drop=True)
        assert earlier_rows_base.equals(earlier_rows_variant)

    def test_recent_win_pct_caps_at_ten_game_window(self):
        # A plays 12 games (home, vs 12 disposable one-off opponents): first 2 wins,
        # then 10 straight losses. If the window weren't capped at 10, the 2 early
        # wins would still be dragging recent_win_pct above 0.
        games = [make_game("2026-03-01", "SEED_A", "A", 0, 1)]  # give A a first prior game
        for i in range(12):
            won = i < 2
            games.append(
                make_game(f"2026-03-{i + 2:02d}", "A", f"OPP{i}", 5 if won else 1, 1 if won else 5)
            )

        # A fresh opponent Z with exactly one prior game with a fully known result
        # (a 1-0 win), so recent_win_pct_diff isolates A's value algebraically.
        games.append(make_game("2026-03-14", "SEED_Z", "Z", 0, 1))
        games.append(make_game("2026-03-15", "Z", "A", 1, 1))  # tie is fine, just need a row

        df = build_features(games)
        final_row = df[(df["home_team"] == "Z") & (df["away_team"] == "A")].iloc[0]

        # Z: 1 prior game, won 1-0 -> recent_win_pct = 1.0
        # A: 12 prior games, last 10 all losses -> recent_win_pct should be 0.0, not 2/12
        assert final_row["recent_win_pct_diff"] == pytest.approx(1.0 - 0.0)


class TestFeatureColumns:
    def test_no_lookahead_ordering_independent_of_input_order(self):
        # Feeding games out of chronological order must produce the same result as feeding
        # them in order -- build_features is responsible for sorting, not the caller.
        games_in_order = [
            make_game("2026-04-01", "A", "B", 5, 3),
            make_game("2026-04-02", "A", "C", 2, 4),
            make_game("2026-04-04", "B", "A", 1, 6),
        ]
        shuffled = [games_in_order[2], games_in_order[0], games_in_order[1]]

        df_ordered = build_features(games_in_order)
        df_shuffled = build_features(shuffled)

        assert df_ordered.equals(df_shuffled)
