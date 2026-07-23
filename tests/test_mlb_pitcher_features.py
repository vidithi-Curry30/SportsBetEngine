from unittest.mock import patch

import pytest

from src.mlb_pitcher_features import build_pitching_features, fetch_all_starter_game_logs


def make_start(date, outs, earned_runs, strikeouts, game_pk=0):
    return {"date": date, "game_pk": game_pk, "outs": outs, "earned_runs": earned_runs, "strikeouts": strikeouts}


def make_game(date, home_team, away_team, home_pitcher_id, away_pitcher_id, game_pk=1):
    return {
        "date": date,
        "game_pk": game_pk,
        "home_team": home_team,
        "away_team": away_team,
        "home_pitcher_id": home_pitcher_id,
        "away_pitcher_id": away_pitcher_id,
    }


class TestBuildPitchingFeatures:
    def test_drops_games_missing_a_probable_pitcher(self):
        games = [make_game("2026-04-15", "A", "B", home_pitcher_id=None, away_pitcher_id=222)]
        df = build_pitching_features(games, pitcher_logs={})
        assert len(df) == 0

    def test_drops_games_with_a_debuting_starter(self):
        # Pitcher 111 has no starts at all recorded before this game
        games = [make_game("2026-04-15", "A", "B", home_pitcher_id=111, away_pitcher_id=222)]
        pitcher_logs = {222: [make_start("2026-04-03", outs=21, earned_runs=2, strikeouts=5)]}
        df = build_pitching_features(games, pitcher_logs)
        assert len(df) == 0

    def test_era_and_k9_diff_hand_computed(self):
        home_id, away_id = 111, 222
        pitcher_logs = {
            home_id: [
                make_start("2026-04-01", outs=15, earned_runs=3, strikeouts=7),
                make_start("2026-04-08", outs=18, earned_runs=1, strikeouts=6),
            ],
            away_id: [
                make_start("2026-04-03", outs=21, earned_runs=2, strikeouts=5),
            ],
        }
        games = [make_game("2026-04-15", "A", "B", home_pitcher_id=home_id, away_pitcher_id=away_id)]

        df = build_pitching_features(games, pitcher_logs)
        row = df.iloc[0]

        home_era = 27 * (3 + 1) / (15 + 18)
        home_k9 = 27 * (7 + 6) / (15 + 18)
        away_era = 27 * 2 / 21
        away_k9 = 27 * 5 / 21

        assert row["starting_pitcher_era_diff"] == pytest.approx(away_era - home_era)
        assert row["starting_pitcher_k9_diff"] == pytest.approx(home_k9 - away_k9)

    def test_only_counts_starts_strictly_before_the_game_date(self):
        home_id, away_id = 111, 222
        pitcher_logs = {
            home_id: [make_start("2026-04-01", outs=15, earned_runs=3, strikeouts=7)],
            away_id: [make_start("2026-04-15", outs=21, earned_runs=2, strikeouts=5)],  # same day as the game
        }
        games = [make_game("2026-04-15", "A", "B", home_pitcher_id=home_id, away_pitcher_id=away_id)]

        # Away pitcher's only "start" is on the game date itself, not before it -> no prior history -> dropped
        df = build_pitching_features(games, pitcher_logs)
        assert len(df) == 0

    def test_a_later_starts_stats_never_affect_an_earlier_games_features(self):
        home_id, away_id = 111, 222
        base_logs = {
            home_id: [make_start("2026-04-01", outs=15, earned_runs=3, strikeouts=7)],
            away_id: [make_start("2026-04-03", outs=21, earned_runs=2, strikeouts=5)],
        }
        games = [make_game("2026-04-15", "A", "B", home_pitcher_id=home_id, away_pitcher_id=away_id)]

        df_before = build_pitching_features(games, base_logs)

        # Add a wildly different LATER start for the home pitcher (after the predicted game)
        blown_up_logs = {
            home_id: base_logs[home_id] + [make_start("2026-04-20", outs=3, earned_runs=9, strikeouts=0)],
            away_id: base_logs[away_id],
        }
        df_after = build_pitching_features(games, blown_up_logs)

        assert df_before["starting_pitcher_era_diff"].iloc[0] == pytest.approx(
            df_after["starting_pitcher_era_diff"].iloc[0]
        )

    def test_positive_era_diff_favors_home_when_home_pitcher_has_lower_era(self):
        home_id, away_id = 111, 222
        pitcher_logs = {
            home_id: [make_start("2026-04-01", outs=27, earned_runs=1, strikeouts=8)],  # ERA 1.0, sharp
            away_id: [make_start("2026-04-01", outs=27, earned_runs=6, strikeouts=8)],  # ERA 6.0, shaky
        }
        games = [make_game("2026-04-15", "A", "B", home_pitcher_id=home_id, away_pitcher_id=away_id)]

        df = build_pitching_features(games, pitcher_logs)
        assert df.iloc[0]["starting_pitcher_era_diff"] > 0


class TestFetchAllStarterGameLogs:
    def test_fetches_each_unique_pitcher_once(self):
        games = [
            make_game("2026-04-01", "A", "B", home_pitcher_id=111, away_pitcher_id=222),
            make_game("2026-04-08", "A", "C", home_pitcher_id=111, away_pitcher_id=333),  # 111 reused
        ]

        with patch("src.mlb_pitcher_features.fetch_pitcher_game_log", return_value=[]) as mock_fetch:
            logs = fetch_all_starter_game_logs(games, season="2026")

        assert mock_fetch.call_count == 3  # 111, 222, 333 -- not 4
        assert set(logs.keys()) == {111, 222, 333}

    def test_skips_games_with_no_probable_pitcher(self):
        games = [make_game("2026-04-01", "A", "B", home_pitcher_id=None, away_pitcher_id=None)]

        with patch("src.mlb_pitcher_features.fetch_pitcher_game_log", return_value=[]) as mock_fetch:
            logs = fetch_all_starter_game_logs(games, season="2026")

        assert mock_fetch.call_count == 0
        assert logs == {}

    def test_retries_transient_connection_errors_then_succeeds(self):
        import requests

        games = [make_game("2026-04-01", "A", "B", home_pitcher_id=111, away_pitcher_id=222)]
        call_count = {111: 0, 222: 0}

        def flaky_fetch(pitcher_id, season):
            call_count[pitcher_id] += 1
            if pitcher_id == 111 and call_count[111] < 2:
                raise requests.exceptions.ConnectionError("simulated reset")
            return [make_start("2026-03-01", outs=15, earned_runs=2, strikeouts=6)]

        with patch("src.mlb_pitcher_features.fetch_pitcher_game_log", side_effect=flaky_fetch):
            logs = fetch_all_starter_game_logs(games, season="2026", retry_backoff_seconds=0)

        assert call_count[111] == 2  # failed once, succeeded on retry
        assert 111 in logs and 222 in logs

    def test_skips_pitcher_that_fails_every_retry_without_crashing(self):
        import requests

        games = [make_game("2026-04-01", "A", "B", home_pitcher_id=111, away_pitcher_id=222)]

        def always_fails(pitcher_id, season):
            if pitcher_id == 111:
                raise requests.exceptions.ConnectionError("simulated reset")
            return [make_start("2026-03-01", outs=15, earned_runs=2, strikeouts=6)]

        with patch("src.mlb_pitcher_features.fetch_pitcher_game_log", side_effect=always_fails):
            logs = fetch_all_starter_game_logs(games, season="2026", max_retries=2, retry_backoff_seconds=0)

        assert 111 not in logs
        assert 222 in logs
