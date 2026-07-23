from unittest.mock import MagicMock, patch

from src.mlb_stats_client import fetch_completed_games, fetch_pitcher_game_log


def make_schedule_response(games_by_date):
    return {
        "dates": [
            {"date": date, "games": games}
            for date, games in games_by_date.items()
        ]
    }


def make_raw_game(
    home, away, home_score, away_score, state="Final",
    game_pk=1, home_pitcher=None, away_pitcher=None,
):
    return {
        "gamePk": game_pk,
        "officialDate": "2026-04-01",
        "status": {"detailedState": state},
        "teams": {
            "home": {
                "team": {"name": home},
                "score": home_score,
                **({"probablePitcher": home_pitcher} if home_pitcher else {}),
            },
            "away": {
                "team": {"name": away},
                "score": away_score,
                **({"probablePitcher": away_pitcher} if away_pitcher else {}),
            },
        },
    }


class TestFetchCompletedGames:
    def test_parses_final_games_only(self):
        payload = make_schedule_response(
            {
                "2026-04-01": [
                    make_raw_game("Boston Red Sox", "New York Yankees", 5, 3),
                    make_raw_game("Chicago Cubs", "St. Louis Cardinals", 2, 2, state="In Progress"),
                ]
            }
        )
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = payload
        mock_response.raise_for_status = MagicMock()

        with patch("src.mlb_stats_client.requests.get", return_value=mock_response):
            games = fetch_completed_games("2026-04-01", "2026-04-01")

        assert len(games) == 1
        assert games[0]["home_team"] == "Boston Red Sox"
        assert games[0]["away_team"] == "New York Yankees"
        assert games[0]["home_win"] == 1

    def test_skips_games_missing_scores(self):
        game = make_raw_game("Boston Red Sox", "New York Yankees", 0, 0)
        del game["teams"]["home"]["score"]
        payload = make_schedule_response({"2026-04-01": [game]})
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = payload
        mock_response.raise_for_status = MagicMock()

        with patch("src.mlb_stats_client.requests.get", return_value=mock_response):
            games = fetch_completed_games("2026-04-01", "2026-04-01")

        assert games == []

    def test_sorts_by_date(self):
        payload = make_schedule_response(
            {
                "2026-04-02": [make_raw_game("Team A", "Team B", 1, 0)],
                "2026-04-01": [make_raw_game("Team C", "Team D", 2, 1)],
            }
        )
        for g in payload["dates"][0]["games"]:
            g["officialDate"] = "2026-04-02"
        for g in payload["dates"][1]["games"]:
            g["officialDate"] = "2026-04-01"

        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = payload
        mock_response.raise_for_status = MagicMock()

        with patch("src.mlb_stats_client.requests.get", return_value=mock_response):
            games = fetch_completed_games("2026-04-01", "2026-04-02")

        assert [g["date"] for g in games] == ["2026-04-01", "2026-04-02"]

    def test_correct_url_and_params(self):
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {"dates": []}
        mock_response.raise_for_status = MagicMock()

        with patch("src.mlb_stats_client.requests.get", return_value=mock_response) as mock_get:
            fetch_completed_games("2026-04-01", "2026-04-30")

        called_url = mock_get.call_args.args[0]
        called_params = mock_get.call_args.kwargs["params"]
        assert called_url.endswith("/schedule")
        assert called_params["sportId"] == 1
        assert called_params["startDate"] == "2026-04-01"
        assert called_params["endDate"] == "2026-04-30"
        assert called_params["gameType"] == "R"
        assert called_params["hydrate"] == "probablePitcher"

    def test_parses_probable_pitchers_when_present(self):
        payload = make_schedule_response(
            {
                "2026-04-01": [
                    make_raw_game(
                        "Boston Red Sox", "New York Yankees", 5, 3, game_pk=555,
                        home_pitcher={"id": 111, "fullName": "Home Starter"},
                        away_pitcher={"id": 222, "fullName": "Away Starter"},
                    )
                ]
            }
        )
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = payload
        mock_response.raise_for_status = MagicMock()

        with patch("src.mlb_stats_client.requests.get", return_value=mock_response):
            games = fetch_completed_games("2026-04-01", "2026-04-01")

        assert games[0]["game_pk"] == 555
        assert games[0]["home_pitcher_id"] == 111
        assert games[0]["home_pitcher_name"] == "Home Starter"
        assert games[0]["away_pitcher_id"] == 222
        assert games[0]["away_pitcher_name"] == "Away Starter"

    def test_pitcher_fields_are_none_when_not_recorded(self):
        payload = make_schedule_response(
            {"2026-04-01": [make_raw_game("Boston Red Sox", "New York Yankees", 5, 3)]}
        )
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = payload
        mock_response.raise_for_status = MagicMock()

        with patch("src.mlb_stats_client.requests.get", return_value=mock_response):
            games = fetch_completed_games("2026-04-01", "2026-04-01")

        assert games[0]["home_pitcher_id"] is None
        assert games[0]["away_pitcher_id"] is None


class TestFetchPitcherGameLog:
    def test_parses_starts_only_skips_relief_appearances(self):
        payload = {
            "stats": [
                {
                    "splits": [
                        {
                            "date": "2026-03-28",
                            "game": {"gamePk": 1},
                            "stat": {"gamesStarted": 1, "outs": 15, "earnedRuns": 3, "strikeOuts": 7},
                        },
                        {
                            "date": "2026-04-02",
                            "game": {"gamePk": 2},
                            "stat": {"gamesStarted": 0, "outs": 3, "earnedRuns": 0, "strikeOuts": 1},
                        },
                    ]
                }
            ]
        }
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = payload
        mock_response.raise_for_status = MagicMock()

        with patch("src.mlb_stats_client.requests.get", return_value=mock_response):
            starts = fetch_pitcher_game_log(605400, "2026")

        assert len(starts) == 1
        assert starts[0]["date"] == "2026-03-28"
        assert starts[0]["outs"] == 15
        assert starts[0]["earned_runs"] == 3
        assert starts[0]["strikeouts"] == 7

    def test_sorts_by_date(self):
        payload = {
            "stats": [
                {
                    "splits": [
                        {
                            "date": "2026-05-01", "game": {"gamePk": 2},
                            "stat": {"gamesStarted": 1, "outs": 18, "earnedRuns": 2, "strikeOuts": 5},
                        },
                        {
                            "date": "2026-04-01", "game": {"gamePk": 1},
                            "stat": {"gamesStarted": 1, "outs": 15, "earnedRuns": 3, "strikeOuts": 7},
                        },
                    ]
                }
            ]
        }
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = payload
        mock_response.raise_for_status = MagicMock()

        with patch("src.mlb_stats_client.requests.get", return_value=mock_response):
            starts = fetch_pitcher_game_log(605400, "2026")

        assert [s["date"] for s in starts] == ["2026-04-01", "2026-05-01"]

    def test_correct_url_and_params(self):
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {"stats": []}
        mock_response.raise_for_status = MagicMock()

        with patch("src.mlb_stats_client.requests.get", return_value=mock_response) as mock_get:
            fetch_pitcher_game_log(605400, "2026")

        called_url = mock_get.call_args.args[0]
        called_params = mock_get.call_args.kwargs["params"]
        assert called_url.endswith("/people/605400/stats")
        assert called_params["stats"] == "gameLog"
        assert called_params["group"] == "pitching"
        assert called_params["season"] == "2026"
