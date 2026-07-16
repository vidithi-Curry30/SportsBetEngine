from unittest.mock import MagicMock, patch

from src.mlb_stats_client import fetch_completed_games


def make_schedule_response(games_by_date):
    return {
        "dates": [
            {"date": date, "games": games}
            for date, games in games_by_date.items()
        ]
    }


def make_raw_game(home, away, home_score, away_score, state="Final"):
    return {
        "officialDate": "2026-04-01",
        "status": {"detailedState": state},
        "teams": {
            "home": {"team": {"name": home}, "score": home_score},
            "away": {"team": {"name": away}, "score": away_score},
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
