import json
from unittest.mock import MagicMock, patch

import pytest

from src.odds_client import OddsAPIClient, OddsAPIError, save_raw_pull


class TestOddsAPIClient:
    def test_requires_api_key(self):
        # Patch out config.ODDS_API_KEY so this doesn't depend on whether a
        # real .env happens to be present in the environment running the test.
        with patch("src.odds_client.config.ODDS_API_KEY", None):
            with pytest.raises(ValueError):
                OddsAPIClient(api_key="")

    def test_get_odds_builds_expected_request(self):
        client = OddsAPIClient(api_key="test-key")
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = [{"id": "game1"}]

        with patch("src.odds_client.requests.get", return_value=mock_response) as mock_get:
            data = client.get_odds(sport="basketball_nba", regions="us", markets="h2h")

        mock_get.assert_called_once()
        called_url = mock_get.call_args.args[0]
        called_params = mock_get.call_args.kwargs["params"]
        assert called_url.endswith("/sports/basketball_nba/odds")
        assert called_params["apiKey"] == "test-key"
        assert called_params["regions"] == "us"
        assert called_params["markets"] == "h2h"
        assert data == [{"id": "game1"}]

    def test_non_200_raises_odds_api_error(self):
        client = OddsAPIClient(api_key="test-key")
        mock_response = MagicMock(status_code=401, reason="Unauthorized", text="bad key")

        with patch("src.odds_client.requests.get", return_value=mock_response):
            with pytest.raises(OddsAPIError):
                client.get_odds()


class TestSaveRawPull:
    def test_writes_timestamped_json_file(self, tmp_path):
        data = [{"id": "game1", "home_team": "Boston Celtics"}]
        out_path = save_raw_pull(data, sport="basketball_nba", raw_dir=tmp_path)

        assert out_path.exists()
        assert out_path.parent == tmp_path
        assert "basketball_nba" in out_path.name
        assert json.loads(out_path.read_text()) == data
