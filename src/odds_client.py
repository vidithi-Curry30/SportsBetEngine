"""Thin wrapper around The Odds API (https://the-odds-api.com/)."""
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

import config


class OddsAPIError(RuntimeError):
    """Raised when The Odds API returns a non-2xx response."""


class OddsAPIClient:
    def __init__(self, api_key: str | None = None, base_url: str = config.ODDS_API_BASE_URL):
        self.api_key = api_key or config.ODDS_API_KEY
        if not self.api_key:
            raise ValueError(
                "ODDS_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        self.base_url = base_url

    def get_sports(self) -> list[dict]:
        """List sports currently available from the API."""
        response = requests.get(
            f"{self.base_url}/sports",
            params={"apiKey": self.api_key},
            timeout=10,
        )
        self._raise_for_status(response)
        return response.json()

    def get_odds(
        self,
        sport: str = config.DEFAULT_SPORT,
        regions: str = config.DEFAULT_REGIONS,
        markets: str = config.DEFAULT_MARKETS,
        odds_format: str = config.DEFAULT_ODDS_FORMAT,
    ) -> list[dict]:
        """Fetch current odds for a sport across books. Free tier returns live odds only."""
        response = requests.get(
            f"{self.base_url}/sports/{sport}/odds",
            params={
                "apiKey": self.api_key,
                "regions": regions,
                "markets": markets,
                "oddsFormat": odds_format,
            },
            timeout=10,
        )
        self._raise_for_status(response)
        return response.json()

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        if response.status_code != 200:
            raise OddsAPIError(f"{response.status_code} {response.reason}: {response.text}")


def save_raw_pull(data: list[dict], sport: str, raw_dir: Path = config.DATA_RAW_DIR) -> Path:
    """Save an unmodified API pull to data/raw/, one timestamped file per pull."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = raw_dir / f"{sport}_{timestamp}.json"
    out_path.write_text(json.dumps(data, indent=2))
    return out_path
