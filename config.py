"""Central configuration: env vars, paths, and default API parameters."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# The Odds API sport key. Swap to whatever US major sport is in-season
# (e.g. "americanfootball_nfl", "baseball_mlb") when NBA is out of season.
DEFAULT_SPORT = os.getenv("DEFAULT_SPORT", "basketball_nba")
DEFAULT_REGIONS = "us"
DEFAULT_MARKETS = "h2h"  # moneyline
DEFAULT_ODDS_FORMAT = "american"
