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

# The Odds API sport key. NBA is off-season as of this writing (regular
# season odds aren't quoted between the Finals and October); MLB is the
# live US major sport right now. Swap via DEFAULT_SPORT env var, or
# --sport on scripts/fetch_odds.py, as the season calendar moves on.
DEFAULT_SPORT = os.getenv("DEFAULT_SPORT", "baseball_mlb")
DEFAULT_REGIONS = "us"
DEFAULT_MARKETS = "h2h"  # moneyline
DEFAULT_ODDS_FORMAT = "american"
