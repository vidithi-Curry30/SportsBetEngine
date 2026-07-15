"""Shared helpers: date handling and team name normalization."""
from datetime import datetime, timezone

# Common sportsbook/media name variants, mapped to a single canonical form
# so odds from different books can be joined on the same game.
TEAM_NAME_ALIASES = {
    "la clippers": "los angeles clippers",
    "la lakers": "los angeles lakers",
    "ny knicks": "new york knicks",
    "gs warriors": "golden state warriors",
    "golden state": "golden state warriors",
    "sixers": "philadelphia 76ers",
    "philadelphia sixers": "philadelphia 76ers",
    "ok city thunder": "oklahoma city thunder",
    "okc thunder": "oklahoma city thunder",
}


def normalize_team_name(name: str) -> str:
    """Normalize a team name to a canonical lowercase form for joining across books."""
    cleaned = " ".join(name.strip().lower().split())
    return TEAM_NAME_ALIASES.get(cleaned, cleaned)


def parse_iso_timestamp(timestamp: str) -> datetime:
    """Parse an ISO-8601 timestamp (as returned by The Odds API) into a UTC datetime."""
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with a trailing 'Z'."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
