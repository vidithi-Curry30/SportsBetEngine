"""Thin wrapper around the free, public MLB Stats API (statsapi.mlb.com) --
real team names, dates, and final scores for completed regular-season games.

Unlike stats.nba.com, this endpoint is reachable without auth and has no
rate-limit hostility problem, which is why the model is trained on real MLB
data (see src/mlb_features.py) rather than NBA -- NBA is off-season anyway.
"""
import requests

MLB_STATS_BASE_URL = "https://statsapi.mlb.com/api/v1"
MLB_SPORT_ID = 1  # statsapi.mlb.com's sportId for the MLB (major league)


def fetch_completed_games(start_date: str, end_date: str) -> list[dict]:
    """Fetch final regular-season games between start_date and end_date
    (both 'YYYY-MM-DD'), parsed into a flat list of dicts:
    {date, home_team, away_team, home_score, away_score, home_win}.
    """
    response = requests.get(
        f"{MLB_STATS_BASE_URL}/schedule",
        params={
            "sportId": MLB_SPORT_ID,
            "startDate": start_date,
            "endDate": end_date,
            "gameType": "R",
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()

    games = []
    for date_entry in payload.get("dates", []):
        for game in date_entry.get("games", []):
            if game["status"]["detailedState"] != "Final":
                continue

            home = game["teams"]["home"]
            away = game["teams"]["away"]
            if "score" not in home or "score" not in away:
                continue

            games.append(
                {
                    "date": game["officialDate"],
                    "home_team": home["team"]["name"],
                    "away_team": away["team"]["name"],
                    "home_score": home["score"],
                    "away_score": away["score"],
                    "home_win": int(home["score"] > away["score"]),
                }
            )

    games.sort(key=lambda g: g["date"])
    return games
