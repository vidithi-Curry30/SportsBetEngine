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
    (both 'YYYY-MM-DD'), parsed into a flat list of dicts: {date, game_pk,
    home_team, away_team, home_score, away_score, home_win, home_pitcher_id,
    home_pitcher_name, away_pitcher_id, away_pitcher_name}. Pitcher fields
    are None when the schedule didn't record a probable starter for that game.
    """
    response = requests.get(
        f"{MLB_STATS_BASE_URL}/schedule",
        params={
            "sportId": MLB_SPORT_ID,
            "startDate": start_date,
            "endDate": end_date,
            "gameType": "R",
            "hydrate": "probablePitcher",
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

            home_pitcher = home.get("probablePitcher") or {}
            away_pitcher = away.get("probablePitcher") or {}

            games.append(
                {
                    "date": game["officialDate"],
                    "game_pk": game["gamePk"],
                    "home_team": home["team"]["name"],
                    "away_team": away["team"]["name"],
                    "home_score": home["score"],
                    "away_score": away["score"],
                    "home_win": int(home["score"] > away["score"]),
                    "home_pitcher_id": home_pitcher.get("id"),
                    "home_pitcher_name": home_pitcher.get("fullName"),
                    "away_pitcher_id": away_pitcher.get("id"),
                    "away_pitcher_name": away_pitcher.get("fullName"),
                }
            )

    games.sort(key=lambda g: g["date"])
    return games


def fetch_pitcher_game_log(pitcher_id: int, season: str) -> list[dict]:
    """Fetch a pitcher's full-season game-by-game pitching log, parsed into
    {date, game_pk, outs, earned_runs, strikeouts}, sorted chronologically.

    Uses `outs` (total outs recorded) rather than the API's "innings pitched"
    string (e.g. "5.1" means 5 and 1/3 innings, NOT 5.1 innings -- a classic
    parsing trap) to keep rolling-ERA arithmetic exact.
    """
    response = requests.get(
        f"{MLB_STATS_BASE_URL}/people/{pitcher_id}/stats",
        params={"stats": "gameLog", "group": "pitching", "season": season},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()

    starts = []
    for group in payload.get("stats", []):
        for split in group.get("splits", []):
            stat = split.get("stat", {})
            if not stat.get("gamesStarted"):
                continue  # relief appearances aren't comparable to a starter's workload

            starts.append(
                {
                    "date": split["date"],
                    "game_pk": split["game"]["gamePk"],
                    "outs": stat.get("outs", 0),
                    "earned_runs": stat.get("earnedRuns", 0),
                    "strikeouts": stat.get("strikeOuts", 0),
                }
            )

    starts.sort(key=lambda s: s["date"])
    return starts
