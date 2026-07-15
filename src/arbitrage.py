"""Cross-book arbitrage scanner for two-way (moneyline) markets."""
from itertools import permutations

from src.probability import american_to_probability


def find_arbitrage(odds_by_book: dict[str, dict[str, int]]) -> list[dict]:
    """Scan a single game's odds across books for a cross-book arbitrage.

    odds_by_book: {book_name: {"team_a": american_odds, "team_b": american_odds}}

    Checks every ordered book pair (X, Y) with X != Y: best price on team_a
    from book X against team_b from book Y. If the summed implied probability
    is < 1, that pair is a guaranteed-profit arbitrage; profit_pct = 1 - sum.
    """
    opportunities = []

    for book_x, book_y in permutations(odds_by_book, 2):
        odds_a = odds_by_book[book_x].get("team_a")
        odds_b = odds_by_book[book_y].get("team_b")
        if odds_a is None or odds_b is None:
            continue

        prob_a = american_to_probability(odds_a)
        prob_b = american_to_probability(odds_b)
        total = prob_a + prob_b

        if total < 1:
            opportunities.append(
                {
                    "book_a": book_x,
                    "team_a_odds": odds_a,
                    "book_b": book_y,
                    "team_b_odds": odds_b,
                    "implied_prob_sum": total,
                    "profit_pct": (1 - total) * 100,
                }
            )

    opportunities.sort(key=lambda o: o["profit_pct"], reverse=True)
    return opportunities


def scan_games_for_arbitrage(games: list[dict]) -> list[dict]:
    """Scan a batch of games in The Odds API's raw response shape for arbitrage.

    Each game is expected to have "home_team", "away_team", and a "bookmakers"
    list of {"title", "markets": [{"key": "h2h", "outcomes": [...]}]}.
    Returns a flat list of opportunities, each tagged with the source game.
    """
    all_opportunities = []

    for game in games:
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        odds_by_book = _extract_h2h_odds_by_book(game, home_team, away_team)

        for opportunity in find_arbitrage(odds_by_book):
            all_opportunities.append(
                {
                    "game_id": game.get("id"),
                    "home_team": home_team,
                    "away_team": away_team,
                    "commence_time": game.get("commence_time"),
                    **opportunity,
                }
            )

    return all_opportunities


def _extract_h2h_odds_by_book(game: dict, home_team: str, away_team: str) -> dict[str, dict[str, int]]:
    odds_by_book: dict[str, dict[str, int]] = {}

    for bookmaker in game.get("bookmakers", []):
        h2h_market = next((m for m in bookmaker.get("markets", []) if m.get("key") == "h2h"), None)
        if h2h_market is None:
            continue

        prices = {outcome["name"]: outcome["price"] for outcome in h2h_market.get("outcomes", [])}
        if home_team not in prices or away_team not in prices:
            continue

        odds_by_book[bookmaker.get("title", bookmaker.get("key"))] = {
            "team_a": prices[home_team],
            "team_b": prices[away_team],
        }

    return odds_by_book
