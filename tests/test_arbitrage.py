import pytest

from src.arbitrage import find_arbitrage, scan_games_for_arbitrage


class TestFindArbitrage:
    def test_detects_known_arbitrage(self):
        # +105 on both sides at different books: 100/205 + 100/205 = 0.97561 < 1
        odds_by_book = {
            "BookA": {"team_a": 105, "team_b": -200},
            "BookB": {"team_a": -200, "team_b": 105},
        }
        opportunities = find_arbitrage(odds_by_book)

        assert len(opportunities) == 1
        opp = opportunities[0]
        assert opp["book_a"] == "BookA"
        assert opp["team_a_odds"] == 105
        assert opp["book_b"] == "BookB"
        assert opp["team_b_odds"] == 105
        assert opp["profit_pct"] == pytest.approx(2.4390, abs=1e-3)

    def test_no_arbitrage_with_standard_vig(self):
        # Standard -110/-110 juice at every book: no arb possible
        odds_by_book = {
            "BookA": {"team_a": -110, "team_b": -110},
            "BookB": {"team_a": -110, "team_b": -110},
            "BookC": {"team_a": -110, "team_b": -110},
        }
        assert find_arbitrage(odds_by_book) == []

    def test_single_book_produces_no_pairs(self):
        odds_by_book = {"BookA": {"team_a": 105, "team_b": 105}}
        assert find_arbitrage(odds_by_book) == []

    def test_missing_side_is_skipped(self):
        # Neither book quotes team_b, so no valid team_a/team_b pairing exists
        odds_by_book = {
            "BookA": {"team_a": 105},
            "BookB": {"team_a": -105},
        }
        assert find_arbitrage(odds_by_book) == []

    def test_best_opportunity_sorted_first(self):
        odds_by_book = {
            "BookA": {"team_a": 105, "team_b": -105},
            "BookB": {"team_a": -105, "team_b": 105},
            "BookC": {"team_a": 130, "team_b": -105},
        }
        opportunities = find_arbitrage(odds_by_book)
        profit_pcts = [o["profit_pct"] for o in opportunities]
        assert profit_pcts == sorted(profit_pcts, reverse=True)


class TestScanGamesForArbitrage:
    def test_extracts_and_flags_arbitrage_across_games(self):
        games = [
            {
                "id": "game1",
                "home_team": "Boston Celtics",
                "away_team": "New York Knicks",
                "commence_time": "2026-01-15T00:30:00Z",
                "bookmakers": [
                    {
                        "title": "BookA",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Boston Celtics", "price": 105},
                                    {"name": "New York Knicks", "price": -200},
                                ],
                            }
                        ],
                    },
                    {
                        "title": "BookB",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Boston Celtics", "price": -200},
                                    {"name": "New York Knicks", "price": 105},
                                ],
                            }
                        ],
                    },
                ],
            },
            {
                "id": "game2",
                "home_team": "Golden State Warriors",
                "away_team": "Los Angeles Lakers",
                "commence_time": "2026-01-15T03:00:00Z",
                "bookmakers": [
                    {
                        "title": "BookA",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Golden State Warriors", "price": -110},
                                    {"name": "Los Angeles Lakers", "price": -110},
                                ],
                            }
                        ],
                    },
                    {
                        "title": "BookB",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Golden State Warriors", "price": -110},
                                    {"name": "Los Angeles Lakers", "price": -110},
                                ],
                            }
                        ],
                    },
                ],
            },
        ]

        opportunities = scan_games_for_arbitrage(games)

        assert len(opportunities) == 1
        assert opportunities[0]["game_id"] == "game1"
        assert opportunities[0]["home_team"] == "Boston Celtics"
        assert opportunities[0]["profit_pct"] == pytest.approx(2.4390, abs=1e-3)

    def test_no_arbitrage_returns_empty_list(self):
        games = [
            {
                "id": "game2",
                "home_team": "Golden State Warriors",
                "away_team": "Los Angeles Lakers",
                "bookmakers": [
                    {
                        "title": "BookA",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Golden State Warriors", "price": -110},
                                    {"name": "Los Angeles Lakers", "price": -110},
                                ],
                            }
                        ],
                    },
                ],
            },
        ]
        assert scan_games_for_arbitrage(games) == []

    def test_ignores_bookmaker_without_h2h_market(self):
        games = [
            {
                "id": "game3",
                "home_team": "Boston Celtics",
                "away_team": "New York Knicks",
                "bookmakers": [
                    {"title": "BookA", "markets": [{"key": "spreads", "outcomes": []}]},
                ],
            },
        ]
        assert scan_games_for_arbitrage(games) == []
