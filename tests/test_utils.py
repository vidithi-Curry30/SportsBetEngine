from datetime import timezone

from src.utils import normalize_team_name, parse_iso_timestamp, utc_now_iso


class TestNormalizeTeamName:
    def test_passthrough_lowercase(self):
        assert normalize_team_name("Boston Celtics") == "boston celtics"

    def test_collapses_whitespace(self):
        assert normalize_team_name("  Boston   Celtics  ") == "boston celtics"

    def test_known_alias(self):
        assert normalize_team_name("LA Clippers") == "los angeles clippers"

    def test_known_alias_case_insensitive(self):
        assert normalize_team_name("gs warriors") == "golden state warriors"

    def test_unmapped_name_is_just_cleaned(self):
        assert normalize_team_name("Denver Nuggets") == "denver nuggets"


class TestParseIsoTimestamp:
    def test_parses_zulu_suffix(self):
        dt = parse_iso_timestamp("2026-01-15T00:30:00Z")
        assert dt.year == 2026
        assert dt.month == 1
        assert dt.tzinfo is not None
        assert dt.utcoffset().total_seconds() == 0


class TestUtcNowIso:
    def test_returns_zulu_suffixed_string(self):
        result = utc_now_iso()
        assert result.endswith("Z")
        # Round-trips through the parser above
        dt = parse_iso_timestamp(result)
        assert dt.tzinfo is not None
