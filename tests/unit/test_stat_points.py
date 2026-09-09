"""Unit tests for Champions Stat Point arithmetic and fallback spreads."""

from __future__ import annotations

import pytest

from p0.teams.stat_points import (
    STAT_NAMES,
    STAT_POINT_TOTAL_LIMIT,
    BaseStats,
    StatPoints,
    calculate_stats,
    fallback_points,
)


class TestStatPoints:
    def test_default_values(self) -> None:
        points = StatPoints()
        assert points.as_tuple() == (0, 0, 0, 0, 0, 0)
        assert sum(points.as_tuple()) == 0

    def test_valid_spread_at_limits(self) -> None:
        points = StatPoints(hp=32, atk=32, spe=2)
        assert points.as_tuple() == (32, 32, 0, 0, 0, 2)
        assert sum(points.as_tuple()) == STAT_POINT_TOTAL_LIMIT

    def test_single_stat_exceeds_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="in \\[0, 32\\]"):
            StatPoints(hp=33)
        with pytest.raises(ValueError, match="in \\[0, 32\\]"):
            StatPoints(atk=-1)

    def test_total_budget_exceeded_raises(self) -> None:
        with pytest.raises(ValueError, match="at most 66 points"):
            StatPoints(hp=32, atk=32, spe=3)

    def test_non_integer_raises(self) -> None:
        with pytest.raises(TypeError, match="must be integers"):
            StatPoints(hp=3.5)  # type: ignore[arg-type]

    def test_as_dict_and_from_dict_roundtrip(self) -> None:
        points = StatPoints(hp=2, atk=0, defense=0, spa=32, spd=0, spe=32)
        d = points.as_dict()
        assert tuple(d.keys()) == STAT_NAMES
        assert StatPoints.from_dict(d) == points

    def test_from_dict_missing_field_raises(self) -> None:
        with pytest.raises(ValueError, match="Missing stat point field"):
            StatPoints.from_dict({"hp": 0, "atk": 0})


class TestBaseStats:
    def test_from_mapping_and_tuple(self) -> None:
        mapping = {"hp": 100, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100}
        stats = BaseStats.from_mapping(mapping)
        assert stats.as_tuple() == (100, 100, 100, 100, 100, 100)

    def test_from_mapping_missing_field_raises(self) -> None:
        with pytest.raises(ValueError, match="Missing base stat"):
            BaseStats.from_mapping({"hp": 100})


class TestCalculateStats:
    def test_level_50_neutral_nature(self) -> None:
        base = BaseStats(100, 100, 100, 100, 100, 100)
        points = StatPoints(hp=0, atk=0, defense=0, spa=0, spd=0, spe=0)
        # HP: (200 + 31) * 50 // 100 + 50 + 10 = 115 + 60 = 175
        # Other stats: (200 + 31) * 50 // 100 + 5 = 115 + 5 = 120
        stats = calculate_stats(base, points, "serious", level=50)
        assert stats == (175, 120, 120, 120, 120, 120)

    def test_level_50_with_max_stat_points(self) -> None:
        base = BaseStats(100, 100, 100, 100, 100, 100)
        points = StatPoints(hp=32, atk=32, defense=0, spa=0, spd=0, spe=2)
        # hp_iv = 2 * 32 - 1 = 63. HP: (200 + 31 + 63) * 50 // 100 + 60 = 147 + 60 = 207
        # atk_iv = 63. Atk raw: (200 + 31 + 63) * 50 // 100 + 5 = 152. Adamant: 152 * 110 // 100 = 167
        # spe_iv = 2 * 2 - 1 = 3. Spe raw: (200 + 31 + 3) * 50 // 100 + 5 = 117 + 5 = 122
        stats = calculate_stats(base, points, "adamant", level=50)
        assert stats[0] == 207
        assert stats[1] == 167
        assert stats[5] == 122

    def test_nature_boost_and_drop(self) -> None:
        base = BaseStats(80, 100, 80, 80, 80, 120)
        points = StatPoints(atk=32, spe=32, hp=2)
        jolly_stats = calculate_stats(base, points, "jolly", level=50)
        serious_stats = calculate_stats(base, points, "serious", level=50)
        # Jolly boosts Spe (110%) and reduces SpA (90%)
        assert jolly_stats[5] == serious_stats[5] * 110 // 100
        assert jolly_stats[3] == serious_stats[3] * 90 // 100
        assert jolly_stats[1] == serious_stats[1]  # Atk unchanged by Jolly

    def test_level_bounds_validation(self) -> None:
        base = BaseStats(100, 100, 100, 100, 100, 100)
        points = StatPoints()
        with pytest.raises(ValueError, match="Level must be in \\[1, 100\\]"):
            calculate_stats(base, points, "serious", level=0)
        with pytest.raises(ValueError, match="Level must be in \\[1, 100\\]"):
            calculate_stats(base, points, "serious", level=101)


class TestFallbackSpreads:
    def test_physical_category_fallback(self) -> None:
        spread = fallback_points(("physical", "physical", "status", "special"))
        assert spread == StatPoints(hp=32, atk=32, spe=2)

    def test_special_category_fallback(self) -> None:
        spread = fallback_points(("special", "special", "status"))
        assert spread == StatPoints(hp=32, spa=32, spe=2)

    def test_status_category_fallback(self) -> None:
        spread = fallback_points(("status", "status", "physical"))
        assert spread == StatPoints(hp=32, defense=17, spd=17)

    def test_insufficient_moves_returns_none(self) -> None:
        spread = fallback_points(("physical", "special", "status"))
        assert spread is None

    def test_case_insensitive_categories(self) -> None:
        spread = fallback_points(("PHYSICAL", "Physical"))
        assert spread == StatPoints(hp=32, atk=32, spe=2)
