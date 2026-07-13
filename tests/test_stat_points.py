import pytest

from p0.team_data.stat_points import (
    BaseStats,
    ImputationInput,
    Role,
    StatPoints,
    calculate_stats,
    classify_role,
    impute_candidates,
    imputed_stats,
    select_candidate,
)

CHARIZARD = BaseStats(78, 84, 78, 109, 85, 100)


def test_champions_stat_formula_matches_pinned_level_50_oracle():
    assert calculate_stats(CHARIZARD, StatPoints(), "serious") == (153, 104, 98, 129, 105, 120)
    assert calculate_stats(CHARIZARD, StatPoints(hp=2, spa=32, spe=32), "modest") == (
        155,
        93,
        98,
        177,
        105,
        152,
    )


def test_first_point_and_nature_truncation_match_showdown():
    base = BaseStats(100, 100, 100, 100, 100, 100)
    zero = calculate_stats(base, StatPoints(), "adamant")
    one = calculate_stats(base, StatPoints(atk=1), "adamant")
    assert zero == (175, 132, 120, 108, 120, 120)
    assert one[1] == zero[1] + 1


@pytest.mark.parametrize(
    "points, error",
    [
        (dict(hp=33), r"\[0, 32\]"),
        (dict(hp=32, atk=32, defense=3), "at most 66"),
    ],
)
def test_stat_point_validation(points, error):
    with pytest.raises(ValueError, match=error):
        StatPoints(**points)


def _input(*, moves=("Heat Wave", "Protect"), categories=("special", "status"), nature="modest"):
    return ImputationInput(
        species="charizard",
        nature=nature,
        item="charizarditey",
        ability="blaze",
        moves=moves,
        move_categories=categories,
        base_stats=CHARIZARD,
    )


def test_imputer_is_legal_deterministic_and_role_sensitive():
    value = _input()
    assert classify_role(value) == Role.SPECIAL
    assert impute_candidates(value) == impute_candidates(value)
    assert impute_candidates(value)[0].points == StatPoints(hp=2, spa=32, spe=32)
    assert all(sum(candidate.points.as_tuple()) <= 66 for candidate in impute_candidates(value))
    assert select_candidate(value, seed=7) == select_candidate(value, seed=7)


def test_imputed_level_stats_are_cached_by_static_team_facts():
    value = _input()
    imputed_stats.cache_clear()
    first = imputed_stats(value)
    second = imputed_stats(value)
    assert first == second
    assert imputed_stats.cache_info().hits == 1


def test_trick_room_and_support_shapes_do_not_assume_fast_offense():
    trick_room = _input(
        moves=("Trick Room", "Heat Wave"),
        categories=("status", "special"),
        nature="quiet",
    )
    support = _input(moves=("Protect", "Follow Me"), categories=("status", "status"), nature="calm")
    assert classify_role(trick_room) == Role.TRICK_ROOM
    assert impute_candidates(trick_room)[0].points.spe == 0
    assert classify_role(support) == Role.SUPPORT
    assert impute_candidates(support)[0].points == StatPoints(hp=32, defense=17, spd=17)
