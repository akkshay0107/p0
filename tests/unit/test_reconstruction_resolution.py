"""Public resolver tests for faint lifetimes, slotless references and Illusion evidence."""

from __future__ import annotations

from dataclasses import replace

import pytest

from p0.replays.identity import ReplayMemberId, ReplaySide
from p0.replays.reconstruction.diagnostics import ReplayRejectionCategory
from p0.replays.reconstruction.events import parse_protocol_events
from p0.replays.reconstruction.resolution import resolve_protocol_events
from p0.replays.schema import OTSData, OTSMember, ProtocolLine


def _sheet(side: ReplaySide, members: tuple[tuple[str, str, tuple[str, ...]], ...]) -> OTSData:
    """Build a team sheet from one (species, ability, moves) entry per member."""
    return OTSData(
        side,
        "]".join(species for species, _, _ in members),
        tuple(
            OTSMember(
                ReplayMemberId(side, index),
                species,
                species,
                "",
                ability,
                moves,
                "Serious",
                "",
                50,
                "",
                species,
            )
            for index, (species, ability, moves) in enumerate(members)
        ),
    )


def _ots(side: ReplaySide, names: tuple[str, ...]) -> OTSData:
    return _sheet(side, tuple((name, "Ability", ("Protect",)) for name in names))


_ILLUSION_SIDE = _sheet(
    ReplaySide.P1,
    (
        ("Simisear", "Gluttony", ("Fire Spin", "Substitute")),
        ("Zoroark", "Illusion", ("Dark Pulse", "Substitute")),
        ("Espeon", "Synchronize", ("Psychic",)),
        ("Mawile", "Hyper Cutter", ("Crunch",)),
        ("Scizor", "Technician", ("Bug Bite",)),
        ("Flapple", "Ripen", ("Aerial Ace",)),
    ),
)
_PLAIN_SIDE = _sheet(
    ReplaySide.P2,
    (
        ("Jolteon", "Volt Absorb", ("Thunderbolt",)),
        ("Houndoom", "Flash Fire", ("Flamethrower",)),
        ("Camerupt", "Anger Point", ("Rock Slide",)),
        ("Grapploct", "Technician", ("Brick Break",)),
        ("Scolipede", "Poison Point", ("Round",)),
        ("Emboar", "Blaze", ("Dig",)),
    ),
)
_SIMISEAR = ReplayMemberId(ReplaySide.P1, 0)
_ZOROARK = ReplayMemberId(ReplaySide.P1, 1)
_SEVERAL_ASSIGNMENTS = "active history has multiple valid assignments"


def _events(*lines: str):
    protocol_lines = tuple(
        ProtocolLine(index, line, tuple(line.split("|")), None) for index, line in enumerate(lines)
    )
    return parse_protocol_events("resolution-test", protocol_lines).events


def test_fainted_slot_resolves_until_the_next_switch() -> None:
    ots = (
        _ots(ReplaySide.P1, ("Alpha", "Bravo", "C", "D", "E", "F")),
        _ots(ReplaySide.P2, ("Golf", "H", "I", "J", "K", "L")),
    )
    events = _events(
        "|switch|p1a: Alpha|Alpha, L50|100/100",
        "|faint|p1a: Alpha",
        "|-ability|p1a: Alpha|Aftermath",
        "|-item|p1a: Alpha|Sitrus Berry",
        "|switch|p1a: Bravo|Bravo, L50|100/100",
    )

    resolved = resolve_protocol_events("resolution-test", ots, events).require_accepted()

    alpha = ReplayMemberId(ReplaySide.P1, 0)
    bravo = ReplayMemberId(ReplaySide.P1, 1)
    assert resolved[2].pokemon_refs[0].member_id == alpha
    assert resolved[3].pokemon_refs[0].member_id == alpha
    assert resolved[4].pokemon_refs[0].member_id == bravo


def test_unique_slotless_roster_reference_resolves_to_a_member() -> None:
    ots = (
        _ots(ReplaySide.P1, ("Alpha", "Bravo", "C", "D", "E", "F")),
        _ots(ReplaySide.P2, ("Golf", "H", "I", "J", "K", "L")),
    )
    events = _events(
        "|switch|p1a: Alpha|Alpha, L50|100/100",
        "|move|p1a: Alpha|Tackle|p2: Golf",
    )

    resolved = resolve_protocol_events("resolution-test", ots, events).require_accepted()

    assert resolved[1].pokemon_refs[1].member_id == ReplayMemberId(ReplaySide.P2, 0)


def test_duplicate_slotless_reference_is_invalid_input() -> None:
    ots = (
        _ots(ReplaySide.P1, ("Alpha", "Bravo", "C", "D", "E", "F")),
        _ots(ReplaySide.P2, ("Golf", "Golf", "I", "J", "K", "L")),
    )
    events = _events(
        "|switch|p1a: Alpha|Alpha, L50|100/100",
        "|move|p1a: Alpha|Tackle|p2: Golf",
    )

    result = resolve_protocol_events("resolution-test", ots, events)

    assert result.events == ()
    assert result.diagnostics[0].category is ReplayRejectionCategory.INVALID_INPUT_CONTRACT
    assert "Golf" in result.diagnostics[0].reason
    assert "2 roster members" in result.diagnostics[0].reason


class TestIdentityRejectionCategories:
    def test_inconsistent_second_side_takes_precedence_over_ambiguous_first_side(self) -> None:
        events = _events(
            "|switch|p1a: Simisear|Simisear, L50|100/100",
            "|switch|p2a: Jolteon|Jolteon, L50|100/100",
            "|faint|p2a: Jolteon",
            "|switch|p2a: Jolteon|Jolteon, L50|100/100",
        )

        result = resolve_protocol_events("resolution-test", (_ILLUSION_SIDE, _PLAIN_SIDE), events)

        assert result.events == ()
        assert result.diagnostics[0].category is ReplayRejectionCategory.INCONSISTENT_IDENTITY

    def test_no_candidate_for_an_own_move_is_inconsistent(self) -> None:
        events = _events(
            "|switch|p1a: Simisear|Simisear, L50|100/100",
            "|move|p1a: Simisear|Protect|p1a: Simisear",
        )

        result = resolve_protocol_events("resolution-test", (_ILLUSION_SIDE, _PLAIN_SIDE), events)

        assert result.events == ()
        assert result.diagnostics[0].category is ReplayRejectionCategory.INCONSISTENT_IDENTITY

    def test_expired_budget_is_a_timeout_even_when_the_first_solution_is_easy(self) -> None:
        events = _events("|switch|p2a: Jolteon|Jolteon, L50|100/100")

        result = resolve_protocol_events(
            "resolution-test", (_ILLUSION_SIDE, _PLAIN_SIDE), events, time_limit_seconds=0
        )

        assert result.events == ()
        assert result.diagnostics[0].category is ReplayRejectionCategory.TIME_LIMIT_EXCEEDED

    @pytest.mark.parametrize("limit", (-1.0, float("nan"), float("inf")))
    def test_invalid_budget_is_rejected_at_the_api_boundary(self, limit: float) -> None:
        with pytest.raises(ValueError, match="finite and nonnegative"):
            resolve_protocol_events(
                "resolution-test", (_ILLUSION_SIDE, _PLAIN_SIDE), (), time_limit_seconds=limit
            )

    def test_long_games_without_illusion_do_not_require_recursive_search(self) -> None:
        first_side = replace(
            _PLAIN_SIDE,
            side=ReplaySide.P1,
            members=tuple(
                replace(member, member_id=ReplayMemberId(ReplaySide.P1, index))
                for index, member in enumerate(_PLAIN_SIDE.members)
            ),
        )
        events = _events(
            *(
                f"|switch|p2a: {name}|{name}, L50|100/100"
                for _ in range(1000)
                for name in ("Jolteon", "Houndoom")
            )
        )

        resolved = resolve_protocol_events(
            "resolution-test", (first_side, _PLAIN_SIDE), events
        ).require_accepted()

        assert len(resolved) == 2000
        assert resolved[-1].pokemon_refs[0].member_id == ReplayMemberId(ReplaySide.P2, 1)


class TestOTSEvidence:
    def test_native_ability_activation_identifies_the_displayed_member(self) -> None:
        events = _events(
            "|switch|p1a: Simisear|Simisear, L50|100/100",
            "|-ability|p1a: Simisear|Gluttony",
        )

        resolved = resolve_protocol_events(
            "resolution-test", (_ILLUSION_SIDE, _PLAIN_SIDE), events
        ).require_accepted()

        assert resolved[0].pokemon_refs[0].member_id == _SIMISEAR

    def test_shell_bell_evidence_belongs_to_the_healed_member(self) -> None:
        sheet = replace(
            _ILLUSION_SIDE,
            members=(
                replace(_ILLUSION_SIDE.members[0], item="Leftovers"),
                replace(_ILLUSION_SIDE.members[1], item="Shell Bell"),
                *_ILLUSION_SIDE.members[2:],
            ),
        )
        events = _events(
            "|switch|p1a: Simisear|Simisear, L50|100/100",
            "|switch|p2a: Jolteon|Jolteon, L50|100/100",
            "|-heal|p1a: Simisear|100/100|[from] item: Shell Bell|[of] p2a: Jolteon",
        )

        resolved = resolve_protocol_events(
            "resolution-test", (sheet, _PLAIN_SIDE), events
        ).require_accepted()

        assert resolved[0].pokemon_refs[0].member_id == _ZOROARK

    def test_rocky_helmet_recoil_does_not_identify_the_attacker(self) -> None:
        events = _events(
            "|switch|p1a: Simisear|Simisear, L50|100/100",
            "|switch|p2a: Jolteon|Jolteon, L50|100/100",
            "|-damage|p1a: Simisear|84/100|[from] item: Rocky Helmet|[of] p2a: Jolteon",
        )

        result = resolve_protocol_events("resolution-test", (_ILLUSION_SIDE, _PLAIN_SIDE), events)

        assert result.diagnostics[0].category is ReplayRejectionCategory.AMBIGUOUS_IDENTITY

    def test_skill_swap_stops_comparing_later_abilities_to_the_initial_sheet(self) -> None:
        events = _events(
            "|switch|p1a: Simisear|Simisear, L50|100/100",
            "|switch|p2a: Jolteon|Jolteon, L50|100/100",
            "|move|p2a: Jolteon|Skill Swap|p1a: Simisear",
            "|-ability|p1a: Simisear|Volt Absorb",
        )

        result = resolve_protocol_events("resolution-test", (_ILLUSION_SIDE, _PLAIN_SIDE), events)

        assert result.diagnostics[0].category is ReplayRejectionCategory.AMBIGUOUS_IDENTITY

    def test_mummy_activation_changes_the_contacts_ability_without_an_ability_line(self) -> None:
        events = _events(
            "|switch|p1a: Simisear|Simisear, L50|100/100",
            "|switch|p2a: Jolteon|Jolteon, L50|100/100",
            "|-activate|p2a: Jolteon|ability: Mummy|p1a: Simisear|[ability] Gluttony",
            "|-activate|p1a: Simisear|ability: Mummy|p2a: Jolteon|[ability] Volt Absorb",
        )

        result = resolve_protocol_events("resolution-test", (_ILLUSION_SIDE, _PLAIN_SIDE), events)

        assert result.diagnostics[0].category is ReplayRejectionCategory.AMBIGUOUS_IDENTITY

    def test_an_item_consumed_after_fling_need_not_be_on_the_targets_sheet(self) -> None:
        events = _events(
            "|switch|p1a: Simisear|Simisear, L50|100/100",
            "|switch|p2a: Jolteon|Jolteon, L50|100/100",
            "|move|p2a: Jolteon|Fling|p1a: Simisear",
            "|-enditem|p1a: Simisear|Sitrus Berry|[eat]",
        )

        result = resolve_protocol_events("resolution-test", (_ILLUSION_SIDE, _PLAIN_SIDE), events)

        assert result.diagnostics[0].category is ReplayRejectionCategory.AMBIGUOUS_IDENTITY


class TestIllusionEvidence:
    def test_hit_without_reveal_resolves_to_the_shown_member(self) -> None:
        events = _events(
            "|switch|p1a: Simisear|Simisear, L50|100/100",
            "|switch|p2a: Jolteon|Jolteon, L50|100/100",
            "|move|p2a: Jolteon|Thunderbolt|p1a: Simisear",
            "|-damage|p1a: Simisear|60/100",
        )

        resolved = resolve_protocol_events(
            "resolution-test", (_ILLUSION_SIDE, _PLAIN_SIDE), events
        ).require_accepted()

        assert resolved[0].pokemon_refs[0].member_id == _SIMISEAR

    def test_hp_cost_of_an_own_move_is_not_a_hit(self) -> None:
        events = _events(
            "|switch|p1a: Simisear|Simisear, L50|100/100",
            "|switch|p2a: Jolteon|Jolteon, L50|100/100",
            "|move|p2a: Jolteon|Thunderbolt|p1a: Simisear|[miss]",
            "|move|p1a: Simisear|Substitute|p1a: Simisear",
            "|-damage|p1a: Simisear|75/100",
        )

        result = resolve_protocol_events("resolution-test", (_ILLUSION_SIDE, _PLAIN_SIDE), events)

        assert result.diagnostics[0].reason == _SEVERAL_ASSIGNMENTS

    def test_damage_with_a_cause_is_not_a_hit(self) -> None:
        events = _events(
            "|switch|p1a: Simisear|Simisear, L50|100/100",
            "|switch|p2a: Jolteon|Jolteon, L50|100/100",
            "|move|p2a: Jolteon|Thunderbolt|p1a: Simisear|[miss]",
            "|-damage|p1a: Simisear|94/100|[from] Sandstorm",
        )

        result = resolve_protocol_events("resolution-test", (_ILLUSION_SIDE, _PLAIN_SIDE), events)

        assert result.diagnostics[0].reason == _SEVERAL_ASSIGNMENTS

    def test_move_of_the_illusion_user_resolves_an_unrevealed_disguise(self) -> None:
        events = _events(
            "|switch|p1a: Simisear|Simisear, L50|100/100",
            "|switch|p2a: Jolteon|Jolteon, L50|100/100",
            "|move|p1a: Simisear|Dark Pulse|p2a: Jolteon",
        )

        resolved = resolve_protocol_events(
            "resolution-test", (_ILLUSION_SIDE, _PLAIN_SIDE), events
        ).require_accepted()

        assert resolved[0].pokemon_refs[0].member_id == _ZOROARK
        assert resolved[2].pokemon_refs[0].member_id == _ZOROARK

    def test_move_of_the_shown_member_resolves_to_it(self) -> None:
        events = _events(
            "|switch|p1a: Simisear|Simisear, L50|100/100",
            "|switch|p2a: Jolteon|Jolteon, L50|100/100",
            "|move|p1a: Simisear|Fire Spin|p2a: Jolteon",
        )

        resolved = resolve_protocol_events(
            "resolution-test", (_ILLUSION_SIDE, _PLAIN_SIDE), events
        ).require_accepted()

        assert resolved[0].pokemon_refs[0].member_id == _SIMISEAR

    def test_move_on_both_sheets_resolves_nothing(self) -> None:
        events = _events(
            "|switch|p1a: Simisear|Simisear, L50|100/100",
            "|move|p1a: Simisear|Substitute|p1a: Simisear",
        )

        result = resolve_protocol_events("resolution-test", (_ILLUSION_SIDE, _PLAIN_SIDE), events)

        assert result.diagnostics[0].reason == _SEVERAL_ASSIGNMENTS

    def test_called_move_resolves_nothing(self) -> None:
        events = _events(
            "|switch|p1a: Simisear|Simisear, L50|100/100",
            "|switch|p2a: Jolteon|Jolteon, L50|100/100",
            "|move|p1a: Simisear|Dark Pulse|p2a: Jolteon|[from] move: Copycat",
        )

        result = resolve_protocol_events("resolution-test", (_ILLUSION_SIDE, _PLAIN_SIDE), events)

        assert result.diagnostics[0].reason == _SEVERAL_ASSIGNMENTS

    def test_move_after_transform_resolves_nothing(self) -> None:
        events = _events(
            "|switch|p1a: Simisear|Simisear, L50|100/100",
            "|switch|p2a: Jolteon|Jolteon, L50|100/100",
            "|-transform|p1a: Simisear|p2a: Jolteon",
            "|move|p1a: Simisear|Thunderbolt|p2a: Jolteon",
        )

        result = resolve_protocol_events("resolution-test", (_ILLUSION_SIDE, _PLAIN_SIDE), events)

        assert result.diagnostics[0].reason == _SEVERAL_ASSIGNMENTS

    def test_entry_hp_separates_the_damaged_illusion_user_from_the_shown_member(self) -> None:
        events = _events(
            "|switch|p1a: Simisear|Simisear, L50|100/100",
            "|switch|p2a: Jolteon|Jolteon, L50|100/100",
            "|move|p2a: Jolteon|Thunderbolt|p1a: Simisear",
            "|-damage|p1a: Simisear|78/100",
            "|replace|p1a: Zoroark|Zoroark, L50",
            "|-damage|p1a: Zoroark|61/100 brn|[from] brn",
            "|switch|p1a: Espeon|Espeon, L50|100/100",
            "|switch|p1b: Simisear|Simisear, L50|100/100",
            "|switch|p1b: Simisear|Simisear, L50|61/100 brn",
        )

        resolved = resolve_protocol_events(
            "resolution-test", (_ILLUSION_SIDE, _PLAIN_SIDE), events
        ).require_accepted()

        assert resolved[7].pokemon_refs[0].member_id == _SIMISEAR
        assert resolved[8].pokemon_refs[0].member_id == _ZOROARK

    def test_heal_before_leaving_sets_the_hp_of_the_next_entry(self) -> None:
        events = _events(
            "|switch|p1a: Simisear|Simisear, L50|100/100",
            "|switch|p2a: Jolteon|Jolteon, L50|100/100",
            "|move|p2a: Jolteon|Thunderbolt|p1a: Simisear",
            "|-damage|p1a: Simisear|40/100",
            "|replace|p1a: Zoroark|Zoroark, L50",
            "|-heal|p1a: Zoroark|73/100|[from] ability: Regenerator|[silent]",
            "|switch|p1a: Espeon|Espeon, L50|100/100",
            "|switch|p1b: Simisear|Simisear, L50|73/100",
        )

        resolved = resolve_protocol_events(
            "resolution-test", (_ILLUSION_SIDE, _PLAIN_SIDE), events
        ).require_accepted()

        assert resolved[7].pokemon_refs[0].member_id == _ZOROARK

    def test_equal_hp_resolves_nothing(self) -> None:
        events = _events(
            "|switch|p1a: Simisear|Simisear, L50|100/100",
            "|switch|p1a: Espeon|Espeon, L50|100/100",
            "|switch|p1b: Simisear|Simisear, L50|100/100",
        )

        result = resolve_protocol_events("resolution-test", (_ILLUSION_SIDE, _PLAIN_SIDE), events)

        assert result.diagnostics[0].reason == _SEVERAL_ASSIGNMENTS


class TestRevivedMemberEntry:
    def test_named_revive_proves_a_disguised_faint_and_sets_its_next_hp(self) -> None:
        events = _events(
            "|switch|p1a: Simisear|Simisear, L50|100/100",
            "|faint|p1a: Simisear",
            "|-heal|p1: Zoroark|50/100|[from] move: Revival Blessing",
            "|switch|p1a: Simisear|Simisear, L50|50/100",
        )

        result = resolve_protocol_events("resolution-test", (_ILLUSION_SIDE, _PLAIN_SIDE), events)

        resolved = result.require_accepted()
        assert resolved[0].pokemon_refs[0].member_id == _ZOROARK
        assert resolved[1].pokemon_refs[0].member_id == _ZOROARK
        assert resolved[3].pokemon_refs[0].member_id == _ZOROARK

    def test_member_revived_by_revival_blessing_enters_again(self) -> None:
        events = _events(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|faint|p1a: Alpha",
            "|-heal|p1: Alpha|50/100|[from] move: Revival Blessing",
            "|switch|p1a: Alpha|Alpha, L50|50/100",
        )
        ots = (
            _ots(ReplaySide.P1, ("Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot")),
            _PLAIN_SIDE,
        )

        resolved = resolve_protocol_events("resolution-test", ots, events).require_accepted()

        assert resolved[2].pokemon_refs[0].member_id == ReplayMemberId(ReplaySide.P1, 0)
        assert resolved[3].pokemon_refs[0].member_id == ReplayMemberId(ReplaySide.P1, 0)

    def test_revive_names_a_member_without_a_nickname_by_its_base_species(self) -> None:
        # Showdown names Vivillon-Sandstorm "Vivillon" when it has no nickname.
        events = _events(
            "|switch|p1a: Vivillon|Vivillon-Sandstorm, L50|100/100",
            "|faint|p1a: Vivillon",
            "|-heal|p1: Vivillon|50/100|[from] move: Revival Blessing",
            "|switch|p1a: Vivillon|Vivillon-Sandstorm, L50|50/100",
        )
        ots = (
            _ots(
                ReplaySide.P1,
                ("Vivillon-Sandstorm", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"),
            ),
            _PLAIN_SIDE,
        )

        resolved = resolve_protocol_events("resolution-test", ots, events).require_accepted()

        assert resolved[3].pokemon_refs[0].member_id == ReplayMemberId(ReplaySide.P1, 0)

    def test_revive_of_another_member_does_not_bring_back_the_fainted_one(self) -> None:
        events = _events(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|faint|p1a: Alpha",
            "|-heal|p1: Bravo|50/100|[from] move: Revival Blessing",
            "|switch|p1a: Alpha|Alpha, L50|50/100",
        )
        ots = (
            _ots(ReplaySide.P1, ("Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot")),
            _PLAIN_SIDE,
        )

        result = resolve_protocol_events("resolution-test", ots, events)

        assert result.events == ()
        # No member has Illusion, so the rejection does not blame Illusion.
        assert result.diagnostics[0].reason == "active history has no valid roster assignment"

    def test_revive_before_a_later_faint_does_not_bring_the_member_back(self) -> None:
        events = _events(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|faint|p1a: Alpha",
            "|-heal|p1: Alpha|50/100|[from] move: Revival Blessing",
            "|switch|p1a: Alpha|Alpha, L50|50/100",
            "|faint|p1a: Alpha",
            "|switch|p1a: Alpha|Alpha, L50|50/100",
        )
        ots = (
            _ots(ReplaySide.P1, ("Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot")),
            _PLAIN_SIDE,
        )

        result = resolve_protocol_events("resolution-test", ots, events)

        assert result.events == ()
        assert result.diagnostics[0].reason == "active history has no valid roster assignment"
