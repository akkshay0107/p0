"""Public resolver tests for faint lifetimes, slotless references and Illusion evidence."""

from __future__ import annotations

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
_SEVERAL_ASSIGNMENTS = "unresolved_illusion: active history has multiple valid assignments"


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


def test_ambiguous_slotless_reference_is_rejected_by_category() -> None:
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
    assert result.diagnostics[0].category is ReplayRejectionCategory.AMBIGUOUS_IDENTITY
    assert "Golf" in result.diagnostics[0].reason
    assert "2 roster members" in result.diagnostics[0].reason


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
