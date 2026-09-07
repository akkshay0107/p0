"""Public resolver regressions for faint lifetimes and slotless references."""

from __future__ import annotations

from p0.replays.identity import ReplayMemberId, ReplaySide
from p0.replays.reconstruction.diagnostics import ReplayRejectionCategory
from p0.replays.reconstruction.events import parse_protocol_events
from p0.replays.reconstruction.resolution import resolve_protocol_events
from p0.replays.schema import OTSData, OTSMember, ProtocolLine


def _ots(side: ReplaySide, names: tuple[str, ...]) -> OTSData:
    return OTSData(
        side,
        "]".join(names),
        tuple(
            OTSMember(
                ReplayMemberId(side, index),
                name,
                name,
                "",
                "Ability",
                ("Protect",),
                "Serious",
                "",
                50,
                "",
                name,
            )
            for index, name in enumerate(names)
        ),
    )


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
    events = _events("|move|p1a: Alpha|Tackle|p2: Golf")

    result = resolve_protocol_events("resolution-test", ots, events)

    assert result.events == ()
    assert result.diagnostics[0].category is ReplayRejectionCategory.AMBIGUOUS_IDENTITY
