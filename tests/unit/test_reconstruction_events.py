"""Tests for the lossless replay reconstruction event layer."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from p0.replays.protocol import parse_replay_payload
from p0.replays.reconstruction.classification import EventClassification
from p0.replays.reconstruction.diagnostics import ReplayEventParseError
from p0.replays.reconstruction.events import parse_protocol_event, parse_protocol_events
from p0.replays.reconstruction.identity import (
    ReplayMemberId,
    ReplaySide,
    parse_protocol_pokemon_reference,
)
from p0.replays.schema import ProtocolLine

_GOLDEN_REPLAY = (
    Path(__file__).parents[2]
    / "src/p0/replays/reconstruction/golden_replays"
    / "gen9championsvgc2026regmbbo3-2641278886.json"
)


def _line(raw: str, *, index: int = 0, turn: int | None = None) -> ProtocolLine:
    return ProtocolLine(index, raw, tuple(raw.split("|")), turn)


def test_member_ids_and_protocol_pokemon_refs_are_distinct_contracts() -> None:
    member = ReplayMemberId(ReplaySide.P2, 4)
    pokemon_ref = parse_protocol_pokemon_reference("p2b: Nickname")

    assert member.side is ReplaySide.P2
    assert member.roster_index == 4
    assert pokemon_ref.side is ReplaySide.P2
    assert pokemon_ref.active_slot == 1
    assert pokemon_ref.displayed_name == "Nickname"

    with pytest.raises(ValueError, match="roster_index"):
        ReplayMemberId(ReplaySide.P1, 6)
    with pytest.raises(ValueError, match="Invalid protocol Pokémon reference"):
        parse_protocol_pokemon_reference("Nickname")


def test_bare_separator_and_empty_tag_text_have_different_classifications() -> None:
    separator = parse_protocol_event("replay-1", _line("|"))
    text = parse_protocol_event("replay-1", _line("||player is ready"))

    assert separator.classification is EventClassification.BOUNDARY_SIGNAL
    assert text.classification is EventClassification.NO_STATE_CHANGE
    assert separator.tag == text.tag == ""


@pytest.mark.parametrize("raw", ("|teampreview", "|teampreview|4"))
def test_team_preview_accepts_standard_and_selected_team_shapes(raw: str) -> None:
    event = parse_protocol_event("replay-1", _line(raw))

    assert event.classification is EventClassification.BOUNDARY_SIGNAL


def test_event_parsing_extracts_effect_cause_and_annotated_pokemon_ref() -> None:
    event = parse_protocol_event(
        "replay-1",
        _line("|-weather|RainDance|[from] ability: Drizzle|[of] p1b: Pelipper"),
    )

    assert event.classification is EventClassification.PUBLIC_STATE
    assert event.effect is not None
    assert event.effect.normalized == "raindance"
    assert event.cause is not None
    assert event.cause.namespace == "ability"
    assert event.cause.normalized == "drizzle"
    assert len(event.pokemon_refs) == 1
    assert event.pokemon_refs[0].argument_index == 2
    assert event.pokemon_refs[0].pokemon_ref.active_slot == 1


@pytest.mark.parametrize(
    ("raw", "effect", "pokemon_ref_count"),
    (
        ("|-activate|move: Trick Room", "trickroom", 0),
        ("|-activate|p1a: Example|move: Protect", "protect", 1),
        ("|-mega|p1a: Mawile|Mawilite", "mawilite", 1),
        ("|-mega|p1a: Mawile|Mawile|Mawilite", "mawilite", 1),
    ),
)
def test_argument_shape_selects_effect_and_pokemon_ref_variants(
    raw: str,
    effect: str,
    pokemon_ref_count: int,
) -> None:
    event = parse_protocol_event("replay-1", _line(raw))

    assert event.classification is EventClassification.PUBLIC_STATE
    assert event.effect is not None
    assert event.effect.normalized == effect
    assert len(event.pokemon_refs) == pokemon_ref_count


def test_malformed_known_event_produces_structured_rejection() -> None:
    result = parse_protocol_events(
        "replay-1",
        (_line("|move|not-a-pokemon-ref|Protect|"),),
    )

    assert result.events[0].classification is EventClassification.MALFORMED
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.replay_id == "replay-1"
    assert diagnostic.line_index == 0
    assert diagnostic.tag == "move"
    assert diagnostic.raw_line == "|move|not-a-pokemon-ref|Protect|"
    assert "required Pokémon reference" in diagnostic.reason

    with pytest.raises(ReplayEventParseError) as caught:
        result.require_accepted()
    assert caught.value.diagnostics == result.diagnostics


def test_unknown_tag_is_unsupported_state_instead_of_a_silent_noop() -> None:
    event = parse_protocol_event(
        "replay-1",
        _line("|-futurestate|p1a: Example|Mystery Effect"),
    )

    assert event.classification is EventClassification.UNSUPPORTED_STATE
    assert event.diagnostic is not None
    assert event.diagnostic.normalized_effect == "mysteryeffect"
    assert event.diagnostic.normalized_cause == ""


@pytest.mark.skipif(not _GOLDEN_REPLAY.is_file(), reason="local golden replay is not present")
def test_local_golden_replay_is_classified_once_without_rejections() -> None:
    document = parse_replay_payload(_GOLDEN_REPLAY.read_bytes())
    result = parse_protocol_events(document.metadata.replay_id, document.protocol_lines)

    assert all(ots.is_complete for ots in document.ots)
    assert tuple(member.member_id for member in document.ots[0].members) == tuple(
        ReplayMemberId(ReplaySide.P1, index) for index in range(6)
    )
    assert len(result.events) == len(document.protocol_lines) == 186
    assert tuple(event.line_index for event in result.events) == tuple(range(186))
    assert result.diagnostics == ()
    assert Counter(event.classification for event in result.events) == {
        EventClassification.NO_STATE_CHANGE: 55,
        EventClassification.PUBLIC_STATE: 70,
        EventClassification.ACTION_EXECUTION: 34,
        EventClassification.BOUNDARY_SIGNAL: 27,
    }
