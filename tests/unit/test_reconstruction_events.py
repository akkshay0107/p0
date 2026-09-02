"""Tests for the lossless replay reconstruction event layer."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
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
from p0.replays.reconstruction.resolution import (
    resolve_protocol_events,
    resolve_replay_events,
)
from p0.replays.schema import OTSData, OTSMember, ProtocolLine

_GOLDEN_REPLAY = (
    Path(__file__).parents[2]
    / "src/p0/replays/reconstruction/golden_replays"
    / "gen9championsvgc2026regmbbo3-2641278886.json"
)


def _line(raw: str, *, index: int = 0, turn: int | None = None) -> ProtocolLine:
    return ProtocolLine(index, raw, tuple(raw.split("|")), turn)


def _complete_ots(
    side: ReplaySide,
    species: tuple[str, str, str, str, str, str],
    *,
    illusion_species: str | None = None,
) -> OTSData:
    members = tuple(
        OTSMember(
            member_id=ReplayMemberId(side, index),
            nickname=name,
            species=name,
            item="",
            ability="Illusion" if name == illusion_species else "Ability",
            moves=("Protect",),
            nature="Serious",
            gender="",
            level=50,
            evs="",
            raw_packed_set=name,
        )
        for index, name in enumerate(species)
    )
    return OTSData(side, "]".join(species), members)


def _test_ots() -> tuple[OTSData, OTSData]:
    return (
        _complete_ots(ReplaySide.P1, ("Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot")),
        _complete_ots(ReplaySide.P2, ("Golf", "Hotel", "India", "Juliet", "Kilo", "Lima")),
    )


class TestReconstructionEvents:
    def test_member_ids_and_protocol_pokemon_refs_are_distinct_contracts(self) -> None:
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

    def test_bare_separator_and_empty_tag_text_have_different_classifications(self) -> None:
        separator = parse_protocol_event("replay-1", _line("|"))
        text = parse_protocol_event("replay-1", _line("||player is ready"))

        assert separator.classification is EventClassification.BOUNDARY_SIGNAL
        assert text.classification is EventClassification.NO_STATE_CHANGE
        assert separator.tag == text.tag == ""

    @pytest.mark.parametrize("raw", ("|teampreview", "|teampreview|4"))
    def test_team_preview_accepts_standard_and_selected_team_shapes(self, raw: str) -> None:
        event = parse_protocol_event("replay-1", _line(raw))

        assert event.classification is EventClassification.BOUNDARY_SIGNAL

    def test_replace_accepts_the_pinned_showdown_shape_only(self) -> None:
        event = parse_protocol_event(
            "replay-1",
            _line("|replace|p1a: Zoroark|Zoroark-Hisui, L50"),
        )
        extra_hp = parse_protocol_event(
            "replay-1",
            _line("|replace|p1a: Zoroark|Zoroark-Hisui, L50|50/100"),
        )

        assert event.classification is EventClassification.ACTION_EXECUTION
        assert event.arguments == ("p1a: Zoroark", "Zoroark-Hisui, L50")
        assert extra_hp.classification is EventClassification.MALFORMED

    def test_player_departure_accepts_an_empty_username(self) -> None:
        event = parse_protocol_event("replay-1", _line("|player|p2|"))

        assert event.classification is EventClassification.PUBLIC_STATE
        assert event.arguments == ("p2", "")

    def test_event_parsing_extracts_effect_cause_and_annotated_pokemon_ref(self) -> None:
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
        self,
        raw: str,
        effect: str,
        pokemon_ref_count: int,
    ) -> None:
        event = parse_protocol_event("replay-1", _line(raw))

        assert event.classification is EventClassification.PUBLIC_STATE
        assert event.effect is not None
        assert event.effect.normalized == effect
        assert len(event.pokemon_refs) == pokemon_ref_count

    def test_transform_accepts_species_and_reference_targets(self) -> None:
        species_target = parse_protocol_event(
            "replay-1",
            _line("|-transform|p1a: Ditto|Pikachu|[from] ability: Imposter"),
        )
        reference_target = parse_protocol_event(
            "replay-1",
            _line("|-transform|p1a: Ditto|p2a: Pikachu|[from] ability: Imposter"),
        )

        assert species_target.classification is EventClassification.PUBLIC_STATE
        assert tuple(reference.argument_index for reference in species_target.pokemon_refs) == (0,)
        assert tuple(reference.argument_index for reference in reference_target.pokemon_refs) == (
            0,
            1,
        )

    def test_malformed_known_event_produces_structured_rejection(self) -> None:
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

    def test_unknown_tag_is_unsupported_state_instead_of_a_silent_noop(self) -> None:
        event = parse_protocol_event(
            "replay-1",
            _line("|-futurestate|p1a: Example|Mystery Effect"),
        )

        assert event.classification is EventClassification.UNSUPPORTED_STATE
        assert event.diagnostic is not None
        assert event.diagnostic.normalized_effect == "mysteryeffect"
        assert event.diagnostic.normalized_cause == ""

    def test_identity_resolution_tracks_slots_and_keeps_side_references_memberless(self) -> None:
        lines = tuple(
            _line(raw, index=index)
            for index, raw in enumerate(
                (
                    "|switch|p1a: Alpha|Alpha, L50|100/100",
                    "|switch|p1b: Bravo|Bravo, L50|100/100",
                    "|swap|p1a: Alpha|1",
                    "|-damage|p1a: Bravo|50/100",
                    "|-sidestart|p1: Player|move: Tailwind",
                    "|faint|p1b: Alpha",
                    "|switch|p1b: Charlie|Charlie, L50|100/100",
                )
            )
        )
        parsed = parse_protocol_events("replay-1", lines)
        result = resolve_protocol_events("replay-1", _test_ots(), parsed.events)
        events = result.require_accepted()

        assert events[0].pokemon_refs[0].member_id == ReplayMemberId(ReplaySide.P1, 0)
        assert events[2].pokemon_refs[0].member_id == ReplayMemberId(ReplaySide.P1, 0)
        assert events[3].pokemon_refs[0].member_id == ReplayMemberId(ReplaySide.P1, 1)
        assert events[4].pokemon_refs[0].member_id is None
        assert events[4].pokemon_refs[0].pokemon_ref.side is ReplaySide.P1
        assert events[6].pokemon_refs[0].member_id == ReplayMemberId(ReplaySide.P1, 2)

    def test_switch_details_are_authoritative_when_a_nickname_matches_another_member(self) -> None:
        p1 = _complete_ots(
            ReplaySide.P1,
            ("Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"),
        )
        members = list(p1.members)
        members[1] = replace(members[1], nickname="Alpha")
        ots = (OTSData(ReplaySide.P1, p1.raw_payload, tuple(members)), _test_ots()[1])
        parsed = parse_protocol_events(
            "nickname-conflict",
            (_line("|switch|p1a: Alpha|Bravo, L50|100/100"),),
        )

        event = resolve_protocol_events("nickname-conflict", ots, parsed.events).require_accepted()[
            0
        ]

        assert event.pokemon_refs[0].member_id == ReplayMemberId(ReplaySide.P1, 1)

    def test_mega_form_reentry_resolves_to_the_base_roster_member(self) -> None:
        ots = (
            _complete_ots(
                ReplaySide.P1,
                ("Greninja", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"),
            ),
            _test_ots()[1],
        )
        parsed = parse_protocol_events(
            "mega-reentry",
            (
                _line("|switch|p1a: Iku Z|Greninja, L50|100/100", index=0),
                _line("|detailschange|p1a: Iku Z|Greninja-Mega, L50", index=1),
                _line("|switch|p1a: Bravo|Bravo, L50|100/100", index=2),
                _line("|switch|p1a: Iku Z|Greninja-Mega, L50|50/100", index=3),
            ),
        )
        dex = {
            "species": (
                {"id": "greninja", "name": "Greninja"},
                {
                    "id": "greninjamega",
                    "name": "Greninja-Mega",
                    "baseSpecies": "Greninja",
                },
            )
        }

        events = resolve_protocol_events(
            "mega-reentry", ots, parsed.events, dex=dex
        ).require_accepted()

        greninja = ReplayMemberId(ReplaySide.P1, 0)
        assert events[0].pokemon_refs[0].member_id == greninja
        assert events[3].pokemon_refs[0].member_id == greninja

    def test_identity_resolution_rejects_unbound_ambiguous_and_impossible_references(self) -> None:
        unbound = parse_protocol_events("unbound", (_line("|-damage|p1a: Alpha|50/100"),))
        result = resolve_protocol_events("unbound", _test_ots(), unbound.events)
        with pytest.raises(ReplayEventParseError, match="no active member"):
            result.require_accepted()
        assert result.events == ()

        duplicate_ots = (
            _complete_ots(
                ReplaySide.P1,
                ("Ditto", "Ditto", "Charlie", "Delta", "Echo", "Foxtrot"),
            ),
            _test_ots()[1],
        )
        ambiguous = parse_protocol_events(
            "ambiguous",
            (_line("|switch|p1a: Ditto|Ditto, L50|100/100"),),
        )
        result = resolve_protocol_events("ambiguous", duplicate_ots, ambiguous.events)
        with pytest.raises(ReplayEventParseError, match="2 available roster members"):
            result.require_accepted()

        illusion = parse_protocol_events(
            "illusion",
            (
                _line("|switch|p1a: Alpha|Alpha, L50|100/100", index=0),
                _line("|replace|p1a: Charlie|Charlie, L50", index=1),
            ),
        )
        result = resolve_protocol_events("illusion", _test_ots(), illusion.events)
        with pytest.raises(ReplayEventParseError, match="no valid roster assignment"):
            result.require_accepted()
        assert result.events == ()
        assert result.diagnostics[0].line_index == 1

    def test_illusion_reveal_resolves_the_complete_active_history(self) -> None:
        ots = (
            _complete_ots(
                ReplaySide.P1,
                ("Alpha", "Zoroark", "Charlie", "Delta", "Echo", "Foxtrot"),
                illusion_species="Zoroark",
            ),
            _test_ots()[1],
        )
        parsed = parse_protocol_events(
            "illusion-reveal",
            tuple(
                _line(raw, index=index)
                for index, raw in enumerate(
                    (
                        "|switch|p1a: Alpha|Alpha, L50|100/100",
                        "|switch|p1b: Charlie|Charlie, L50|100/100",
                        "|-damage|p1a: Alpha|50/100",
                        "|replace|p1a: Zoroark|Zoroark, L50",
                        "|-end|p1a: Zoroark|Illusion",
                    )
                )
            ),
        )

        events = resolve_protocol_events("illusion-reveal", ots, parsed.events).require_accepted()
        zoroark = ReplayMemberId(ReplaySide.P1, 1)

        assert events[0].pokemon_refs[0].member_id == zoroark
        assert events[2].pokemon_refs[0].member_id == zoroark
        assert events[3].pokemon_refs[0].member_id == zoroark
        assert events[4].pokemon_refs[0].member_id == zoroark

    def test_reserve_illusion_reveal_resolves_only_the_new_history(self) -> None:
        ots = (
            _complete_ots(
                ReplaySide.P1,
                ("Alpha", "Zoroark", "Charlie", "Delta", "Echo", "Foxtrot"),
                illusion_species="Zoroark",
            ),
            _test_ots()[1],
        )
        parsed = parse_protocol_events(
            "reserve-illusion",
            tuple(
                _line(raw, index=index)
                for index, raw in enumerate(
                    (
                        "|switch|p1a: Alpha|Alpha, L50|100/100",
                        "|switch|p1b: Charlie|Charlie, L50|100/100",
                        "|faint|p1a: Alpha",
                        "|switch|p1a: Delta|Delta, L50|100/100",
                        "|replace|p1a: Zoroark|Zoroark, L50",
                    )
                )
            ),
        )

        events = resolve_protocol_events("reserve-illusion", ots, parsed.events).require_accepted()

        assert events[0].pokemon_refs[0].member_id == ReplayMemberId(ReplaySide.P1, 0)
        assert events[1].pokemon_refs[0].member_id == ReplayMemberId(ReplaySide.P1, 2)
        assert events[3].pokemon_refs[0].member_id == ReplayMemberId(ReplaySide.P1, 1)
        assert events[4].pokemon_refs[0].member_id == ReplayMemberId(ReplaySide.P1, 1)

    def test_illusion_resolution_uses_overlap_faint_and_swap_constraints(self) -> None:
        ots = (
            _complete_ots(
                ReplaySide.P1,
                ("Alpha", "Zoroark", "Charlie", "Delta", "Echo", "Foxtrot"),
                illusion_species="Zoroark",
            ),
            _test_ots()[1],
        )
        parsed = parse_protocol_events(
            "illusion-constraints",
            tuple(
                _line(raw, index=index)
                for index, raw in enumerate(
                    (
                        "|switch|p1a: Alpha|Alpha, L50|100/100",
                        "|switch|p1b: Alpha|Alpha, L50|100/100",
                        "|swap|p1b: Alpha|0",
                        "|replace|p1a: Zoroark|Zoroark, L50",
                        "|faint|p1a: Zoroark",
                    )
                )
            ),
        )

        events = resolve_protocol_events(
            "illusion-constraints", ots, parsed.events
        ).require_accepted()
        alpha = ReplayMemberId(ReplaySide.P1, 0)
        zoroark = ReplayMemberId(ReplaySide.P1, 1)

        assert events[0].pokemon_refs[0].member_id == alpha
        assert events[1].pokemon_refs[0].member_id == zoroark
        assert events[2].pokemon_refs[0].member_id == zoroark
        assert events[3].pokemon_refs[0].member_id == zoroark

    def test_faint_before_reveal_can_resolve_an_illusion_history(self) -> None:
        ots = (
            _complete_ots(
                ReplaySide.P1,
                ("Alpha", "Zoroark", "Charlie", "Delta", "Echo", "Foxtrot"),
                illusion_species="Zoroark",
            ),
            _test_ots()[1],
        )
        parsed = parse_protocol_events(
            "illusion-faint",
            tuple(
                _line(raw, index=index)
                for index, raw in enumerate(
                    (
                        "|switch|p1a: Alpha|Alpha, L50|100/100",
                        "|faint|p1a: Alpha",
                        "|switch|p1a: Alpha|Alpha, L50|100/100",
                    )
                )
            ),
        )

        events = resolve_protocol_events("illusion-faint", ots, parsed.events).require_accepted()

        assert events[0].pokemon_refs[0].member_id == ReplayMemberId(ReplaySide.P1, 1)
        assert events[1].pokemon_refs[0].member_id == ReplayMemberId(ReplaySide.P1, 1)
        assert events[2].pokemon_refs[0].member_id == ReplayMemberId(ReplaySide.P1, 0)

    def test_replacement_switch_cannot_reuse_the_outgoing_actual_member(self) -> None:
        ots = (
            _complete_ots(
                ReplaySide.P1,
                ("Alpha", "Zoroark", "Charlie", "Delta", "Echo", "Foxtrot"),
                illusion_species="Zoroark",
            ),
            _test_ots()[1],
        )
        parsed = parse_protocol_events(
            "illusion-predecessor",
            (
                _line("|switch|p1a: Alpha|Alpha, L50|100/100", index=0),
                _line("|switch|p1a: Alpha|Alpha, L50|100/100", index=1),
                _line("|replace|p1a: Zoroark|Zoroark, L50", index=2),
            ),
        )

        events = resolve_protocol_events(
            "illusion-predecessor", ots, parsed.events
        ).require_accepted()

        assert events[0].pokemon_refs[0].member_id == ReplayMemberId(ReplaySide.P1, 0)
        assert events[1].pokemon_refs[0].member_id == ReplayMemberId(ReplaySide.P1, 1)
        assert events[2].pokemon_refs[0].member_id == ReplayMemberId(ReplaySide.P1, 1)

    def test_unrevealed_illusion_ambiguity_rejects_without_partial_events(self) -> None:
        ots = (
            _complete_ots(
                ReplaySide.P1,
                ("Alpha", "Zoroark", "Charlie", "Delta", "Echo", "Foxtrot"),
                illusion_species="Zoroark",
            ),
            _test_ots()[1],
        )
        parsed = parse_protocol_events(
            "unresolved-illusion",
            (_line("|switch|p1a: Alpha|Alpha, L50|100/100"),),
        )

        result = resolve_protocol_events("unresolved-illusion", ots, parsed.events)

        assert result.events == ()
        assert len(result.diagnostics) == 1
        assert result.diagnostics[0].line_index == 0
        assert result.diagnostics[0].reason.startswith("unresolved_illusion:")

    def test_switch_out_before_reveal_remains_unresolved(self) -> None:
        ots = (
            _complete_ots(
                ReplaySide.P1,
                ("Alpha", "Zoroark", "Charlie", "Delta", "Echo", "Foxtrot"),
                illusion_species="Zoroark",
            ),
            _test_ots()[1],
        )
        parsed = parse_protocol_events(
            "illusion-switch-out",
            (
                _line("|switch|p1a: Alpha|Alpha, L50|100/100", index=0),
                _line("|switch|p1a: Charlie|Charlie, L50|100/100", index=1),
            ),
        )

        result = resolve_protocol_events("illusion-switch-out", ots, parsed.events)

        assert result.events == ()
        assert result.diagnostics[0].reason.startswith("unresolved_illusion:")

    @pytest.mark.skipif(not _GOLDEN_REPLAY.is_file(), reason="local golden replay is not present")
    def test_local_golden_replay_is_classified_once_without_rejections(self) -> None:
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

        resolved = resolve_replay_events(document).require_accepted()
        first_switch = next(event for event in resolved if event.event.tag == "switch")
        assert first_switch.pokemon_refs[0].member_id == ReplayMemberId(ReplaySide.P1, 3)
        assert len(resolved) == 186
