"""Tests for the lossless replay reconstruction event layer."""

from __future__ import annotations

import pytest

from p0.replays.reconstruction.classification import EventClassification
from p0.replays.reconstruction.diagnostics import ReplayEventParseError
from p0.replays.reconstruction.events import parse_protocol_event, parse_protocol_events
from p0.replays.reconstruction.identity import (
    ReplayMemberId,
    ReplaySide,
    parse_protocol_pokemon_reference,
)
from p0.replays.schema import ProtocolLine


def _line(raw: str, *, index: int = 0, turn: int | None = None) -> ProtocolLine:
    return ProtocolLine(index, raw, tuple(raw.split("|")), turn)


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

    def test_activation_unknown_and_wrong_move_fail_closed(self) -> None:
        for raw in ("|-activate|p1a: A|move: UnknownMadeUp", "|-activate|p1a: A|move: Thunderbolt"):
            event = parse_protocol_event("activation", _line(raw))
            assert event.classification is EventClassification.UNSUPPORTED_STATE

    def test_effect_valid_for_another_tag_fails_closed(self) -> None:
        event = parse_protocol_event("effect-family", _line("|-singlemove|p1a: A|move: Protect"))

        assert event.classification is EventClassification.UNSUPPORTED_STATE

    def test_source_causes_are_catalog_checked_without_rejecting_of_annotations(self) -> None:
        legal = (
            "|-damage|p1a: A|50/100|[from] item: White Herb",
            "|-activate|p1a: A|ability: Hospitality|[from] ability: Hospitality",
            "|-start|p1a: A|Chilly Reception|[from] move: Chilly Reception",
            "|-end|p1a: A|Chilly Reception|[of] p1a: A",
        )
        assert (
            parse_protocol_events(
                "causes", tuple(_line(raw, index=i) for i, raw in enumerate(legal))
            ).diagnostics
            == ()
        )

        invalid = (
            "|-damage|p1a: A|50/100|[from] move: Made Up",
            "|-damage|p1a: A|50/100|[from] item: Made Up",
            "|-damage|p1a: A|50/100|[from] ability: Made Up",
            "|-damage|p1a: A|50/100|[from] Made Up",
        )
        parsed = parse_protocol_events(
            "causes", tuple(_line(raw, index=i) for i, raw in enumerate(invalid))
        )
        assert len(parsed.diagnostics) == len(invalid)
        assert all(
            diagnostic.category.value == "UNSUPPORTED_EVENT" for diagnostic in parsed.diagnostics
        )

    def test_activation_pp_amounts_require_bounded_positive_values(self) -> None:
        for raw in (
            "|-activate|p1a: A|move: Spite|Tackle|0",
            "|-activate|p1a: A|move: Spite|Tackle|5",
            "|-activate|p1a: A|move: Eerie Spell|Tackle|4",
        ):
            event = parse_protocol_event("activation", _line(raw))
            assert event.classification is EventClassification.MALFORMED

    def test_activation_leppa_requires_named_move(self) -> None:
        event = parse_protocol_event(
            "activation", _line("|-activate|p1a: A|item: Leppa Berry|[consumed]")
        )
        assert event.classification is EventClassification.MALFORMED

    def test_activation_mummy_and_symbiosis_retain_source_references(self) -> None:
        mummy = parse_protocol_event(
            "activation", _line("|-activate|p1a: A|ability: Mummy|p2a: B|[ability]")
        )
        symbiosis = parse_protocol_event(
            "activation", _line("|-activate|p1a: A|ability: Symbiosis|Leftovers|[of] p2a: B")
        )
        assert mummy.classification is EventClassification.PUBLIC_STATE
        assert len(mummy.pokemon_refs) == 2
        assert symbiosis.classification is EventClassification.PUBLIC_STATE
        assert len(symbiosis.pokemon_refs) == 2

    def test_activation_substitute_is_known_source_effect(self) -> None:
        event = parse_protocol_event("activation", _line("|-activate|p1a: A|move: Substitute"))
        assert event.classification is EventClassification.PUBLIC_STATE

    def test_instruct_single_turn_marker_is_accepted(self) -> None:
        event = parse_protocol_event(
            "activation", _line("|-singleturn|p1a: A|move: Instruct|[of] p1b: B")
        )
        assert event.classification is EventClassification.PUBLIC_STATE

    def test_activation_stored_stat_family_is_accepted(self) -> None:
        for raw in (
            "|-activate|p1a: A|move: Power Split|[of] p2a: B",
            "|-activate|p1a: A|move: Guard Split|[of] p2a: B",
            "|-activate|p1a: A|move: Speed Swap|[of] p2a: B",
            "|-start|p1a: A|Power Trick",
        ):
            event = parse_protocol_event("activation", _line(raw))
            assert event.classification is EventClassification.PUBLIC_STATE

    def test_bare_separator_and_empty_tag_text_have_different_classifications(self) -> None:
        separator = parse_protocol_event("replay-1", _line("|"))
        text = parse_protocol_event("replay-1", _line("||player is ready"))

        assert separator.classification is EventClassification.BOUNDARY_SIGNAL
        assert text.classification is EventClassification.NO_STATE_CHANGE
        assert separator.tag == text.tag == ""

    @pytest.mark.parametrize("raw", ("|teampreview", "|teampreview|4"))
    def test_team_preview_accepts_standard_and_selected_team_shapes(self, raw: str) -> None:
        event = parse_protocol_event("replay-1", _line(raw))

        assert event.classification is EventClassification.NO_STATE_CHANGE

    def test_replace_accepts_the_pinned_showdown_shape_only(self) -> None:
        event = parse_protocol_event(
            "replay-1",
            _line("|replace|p1a: Zoroark|Zoroark-Hisui, L50"),
        )
        extra_hp = parse_protocol_event(
            "replay-1",
            _line("|replace|p1a: Zoroark|Zoroark-Hisui, L50|50/100"),
        )

        assert event.classification is EventClassification.PUBLIC_STATE
        assert event.arguments == ("p1a: Zoroark", "Zoroark-Hisui, L50")
        assert extra_hp.classification is EventClassification.MALFORMED

    def test_player_departure_accepts_an_empty_username(self) -> None:
        event = parse_protocol_event("replay-1", _line("|player|p2|"))

        assert event.classification is EventClassification.NO_STATE_CHANGE
        assert event.arguments == ("p2", "")

    def test_round_moved_up_by_round_is_the_users_own_move(self) -> None:
        boosted = parse_protocol_event(
            "replay-1", _line("|move|p1b: Politoed|Round|p2b: Espeon|[from] move: Round")
        )
        called = parse_protocol_event(
            "replay-1", _line("|move|p1b: Politoed|Round|p2b: Espeon|[from] move: Sleep Talk")
        )

        assert boosted.cause is None
        assert called.cause is not None and called.cause.normalized == "sleeptalk"

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

    def test_transform_requires_a_reference_target(self) -> None:
        species_target = parse_protocol_event(
            "replay-1",
            _line("|-transform|p1a: Ditto|Pikachu|[from] ability: Imposter"),
        )
        reference_target = parse_protocol_event(
            "replay-1",
            _line("|-transform|p1a: Ditto|p2a: Pikachu|[from] ability: Imposter"),
        )

        assert species_target.classification is EventClassification.MALFORMED
        assert reference_target.classification is EventClassification.PUBLIC_STATE
        assert tuple(reference.argument_index for reference in reference_target.pokemon_refs) == (
            0,
            1,
        )

    def test_state_neutral_source_shapes(self) -> None:
        assert (
            parse_protocol_event("r", _line("|-ohko")).classification
            is EventClassification.NO_STATE_CHANGE
        )
        assert (
            parse_protocol_event("r", _line("|bigerror|warning")).classification
            is EventClassification.NO_STATE_CHANGE
        )
        assert (
            parse_protocol_event("r", _line("|-sethp|p1a: A|50/100|[silent]")).classification
            is EventClassification.PUBLIC_STATE
        )

    def test_malformed_shapes_fail_closed(self) -> None:
        assert (
            parse_protocol_event("r", _line("|-ohko|p1a: A")).classification
            is EventClassification.MALFORMED
        )
        assert (
            parse_protocol_event("r", _line("|-sethp|p1a: A|50/100|extra")).classification
            is EventClassification.MALFORMED
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
