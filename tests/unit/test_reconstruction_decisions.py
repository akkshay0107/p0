"""Tests for v2 replay decision-window reconstruction."""

from __future__ import annotations

import json

import pytest

from p0.battle.actions import ActionKind, SlotAction, encode_action
from p0.battle.legality import DecisionView, SlotDecision, legal_actions
from p0.replays.compile import compile_payloads
from p0.replays.evidence import EvidenceRequest, ObservedAction, extract_action_evidence
from p0.replays.protocol import parse_replay_payload
from p0.replays.reconstruction.decisions import (
    build_decision_view,
    infer_decision_windows,
)
from p0.replays.reconstruction.resolution import resolve_replay_events
from p0.replays.reconstruction.state import reduce_replay_state
from p0.replays.schema import DecisionType, LabelKind
from tests.unit.replay_fixtures import decision_payload, sample_replay_payload


def _decision_view_with_ots(
    *,
    p1_species: str = "Pikachu",
    p1_moves: list[str] | None = None,
    p1_b_moves: list[str] | None = None,
    p1_item: str = "Leftovers",
    p1_ability: str = "Static",
    extra_lines: tuple[str, ...] = (),
):
    payload = decision_payload()
    lines = str(payload["log"]).splitlines()
    for index, line in enumerate(lines):
        if line.startswith("|showteam|p1|"):
            members = json.loads(line.split("|", 3)[3])
            members[0].update(
                {"species": p1_species, "name": p1_species, "item": p1_item, "ability": p1_ability}
            )
            if p1_moves is not None:
                members[0]["moves"] = p1_moves
            if p1_b_moves is not None:
                members[1]["moves"] = p1_b_moves
            lines[index] = f"|showteam|p1|{json.dumps(members, separators=(',', ':'))}"
    if extra_lines:
        turn_index = next(index for index, line in enumerate(lines) if line.startswith("|turn|"))
        lines[turn_index:turn_index] = list(extra_lines)
    payload["log"] = "\n".join(lines)
    if p1_species != "Pikachu":
        payload["log"] = str(payload["log"]).replace("Pikachu", p1_species)
    document = parse_replay_payload(payload)
    events = resolve_replay_events(document).require_accepted()
    snapshots = reduce_replay_state(
        document.metadata.replay_id, document.ots, events
    ).require_accepted()
    turn_index = next(index for index, line in enumerate(lines) if line.startswith("|turn|"))
    separator_index = next(
        index for index in range(turn_index + 1, len(events)) if events[index].event.tag == ""
    )
    return build_decision_view(
        snapshots[turn_index - 1],
        document.ots[0],
        0,
        events[turn_index:separator_index],
        preview=False,
    )


class TestReconstructionDecisions:
    def test_decision_reconstruction_shares_boundaries_across_perspectives(self) -> None:
        payload = decision_payload()
        events = resolve_replay_events(parse_replay_payload(payload)).require_accepted()

        windows = infer_decision_windows(events)
        first, second = compile_payloads((payload,)).accepted_series[0].games[0].perspectives

        assert [window.decision_type for window in windows] == [
            DecisionType.UNSPECIFIED,
            DecisionType.TEAM_PREVIEW,
            DecisionType.TURN,
            DecisionType.UNSPECIFIED,
        ]
        for perspective in (first, second):
            assert [decision.decision_type.name for decision in perspective.decisions] == [
                "TEAM_PREVIEW",
                "TURN",
            ]
            assert [decision.pre_line_index for decision in perspective.decisions] == [4, 10]
            assert [decision.post_line_index for decision in perspective.decisions] == [10, 15]
            assert all(decision.evidence.candidates for decision in perspective.decisions)

    def test_decision_reconstruction_preserves_unknown_action_evidence(self) -> None:
        payload = decision_payload()
        payload["log"] = str(payload["log"])
        payload["log"] = payload["log"].replace(
            "|move|p1a: Pikachu|Protect|p1a: Pikachu", "|cant|p1a: Pikachu|par|"
        )
        payload["log"] = payload["log"].replace(
            "|move|p1b: Eevee|Tackle|p2b: Charmander", "|cant|p1b: Eevee|par|"
        )

        result = compile_payloads((payload,)).accepted_series[0].games[0].perspectives[0]

        assert result.decisions[-1].evidence.label_kind.name == "UNKNOWN"
        assert "no_observed_order" in result.decisions[-1].evidence.tags

    def test_actionful_replay_without_separators_is_unrecoverable(self) -> None:
        payload = decision_payload()
        payload["log"] = str(payload["log"]).replace("\n|\n", "\n")
        resolved = resolve_replay_events(parse_replay_payload(payload))

        with pytest.raises(ValueError, match="actions but no update separators"):
            infer_decision_windows(resolved.require_accepted())


class TestDecisionTargets:
    @pytest.mark.parametrize(
        (
            "p1_species",
            "p1_moves",
            "p1_b_moves",
            "extra_lines",
            "slot_idx",
            "move_idx",
            "expected_targets",
        ),
        (
            ("Pikachu", ["Protect", "Tackle"], None, (), 0, 1, (-2, 1, 2)),
            ("Pikachu", ["Protect", "Tackle"], None, (), 1, 1, (-1, 1, 2)),
            ("Pikachu", ["Protect", "Acupressure"], None, (), 0, 1, (-2, -1)),
            ("Pikachu", None, ["Protect", "Acupressure", "Tackle"], (), 1, 1, (-1, -2)),
            ("Pikachu", ["Protect", "Curse"], None, (), 0, 1, (0,)),
            ("Gengar", ["Protect", "Curse"], None, (), 0, 1, (-2, 1, 2)),
            (
                "Pikachu",
                ["Protect", "Pollen Puff"],
                None,
                ("|-start|p1a: Pikachu|Heal Block",),
                0,
                1,
                (-2, 1, 2),
            ),
            (
                "Pikachu",
                ["Protect", "Pollen Puff"],
                None,
                ("|-start|p2a: Bulbasaur|Heal Block",),
                0,
                1,
                (-2, 1, 2),
            ),
        ),
    )
    def test_normal_move_targets_adjacent_ally_and_foes_in_doubles(
        self,
        p1_species: str,
        p1_moves: list[str] | None,
        p1_b_moves: list[str] | None,
        extra_lines: tuple[str, ...],
        slot_idx: int,
        move_idx: int,
        expected_targets: tuple[int, ...],
    ) -> None:
        view = _decision_view_with_ots(
            p1_species=p1_species,
            p1_moves=p1_moves,
            p1_b_moves=p1_b_moves,
            extra_lines=extra_lines,
        )
        assert view.slots[slot_idx].move_targets[move_idx] == expected_targets

    def test_observed_self_target_label_matches_the_live_self_slot_order(self) -> None:
        payload = decision_payload()
        payload["log"] = (
            str(payload["log"])
            .replace('"moves":["Protect","Tackle"]', '"moves":["Protect","Acupressure"]', 1)
            .replace(
                "|move|p1a: Pikachu|Protect|p1a: Pikachu",
                "|move|p1a: Pikachu|Acupressure|p1a: Pikachu",
            )
        )

        game = compile_payloads((payload,)).accepted_series[0].games[0]
        evidence = game.perspectives[0].decisions[-1].evidence

        live_self_order = encode_action(SlotAction(ActionKind.MOVE, move_slot=1, target=-1))
        assert live_self_order in {candidate[0] for candidate in evidence.candidates}

    @pytest.mark.parametrize(
        ("species", "item", "extra_lines", "expected_can_mega"),
        (
            ("Charizard", "Charizardite X", (), True),
            ("Pikachu", "Charizardite X", (), False),
            ("Charizard", "Charizardite X", ("|-mega|p1a: Charizard|Charizardite X",), False),
        ),
    )
    def test_mega_requires_compatible_species_and_is_consumed_once(
        self,
        species: str,
        item: str,
        extra_lines: tuple[str, ...],
        expected_can_mega: bool,
    ) -> None:
        view = _decision_view_with_ots(p1_species=species, p1_item=item, extra_lines=extra_lines)
        assert view.slots[0].can_mega is expected_can_mega
        assert any(27 <= action < 48 for action in legal_actions(view, 0)) is expected_can_mega

    @pytest.mark.parametrize(
        ("extra_lines", "expected_can_mega"),
        (
            ((), False),
            (("|detailschange|p1a: Zygarde|Zygarde-Complete, L50",), True),
        ),
    )
    def test_zygarde_mega_requires_complete_forme(
        self,
        extra_lines: tuple[str, ...],
        expected_can_mega: bool,
    ) -> None:
        view = _decision_view_with_ots(
            p1_species="Zygarde",
            p1_item="Zygardite",
            extra_lines=extra_lines,
        )
        assert view.slots[0].can_mega is expected_can_mega
        assert any(27 <= action < 48 for action in legal_actions(view, 0)) is expected_can_mega

    def test_candidate_cap_degrades_to_explicit_unknown_evidence(self) -> None:
        """Verify that when candidate action space exceeds max_candidates, the label degrades to LabelKind.UNKNOWN with empty candidates, while uncapped retains partial candidates."""
        view = DecisionView(
            slots=(
                SlotDecision(move_targets=((-2, -1),)),
                SlotDecision(move_targets=((-2,),)),
            )
        )
        capped = extract_action_evidence(
            EvidenceRequest(
                view=view,
                slots=(
                    ObservedAction(alternatives=(7, 8), exact=False),
                    ObservedAction(action=7),
                ),
                max_candidates=1,
            )
        )
        assert capped.label_kind is LabelKind.UNKNOWN
        assert capped.candidates == ()
        assert "candidate_cap_or_illegal" in capped.tags

        uncapped = extract_action_evidence(
            EvidenceRequest(
                view=view,
                slots=(
                    ObservedAction(alternatives=(7, 8), exact=False),
                    ObservedAction(action=7),
                ),
                max_candidates=256,
            )
        )
        assert uncapped.label_kind is LabelKind.PARTIAL
        assert uncapped.candidates == ((7, 7), (8, 7))
        assert uncapped.tags == ()

    @pytest.mark.parametrize(
        "boost_line",
        ("|-boost|p1a: Pikachu|atk|1", "|-boost|p2a: Bulbasaur|spe|1"),
    )
    def test_faster_state_update_does_not_erase_slower_submitted_move(
        self, boost_line: str
    ) -> None:
        tackle = "|move|p1b: Eevee|Tackle|p2b: Charmander"
        payload = decision_payload()
        payload["log"] = str(payload["log"]).replace(tackle, f"{boost_line}\n{tackle}")

        result = compile_payloads((payload,)).accepted_series[0].games[0].perspectives[0]

        evidence = result.decisions[-1].evidence
        assert evidence.candidates == ((9, 13), (9, 15), (9, 16))
        assert evidence.tags == ("execution_target", "move")

    def test_mid_turn_encore_keeps_original_submitted_move_possible(self) -> None:
        payload = sample_replay_payload("encore-override")
        payload["log"] = str(payload["log"]).replace(
            "|move|p1b: Eevee|Tackle|p2b: Charmander",
            "|-start|p1b: Eevee|Encore\n|move|p1b: Eevee|Tackle|p2b: Charmander",
        )

        result = compile_payloads((payload,))

        assert result.metrics.counters["accepted_games"] == 1
        affected = result.accepted_series[0].games[0].perspectives[0].decisions[-1]
        assert (9, 13) in affected.evidence.candidates
        assert (9, 11) in affected.evidence.candidates

    def test_struggle_under_opposing_imprison_keeps_submitted_moves_possible(self) -> None:
        payload = decision_payload()
        payload["log"] = (
            str(payload["log"])
            .replace("|turn|1", "|-start|p2a: Bulbasaur|move: Imprison\n|turn|1")
            .replace(
                "|move|p1a: Pikachu|Protect|p1a: Pikachu",
                "|-activate|p1a: Pikachu|move: Struggle\n|move|p1a: Pikachu|Struggle|p2a: Bulbasaur",
            )
        )

        result = compile_payloads((payload,)).accepted_series[0].games[0].perspectives[0]

        evidence = result.decisions[-1].evidence
        assert "hidden_disable_struggle" in evidence.tags
        assert (9, 13) in evidence.candidates

    def test_encore_after_actor_does_not_erase_observed_choice(self) -> None:
        payload = sample_replay_payload("encore-after-actor")
        payload["log"] = str(payload["log"]).replace(
            "|move|p1b: Eevee|Tackle|p2b: Charmander",
            "|move|p1b: Eevee|Tackle|p2b: Charmander\n|-start|p1b: Eevee|Encore",
        )

        result = compile_payloads((payload,))

        affected = result.accepted_series[0].games[0].perspectives[0].decisions[-1]
        assert "mid_turn_encore_override" not in affected.evidence.tags
        assert (9, 11) in affected.evidence.candidates
        assert (9, 13) not in affected.evidence.candidates


class TestForcedMoves:
    def test_locked_move_is_the_forced_action(self) -> None:
        # Eevee does not own Dig: it copied the move and is on its second turn.
        payload = decision_payload()
        payload["log"] = str(payload["log"]).replace(
            "|move|p1b: Eevee|Tackle|p2b: Charmander",
            "|move|p1b: Eevee|Dig|p2b: Charmander|[from] lockedmove",
        )

        result = compile_payloads((payload,)).accepted_series[0].games[0].perspectives[0]

        evidence = result.decisions[-1].evidence
        assert evidence.label_kind is LabelKind.EXACT
        # Pikachu chose Protect (move slot 0, no target); Eevee had the forced action.
        assert evidence.candidates == ((9, 48),)

    def test_struggle_after_ally_switch_belongs_to_the_slot_held_at_the_request(self) -> None:
        payload = decision_payload()
        payload["log"] = (
            str(payload["log"])
            .replace(
                '"species":"Eevee","name":"Eevee","ability":"Static","moves":["Protect","Tackle"]',
                '"species":"Eevee","name":"Eevee","ability":"Static","moves":["Ally Switch"]',
            )
            .replace(
                "|move|p1a: Pikachu|Protect|p1a: Pikachu\n|move|p1b: Eevee|Tackle|p2b: Charmander",
                "|move|p1b: Eevee|Ally Switch|p1b: Eevee\n"
                "|swap|p1b: Eevee|0|[from] move: Ally Switch\n"
                "|-activate|p1b: Pikachu|move: Struggle\n"
                "|move|p1b: Pikachu|Struggle|p2a: Bulbasaur",
            )
        )

        result = compile_payloads((payload,)).accepted_series[0].games[0].perspectives[0]

        evidence = result.decisions[-1].evidence
        assert evidence.label_kind is LabelKind.EXACT
        # Pikachu was in the first slot when it was asked, so the forced action is there.
        assert evidence.candidates == ((48, 9),)
