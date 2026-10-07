"""Tests for v2 replay decision-window reconstruction."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from p0.battle.actions import ActionKind, SlotAction, encode_action
from p0.battle.legality import DecisionView, SlotDecision, legal_actions
from p0.model.resources import default_runtime_resources
from p0.replays.compile import compile_payloads
from p0.replays.evidence import EvidenceRequest, ObservedAction, extract_action_evidence
from p0.replays.protocol import parse_replay_payload
from p0.replays.reconstruction.decisions import (
    BoundaryKind,
    build_decision_view,
    infer_decision_windows,
    reconstruct_replay_decisions_both,
)
from p0.replays.reconstruction.events import parse_replay_events
from p0.replays.reconstruction.resolution import resolve_protocol_events
from p0.replays.reconstruction.state import reduce_replay_state
from p0.replays.schema import LabelKind
from tests.unit.replay_fixtures import decision_payload, sample_replay_payload


def _decision_view_with_ots(
    *,
    p1_species: str = "Pikachu",
    p1_moves: list[str] | None = None,
    p1_b_moves: list[str] | None = None,
    p1_item: str = "Leftovers",
    p1_ability: str = "Static",
    extra_lines: tuple[str, ...] = (),
    dex: Mapping[str, Any] | None = None,
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
    if dex is None:
        dex = default_runtime_resources().dex
    parsed = parse_replay_events(document)
    resolved = resolve_protocol_events(
        document.metadata.replay_id, document.ots, parsed.events, dex=dex
    )
    snapshots = reduce_replay_state(
        document.metadata.replay_id, document.ots, resolved.require_accepted(), dex=dex
    ).require_accepted()
    events = resolved.require_accepted()
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
        dex=dex,
    )


class TestReconstructionDecisions:
    def test_decision_reconstruction_shares_boundaries_across_perspectives(self) -> None:
        document = parse_replay_payload(decision_payload())

        first, second = reconstruct_replay_decisions_both(document)

        assert not first.diagnostics
        assert not second.diagnostics
        assert first.decisions
        assert second.decisions
        assert first.windows == second.windows
        assert [window.kind for window in first.windows] == [
            BoundaryKind.RESIDUAL_TERMINAL,
            BoundaryKind.TEAM_PREVIEW,
            BoundaryKind.NORMAL_TURN,
            BoundaryKind.RESIDUAL_TERMINAL,
        ]
        assert [decision.decision_type.name for decision in first.decisions] == [
            "TEAM_PREVIEW",
            "TURN",
        ]
        assert [decision.pre_line_index for decision in first.decisions] == [4, 10]
        assert [decision.post_line_index for decision in first.decisions] == [10, 15]
        assert all(decision.evidence.candidates for decision in first.decisions)

    def test_decision_reconstruction_preserves_unknown_action_evidence(self) -> None:
        payload = decision_payload()
        payload["log"] = str(payload["log"])
        payload["log"] = payload["log"].replace(
            "|move|p1a: Pikachu|Protect|p1a: Pikachu", "|cant|p1a: Pikachu|par|"
        )
        payload["log"] = payload["log"].replace(
            "|move|p1b: Eevee|Tackle|p2b: Charmander", "|cant|p1b: Eevee|par|"
        )
        document = parse_replay_payload(payload)

        result = reconstruct_replay_decisions_both(document)[0]

        assert not result.diagnostics
        assert result.decisions[-1].evidence.label_kind.name == "UNKNOWN"
        assert "no_observed_order" in result.decisions[-1].evidence.tags

    def test_actionful_replay_without_separators_is_unrecoverable(self) -> None:
        payload = decision_payload()
        payload["log"] = str(payload["log"]).replace("\n|\n", "\n")
        document = parse_replay_payload(payload)
        parsed = parse_replay_events(document)
        resolved = resolve_protocol_events(
            document.metadata.replay_id,
            document.ots,
            parsed.events,
        )

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
        document = parse_replay_payload(payload)

        evidence = reconstruct_replay_decisions_both(document)[0].decisions[-1].evidence

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
        ("species", "item", "moves", "expected_can_mega"),
        (
            ("Rayquaza", "Leftovers", ["Protect", "Dragon Ascent"], True),
            ("Rayquaza", "Leftovers", ["Protect", "Tackle"], False),
            ("Rayquaza", "Flyinium Z", ["Protect", "Dragon Ascent"], False),
        ),
    )
    def test_mega_rayquaza_requires_dragon_ascent_and_non_z_item(
        self,
        species: str,
        item: str,
        moves: list[str],
        expected_can_mega: bool,
    ) -> None:
        base_dex = default_runtime_resources().dex
        dex = dict(base_dex)
        dex["items"] = list(base_dex["items"]) + [
            {"id": "flyiniumz", "name": "Flyinium Z", "zMove": True}
        ]
        view = _decision_view_with_ots(
            p1_species=species,
            p1_item=item,
            p1_moves=moves,
            dex=dex,
        )
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
        baseline = reconstruct_replay_decisions_both(parse_replay_payload(decision_payload()))[0]
        payload = decision_payload()
        payload["log"] = str(payload["log"]).replace(tackle, f"{boost_line}\n{tackle}")

        result = reconstruct_replay_decisions_both(parse_replay_payload(payload))[0]

        evidence = result.decisions[-1].evidence
        assert evidence.candidates == ((9, 13), (9, 15), (9, 16))
        assert evidence.candidates == baseline.decisions[-1].evidence.candidates
        assert evidence.tags == baseline.decisions[-1].evidence.tags == ("execution_target", "move")

    @pytest.mark.parametrize(
        ("singleturn_effect", "expected_tags", "expected_first_slot_actions"),
        (
            ("Protect", ("execution_target", "move"), {9}),
            (
                "Instruct",
                ("externally_generated_move", "execution_target"),
                {0, 3, 4, 5, 6, 9, 12, 15, 16, 48},
            ),
        ),
    )
    def test_only_instruct_singleturn_marks_following_move_as_generated(
        self,
        singleturn_effect: str,
        expected_tags: tuple[str, ...],
        expected_first_slot_actions: set[int],
    ) -> None:
        dex = default_runtime_resources().dex
        legal_effects = {
            name: list(effects) for name, effects in dex["legalProtocolEffects"].items()
        }
        legal_effects["effect"].append("instruct")
        payload = decision_payload()
        protect = "|move|p1a: Pikachu|Protect|p1a: Pikachu"
        payload["log"] = str(payload["log"]).replace(
            protect, f"|-singleturn|p1a: Pikachu|move: {singleturn_effect}\n{protect}"
        )

        result = reconstruct_replay_decisions_both(
            parse_replay_payload(payload), dex={**dex, "legalProtocolEffects": legal_effects}
        )[0]

        evidence = result.decisions[-1].evidence
        assert evidence.tags == expected_tags
        assert {first for first, _ in evidence.candidates} == expected_first_slot_actions
        assert {second for _, second in evidence.candidates} == {13, 15, 16}

    def test_mid_turn_encore_keeps_original_submitted_move_possible(self) -> None:
        payload = sample_replay_payload("encore-override")
        payload["log"] = str(payload["log"]).replace(
            "|move|p1b: Eevee|Tackle|p2b: Charmander",
            "|-start|p1b: Eevee|Encore\n|move|p1b: Eevee|Tackle|p2b: Charmander",
        )

        result = compile_payloads((payload,), chunksize=0)

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

        result = reconstruct_replay_decisions_both(parse_replay_payload(payload))[0]

        evidence = result.decisions[-1].evidence
        assert "hidden_disable_struggle" in evidence.tags
        assert (9, 13) in evidence.candidates

    def test_encore_after_actor_does_not_erase_observed_choice(self) -> None:
        payload = sample_replay_payload("encore-after-actor")
        payload["log"] = str(payload["log"]).replace(
            "|move|p1b: Eevee|Tackle|p2b: Charmander",
            "|move|p1b: Eevee|Tackle|p2b: Charmander\n|-start|p1b: Eevee|Encore",
        )

        result = compile_payloads((payload,), chunksize=0)

        affected = result.accepted_series[0].games[0].perspectives[0].decisions[-1]
        assert "mid_turn_encore_override" not in affected.evidence.tags
        assert (9, 11) in affected.evidence.candidates
        assert (9, 13) not in affected.evidence.candidates
