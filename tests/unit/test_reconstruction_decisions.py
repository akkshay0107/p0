"""Tests for v2 replay decision-window reconstruction."""

from __future__ import annotations

import json

import pytest

from p0.model.resources import default_runtime_resources
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
from tests.unit.replay_fixtures import decision_payload


def _decision_view_with_ots(
    *,
    p1_species: str = "Pikachu",
    p1_moves: list[str] | None = None,
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
            lines[index] = f"|showteam|p1|{json.dumps(members, separators=(',', ':'))}"
    if extra_lines:
        turn_index = next(index for index, line in enumerate(lines) if line.startswith("|turn|"))
        lines[turn_index:turn_index] = list(extra_lines)
    payload["log"] = "\n".join(lines)
    if p1_species != "Pikachu":
        payload["log"] = str(payload["log"]).replace("Pikachu", p1_species)
    document = parse_replay_payload(payload)
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
    def test_normal_move_targets_adjacent_ally_and_foes_in_doubles(self) -> None:
        view = _decision_view_with_ots()

        assert view.slots[0].move_targets[1] == (-2, 1, 2)
        assert view.slots[1].move_targets[1] == (-1, 1, 2)

    def test_adjacent_ally_or_self_targets_self_and_only_adjacent_ally(self) -> None:
        view = _decision_view_with_ots(p1_moves=["Protect", "Acupressure"])

        assert view.slots[0].move_targets[1] == (0, -2)

    def test_curse_targets_self_for_non_ghost_and_any_for_ghost(self) -> None:
        normal = _decision_view_with_ots(p1_moves=["Protect", "Curse"])
        ghost = _decision_view_with_ots(p1_species="Gengar", p1_moves=["Protect", "Curse"])

        assert normal.slots[0].move_targets[1] == (0,)
        assert ghost.slots[0].move_targets[1] == (-2, 1, 2)

    def test_mega_requires_compatible_species_and_is_consumed_once(self) -> None:
        compatible = _decision_view_with_ots(p1_species="Charizard", p1_item="Charizardite X")
        mismatch = _decision_view_with_ots(p1_species="Pikachu", p1_item="Charizardite X")
        used = _decision_view_with_ots(
            p1_species="Charizard",
            p1_item="Charizardite X",
            extra_lines=("|-mega|p1a: Pikachu|Charizardite X",),
        )

        assert compatible.slots[0].can_mega
        assert not mismatch.slots[0].can_mega
        assert not used.slots[0].can_mega

    def test_mega_rayquaza_requires_dragon_ascent_and_non_z_item(self) -> None:
        eligible = _decision_view_with_ots(
            p1_species="Rayquaza", p1_item="Leftovers", p1_moves=["Protect", "Dragon Ascent"]
        )
        missing_move = _decision_view_with_ots(
            p1_species="Rayquaza", p1_item="Leftovers", p1_moves=["Protect", "Tackle"]
        )

        assert eligible.slots[0].can_mega
        assert not missing_move.slots[0].can_mega

    def test_pollen_puff_keeps_ally_target_available_under_heal_block(self) -> None:
        own_block = _decision_view_with_ots(
            p1_moves=["Protect", "Pollen Puff"],
            extra_lines=("|-start|p1a: Pikachu|Heal Block",),
        )
        opponent_block = _decision_view_with_ots(
            p1_moves=["Protect", "Pollen Puff"],
            extra_lines=("|-start|p2a: Bulbasaur|Heal Block",),
        )

        assert own_block.slots[0].move_targets[1] == (-2, 1, 2)
        assert opponent_block.slots[0].move_targets[1] == (-2, 1, 2)

    def test_zygarde_mega_requires_complete_forme(self) -> None:
        base = _decision_view_with_ots(p1_species="Zygarde", p1_item="Zygardite")
        complete = _decision_view_with_ots(
            p1_species="Zygarde",
            p1_item="Zygardite",
            extra_lines=("|detailschange|p1a: Zygarde|Zygarde-Complete, L50",),
        )

        assert not base.slots[0].can_mega
        assert complete.slots[0].can_mega

    def test_faster_state_update_does_not_erase_slower_submitted_move(self) -> None:
        payload = decision_payload()
        payload["log"] = str(payload["log"]).replace(
            "|move|p1b: Eevee|Tackle|p2b: Charmander",
            "|-boost|p1a: Pikachu|atk|1\n|move|p1b: Eevee|Tackle|p2b: Charmander",
        )

        result = reconstruct_replay_decisions_both(parse_replay_payload(payload))[0]

        assert result.decisions[-1].evidence.candidates
        assert "submission_state_changed" not in result.decisions[-1].evidence.tags

    def test_only_instruct_singleturn_marks_following_move_as_generated(self) -> None:
        payload = decision_payload()
        payload["log"] = str(payload["log"]).replace(
            "|move|p1b: Eevee|Tackle|p2b: Charmander",
            "|-singleturn|p1a: Pikachu|move: Protect\n|move|p1b: Eevee|Tackle|p2b: Charmander",
        )

        result = reconstruct_replay_decisions_both(parse_replay_payload(payload))[0]

        assert result.decisions[-1].evidence.candidates
        assert "externally_generated_move" not in result.decisions[-1].evidence.tags
