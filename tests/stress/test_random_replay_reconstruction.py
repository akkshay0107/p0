"""Stress the live observation -> replay -> reconstructed observation path.

The important assertion in this test is deliberately made before tensor fusion:
the observation captured at a live request must be the same observation that the
replay reconstructs.  The test also keeps the action selected by the live bot so
that PARTIAL and UNKNOWN labels are checked against the thing that actually
happened in the battle, rather than only against their tensor schema.

For the pinned champions Showdown mod, a missing action can be represented by
cant.  The current engine emits that outcome for paralysis, sleep, freeze,
flinch, recharge, no PP, the move-prevention abilities, and the move/volatile
prevention effects listed in _CANT_REASONS below.  This inventory was checked
against the pinned Showdown source while adding this test; it is intentionally
format-specific rather than a claim about every historical Showdown mod.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from pathlib import Path
from typing import Any, cast

import orjson
import pytest
import torch
from poke_env import AccountConfiguration
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import RandomPlayer
from poke_env.player.battle_order import BattleOrder, SingleBattleOrder

from p0.battle.actions import PASS_ACTION
from p0.battle.events import get_hp_fraction
from p0.format_config import FORMAT
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import (
    NUM_IDX_CAN_MEGA,
    NUM_IDX_CAN_SWITCH_OUT,
    NUM_IDX_HP_FRACTION,
    NUM_IDX_LEGALITY_UNKNOWN,
    NUM_IDX_LEVEL_STATS,
    NUM_IDX_MOVE_LEGAL,
    NUM_IDX_SLOT_CONDITION_UNKNOWN,
    NUM_IDX_SLOT_LEGALITY_UNKNOWN,
    NUM_IDX_STATUS_COUNTER,
    TOKEN_IDX_ALLY_SIDE,
    StructuredObservation,
)
from p0.replays.compile import compile_documents, write_tensor_shards
from p0.replays.protocol import ReplayDocument, parse_replay_payload
from p0.replays.reconstruct import (
    ReconstructedPerspective,
    ReconstructedSnapshot,
    normalize_id,
    reconstruct_both,
)
from p0.replays.schema import LabelKind
from p0.replays.shards import validate_shard_tensors
from p0.rl_player import TeamPlayerMixin
from p0.runtime import poke_env_patches
from p0.runtime.poke_env_action_adapter import order_to_action
from p0.runtime.poke_env_battle_adapter import battle_view
from p0.teams.source import FileTeamSource
from p0.teams.stat_points import StatPoints
from p0.teams.team import TeamMember
from tests.stress._helpers import stress_count, stress_random_team_record

# Cant reasons emitted by the checked-in Showdown commit for the champions
# format.  The source locations are data/conditions.ts, data/abilities.ts,
# data/moves.ts, and sim/battle-actions.ts in the pokemon-showdown submodule.
_CANT_REASONS = frozenset(
    {
        "par",
        "slp",
        "frz",
        "flinch",
        "recharge",
        "nopp",
        "attract",
        "disable",
        "focus punch",
        "shell trap",
        "ability: armor tail",
        "ability: damp",
        "ability: dazzling",
        "ability: queenly majesty",
        "ability: truant",
        "move: gravity",
        "move: heal block",
        "move: imprison",
        "move: taunt",
        "move: throat chop",
    }
)

_REPLAY_AMBIGUITY_TAGS = frozenset(
    {
        "move_slot_or_target_unknown",
        "switch_slot_unknown",
        "multiple_moves_same_slot",
        "candidate_cap_or_illegal",
        "preview_leads_unknown",
        "preview_roster_unknown",
        "preview_duplicate_lead",
        "preview_reserves_unknown",
        "externally_generated_move",
        # the submitted order left no trace: the mon was KO'd before it acted, or the
        # request was answered with a pass that the protocol never renders
        "no_observed_order",
    }
)

_OBSERVATION_EVENT_FIELDS = ("spatial_cat",)


def _message_tag(line: str) -> str:
    """Extract Showdown protocol message identifier (e.g. '|move|' -> 'move')."""
    parts = line.split("|")
    return parts[1] if len(parts) > 1 else ""


def _endpoint_role(endpoint: str) -> str:
    """Extract player role prefix from an entity string (e.g. 'p1a: Pikachu' -> 'p1')."""
    return endpoint.split("a", 1)[0].strip()


def _cant_reasons(snapshot: ReconstructedSnapshot, role: int) -> tuple[str, ...]:
    """Collect all '|cant|' failure reasons recorded in a turn snapshot for a specific player role."""
    reasons: list[str] = []
    for raw_line in snapshot.raw_lines:
        parts = raw_line.split("|")
        if len(parts) < 4 or parts[1] != "cant":
            continue
        if _endpoint_role(parts[2]) != f"p{role + 1}":
            continue
        reasons.append(parts[3].casefold().strip())
    return tuple(dict.fromkeys(reasons))


def _showteam_offset(document: ReplayDocument) -> tuple[int, int]:
    """Where the harness spliced open-team-sheet lines into the captured log.

    The live cursor counts only lines poke-env parsed, and showteam is not one of
    them, so cursors at or past the splice point are shifted by the inserted lines.
    """
    indices = [
        line.index
        for line in document.protocol_lines
        if len(line.parts) > 1 and line.parts[1] == "showteam"
    ]
    return (len(document.protocol_lines), 0) if not indices else (indices[0], len(indices))


def _live_boundary(record: dict[str, Any], splice_index: int, splice_count: int) -> int:
    """Adjust live replay cursor index to align with spliced replay document protocol lines."""
    cursor = int(record["replay_cursor"])
    return cursor if cursor <= splice_index else cursor + splice_count


class JsonCapturingRandomPlayer(TeamPlayerMixin, RandomPlayer):
    """Random player that saves pre-fusion decisions and completed replay JSON.

    Captures exact raw observation tensors, action decisions, and replay line stream offsets
    during live battle execution so offline replay reconstruction can be verified 1-to-1 against ground truth.
    """

    def __init__(
        self,
        *,
        observation_builder: ObservationBuilder,
        observation_dir: Path,
        replay_dir: Path | None = None,
        write_replays: bool = False,
        team_source: FileTeamSource,
        team_rng: random.Random,
        **kwargs: Any,
    ) -> None:
        self.observation_builder = observation_builder
        self.observation_dir = observation_dir
        self.replay_dir = replay_dir
        self.write_replays = write_replays
        self.replay_paths: list[Path] = []
        self.observation_paths: list[Path] = []
        self.teams_seen: list[str] = []
        self._showteam_lines: dict[str, list[str]] = {}
        self._live_records: dict[str, list[dict[str, Any]]] = {}
        super().__init__(team_source=team_source, team_rng=team_rng, **kwargs)

    async def _handle_battle_message(self, split_messages: list[list[str]]) -> None:
        battle_tag = split_messages[0][0].removeprefix(">")
        for split_message in split_messages[1:]:
            if len(split_message) > 1 and split_message[1] == "showteam":
                self._showteam_lines.setdefault(battle_tag, []).append("|".join(split_message))

        await super()._handle_battle_message(split_messages)

    def teampreview(self, battle: AbstractBattle) -> str:
        result = super().teampreview(battle)
        if not isinstance(result, str):
            raise TypeError(f"RandomPlayer returned a non-string preview order: {result!r}")
        double_battle = cast(DoubleBattle, battle)
        action_values = order_to_action(SingleBattleOrder(result), double_battle)
        action = int(action_values[0]), int(action_values[1])
        self._record_decision(double_battle, action)
        return result

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        result = super().choose_move(battle)
        if not isinstance(result, BattleOrder):
            raise TypeError(f"RandomPlayer returned a non-order move: {result!r}")
        double_battle = cast(DoubleBattle, battle)
        action_values = order_to_action(result, double_battle)
        action = int(action_values[0]), int(action_values[1])
        self._record_decision(double_battle, action)
        return result

    def _record_decision(self, battle: DoubleBattle, action: tuple[int, int]) -> None:
        observation = self.observation_builder.build(battle_view(battle)).cpu()
        self._live_records.setdefault(battle.battle_tag, []).append(
            {
                "turn": int(battle.turn),
                "teampreview": bool(battle.teampreview),
                "player_role": str(battle.player_role),
                "action": list(action),
                # The count of replay-visible lines consumed before this request is the
                # shared boundary: reconstruction reaches the same line index when it
                # segments the log, so the two sides are matched exactly rather than
                # by a turn heuristic.
                "replay_cursor": len(battle._replay_data),
                # Save the individual pre-fusion fields, not a fused model input.
                "tensors": [tensor.clone() for tensor in observation.tensors()],
            }
        )

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        self._write_observation_tensors(battle)
        if self.write_replays:
            self._write_replay_json(battle)
        if self.current_team_packed is not None:
            self.teams_seen.append(self.current_team_packed)
        super()._battle_finished_callback(battle)

    def _write_observation_tensors(self, battle: AbstractBattle) -> None:
        replay_id = battle.battle_tag.removeprefix("battle-")
        role = str(battle.player_role)
        payload = {
            "records": self._live_records.pop(battle.battle_tag, []),
            # This side's own copy of the protocol stream. For the player that does not
            # write the replay it is an independent capture of the same battle, which
            # is what makes the line-stream comparison a real check.
            "lines": ["|".join(message) for message in battle._replay_data],
        }
        self.observation_dir.mkdir(parents=True, exist_ok=True)
        path = self.observation_dir / f"{replay_id}-{role}.pt"
        torch.save(payload, path)
        self.observation_paths.append(path)

    def _write_replay_json(self, battle: AbstractBattle) -> None:
        if self.replay_dir is None:
            raise RuntimeError("Replay output is enabled without a replay directory")

        replay_id = battle.battle_tag.removeprefix("battle-")
        replay_events = list(getattr(battle, "_build_replay_events")())
        showteam_lines = self._showteam_lines.pop(battle.battle_tag, [])

        if showteam_lines:
            insertion_point = next(
                (
                    index
                    for index, line in enumerate(replay_events)
                    if _message_tag(line) == "switch"
                ),
                len(replay_events),
            )
            replay_events[insertion_point:insertion_point] = showteam_lines

        players = battle.players
        winner = next(
            (line.split("|", 2)[2] for line in replay_events if _message_tag(line) == "win"),
            "",
        )
        payload = {
            "id": replay_id,
            "formatid": FORMAT.battle_format,
            "p1": players[0],
            "p2": players[1],
            "winner": winner,
            "roomid": battle.battle_tag,
            "parent": replay_id,
            "game_number": 1,
            "uploadtime": 0,
            "log": "\n".join(replay_events),
        }

        self.replay_dir.mkdir(parents=True, exist_ok=True)
        path = self.replay_dir / f"{replay_id}.json"
        path.write_bytes(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS) + b"\n")
        self.replay_paths.append(path)


def _stress_game_count() -> int:
    return stress_count("P0_STRESS_GAME_COUNT", 100)


def _stress_concurrency() -> int:
    return stress_count("P0_STRESS_CONCURRENCY", 4)


def _stress_timeout(game_count: int) -> float:
    value = float(os.getenv("P0_STRESS_TIMEOUT_SECONDS", str(max(120, game_count * 15))))
    if value <= 0:
        raise ValueError("P0_STRESS_TIMEOUT_SECONDS must be positive")
    return value


def _showdown_team_text(team: tuple[TeamMember, ...], spreads: tuple[StatPoints, ...]) -> str:
    """Serialize a generated team into the Showdown text format used by FileTeamSource."""
    stat_names = {
        "hp": "HP",
        "atk": "Atk",
        "def": "Def",
        "spa": "SpA",
        "spd": "SpD",
        "spe": "Spe",
    }
    sets = []
    for member, spread in zip(team, spreads, strict=True):
        evs = " / ".join(
            f"{value} {stat_names[name]}" for name, value in spread.as_dict().items() if value
        )
        lines = [
            f"{member.species} @ {member.item}",
            f"Ability: {member.ability}",
            f"Level: {member.level}",
        ]
        if evs:
            lines.append(f"EVs: {evs}")
        lines.extend((f"{member.nature} Nature", *(f"- {move}" for move in member.moves)))
        sets.append("\n".join(lines))
    return "\n\n".join(sets) + "\n"


def _random_team_source(directory: Path, *, seed: int, count: int) -> FileTeamSource:
    """Create a temporary FileTeamSource populated only with generated legal teams."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for index in range(count):
        record = stress_random_team_record(rng, label=f"live-random-{index}")
        (directory / f"random-{index:04d}.txt").write_text(
            _showdown_team_text(record.team.members, record.spreads), encoding="utf-8"
        )
    return FileTeamSource(directory)


def _load_live_artifact(path: Path) -> dict[str, Any]:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(artifact, dict)
        or not isinstance(artifact.get("records"), list)
        or not isinstance(artifact.get("lines"), list)
    ):
        raise AssertionError(f"Malformed live observation artifact: {path}")
    return {
        "records": tuple(cast(dict[str, Any], record) for record in artifact["records"]),
        "lines": tuple(str(line) for line in artifact["lines"]),
    }


def _live_tensors(record: dict[str, Any]) -> StructuredObservation:
    tensors = record.get("tensors")
    if not isinstance(tensors, list) or len(tensors) != len(StructuredObservation._FIELD_NAMES):
        raise AssertionError("Live record does not contain all pre-fusion observation fields")
    return StructuredObservation._from_values([cast(torch.Tensor, tensor) for tensor in tensors])


def _live_action(record: dict[str, Any]) -> tuple[int, int] | None:
    value = record.get("action")
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise AssertionError(f"Malformed live action: {value!r}")
    return int(value[0]), int(value[1])


def _assert_expected_observation_fields(
    live: StructuredObservation,
    reconstructed: StructuredObservation,
    *,
    compare_event_fields: bool = True,
) -> None:
    """Compare every field a public replay can reproduce at a shared boundary."""
    fields = (
        "token_type_ids",
        "side_ids",
        "slot_ids",
        "categorical",
    )
    for name in fields:
        torch.testing.assert_close(getattr(live, name), getattr(reconstructed, name))
    if compare_event_fields:
        for name in _OBSERVATION_EVENT_FIELDS:
            torch.testing.assert_close(getattr(live, name), getattr(reconstructed, name))

    # Replays store the player's own HP as a percentage, so the HP column and the
    # event HP deltas are quantized relative to the live capture.
    torch.testing.assert_close(live.spatial_num, reconstructed.spatial_num, atol=0.02, rtol=0)
    torch.testing.assert_close(
        live.numerical[:, NUM_IDX_HP_FRACTION],
        reconstructed.numerical[:, NUM_IDX_HP_FRACTION],
        atol=0.02,
        rtol=0,
    )

    # Status age is not fully observable in a public replay: sleep duration includes
    # hidden randomness, while toxic stage is reconstructed from public damage turns.
    # Compare its bounded representation rather than requiring hidden state equality.
    for observation in (live, reconstructed):
        status_counter = observation.numerical[..., NUM_IDX_STATUS_COUNTER]
        assert torch.isfinite(status_counter).all()
        assert torch.all((status_counter >= 0) & (status_counter <= 1))

    # Which reserves the player brought is private until one appears, so a row the
    # replay cannot place yet reports the unknown slot condition instead of guessing,
    # and its side-token mega availability rides the same gate.
    known_rows = reconstructed.numerical[:, NUM_IDX_SLOT_CONDITION_UNKNOWN] == 0
    known_rows[TOKEN_IDX_ALLY_SIDE] = (
        reconstructed.numerical[TOKEN_IDX_ALLY_SIDE, NUM_IDX_LEGALITY_UNKNOWN] == 0
    )
    columns = [
        index
        for index in range(NUM_IDX_LEVEL_STATS)
        if index not in {NUM_IDX_CAN_MEGA, NUM_IDX_STATUS_COUNTER}
    ]
    columns.remove(NUM_IDX_HP_FRACTION)
    torch.testing.assert_close(
        live.numerical[known_rows][:, columns], reconstructed.numerical[known_rows][:, columns]
    )
    _assert_legality_provenance(live, reconstructed)


def _assert_legality_provenance(
    live: StructuredObservation,
    reconstructed: StructuredObservation,
) -> None:
    """A reconstructed legality cell must match live exactly, or be marked unproven.

    Hidden level/stat values and their provenance intentionally differ: the live battle
    knows the player's actual values, while a replay reconstructs them from the public
    species/OTS information. Legality is different — it must never be wrong, so a
    replay either proves it or raises the unknown gate and writes zeros.
    """
    gates = slice(NUM_IDX_SLOT_LEGALITY_UNKNOWN, NUM_IDX_SLOT_LEGALITY_UNKNOWN + 2)
    assert torch.all(live.numerical[:, NUM_IDX_LEGALITY_UNKNOWN] == 0)
    assert torch.all(live.numerical[TOKEN_IDX_ALLY_SIDE, gates] == 0)

    unproven = reconstructed.numerical[:, NUM_IDX_LEGALITY_UNKNOWN] > 0
    legality = slice(NUM_IDX_MOVE_LEGAL, NUM_IDX_CAN_SWITCH_OUT + 1)
    assert torch.all(reconstructed.numerical[unproven][:, legality] == 0), (
        "An unproven legality row must carry zeros, never a guessed value"
    )
    assert torch.all(reconstructed.numerical[unproven, NUM_IDX_CAN_MEGA] == 0)

    proven = ~unproven
    torch.testing.assert_close(
        live.numerical[proven][:, legality], reconstructed.numerical[proven][:, legality]
    )
    torch.testing.assert_close(
        live.numerical[proven, NUM_IDX_CAN_MEGA], reconstructed.numerical[proven, NUM_IDX_CAN_MEGA]
    )


def _assert_reconstruction_labels(document: ReplayDocument) -> tuple[ReconstructedPerspective, ...]:
    perspectives = reconstruct_both(
        document,
        max_candidates=256,
        dex=default_runtime_resources().dex,
    )
    for perspective in perspectives:
        assert perspective.decisions
        assert len(perspective.snapshots) == len(perspective.decisions)
        for decision in perspective.decisions:
            evidence = decision.evidence
            if evidence.label_kind is LabelKind.EXACT:
                assert len(evidence.candidates) == 1
            elif evidence.label_kind is LabelKind.PARTIAL:
                assert len(evidence.candidates) >= 2
            elif evidence.label_kind is LabelKind.UNKNOWN:
                assert not evidence.candidates
            else:
                pytest.fail(f"Unsupported reconstructed label kind: {evidence.label_kind!r}")
    return perspectives


def _assert_live_action_is_observable(
    live_action: tuple[int, int],
    evidence: Any,
) -> None:
    """Check only the live action components that a public replay can expose.

    Showdown does not emit a protocol line for a slot that submitted pass.
    A replacement request can therefore produce a live pair such as
    (switch, pass) while the replay only provides candidates for the switch
    slot. Treating that invisible component as a required joint-action match
    manufactures a reconstruction failure at an otherwise shared boundary.
    """
    if evidence.label_kind is LabelKind.EXACT and all(
        action != PASS_ACTION for action in live_action
    ):
        assert live_action == evidence.candidates[0]
        return

    assert evidence.candidates
    for slot, action in enumerate(live_action):
        if action == PASS_ACTION:
            continue
        assert any(candidate[slot] == action for candidate in evidence.candidates), (
            f"Live action component {action} for slot {slot} was not candidate-contained: "
            f"candidates={evidence.candidates!r}"
        )


def _match_live_records(
    perspective: ReconstructedPerspective,
    live_records: tuple[dict[str, Any], ...],
    document: ReplayDocument,
) -> tuple[tuple[ReconstructedSnapshot, dict[str, Any]], ...]:
    """Pair requests with decisions at the boundary both sides can name.

    Reconstruction is not required to reproduce the live request schedule, so waits and
    requests answered with no order are dropped. Every request that carried an order is
    a shared boundary and must be represented by a decision at the same line index.
    """
    splice_index, splice_count = _showteam_offset(document)
    boundaries = {snapshot.pre_line_index: snapshot for snapshot in perspective.snapshots}

    matched: list[tuple[ReconstructedSnapshot, dict[str, Any]]] = []
    for record in live_records:
        if _live_action(record) is None:
            continue
        boundary = _live_boundary(record, splice_index, splice_count)
        snapshot = boundaries.get(boundary)
        if snapshot is None:
            raise AssertionError(
                f"No reconstructed decision at line {boundary} for "
                f"{perspective.game_id} p{perspective.player + 1} turn {record['turn']}; "
                f"decision boundaries were {sorted(boundaries)}"
            )
        matched.append((snapshot, record))
    return tuple(matched)


def _assert_live_truth(
    perspective: ReconstructedPerspective,
    live_records: tuple[dict[str, Any], ...],
    document: ReplayDocument,
    builder: ObservationBuilder,
) -> None:
    seen_replay_cursors: set[int] = set()
    for snapshot, record in _match_live_records(perspective, live_records, document):
        decision = perspective.decisions[snapshot.decision_index]
        evidence = decision.evidence
        live_action = _live_action(record)
        assert live_action is not None

        if evidence.label_kind in (LabelKind.EXACT, LabelKind.PARTIAL):
            _assert_live_action_is_observable(live_action, evidence)
        else:
            reasons = _cant_reasons(snapshot, perspective.player)
            if reasons:
                assert set(reasons) <= _CANT_REASONS, (
                    f"Unknown cant reason was not in the researched taxonomy: {reasons}"
                )
            else:
                assert set(evidence.tags) & _REPLAY_AMBIGUITY_TAGS, (
                    f"Selected live action {live_action} became UNKNOWN without a known "
                    f"reconstruction reason: tags={evidence.tags!r}"
                )

        reconstructed = builder.build(snapshot.view, {}).cpu()
        replay_cursor = int(record["replay_cursor"])
        _assert_expected_observation_fields(
            _live_tensors(record),
            reconstructed,
            compare_event_fields=replay_cursor not in seen_replay_cursors,
        )
        seen_replay_cursors.add(replay_cursor)


_ORACLE_SKIPPED_TAGS = frozenset({"", "t:", "expire", "uhtmlchange", "showteam", "win", "tie"})


def _oracle_battle(document: ReplayDocument, perspective: int) -> DoubleBattle:
    """A real poke-env battle to diff the pure state machine against."""
    # poke-env derives the role from the |player| lines by username, so the oracle
    # must impersonate the perspective's player rather than set the role directly.
    return DoubleBattle(
        document.metadata.replay_id,
        document.metadata.player_names[perspective],
        logging.getLogger("p0.stress.oracle"),
        gen=9,
    )


def _assert_poke_env_state_agreement(
    perspective: ReconstructedPerspective,
    document: ReplayDocument,
) -> None:
    """Diff the pure replay state machine against poke-env over the same lines.

    Only fields both can know from public protocol are compared. This isolates a state
    tracking regression (Illusion, boosts, field state) from a boundary or tensor
    problem, and covers the Illusion handling that poke-env and p0 implement apart.
    """
    battle = _oracle_battle(document, perspective.player)
    boundaries = {snapshot.pre_line_index: snapshot for snapshot in perspective.snapshots}
    ally = f"p{perspective.player + 1}"

    for line in document.protocol_lines:
        snapshot = boundaries.get(line.index)
        if snapshot is not None and battle.active_pokemon != [None, None]:
            for slot, mon in enumerate(battle.active_pokemon):
                reconstructed = snapshot.view.active_pokemon[slot]
                if mon is None or reconstructed is None:
                    assert (mon is None) == (reconstructed is None), (
                        f"Active slot {slot} disagrees at line {line.index}"
                    )
                    continue
                assert mon.fainted == reconstructed.fainted
                assert mon.current_hp_fraction == pytest.approx(
                    reconstructed.current_hp_fraction, abs=0.02
                )
                assert dict(mon.boosts) == dict(reconstructed.boosts)
                live_status = None if mon.status is None else mon.status.name
                own_status = getattr(reconstructed.status, "name", reconstructed.status)
                assert live_status == (None if own_status is None else str(own_status).upper())

            assert {weather.name for weather in battle.weather} == {
                weather.name for weather in snapshot.view.weather
            }
            assert {field.name for field in battle.fields} == {
                field.name for field in snapshot.view.fields
            }
            assert {normalize_id(condition.name) for condition in battle.side_conditions} == {
                normalize_id(condition.name) for condition in snapshot.view.side_conditions
            }

        if line.parts[1] in _ORACLE_SKIPPED_TAGS:
            continue
        try:
            battle.parse_message(list(line.parts))
        except (AssertionError, IndexError, KeyError, NotImplementedError, ValueError) as error:
            raise AssertionError(
                f"poke-env rejected {ally} line {line.index}: {line.raw!r} ({error})"
            ) from error


def _hp_fields_agree(left: str, right: str) -> bool:
    """Whether two HP fields describe the same health within percent quantization.

    Each player receives its own mons' HP exactly and the opponent's rounded to a
    percentage, so the same battle line differs in precision between the two streams.
    """
    left_value, _, left_status = left.partition(" ")
    right_value, _, right_status = right.partition(" ")
    if left_status != right_status or "/" not in left_value or "/" not in right_value:
        return False
    return abs(get_hp_fraction(left) - get_hp_fraction(right)) <= 0.02


def _protocol_lines_agree(left: str, right: str) -> bool:
    """Whether two captures of one protocol line differ only in HP precision."""
    left_parts = left.split("|")
    right_parts = right.split("|")
    if len(left_parts) != len(right_parts):
        return False
    return all(
        expected == actual or _hp_fields_agree(expected, actual)
        for expected, actual in zip(left_parts, right_parts, strict=True)
    )


def _assert_line_stream_fidelity(
    document: ReplayDocument,
    live_lines: tuple[str, ...],
) -> None:
    """The parsed replay must be the line stream this player actually received.

    Isolating this from grouping turns a downstream tensor mismatch into a line-level
    diff, which is where a parsing or capture regression actually originates.
    """
    splice_index, splice_count = _showteam_offset(document)
    parsed = [line.raw for line in document.protocol_lines]
    spliced = parsed[splice_index : splice_index + splice_count]
    assert all(_message_tag(line) == "showteam" for line in spliced)

    without_splice = parsed[:splice_index] + parsed[splice_index + splice_count :]
    assert len(without_splice) == len(live_lines)
    for parsed_line, live_line in zip(without_splice, live_lines, strict=True):
        assert _protocol_lines_agree(parsed_line, live_line), (
            f"Parsed line {parsed_line!r} is not the captured line {live_line!r}"
        )


def _assert_decision_boundaries_partition_the_log(
    perspective: ReconstructedPerspective,
    document: ReplayDocument,
) -> None:
    """Decisions must be ordered, disjoint, and quote the log they point at."""
    cursor = 0
    for snapshot in perspective.snapshots:
        assert cursor <= snapshot.pre_line_index < snapshot.post_line_index
        assert snapshot.raw_lines == tuple(
            line.raw
            for line in document.protocol_lines[snapshot.pre_line_index : snapshot.post_line_index]
        )
        cursor = snapshot.post_line_index
    assert cursor <= len(document.protocol_lines)


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.asyncio
async def test_random_local_games_reconstruct_to_valid_tensors(
    showdown_server, tmp_path: Path
) -> None:
    """End-to-end stress test: live Showdown battles -> replay JSON -> offline observation reconstruction -> tensor shards.

    Verifies that:
    1. 100+ random live battles run concurrently against a local Showdown server.
    2. Spliced replay documents capture all turns with valid terminal line indices.
    3. Reconstructed observations match ground truth live captured tensors at exact line boundaries.
    4. Offline reconstructed state machines perfectly agree with poke-env active pokemon/field state.
    5. Reconstructed action candidates encapsulate the actual live chosen actions.
    6. Shard compilation produces valid training tensors with correct loss mask semantics:
       - UNKNOWN label kinds have loss_mask == 0 (excluded from policy gradient / imitation loss)
       - EXACT and PARTIAL label kinds have loss_mask > 0 (included in loss calculation).
    """
    seed = int(os.getenv("P0_STRESS_SEED", "20260802"))
    game_count = _stress_game_count()
    concurrency = _stress_concurrency()
    replay_dir = tmp_path / "local-replays"
    observation_dir = tmp_path / "live-observations"
    resources = default_runtime_resources()
    builder = ObservationBuilder(resources)
    team_source = _random_team_source(
        tmp_path / "random-teams",
        seed=seed + 1,
        count=max(32, game_count * 2),
    )
    random_state = random.getstate()
    random.seed(seed)
    poke_env_patches.install()

    player_a = JsonCapturingRandomPlayer(
        account_configuration=AccountConfiguration("StressRandomA", None),
        battle_format=FORMAT.battle_format,
        server_configuration=showdown_server,
        team_source=team_source,
        team_rng=random.Random(seed + 1),
        accept_open_team_sheet=True,
        max_concurrent_battles=concurrency,
        observation_builder=builder,
        observation_dir=observation_dir,
        replay_dir=replay_dir,
        write_replays=True,
    )
    player_b = JsonCapturingRandomPlayer(
        account_configuration=AccountConfiguration("StressRandomB", None),
        battle_format=FORMAT.battle_format,
        server_configuration=showdown_server,
        team_source=team_source,
        team_rng=random.Random(seed + 2),
        accept_open_team_sheet=True,
        max_concurrent_battles=concurrency,
        observation_builder=builder,
        observation_dir=observation_dir,
    )

    try:
        await asyncio.wait_for(
            player_a.battle_against(player_b, n_battles=game_count),
            timeout=_stress_timeout(game_count),
        )
    finally:
        await player_a.ps_client.stop_listening()
        await player_b.ps_client.stop_listening()
        poke_env_patches.uninstall_for_tests()
        random.setstate(random_state)

    replay_paths = tuple(sorted(player_a.replay_paths))
    assert len(replay_paths) == game_count
    assert all(path.is_file() and path.suffix == ".json" for path in replay_paths)

    documents = tuple(
        parse_replay_payload(
            path.read_bytes(),
            replay_id=path.stem,
            format_id=FORMAT.battle_format,
        )
        for path in replay_paths
    )
    assert len(documents) == game_count
    assert all(document.outcome.terminal_line_index is not None for document in documents)

    # For every completed game, verify offline reconstruction against the live recorded ground truth
    for document in documents:
        replay_id = document.metadata.replay_id
        for perspective in _assert_reconstruction_labels(document):
            role = f"p{perspective.player + 1}"
            path = observation_dir / f"{replay_id}-{role}.pt"
            assert path.is_file(), f"Missing live observation capture: {path}"
            artifact = _load_live_artifact(path)
            # Verify protocol line stream matches what player received over websocket
            _assert_line_stream_fidelity(document, artifact["lines"])
            # Verify reconstructed decision snapshots form an ordered partition of the log
            _assert_decision_boundaries_partition_the_log(perspective, document)
            # Verify agreement with poke-env oracle battle state
            _assert_poke_env_state_agreement(perspective, document)
            # Verify reconstructed structured observations match live pre-fusion observation tensors
            _assert_live_truth(perspective, artifact["records"], document, builder)

    # Compile replay documents into training dataset representation
    compilation = compile_documents(
        documents,
        format_id=FORMAT.battle_format,
        max_candidates=256,
        dex=resources.dex,
    )
    assert len(compilation.games) == game_count
    assert compilation.metrics.counters["accepted_games"] == game_count
    assert compilation.metrics.counters["rejected_games"] == 0

    # Write out serialized PyTorch tensor shards
    build = write_tensor_shards(
        compilation,
        tmp_path / "tensor-shards",
        resources=resources,
        max_candidates=256,
        max_decisions_per_shard=4096,
    )
    assert build.manifest.accepted_games == game_count
    assert build.manifest.rejected_games == 0

    # Verify tensor contracts and loss masking across all generated shard files
    for shard in build.manifest.shards:
        artifact = torch.load(
            build.manifest_path.parent / shard.filename,
            map_location="cpu",
            weights_only=True,
        )
        validate_shard_tensors(artifact["tensors"])

        tensors = artifact["tensors"]
        label_kind = tensors["label_kind"]
        loss_mask = tensors["loss_mask"]
        # UNKNOWN labels must never contribute to training loss (loss_mask == 0)
        assert torch.all(loss_mask[label_kind == int(LabelKind.UNKNOWN)] == 0)
        # EXACT and PARTIAL candidate labels must have active training loss weights
        assert torch.all(loss_mask[label_kind == int(LabelKind.EXACT)] > 0)
        assert torch.all(loss_mask[label_kind == int(LabelKind.PARTIAL)] > 0)
