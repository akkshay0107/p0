"""
Compare live observations with replay reconstruction and disk shards.

Capture live requests, chosen orders, and observations before tensor fusion.
Match reconstructed decisions to those requests, check full joint-action labels
and request flags, and compare public observation fields at shared boundaries.
Event values are compared when both ends of the captured intervals match.

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
import sys
from bisect import bisect_left
from collections import Counter
from concurrent.futures import Future
from functools import partial
from pathlib import Path
from typing import Any, cast
from weakref import WeakKeyDictionary

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
    CAT_IDX_STAT_PROVENANCE,
    CATEGORICAL_WIDTH,
    NUM_IDX_CAN_MEGA,
    NUM_IDX_CAN_SWITCH_OUT,
    NUM_IDX_HP_FRACTION,
    NUM_IDX_LEGALITY_UNKNOWN,
    NUM_IDX_LEVEL_STATS,
    NUM_IDX_MOVE_LEGAL,
    NUM_IDX_REVEALED,
    NUM_IDX_SLOT_CONDITION_UNKNOWN,
    NUM_IDX_SLOT_LEGALITY_UNKNOWN,
    NUM_IDX_STATUS_COUNTER,
    TEAM_SIZE,
    TOKEN_IDX_ALLY_SIDE,
    StructuredObservation,
)
from p0.replays.compile import compile_documents, write_tensor_shards
from p0.replays.identity import normalize_showdown_id
from p0.replays.protocol import ReplayDocument, parse_replay_payload
from p0.replays.reconstruction.projection import ProjectedPerspective, ProjectedSnapshot
from p0.replays.schema import DecisionType, LabelKind
from p0.replays.shards import validate_shard_tensors
from p0.rl_player import TeamPlayerMixin
from p0.runtime import poke_env_patches
from p0.runtime.live_event_capture import consume_events
from p0.runtime.poke_env_battle_adapter import battle_view
from p0.teams.corpus import TeamCorpus, corpus_from_team_files
from p0.teams.stat_points import StatPoints
from p0.teams.team import TeamMember
from tests.integration.helpers import order_to_action
from tests.stress._helpers import stress_count, stress_random_team_record
from tests.team_fixtures import default_test_corpus

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
        "unsupported",
        # the submitted order left no trace: the mon was KO'd before it acted, or the
        # request was answered with a pass that the protocol never renders
        "no_observed_order",
    }
)

_PUBLIC_CATEGORICAL_INDICES = (0, 3, 4)
# Public HP is rounded to percent; the owner's live request has exact HP totals.
_PUBLIC_HP_TOLERANCE = 0.011
_PUBLIC_OPPONENT_CATEGORICAL_INDICES = tuple(
    index for index in range(CATEGORICAL_WIDTH) if index != CAT_IDX_STAT_PROVENANCE
)
_PUBLIC_OPPONENT_NUMERICAL_INDICES = (
    *range(NUM_IDX_HP_FRACTION, NUM_IDX_CAN_MEGA),
    *range(NUM_IDX_CAN_MEGA + 1, NUM_IDX_LEVEL_STATS),
    NUM_IDX_REVEALED,
)


def _message_tag(line: str) -> str:
    """Extract Showdown protocol message identifier (e.g. '|move|' -> 'move')."""
    parts = line.split("|")
    return parts[1] if len(parts) > 1 else ""


# Protocol lines each live battle parsed, in arrival order.
_PROTOCOL_LINES: WeakKeyDictionary[DoubleBattle, list[str]] = WeakKeyDictionary()
# Best-of room notices that are not battle events.
_ROOM_NOTICES = frozenset({"tempnotify", "tempnotifyoff"})
# The "|"-split field holding the HP of each line that shows HP.
_HP_FIELD_INDICES = {"switch": 4, "drag": 4, "-damage": 3, "-heal": 3, "-sethp": 3}
_HP_BAR_COLOURS = "gyr"


def _install_patches_and_record_lines() -> None:
    """Install the p0 patches and record every line a battle parses; conftest undoes both."""
    poke_env_patches.install()
    parse_message = DoubleBattle.parse_message

    def record_and_parse(battle: DoubleBattle, split_message: list[str]):
        tag = split_message[1] if len(split_message) > 1 else ""
        if tag not in _ROOM_NOTICES and tag != "showteam":
            lines = _PROTOCOL_LINES.get(battle)
            if lines is not None:
                lines.append("|".join(split_message))
            elif tag in {"start", "teampreview"}:
                _PROTOCOL_LINES[battle] = ["|".join(split_message)]
        return parse_message(battle, split_message)

    setattr(DoubleBattle, "parse_message", record_and_parse)


def _battle_log_lines(lines: tuple[str, ...]) -> tuple[str, ...]:
    """Keep the battle event stream and its optional team-preview boundary."""
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if _message_tag(line) in {"start", "teampreview"}
        ),
        0,
    )
    return lines[start:]


def _cant_reasons(snapshot: ProjectedSnapshot, role: int) -> tuple[str, ...]:
    """Collect all '|cant|' failure reasons recorded in a turn snapshot for a specific player role."""
    reasons: list[str] = []
    for raw_line in snapshot.raw_lines:
        parts = raw_line.split("|")
        if len(parts) < 4 or parts[1] != "cant":
            continue
        if parts[2][:2] != f"p{role + 1}":
            continue
        reasons.append(parts[3].casefold().strip())
    return tuple(dict.fromkeys(reasons))


def _showteam_offset(document: ReplayDocument) -> tuple[int, int]:
    """
    Where the harness spliced open-team-sheet lines into the captured log.

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
    """
    Random player that saves pre-fusion decisions, the open team sheets it receives
    and its finished battles.

    Captures exact raw observation tensors, action decisions, and replay line stream offsets
    during live battle execution so offline replay reconstruction can be verified 1-to-1 against ground truth.
    """

    def __init__(
        self,
        *,
        observation_builder: ObservationBuilder,
        observation_dir: Path,
        team_source: TeamCorpus,
        team_rng: random.Random,
        handler_failure: Future[None],
        **kwargs: Any,
    ) -> None:
        self.observation_builder = observation_builder
        self.observation_dir = observation_dir
        self.showteam_lines: dict[str, list[str]] = {}
        # The team mixin forgets a battle when it ends; the public replay needs both captures.
        self.finished_battles: dict[str, DoubleBattle] = {}
        self._live_records: dict[str, list[dict[str, Any]]] = {}
        self.handler_failure = handler_failure
        super().__init__(team_source=team_source, team_rng=team_rng, **kwargs)

    async def _handle_battle_message(self, split_messages: list[list[str]]) -> None:
        """Record poke-env's UI-only open-sheet messages before its normal callback."""
        battle_tag = split_messages[0][0].removeprefix(">")
        try:
            for split_message in split_messages[1:]:
                if len(split_message) > 1 and split_message[1] == "showteam":
                    self.showteam_lines.setdefault(battle_tag, []).append("|".join(split_message))
            await super()._handle_battle_message(split_messages)
        except Exception as exc:
            if not self.handler_failure.done():
                failure = RuntimeError(
                    f"Player {self.username} failed in {battle_tag} while handling "
                    f"{split_messages[-1]!r}: {exc}"
                )
                failure.__cause__ = exc
                self.handler_failure.set_exception(failure)
            raise

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
        view = battle_view(battle)
        observation = self.observation_builder.build(view).cpu()
        forced_switch_slots = tuple(
            index for index, forced in enumerate(battle.force_switch) if forced
        )
        records = self._live_records.setdefault(battle.battle_tag, [])
        replay_cursor = len(_PROTOCOL_LINES.get(battle, ()))
        # Like RLPlayer, submitting this order closes the event interval it observed.
        consume_events(battle)
        records.append(
            {
                "teampreview": bool(battle.teampreview),
                "wait": bool(view.wait),
                "forced_switch_slots": list(forced_switch_slots),
                "action": list(action),
                # The count of replay-visible lines consumed before this request is the
                # shared boundary: reconstruction reaches the same line index when it
                # segments the log, so the two sides are matched exactly rather than
                # by a turn heuristic.
                "replay_cursor": replay_cursor,
                # Where this decision's event interval began: the previous submission.
                "interval_start_cursor": records[-1]["replay_cursor"] if records else 0,
                # Save the individual pre-fusion fields, not a fused model input.
                "tensors": [tensor.clone() for tensor in observation.tensors()],
            }
        )

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        self._write_observation_tensors(battle)
        self.finished_battles[battle.battle_tag] = cast(DoubleBattle, battle)
        super()._battle_finished_callback(battle)

    def _write_observation_tensors(self, battle: AbstractBattle) -> None:
        double_battle = cast(DoubleBattle, battle)
        replay_id = battle.battle_tag.removeprefix("battle-")
        role = str(battle.player_role)
        payload = {
            "records": self._live_records.pop(battle.battle_tag, []),
            # This side's own copy of the protocol stream, which the line-stream check
            # compares with the public replay.
            "lines": _PROTOCOL_LINES.get(double_battle, []),
        }
        self.observation_dir.mkdir(parents=True, exist_ok=True)
        path = self.observation_dir / f"{replay_id}-{role}.pt"
        torch.save(payload, path)


def _stress_game_count() -> int:
    return stress_count("P0_STRESS_GAME_COUNT", 50)


def _stress_concurrency() -> int:
    return stress_count("P0_STRESS_CONCURRENCY", 4)


def _stress_timeout(game_count: int) -> float:
    value = float(os.getenv("P0_STRESS_TIMEOUT_SECONDS", str(max(120, game_count * 15))))
    if value <= 0:
        raise ValueError("P0_STRESS_TIMEOUT_SECONDS must be positive")
    return value


def _showdown_team_text(team: tuple[TeamMember, ...], spreads: tuple[StatPoints, ...]) -> str:
    """Serialize a generated team into the Showdown text format read by corpus_from_team_files."""
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


def _random_team_source(directory: Path, *, seed: int, count: int) -> TeamCorpus:
    """Create a corpus populated only with generated legal teams."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for index in range(count):
        record = stress_random_team_record(rng, label=f"live-random-{index}")
        (directory / f"random-{index:04d}.txt").write_text(
            _showdown_team_text(record.team.members, record.spreads), encoding="utf-8"
        )
    return corpus_from_team_files(sorted(directory.iterdir()), FORMAT.battle_format)


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
    return StructuredObservation(*record["tensors"])


def _live_action(record: dict[str, Any]) -> tuple[int, int]:
    value = record["action"]
    return int(value[0]), int(value[1])


def _assert_expected_observation_fields(
    live: StructuredObservation,
    reconstructed: StructuredObservation,
    *,
    compare_event_values: bool = False,
) -> None:
    """Compare every field a public replay can reproduce at a shared boundary."""
    fields = (
        "token_type_ids",
        "side_ids",
        "slot_ids",
    )
    for name in fields:
        torch.testing.assert_close(getattr(live, name), getattr(reconstructed, name))
    for index in _PUBLIC_CATEGORICAL_INDICES:
        torch.testing.assert_close(
            live.categorical[..., index], reconstructed.categorical[..., index]
        )
    if compare_event_values:
        # Both sides read the same protocol lines between the same two decisions, so the
        # categorical record columns (kind, source, target, move, detail) must agree.
        # Amounts are excluded: live pre-event HP comes from poke-env, replay HP from
        # public damage text.
        torch.testing.assert_close(live.spatial_cat, reconstructed.spatial_cat)

    # Replays store HP from public damage messages, while the live callback can
    # run before the same message batch has fully settled. Spatial and HP values
    # are therefore checked for valid bounded tensors rather than exact equality.
    assert live.spatial_num.shape == reconstructed.spatial_num.shape
    assert torch.isfinite(live.spatial_num).all()
    assert torch.isfinite(reconstructed.spatial_num).all()
    live_hp = live.numerical[:, NUM_IDX_HP_FRACTION]
    replay_hp = reconstructed.numerical[:, NUM_IDX_HP_FRACTION]
    assert torch.isfinite(live_hp).all()
    assert torch.isfinite(replay_hp).all()
    assert torch.all((live_hp >= 0) & (live_hp <= 1))
    assert torch.all((replay_hp >= 0) & (replay_hp <= 1))

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
    known_rows = (live.numerical[:, NUM_IDX_SLOT_CONDITION_UNKNOWN] == 0) & (
        reconstructed.numerical[:, NUM_IDX_SLOT_CONDITION_UNKNOWN] == 0
    )
    known_rows[TOKEN_IDX_ALLY_SIDE] = (
        reconstructed.numerical[TOKEN_IDX_ALLY_SIDE, NUM_IDX_LEGALITY_UNKNOWN] == 0
    )
    # The request-derived and hidden-stat columns are intentionally excluded:
    # a public replay cannot reproduce them at the exact point where poke-env
    # invokes the live order callback. The remaining columns are stable public
    # state (species base stats). HP is compared above with replay-aware
    # quantization. Slot conditions, boosts, roster state, and active-state
    # flags are public eventually, but can still differ at the instant the live
    # callback is invoked within a websocket message batch.
    columns = tuple(range(6, 12))
    torch.testing.assert_close(
        live.numerical[known_rows][:, columns], reconstructed.numerical[known_rows][:, columns]
    )
    _assert_legality_provenance(live, reconstructed)


def _assert_legality_provenance(
    live: StructuredObservation,
    reconstructed: StructuredObservation,
) -> None:
    """
    A reconstructed legality cell must match live exactly, or be marked unproven.

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


def _match_live_records(
    perspective: ProjectedPerspective,
    live_records: tuple[dict[str, Any], ...],
    document: ReplayDocument,
) -> tuple[tuple[ProjectedSnapshot, dict[str, Any], bool], ...]:
    """
    Pair requests with decisions at the boundary both sides can name.

    Orders that never produce a replay-visible event before the battle ends are dropped.
    Retries retain only the final submitted order.
    Ordered requests use exact boundaries when available; a later boundary handles callbacks
    before trailing animation lines have been parsed.
    """
    splice_index, splice_count = _showteam_offset(document)
    boundaries = tuple(snapshot.pre_line_index for snapshot in perspective.snapshots)
    final_records = {
        _live_boundary(record, splice_index, splice_count): record for record in live_records
    }

    matched: list[tuple[ProjectedSnapshot, dict[str, Any], bool]] = []
    previous_index = -1
    for boundary, record in final_records.items():
        index = bisect_left(boundaries, boundary)
        if index < len(boundaries) and boundaries[index] != boundary:
            index = max(index, previous_index + 1)
        if index == len(boundaries):
            continue
        snapshot = perspective.snapshots[index]
        matched.append(
            (
                snapshot,
                record,
                snapshot.pre_line_index == boundary,
            )
        )
        previous_index = index
    return tuple(matched)


def _assert_live_truth(
    perspective: ProjectedPerspective,
    live_records: tuple[dict[str, Any], ...],
    document: ReplayDocument,
    builder: ObservationBuilder,
) -> tuple[int, Counter[DecisionType]]:
    """
    Check reconstructed requests and count full action comparisons by decision type.

    Arguments:
        perspective: Reconstructed decisions and observations for one player.
        live_records: Orders and observations captured from that player's callbacks.
        document: Public replay corresponding to the live captures.
        builder: Observation builder shared by live and reconstructed views.

    Returns:
        Matching event interval count and full action comparison counts by decision type.
    """
    # A rejected order is retried before any new battle line arrives, so every
    # retry of one request observes the events its first attempt observed.
    first_events: dict[int, torch.Tensor] = {}
    for record in live_records:
        events = _live_tensors(record).spatial_cat
        cursor = int(record["replay_cursor"])
        if cursor in first_events:
            torch.testing.assert_close(events, first_events[cursor])
        else:
            first_events[cursor] = events

    compared_intervals = 0
    compared_actions: Counter[DecisionType] = Counter()
    splice_index, splice_count = _showteam_offset(document)
    starts = tuple(snapshot.pre_line_index for snapshot in perspective.snapshots)
    replay_interval_starts = dict(zip(starts, (0, *starts[:-1]), strict=True))
    opponent_has_illusion = any(
        normalize_showdown_id(member.ability) == "illusion"
        for member in document.ots[1 - perspective.player].members
    )
    matched = _match_live_records(perspective, live_records, document)
    assert matched, "No live requests matched this accepted perspective"
    matched_indices = {snapshot.decision_index for snapshot, _, _ in matched}
    assert len(matched_indices) == len(matched), "Different live requests matched the same decision"
    assert matched_indices == set(range(len(perspective.decisions))), (
        f"Reconstructed decisions without a live request: "
        f"{set(range(len(perspective.decisions))) - matched_indices}"
    )
    for snapshot, record, exact_boundary in matched:
        decision = perspective.decisions[snapshot.decision_index]
        evidence = decision.evidence
        live_action = _live_action(record)
        request = snapshot.view.decision
        assert request.team_preview == record["teampreview"]
        assert request.wait == record["wait"]
        assert [i for i, slot in enumerate(request.slots) if slot.force_switch] == record[
            "forced_switch_slots"
        ]

        if evidence.label_kind in (LabelKind.EXACT, LabelKind.PARTIAL):
            assert live_action in evidence.candidates, (
                f"Live action {live_action} is absent from joint candidates {evidence.candidates}"
            )
            compared_actions[decision.decision_type] += int(
                any(action != PASS_ACTION for action in live_action)
            )
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

        if not exact_boundary:
            continue

        reconstructed = builder.build(snapshot.view, {}).cpu()
        replay_cursor = int(record["replay_cursor"])
        # Live play consumes events at every submitted order and replay at every
        # reconstructed decision; the intervals agree when both of their ends do.
        live_interval_start = _live_boundary(
            {"replay_cursor": record["interval_start_cursor"]}, splice_index, splice_count
        )
        same_interval = live_interval_start == replay_interval_starts[snapshot.pre_line_index]
        try:
            live = _live_tensors(record)
            _assert_expected_observation_fields(
                live,
                reconstructed,
                compare_event_values=same_interval,
            )
            if opponent_has_illusion:
                # These are public opponent fields, compared with the actual live
                # capture at the same cursor. Private stat estimates differ by source.
                torch.testing.assert_close(
                    live.categorical[
                        TEAM_SIZE : 2 * TEAM_SIZE, _PUBLIC_OPPONENT_CATEGORICAL_INDICES
                    ],
                    reconstructed.categorical[
                        TEAM_SIZE : 2 * TEAM_SIZE, _PUBLIC_OPPONENT_CATEGORICAL_INDICES
                    ],
                )
                torch.testing.assert_close(
                    live.numerical[TEAM_SIZE : 2 * TEAM_SIZE, _PUBLIC_OPPONENT_NUMERICAL_INDICES],
                    reconstructed.numerical[
                        TEAM_SIZE : 2 * TEAM_SIZE, _PUBLIC_OPPONENT_NUMERICAL_INDICES
                    ],
                    atol=_PUBLIC_HP_TOLERANCE,
                    rtol=0,
                )
        except AssertionError as exc:
            raise AssertionError(
                f"Replay {document.metadata.replay_id}, observer p{perspective.player + 1}, "
                f"decision {snapshot.decision_index}, line {snapshot.pre_line_index}, "
                f"live cursor {replay_cursor}: {exc}"
            ) from exc
        compared_intervals += same_interval
    return compared_intervals, compared_actions


_ORACLE_SKIPPED_TAGS = frozenset({"", "t:", "expire", "uhtmlchange", "showteam", "win", "tie"})


def _oracle_battle(document: ReplayDocument, perspective: int) -> DoubleBattle:
    """A real poke-env battle to diff the pure state machine against."""
    # poke-env derives the role from the |player| lines by username, so the oracle
    # must impersonate the perspective's player rather than set the role directly.
    battle = DoubleBattle(
        document.metadata.replay_id,
        document.metadata.player_names[perspective],
        logging.getLogger("p0.stress.oracle"),
        gen=9,
    )
    for player_index, username in enumerate(document.metadata.player_names):
        battle.parse_message(["", "player", f"p{player_index + 1}", username, "", ""])

    for ots in document.ots:
        role = f"p{ots.side.side_index + 1}"
        for member in ots.members:
            identifier = f"{role}: {member.species}"
            battle.get_pokemon(
                identifier,
                request={
                    "active": False,
                    "baseAbility": member.ability,
                    "condition": "100/100",
                    "details": f"{member.species}, L{member.level}",
                    "ident": identifier,
                    "item": member.item,
                    "moves": list(member.moves),
                },
            )
    return battle


def _assert_poke_env_state_agreement(
    perspective: ProjectedPerspective,
    document: ReplayDocument,
) -> None:
    """
    Diff the pure replay state machine against poke-env over the same lines.

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
                # poke-env can lose a slot when two Illusion displays share a
                # species; the replay reducer still retains the active member.
                if reconstructed is not None and (
                    any(effect.name == "ILLUSION" for effect in reconstructed.effects)
                ):
                    continue
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
            assert {normalize_showdown_id(field.name) for field in battle.fields} == {
                normalize_showdown_id(field.name) for field in snapshot.view.fields
            }
            assert {
                normalize_showdown_id(condition.name) for condition in battle.side_conditions
            } == {
                normalize_showdown_id(condition.name) for condition in snapshot.view.side_conditions
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
    """
    Whether two HP fields describe the same health within percent quantization.

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


def _public_replay_events(
    battles: tuple[DoubleBattle, DoubleBattle], replay_dir: Path
) -> list[str]:
    """
    Merge both players' captures of one battle into the log a spectator receives.

    Each player sees its own Pokemon's HP exactly and the opponent's as a percentage,
    so a line naming one side is taken from the other side's capture.
    """
    captures = {}
    for battle in battles:
        path = battle.save_replay(replay_dir / f"{battle.player_role}.html")
        html = path.read_text(encoding="utf-8")
        log = html.split('<script type="text/plain" class="battle-log-data">', 1)[1]
        lines = tuple(log.split("</script>", 1)[0].strip().splitlines())
        captures[battle.player_role] = _battle_log_lines(lines)
    p1_lines, p2_lines = captures["p1"], captures["p2"]
    if len(p1_lines) != len(p2_lines):
        raise AssertionError(
            f"The players captured {len(p1_lines)} and {len(p2_lines)} lines of one battle"
        )

    public = []
    for p1_line, p2_line in zip(p1_lines, p2_lines, strict=True):
        if p1_line == p2_line:
            public.append(p1_line)
            continue
        if not _protocol_lines_agree(p1_line, p2_line):
            raise AssertionError(f"The players captured {p1_line!r} and {p2_line!r}")
        public.append(p2_line if p1_line.split("|")[2].startswith("p1") else p1_line)
    return public


def _write_public_replay(
    battles: tuple[DoubleBattle, DoubleBattle], showteam_lines: list[str], replay_dir: Path
) -> Path:
    """Write one battle as replay JSON in the public form of Showdown's replay site."""
    battle = battles[0]
    replay_id = battle.battle_tag.removeprefix("battle-")
    replay_events = _public_replay_events(battles, replay_dir / replay_id)
    insertion_point = next(
        (index for index, line in enumerate(replay_events) if _message_tag(line) == "switch"),
        len(replay_events),
    )
    replay_events[insertion_point:insertion_point] = showteam_lines

    winner = next(
        (line.split("|", 2)[2] for line in replay_events if _message_tag(line) == "win"),
        "",
    )
    payload = {
        "id": replay_id,
        "formatid": FORMAT.battle_format,
        "p1": battle.players[0],
        "p2": battle.players[1],
        "winner": winner,
        "roomid": battle.battle_tag,
        "parent": replay_id,
        "game_number": 1,
        "uploadtime": 0,
        "log": "\n".join(replay_events),
    }
    replay_dir.mkdir(parents=True, exist_ok=True)
    path = replay_dir / f"{replay_id}.json"
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS) + b"\n")
    return path


def _shows_exact_hp(line: str) -> bool:
    """Whether a line gives HP out of a maximum other than 100, which only its owner sees."""
    parts = line.split("|")
    index = _HP_FIELD_INDICES.get(_message_tag(line))
    if index is None or len(parts) <= index:
        return False
    hp = parts[index].split(" ", 1)[0]
    # Showdown may add the colour of the HP bar to a percentage (50/100y).
    return "/" in hp and hp.split("/", 1)[1].rstrip(_HP_BAR_COLOURS) != "100"


def _assert_line_stream_fidelity(
    document: ReplayDocument,
    live_lines: tuple[str, ...],
) -> None:
    """
    The parsed replay must be the line stream this player actually received.

    Isolating this from grouping turns a downstream tensor mismatch into a line-level
    diff, which is where a parsing or capture regression actually originates.
    """
    splice_index, splice_count = _showteam_offset(document)
    parsed = [line.raw for line in document.protocol_lines]
    spliced = parsed[splice_index : splice_index + splice_count]
    assert all(_message_tag(line) == "showteam" for line in spliced)

    without_splice = parsed[:splice_index] + parsed[splice_index + splice_count :]
    # poke-env synthesizes the terminal win/tie event when it knows the outcome,
    # but that event is not part of the websocket stream captured by the player.
    without_splice = [line for line in without_splice if _message_tag(line) not in {"win", "tie"}]
    assert len(without_splice) == len(live_lines)
    for parsed_line, live_line in zip(without_splice, live_lines, strict=True):
        assert _protocol_lines_agree(parsed_line, live_line), (
            f"Parsed line {parsed_line!r} is not the captured line {live_line!r}"
        )


def _assert_decision_boundaries_partition_the_log(
    perspective: ProjectedPerspective,
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


class TestReplayCaptureHarness:
    @pytest.mark.integration
    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_artifact_write_failure_reaches_waiter_through_live_callback(
        self, showdown_server, tmp_path: Path
    ) -> None:
        # A real filesystem error exercises failure delivery from poke-env's completion callback.
        team_source = default_test_corpus()
        blocked_output = tmp_path / "observations"
        blocked_output.write_text("An existing file cannot be an artifact directory")
        failure: Future[None] = Future()
        _install_patches_and_record_lines()
        player = JsonCapturingRandomPlayer(
            account_configuration=AccountConfiguration("StressFailureA", None),
            battle_format=FORMAT.battle_format,
            server_configuration=showdown_server,
            team_source=team_source,
            team_rng=random.Random(3),
            accept_open_team_sheet=True,
            max_concurrent_battles=1,
            observation_builder=ObservationBuilder(default_runtime_resources()),
            observation_dir=tmp_path / "observations",
            handler_failure=failure,
        )
        opponent = RandomPlayer(
            account_configuration=AccountConfiguration("StressFailureB", None),
            battle_format=FORMAT.battle_format,
            server_configuration=showdown_server,
            team=team_source.sample(random.Random(4)),
            accept_open_team_sheet=True,
            max_concurrent_battles=1,
        )
        battle_task = asyncio.create_task(player.battle_against(opponent, n_battles=1))
        try:
            with pytest.raises(RuntimeError, match="StressFailureA failed in battle-") as error:
                await asyncio.wait_for(asyncio.wrap_future(failure), timeout=60)
            assert isinstance(error.value.__cause__, FileExistsError)
        finally:
            battle_task.cancel()
            await asyncio.gather(battle_task, return_exceptions=True)
            await asyncio.gather(
                player.ps_client.stop_listening(), opponent.ps_client.stop_listening()
            )


class TestRandomReplayReconstruction:
    @pytest.mark.integration
    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_random_local_games_reconstruct_to_valid_tensors(
        self, showdown_server, tmp_path: Path, request: pytest.FixtureRequest
    ) -> None:
        """
        Compare real battles with reconstructed decisions, observations, and disk shards.

        Arguments:
            showdown_server: Local Showdown server configuration.
            tmp_path: Directory for teams, live captures, replay JSON, and shards.
            request: Pytest request used to restore the random state after the workload.

        Returns:
            None.
        """
        seed = int(os.getenv("P0_STRESS_SEED", "20260802"))
        game_count = _stress_game_count()
        concurrency = _stress_concurrency()
        replay_dir = tmp_path / "local-replays"
        observation_dir = tmp_path / "live-observations"
        resources = default_runtime_resources()
        builder = ObservationBuilder(resources)
        max_candidates = stress_count("P0_STRESS_MAX_CANDIDATES", 256)
        team_source = _random_team_source(
            tmp_path / "random-teams",
            seed=seed + 1,
            count=max(32, game_count * 2),
        )
        random_state = random.getstate()
        random.seed(seed)
        _install_patches_and_record_lines()
        request.addfinalizer(partial(random.setstate, random_state))
        handler_failure: Future[None] = Future()
        players: list[JsonCapturingRandomPlayer] = []
        try:
            common_kwargs = {
                "battle_format": FORMAT.battle_format,
                "server_configuration": showdown_server,
                "team_source": team_source,
                "accept_open_team_sheet": True,
                "max_concurrent_battles": concurrency,
                "observation_builder": builder,
                "observation_dir": observation_dir,
                "handler_failure": handler_failure,
            }
            player_a = JsonCapturingRandomPlayer(
                account_configuration=AccountConfiguration("StressRandomA", None),
                team_rng=random.Random(seed + 1),
                **common_kwargs,
            )
            player_b = JsonCapturingRandomPlayer(
                account_configuration=AccountConfiguration("StressRandomB", None),
                team_rng=random.Random(seed + 2),
                **common_kwargs,
            )
            players.extend([player_a, player_b])

            battle_task = asyncio.create_task(
                player_a.battle_against(player_b, n_battles=game_count)
            )
            failure_task = asyncio.wrap_future(handler_failure)
            try:
                done, _ = await asyncio.wait(
                    {battle_task, failure_task},
                    timeout=_stress_timeout(game_count),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if failure_task in done:
                    await failure_task
                if battle_task not in done:
                    raise TimeoutError(f"Timed out before {game_count} stress games finished")
                await battle_task
            finally:
                if not battle_task.done():
                    battle_task.cancel()
                failure_task.cancel()
                await asyncio.gather(battle_task, return_exceptions=True)
        finally:
            closed = await asyncio.gather(
                *(player.ps_client.stop_listening() for player in players),
                return_exceptions=True,
            )
            failures = [result for result in closed if isinstance(result, BaseException)]
            if failures:
                if sys.exc_info()[0] is None:
                    raise RuntimeError(f"Failed to stop stress players: {failures}")
                logging.getLogger(__name__).error("Failed to stop stress players: %s", failures)

        # Each player sees only its own HP exactly, so the replay merges both captures.
        replay_paths = tuple(
            sorted(
                _write_public_replay(
                    (battle, player_b.finished_battles[battle_tag]),
                    player_a.showteam_lines.pop(battle_tag, []),
                    replay_dir,
                )
                for battle_tag, battle in player_a.finished_battles.items()
            )
        )
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
        # A public replay shows every HP as a percentage.
        assert not any(
            _shows_exact_hp(line.raw) for document in documents for line in document.protocol_lines
        )

        compilation = compile_documents(
            documents,
            format_id=FORMAT.battle_format,
            max_candidates=max_candidates,
        )
        accepted_games = {
            game.replay_id: game for series in compilation.accepted_series for game in series.games
        }
        rejected_ids = {
            document.metadata.replay_id for document in documents
        } - accepted_games.keys()
        compared_intervals = 0
        compared_actions: Counter[DecisionType] = Counter()
        compared_perspectives = 0
        for document in documents:
            replay_id = document.metadata.replay_id
            game = accepted_games.get(replay_id)
            if game is None:
                continue
            for perspective in game.perspectives:
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
                intervals, actions = _assert_live_truth(
                    perspective, artifact["records"], document, builder
                )
                compared_intervals += intervals
                compared_actions.update(actions)
                compared_perspectives += 1
        assert compared_intervals > 0
        assert all(
            compared_actions[decision_type] > 0
            for decision_type in (
                DecisionType.TEAM_PREVIEW,
                DecisionType.TURN,
                DecisionType.FORCED_SWITCH,
                DecisionType.PIVOT_SWITCH,
            )
        ), (
            f"Workload must compare every request type; increase P0_STRESS_GAME_COUNT: {compared_actions}"
        )

        accepted_count = len(accepted_games)
        assert accepted_count > 0
        assert compared_perspectives == 2 * accepted_count
        assert accepted_count + len(rejected_ids) == game_count
        assert compilation.metrics.counters["accepted_games"] == accepted_count
        assert compilation.metrics.counters["rejected_games"] == len(rejected_ids)

        # Write out serialized PyTorch tensor shards
        build = write_tensor_shards(
            compilation,
            tmp_path / "tensor-shards",
            resources=resources,
            max_candidates=max_candidates,
        )
        assert build.manifest.source_games == game_count
        assert build.manifest.accepted_games == accepted_count
        assert build.manifest.rejected_games == len(rejected_ids)

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
