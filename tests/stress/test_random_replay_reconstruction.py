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
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import orjson
import pytest
import torch
from poke_env import AccountConfiguration
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import RandomPlayer
from poke_env.player.battle_order import BattleOrder, SingleBattleOrder

from p0.evaluation.harness import DEFAULT_TEST_TEAM
from p0.format_config import FORMAT
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.replays.compile import compile_documents, write_tensor_shards
from p0.replays.protocol import ReplayDocument, parse_replay_payload
from p0.replays.reconstruct import ReconstructedPerspective, ReconstructedSnapshot, reconstruct_both
from p0.replays.schema import LabelKind
from p0.replays.shards import validate_shard_tensors
from p0.runtime import poke_env_patches
from p0.runtime.poke_env_action_adapter import order_to_action
from p0.runtime.poke_env_battle_adapter import battle_view
from tests.stress._helpers import stress_count

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
    }
)


def _message_tag(line: str) -> str:
    parts = line.split("|")
    return parts[1] if len(parts) > 1 else ""


def _endpoint_role(endpoint: str) -> str:
    return endpoint.split("a", 1)[0].strip()


def _line_has_observed_action(line: str, role: int) -> bool:
    """Whether a replay line contains a submitted action for this perspective."""
    parts = line.split("|")
    if len(parts) < 3 or parts[1] not in {"move", "switch", "cant"}:
        return False
    endpoint = parts[2]
    if _endpoint_role(endpoint) != f"p{role + 1}":
        return False
    if parts[1] == "cant":
        # A cant is the server's result of a submitted order.  There is no
        # preceding move line for BeforeMove failures in Showdown.
        return True
    suffix_start = 5 if parts[1] == "move" else 4
    return not any(part.startswith("[from]") for part in parts[suffix_start:])


def _cant_reasons(snapshot: ReconstructedSnapshot, role: int) -> tuple[str, ...]:
    reasons: list[str] = []
    for raw_line in snapshot.raw_lines:
        parts = raw_line.split("|")
        if len(parts) < 4 or parts[1] != "cant":
            continue
        if _endpoint_role(parts[2]) != f"p{role + 1}":
            continue
        reasons.append(parts[3].casefold().strip())
    return tuple(dict.fromkeys(reasons))


def _record_key(record: dict[str, Any]) -> tuple[bool, int]:
    return bool(record["teampreview"]), int(record["turn"])


def _snapshot_key(snapshot: ReconstructedSnapshot) -> tuple[bool, int]:
    return snapshot.view.teampreview, snapshot.turn


class JsonCapturingRandomPlayer(RandomPlayer):
    """Random player that saves pre-fusion decisions and completed replay JSON."""

    def __init__(
        self,
        *,
        observation_builder: ObservationBuilder,
        observation_dir: Path,
        replay_dir: Path | None = None,
        write_replays: bool = False,
        **kwargs: Any,
    ) -> None:
        self.observation_builder = observation_builder
        self.observation_dir = observation_dir
        self.replay_dir = replay_dir
        self.write_replays = write_replays
        self.replay_paths: list[Path] = []
        self.observation_paths: list[Path] = []
        self._showteam_lines: dict[str, list[str]] = {}
        self._live_records: dict[str, list[dict[str, Any]]] = {}
        self._wait_request_keys: set[tuple[str, int, int]] = set()
        super().__init__(**kwargs)

    async def _handle_battle_message(self, split_messages: list[list[str]]) -> None:
        battle_tag = split_messages[0][0].removeprefix(">")
        for split_message in split_messages[1:]:
            if len(split_message) > 1 and split_message[1] == "showteam":
                self._showteam_lines.setdefault(battle_tag, []).append("|".join(split_message))

        await super()._handle_battle_message(split_messages)

    async def _handle_battle_request(
        self, battle: AbstractBattle, maybe_default_order: bool = False
    ) -> None:
        # poke-env uses _wait when this side received the request but does not
        # get to submit an order.  Keep that observation as the ground truth
        # for a replay segment that reconstructs to UNKNOWN with no action.
        if battle._wait and battle.last_request is not None:
            key = (battle.battle_tag, battle.turn, id(battle.last_request))
            if key not in self._wait_request_keys:
                self._wait_request_keys.add(key)
                self._record_decision(cast(DoubleBattle, battle), None)

        await super()._handle_battle_request(battle, maybe_default_order)

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

    def _record_decision(self, battle: DoubleBattle, action: tuple[int, int] | None) -> None:
        observation = self.observation_builder.build(battle_view(battle)).cpu()
        self._live_records.setdefault(battle.battle_tag, []).append(
            {
                "turn": int(battle.turn),
                "teampreview": bool(battle.teampreview),
                "player_role": str(battle.player_role),
                "action": None if action is None else list(action),
                # Save the individual pre-fusion fields, not a fused model input.
                "tensors": [tensor.clone() for tensor in observation.tensors()],
            }
        )

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        self._write_observation_tensors(battle)
        if self.write_replays:
            self._write_replay_json(battle)
        super()._battle_finished_callback(battle)

    def _write_observation_tensors(self, battle: AbstractBattle) -> None:
        replay_id = battle.battle_tag.removeprefix("battle-")
        role = str(battle.player_role)
        payload = {"records": self._live_records.pop(battle.battle_tag, [])}
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


def _load_live_records(path: Path) -> tuple[dict[str, Any], ...]:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("records"), list):
        raise AssertionError(f"Malformed live observation artifact: {path}")
    return tuple(cast(dict[str, Any], record) for record in artifact["records"])


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
) -> None:
    for name in (
        "token_type_ids",
        "side_ids",
        "slot_ids",
        "categorical",
        "events_cat",
        "events_num",
        "events_side_ids",
        "events_slot_ids",
        "events_metadata",
    ):
        torch.testing.assert_close(getattr(live, name), getattr(reconstructed, name))

    # Hidden level/stat values and their provenance intentionally differ: the
    # live battle knows the player's actual values, while a replay reconstructs
    # them from the public species/OTS information.  All other numeric columns
    # are expected to be identical, including legality and dynamic state.
    torch.testing.assert_close(live.numerical[:, :43], reconstructed.numerical[:, :43])
    torch.testing.assert_close(live.numerical[:, 50:56], reconstructed.numerical[:, 50:56])


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


def _match_live_records(
    perspective: ReconstructedPerspective,
    live_records: tuple[dict[str, Any], ...],
) -> tuple[tuple[ReconstructedSnapshot, dict[str, Any]], ...]:
    """Match replay decisions to request-time captures, allowing no-request segments."""
    queues: dict[tuple[bool, int], list[dict[str, Any]]] = defaultdict(list)
    for record in live_records:
        queues[_record_key(record)].append(record)

    matched: list[tuple[ReconstructedSnapshot, dict[str, Any]]] = []
    for snapshot, decision in zip(perspective.snapshots, perspective.decisions, strict=True):
        key = _snapshot_key(snapshot)
        queue = queues[key]
        causes = _cant_reasons(snapshot, perspective.player)
        has_protocol_action = any(
            _line_has_observed_action(line, perspective.player) for line in snapshot.raw_lines
        )

        record: dict[str, Any] | None = None
        if queue and (
            decision.evidence.label_kind is not LabelKind.UNKNOWN or has_protocol_action or causes
        ):
            record = queue.pop(0)
        elif queue and decision.evidence.label_kind is LabelKind.UNKNOWN:
            # A request can be visible in the replay while poke-env was waiting
            # for this side, or while both slots had only implicit/default passes.
            candidate = queue[0]
            action = _live_action(candidate)
            if action is None or all(value in {-2, 0} for value in action):
                record = queue.pop(0)

        if record is None:
            if decision.evidence.label_kind is not LabelKind.UNKNOWN:
                raise AssertionError(
                    f"No live request matched {perspective.game_id} p{perspective.player + 1} "
                    f"decision {decision.decision_index} at turn {snapshot.turn}"
                )
            # This is the intended no-request UNKNOWN: the protocol segment has
            # no order from this player and no live order was captured for it.
            assert not has_protocol_action and not causes
            continue
        matched.append((snapshot, record))

    leftovers = [record for records in queues.values() for record in records]
    if leftovers:
        raise AssertionError(
            f"{len(leftovers)} live requests were not represented by replay decisions for "
            f"p{perspective.player + 1}: {leftovers[0]!r}"
        )
    return tuple(matched)


def _assert_live_truth(
    perspective: ReconstructedPerspective,
    live_records: tuple[dict[str, Any], ...],
    builder: ObservationBuilder,
) -> None:
    for snapshot, record in _match_live_records(perspective, live_records):
        decision = perspective.decisions[snapshot.decision_index]
        evidence = decision.evidence
        live_action = _live_action(record)

        if evidence.label_kind is LabelKind.EXACT:
            assert live_action is not None
            assert live_action == evidence.candidates[0]
        elif evidence.label_kind is LabelKind.PARTIAL:
            assert live_action is not None
            assert live_action in evidence.candidates
        elif evidence.label_kind is LabelKind.UNKNOWN:
            reasons = _cant_reasons(snapshot, perspective.player)
            tags = set(evidence.tags)
            if live_action is None:
                assert not any(
                    _line_has_observed_action(line, perspective.player)
                    for line in snapshot.raw_lines
                )
            elif reasons:
                assert set(reasons) <= _CANT_REASONS, (
                    f"Unknown cant reason was not in the researched taxonomy: {reasons}"
                )
            else:
                assert tags & _REPLAY_AMBIGUITY_TAGS, (
                    f"Selected live action {live_action} became UNKNOWN without a known "
                    f"reconstruction reason: tags={evidence.tags!r}"
                )

        reconstructed = builder.build(snapshot.view, {}).cpu()
        live = _live_tensors(record)
        _assert_expected_observation_fields(live, reconstructed)


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.asyncio
async def test_random_local_games_reconstruct_to_valid_tensors(showdown_server, tmp_path: Path):
    """Play random games, persist replays/captures, and validate round-trip tensors."""
    seed = int(os.getenv("P0_STRESS_SEED", "20260802"))
    game_count = _stress_game_count()
    concurrency = _stress_concurrency()
    replay_dir = tmp_path / "local-replays"
    observation_dir = tmp_path / "live-observations"
    resources = default_runtime_resources()
    builder = ObservationBuilder(resources)
    random.seed(seed)
    poke_env_patches.install()

    player_a = JsonCapturingRandomPlayer(
        account_configuration=AccountConfiguration("StressRandomA", None),
        battle_format=FORMAT.battle_format,
        server_configuration=showdown_server,
        team=DEFAULT_TEST_TEAM,
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
        team=DEFAULT_TEST_TEAM,
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

    perspectives_by_game = {
        document.metadata.replay_id: _assert_reconstruction_labels(document)
        for document in documents
    }
    for replay_id, perspectives in perspectives_by_game.items():
        for perspective in perspectives:
            role = f"p{perspective.player + 1}"
            path = observation_dir / f"{replay_id}-{role}.pt"
            assert path.is_file(), f"Missing live observation capture: {path}"
            _assert_live_truth(perspective, _load_live_records(path), builder)

    compilation = compile_documents(
        documents,
        format_id=FORMAT.battle_format,
        max_candidates=256,
        dex=resources.dex,
    )
    assert len(compilation.games) == game_count
    assert compilation.metrics.counters["accepted_games"] == game_count
    assert compilation.metrics.counters["rejected_games"] == 0

    build = write_tensor_shards(
        compilation,
        tmp_path / "tensor-shards",
        resources=resources,
        max_candidates=256,
        max_decisions_per_shard=4096,
    )
    assert build.manifest.accepted_games == game_count
    assert build.manifest.rejected_games == 0

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
        assert torch.all(loss_mask[label_kind == int(LabelKind.UNKNOWN)] == 0)
        assert torch.all(loss_mask[label_kind == int(LabelKind.EXACT)] > 0)
        assert torch.all(loss_mask[label_kind == int(LabelKind.PARTIAL)] > 0)
