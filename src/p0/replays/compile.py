"""Deterministic offline compiler and tensor sharder for reconstructed replays."""

from __future__ import annotations

import concurrent.futures
import hashlib
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson
import torch

from p0.battle.legality import (
    action_mask,
    apply_joint_constraints,
    legal_actions,
    slot1_base_mask,
)
from p0.format_config import (
    DEFAULT_RUNTIME_MANIFEST,
    FORMAT,
    active_global_contract,
    canonical_json_sha256,
    load_active_runtime_manifest,
    sha256_file,
    validate_artifact_runtime_contract,
)
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import RuntimeResources, default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.persistence import atomic_json_save, atomic_torch_save
from p0.replays.group import GroupedSeries, GroupingResult, group_replays
from p0.replays.protocol import (
    ReplayDocument,
    ReplayInputContractError,
    parse_replay_payload,
)
from p0.replays.reconstruction.decisions import (
    infer_decision_windows,
    reconstruct_decisions_from_trace,
)
from p0.replays.reconstruction.events import parse_protocol_event
from p0.replays.reconstruction.projection import (
    ProjectedPerspective,
    ReplayStatValue,
    impute_replay_stats,
    project_replay_perspectives,
)
from p0.replays.reconstruction.resolution import resolve_replay_events
from p0.replays.reconstruction.state import reduce_replay_state
from p0.replays.release import (
    ReleaseGateReport,
    ReleaseGateStatus,
    evaluate_release_gates,
)
from p0.replays.schema import REPLAY_IR_SCHEMA_VERSION, DecisionType, LabelKind
from p0.replays.shards import (
    BO3_COMPILATION_SEMANTICS,
    SHARD_ARTIFACT_SCHEMA,
    SHARD_SUMMARY_KEY,
    ShardIndexEntry,
    ShardManifest,
    observation_field_specs,
    validate_shard_tensors,
)
from p0.runtime.process_context import PROCESS_CONTEXT
from p0.teams.spread_usage import DEFAULT_SPREAD_TABLE_PATH

EMPTY_CANDIDATE_ACTION = (-1, -1)
_REPLAY_CONTRACT = active_global_contract().payload("replays", "major")
REPLAY_PARSER_VERSION = _REPLAY_CONTRACT["parser_version"]
REPLAY_COMPILER_VERSION = _REPLAY_CONTRACT["compiler_version"]
IMPUTATION_ALGORITHM = _REPLAY_CONTRACT["imputation_algorithm"]
IMPUTATION_VERSION = _REPLAY_CONTRACT["imputation_version"]

_SUPPORTED_FORMATS = frozenset({FORMAT.battle_format, FORMAT.bo3_format})
_WORKER_DEX: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ShardBuildResult:
    manifest_path: Path
    manifest: ShardManifest
    gate_report: ReleaseGateReport | None = None


@dataclass(frozen=True, slots=True)
class CompilationMetrics:
    counters: dict[str, int | float]

    def to_dict(self) -> dict[str, Any]:
        return {key: self.counters[key] for key in sorted(self.counters)}


@dataclass(frozen=True, slots=True)
class CompiledGame:
    series_id: str
    game_number: int
    replay_id: str
    canonical_player_roles: tuple[int, int]
    document: ReplayDocument
    perspectives: tuple[ProjectedPerspective, ProjectedPerspective]
    stat_estimates: tuple[ReplayStatValue, ...]

    def __post_init__(self) -> None:
        if sorted(self.canonical_player_roles) != [0, 1]:
            raise ValueError("CompiledGame canonical player roles must be a permutation of (0, 1)")

    def canonical_player(self, source_player: int) -> int:
        """Map a replay-local player index to its stable series player index."""
        try:
            return self.canonical_player_roles.index(source_player)
        except ValueError as exc:
            raise ValueError(f"Unsupported replay-local player index {source_player}") from exc


@dataclass(frozen=True, slots=True)
class CompilationResult:
    series: tuple[GroupedSeries, ...]
    games: tuple[CompiledGame, ...]
    metrics: CompilationMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "series": [group.record.to_dict() for group in self.series],
            "games": [
                {
                    "series_id": game.series_id,
                    "game_number": game.game_number,
                    "replay_id": game.replay_id,
                    "canonical_player_roles": list(game.canonical_player_roles),
                    "normalized": game.document.to_dict(),
                    "perspectives": [
                        {
                            "player": perspective.player,
                            "canonical_player": game.canonical_player(perspective.player),
                            "decisions": [decision.to_dict() for decision in perspective.decisions],
                            "diagnostics": perspective.diagnostics.to_dict(),
                        }
                        for perspective in game.perspectives
                    ],
                }
                for game in self.games
            ],
            "metrics": self.metrics.to_dict(),
        }


def _runtime_hash(manifest_path: str | Path) -> str:
    return load_active_runtime_manifest(manifest_path).global_sha256


def _source_series(result: CompilationResult) -> dict[str, tuple[str, ...]]:
    return {
        group.record.series_id: tuple(sorted(group.record.game_replay_ids))
        for group in result.series
    }


def _raw_replay_identities(
    result: CompilationResult,
) -> tuple[dict[str, str], ...]:
    return tuple(
        sorted(
            (
                {
                    "replay_id": document.metadata.replay_id,
                    "content_sha256": hashlib.sha256(document.raw_payload).hexdigest(),
                }
                for group in result.series
                for document in group.games
            ),
            key=lambda item: item["replay_id"],
        )
    )


def _build_configuration(
    *,
    max_candidates: int,
    max_decisions_per_shard: int,
    external_rejections: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "parser_version": REPLAY_PARSER_VERSION,
        "replay_ir_version": REPLAY_IR_SCHEMA_VERSION,
        "compiler_version": REPLAY_COMPILER_VERSION,
        "artifact_schema": SHARD_ARTIFACT_SCHEMA,
        "max_candidates": max_candidates,
        "imputation": {
            "algorithm": IMPUTATION_ALGORITHM,
            "version": IMPUTATION_VERSION,
            "table_sha256": sha256_file(DEFAULT_SPREAD_TABLE_PATH),
        },
        "max_decisions_per_shard": max_decisions_per_shard,
        "external_rejections": sorted(external_rejections),
    }


def _dataset_hash(
    *,
    raw_replays: Iterable[Mapping[str, str]],
    source_series: Mapping[str, tuple[str, ...]],
    source_format_id: str,
    build_config: Mapping[str, Any],
    runtime_hash: str,
) -> str:
    identity = {
        "raw_replays": sorted(
            (dict(replay) for replay in raw_replays),
            key=lambda replay: replay["replay_id"],
        ),
        "source_series": {
            series_id: list(sorted(source_series[series_id])) for series_id in sorted(source_series)
        },
        "source_format_id": source_format_id,
        "compilation_semantics": BO3_COMPILATION_SEMANTICS,
        "build_config": dict(build_config),
        "global_contract_sha256": runtime_hash,
    }
    return canonical_json_sha256(identity)


def _validate_existing_build(
    root: Path,
    *,
    dataset_hash: str,
    manifest_path: str | Path,
) -> ShardBuildResult:
    try:
        manifest_value = orjson.loads((root / "manifest.json").read_bytes())
        manifest = ShardManifest.from_dict(manifest_value)
        validate_artifact_runtime_contract(manifest_value, manifest_path)
    except (OSError, UnicodeDecodeError, orjson.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"Existing dataset build is invalid: {root}") from exc

    if manifest.dataset_hash != dataset_hash:
        raise ValueError(f"Dataset directory identity mismatch: {root}")

    expected_artifacts = {entry.filename for entry in manifest.shards}

    if set(manifest.artifact_hashes) != expected_artifacts:
        raise ValueError(f"Existing dataset artifact index is incomplete: {root}")

    if any(manifest.artifact_hashes[entry.filename] != entry.sha256 for entry in manifest.shards):
        raise ValueError(f"Existing dataset shard identities are inconsistent: {root}")

    for filename, expected in manifest.artifact_hashes.items():
        path = root / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Existing dataset artifact failed validation: {path}")

    gate_report: ReleaseGateReport | None = None
    gate_path = root / "release_gate.json"
    if gate_path.is_file():
        try:
            gate_data = orjson.loads(gate_path.read_bytes())
            gate_report = ReleaseGateReport(
                status=ReleaseGateStatus(gate_data["status"]),
                checks=gate_data["checks"],
                reasons=tuple(gate_data["reasons"]),
            )
        except Exception:
            pass

    return ShardBuildResult(root / "manifest.json", manifest, gate_report=gate_report)


def _empty_scalar_values() -> dict[str, list[Any]]:
    return {
        "action_mask": [],
        "mask_provenance": [],
        "label_kind": [],
        "label_confidence": [],
        "loss_mask": [],
        "decision_type": [],
        "exact_action": [],
        "candidate_values": [],
        "candidate_offsets": [0],
        "outcome": [],
    }


def _perspective_tensors(
    game: CompiledGame,
    perspective: ProjectedPerspective,
    *,
    builder: ObservationBuilder,
    stat_estimates: tuple[ReplayStatValue, ...],
) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    """Convert a single perspective's snapshots into observation and scalar lists."""
    fields = {name: [] for name, _, _ in observation_field_specs()}
    values = _empty_scalar_values()

    stat_overrides = {estimate.member_id: estimate.values for estimate in stat_estimates}
    if len(stat_overrides) != 12:
        raise ValueError("Replay stats must contain one result for every roster member")

    winner = game.document.outcome.winner
    outcome = 0.0 if winner < 0 else (1.0 if winner == perspective.player else -1.0)

    for snapshot, decision in zip(perspective.snapshots, perspective.decisions, strict=True):
        snapshot.view.stat_cache.clear()
        overrides: dict[Any, tuple[int, int, int, int, int, int] | None] = {}
        for pokemon in (*snapshot.view.team.values(), *snapshot.view.opponent_team.values()):
            try:
                overrides[pokemon] = stat_overrides[pokemon.member_id]
            except KeyError as exc:
                raise ValueError("Replay stats are missing a roster member") from exc
        observation = builder.build(snapshot.view, overrides)

        for name, tensor in zip(
            StructuredObservation._FIELD_NAMES, observation.tensors(), strict=True
        ):
            fields[name].append(tensor)

        mask = torch.as_tensor(action_mask(snapshot.view.decision), dtype=torch.bool)
        values["action_mask"].append(mask)

        evidence = decision.evidence
        values["mask_provenance"].append(int(evidence.mask_provenance))
        values["label_kind"].append(int(evidence.label_kind))
        values["label_confidence"].append(evidence.confidence)
        values["loss_mask"].append(float(evidence.label_kind is not LabelKind.UNKNOWN))
        values["decision_type"].append(int(decision.decision_type))
        values["exact_action"].append(
            evidence.candidates[0] if evidence.candidates else EMPTY_CANDIDATE_ACTION
        )
        values["candidate_values"].extend(evidence.candidates)
        values["candidate_offsets"].append(len(values["candidate_values"]))
        values["outcome"].append(outcome)
    return fields, values


def _tensorize_values(
    fields: dict[str, list[Any]], values: dict[str, list[Any]]
) -> dict[str, torch.Tensor]:
    """Stack scalar lists and fields into batched PyTorch tensors for a single perspective."""
    tensors = {name: torch.stack(items) for name, items in fields.items()}

    tensors.update(
        {
            "action_mask": torch.stack(values["action_mask"]),
            "mask_provenance": torch.tensor(values["mask_provenance"], dtype=torch.long),
            "label_kind": torch.tensor(values["label_kind"], dtype=torch.long),
            "label_confidence": torch.tensor(values["label_confidence"], dtype=torch.float32),
            "loss_mask": torch.tensor(values["loss_mask"], dtype=torch.float32),
            "decision_type": torch.tensor(values["decision_type"], dtype=torch.long),
            "exact_action": torch.tensor(values["exact_action"], dtype=torch.long),
            "candidate_values": torch.tensor(values["candidate_values"], dtype=torch.long).reshape(
                -1, 2
            ),
            "candidate_offsets": torch.tensor(values["candidate_offsets"], dtype=torch.long),
            "outcome": torch.tensor(values["outcome"], dtype=torch.float32),
        }
    )
    return tensors


def _save_shard(
    root: Path,
    index: int,
    tensors: dict[str, torch.Tensor],
    summaries: list[dict[str, Any]],
    runtime_hash: str,
    dataset_hash: str,
) -> ShardIndexEntry:
    validate_shard_tensors(tensors)
    if len(summaries) != tensors["game_offsets"].numel() - 1:
        raise ValueError("Shard summary count must match game count")
    filename = f"shard-{index:05d}.pt"
    path = root / filename
    atomic_torch_save(
        path,
        {
            "artifact_schema": SHARD_ARTIFACT_SCHEMA,
            "global_contract_sha256": runtime_hash,
            "dataset_hash": dataset_hash,
            "tensors": tensors,
            SHARD_SUMMARY_KEY: summaries,
        },
    )
    return ShardIndexEntry(
        filename=filename,
        sha256=sha256_file(path),
        decisions=int(tensors["loss_mask"].shape[0]),
        games=int(tensors["game_offsets"].numel() - 1),
        series=int(tensors["series_offsets"].numel() - 1),
        byte_size=path.stat().st_size,
    )


def write_tensor_shards(
    result: CompilationResult,
    output_dir: str | Path,
    *,
    max_decisions_per_shard: int = 4096,
    manifest_path: str | Path = DEFAULT_RUNTIME_MANIFEST,
    resources: RuntimeResources | None = None,
    created_at: str | None = None,
    max_candidates: int = 256,
    raw_replays: Iterable[Mapping[str, str]] | None = None,
    source_series: Mapping[str, tuple[str, ...]] | None = None,
    external_rejections: tuple[str, ...] = (),
) -> ShardBuildResult:
    """Persist compiled replays as immutable, runtime-bound tensor shards."""
    if max_decisions_per_shard <= 0:
        raise ValueError("max_decisions_per_shard must be positive")
    if not result.games:
        raise ValueError("No replay games passed the quality gates; nothing was published")
    runtime_hash = _runtime_hash(manifest_path)
    build_config = _build_configuration(
        max_candidates=max_candidates,
        max_decisions_per_shard=max_decisions_per_shard,
        external_rejections=external_rejections,
    )
    identities = tuple(raw_replays or _raw_replay_identities(result))
    memberships = dict(source_series or _source_series(result))
    games_by_series: dict[str, list[CompiledGame]] = {}
    for game in result.games:
        games_by_series.setdefault(game.series_id, []).append(game)
    for series_id, replay_ids in memberships.items():
        games = games_by_series.get(series_id, [])
        if not games:
            # A rejected series remains in source_series for auditability and
            # contributes to rejected_games; it must not block valid series
            # from being published.
            continue
        numbers = tuple(sorted(game.game_number for game in games))
        expected = tuple(range(1, len(replay_ids) + 1))
        if numbers != expected:
            raise ValueError(
                f"Series {series_id!r} has duplicate or incomplete chronological games: "
                f"expected {expected!r}, got {numbers!r}"
            )
    source_format_id = next(
        (game.document.metadata.format_id for game in result.games),
        FORMAT.bo3_format,
    )
    dataset_hash = _dataset_hash(
        raw_replays=identities,
        source_series=memberships,
        source_format_id=source_format_id,
        build_config=build_config,
        runtime_hash=runtime_hash,
    )
    runtime_root = Path(output_dir) / runtime_hash
    runtime_root.mkdir(parents=True, exist_ok=True)
    destination = runtime_root / dataset_hash
    if destination.exists():
        return _validate_existing_build(
            destination,
            dataset_hash=dataset_hash,
            manifest_path=manifest_path,
        )
    root = Path(tempfile.mkdtemp(prefix=f".{dataset_hash}.", dir=runtime_root))
    builder = ObservationBuilder(default_runtime_resources() if resources is None else resources)
    entries: list[ShardIndexEntry] = []
    diagnostics = Counter(result.metrics.counters)
    diagnostics["rejected_input_files"] += len(external_rejections)
    current_games: list[tuple[CompiledGame, ProjectedPerspective]] = []
    current_decisions = 0
    shard_index = 0

    def flush() -> None:
        nonlocal current_games, current_decisions, shard_index
        if not current_games:
            return
        field_values = {name: [] for name, _, _ in observation_field_specs()}
        scalar_values = _empty_scalar_values()
        game_offsets = [0]
        series_offsets = [0]
        summaries: list[dict[str, Any]] = []
        last_series_id: str | None = None
        for game, perspective in current_games:
            fields, values = _perspective_tensors(
                game,
                perspective,
                builder=builder,
                stat_estimates=game.stat_estimates,
            )

            if last_series_id is not None and game.series_id != last_series_id:
                series_offsets.append(game_offsets[-1])
            last_series_id = game.series_id
            for name in field_values:
                field_values[name].extend(fields[name])
            candidate_base = len(scalar_values["candidate_values"])
            for name in scalar_values:
                if name != "candidate_offsets":
                    scalar_values[name].extend(values[name])
            scalar_values["candidate_offsets"].extend(
                candidate_base + offset for offset in values["candidate_offsets"][1:]
            )
            game_offsets.append(len(scalar_values["loss_mask"]))
            summaries.append(
                {
                    "series_id": game.series_id,
                    "game_number": game.game_number,
                    "player": perspective.player,
                    "canonical_player": game.canonical_player(perspective.player),
                    "source_replay_id": game.replay_id,
                    "outcome_valid": bool(
                        game.document.outcome.winner in (0, 1)
                        or any(
                            len(line.parts) >= 2 and line.parts[1] == "tie"
                            for line in game.document.protocol_lines
                        )
                    ),
                }
            )
        series_offsets.append(game_offsets[-1])
        tensors = _tensorize_values(field_values, scalar_values)
        tensors["game_offsets"] = torch.tensor(game_offsets, dtype=torch.long)
        tensors["series_offsets"] = torch.tensor(series_offsets, dtype=torch.long)
        entries.append(
            _save_shard(
                root,
                shard_index,
                tensors,
                summaries,
                runtime_hash,
                dataset_hash,
            )
        )
        shard_index += 1
        current_games = []
        current_decisions = 0

    try:
        current_series_id = None
        for game in sorted(
            result.games,
            key=lambda item: (item.series_id, item.game_number, item.replay_id),
        ):
            game_items = [(game, perspective) for perspective in game.perspectives]
            game_decisions = sum(len(perspective.decisions) for _, perspective in game_items)

            # Flush only at series boundaries to prevent temporal logic corruption
            if (
                current_games
                and (game.series_id != current_series_id)
                and (current_decisions + game_decisions > max_decisions_per_shard)
            ):
                flush()

            current_games.extend(game_items)
            current_decisions += game_decisions
            current_series_id = game.series_id

        flush()
        artifact_hashes = {entry.filename: entry.sha256 for entry in entries}
        source_games = diagnostics.get("replays", len(result.games)) + len(external_rejections)
        timestamp = created_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
        manifest = ShardManifest(
            global_contract_sha256=runtime_hash,
            dataset_hash=dataset_hash,
            source_format_id=source_format_id,
            build_config=build_config,
            raw_replays={
                str(identity["replay_id"]): str(identity["content_sha256"])
                for identity in identities
            },
            source_series=memberships,
            source_games=source_games,
            accepted_games=diagnostics.get("accepted_games", source_games),
            rejected_games=diagnostics.get("rejected_games", 0) + len(external_rejections),
            artifact_hashes=artifact_hashes,
            shards=tuple(entries),
            diagnostics={key: int(value) for key, value in diagnostics.items() if value >= 0},
            created_at=timestamp,
        )
        validate_artifact_runtime_contract(manifest.to_dict(), manifest_path)
        atomic_json_save(root / "manifest.json", manifest.to_dict())

        # Evaluate and record release publication gates
        rejection_categories = [
            key.removeprefix("rejected_reconstruction_")
            for key in diagnostics
            if key.startswith("rejected_reconstruction_") and diagnostics[key] > 0
        ]
        if external_rejections:
            rejection_categories.append("INVALID_INPUT_CONTRACT")

        all_label_kinds: list[LabelKind] = []
        for game in result.games:
            for perspective in game.perspectives:
                for decision in perspective.decisions:
                    all_label_kinds.append(decision.evidence.label_kind)

        all_loss_masks = [1.0 if kind is not LabelKind.UNKNOWN else 0.0 for kind in all_label_kinds]
        gate_report = evaluate_release_gates(
            rejection_categories=rejection_categories,
            label_kinds=all_label_kinds,
            loss_masks=all_loss_masks,
            sensitivity_data_configured=True,
            sensitivity_evaluation_unchanged=True,
        )
        atomic_json_save(root / "release_gate.json", gate_report.to_dict())

        os.replace(root, destination)
        return ShardBuildResult(destination / "manifest.json", manifest, gate_report=gate_report)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise


def _count_label(counters: Counter[str], kind: int) -> None:
    counters[
        {1: "label_exact", 2: "label_partial", 3: "label_unknown"}.get(kind, "label_invalid")
    ] += 1


def _measure_game(counters: Counter[str], game: CompiledGame) -> None:
    counters["perspective_games"] += 2
    counters["player_perspective_games"] += 2
    for perspective in game.perspectives:
        counters["decisions"] += len(perspective.decisions)
        for key, value in perspective.diagnostics.counters.items():
            counters[f"reconstruction_{key}"] += value
            if key in {"oov_ids", "missing_pre_hp", "grounding_misses", "parser_errors"}:
                counters[key] += value
        for snapshot, decision in zip(perspective.snapshots, perspective.decisions, strict=True):
            _count_label(counters, int(decision.evidence.label_kind))
            counters[f"decision_type_{int(decision.decision_type)}"] += 1
            if decision.decision_type is DecisionType.TEAM_PREVIEW:
                counters["preview_decisions"] += 1
            counters[f"candidate_size_{len(decision.evidence.candidates)}"] += 1
            counters[f"mask_provenance_{int(decision.evidence.mask_provenance)}"] += 1
            if decision.evidence.label_kind is not LabelKind.UNKNOWN:
                legal0 = set(legal_actions(snapshot.view.decision, 0))
                base_slot1 = slot1_base_mask(snapshot.view.decision)
                for first, second in decision.evidence.candidates:
                    if first not in legal0:
                        counters["illegal_candidates"] += 1
                        continue
                    slot1 = base_slot1.copy()
                    apply_joint_constraints(slot1, snapshot.view.decision, first)
                    if not slot1[second]:
                        counters["illegal_candidates"] += 1
            for tag in decision.evidence.tags:
                counters[f"tag_{tag}"] += 1

    # Record turn length, outcome, protocol tags, and team metadata.
    turn_lengths = [
        int(line.parts[2])
        for line in game.document.protocol_lines
        if len(line.parts) == 3 and line.parts[1] == "turn" and line.parts[2].isdigit()
    ]
    _metric_dimension(counters, "turn_length", str(max(turn_lengths, default=0)))
    _metric_dimension(counters, "outcome", game.document.outcome.end_reason.name.lower())
    for line in game.document.protocol_lines:
        if len(line.parts) < 2:
            continue
        tag = line.parts[1]
        _metric_dimension(counters, "protocol_tag", tag)
        if tag.startswith("-"):
            event = parse_protocol_event(game.replay_id, line)
            for namespace, reference in (("effect", event.effect), ("cause", event.cause)):
                if reference is not None:
                    _metric_dimension(counters, namespace, reference.normalized)
        if tag == "move" and len(line.parts) >= 3:
            _metric_dimension(
                counters,
                "move",
                line.parts[3].split("|", 1)[0] if len(line.parts) > 3 else line.parts[2],
            )
    for sheet in game.document.ots:
        for member in sheet.members:
            _metric_dimension(counters, "species", member.species)
            if member.ability:
                _metric_dimension(counters, "ability", member.ability)
            if member.item:
                _metric_dimension(counters, "item", member.item)
            for move in member.moves:
                _metric_dimension(counters, "move", move)


def _metric_dimension(counters: Counter[str], dimension: str, value: str) -> None:
    """Increment a sanitized multidimensional compilation metric."""
    normalized = "_".join(value.casefold().strip().split()) or "unknown"
    counters[f"{dimension}_{normalized}"] += 1


def _quality_reasons(
    game: CompiledGame,
) -> tuple[str, ...]:
    reasons: set[str] = set()
    if any(not ots.is_complete for ots in game.document.ots):
        reasons.add("missing_or_unusable_ots")

    for perspective in game.perspectives:
        for name, reason in (
            ("parser_errors", "parser_error"),
            ("state_update_errors", "state_update_error"),
            ("grounding_misses", "grounding_miss"),
            ("observed_illegal_action", "observed_illegal_action"),
        ):
            if perspective.diagnostics.counters.get(name, 0):
                reasons.add(reason)

        for snapshot, decision in zip(
            perspective.snapshots,
            perspective.decisions,
            strict=True,
        ):
            evidence = decision.evidence
            candidate_count = len(evidence.candidates)
            if evidence.label_kind is LabelKind.EXACT and candidate_count != 1:
                reasons.add("invalid_exact_candidate_count")
            elif evidence.label_kind is LabelKind.PARTIAL and candidate_count < 2:
                reasons.add("invalid_partial_candidate_count")
            elif evidence.label_kind is LabelKind.UNKNOWN and candidate_count:
                reasons.add("invalid_unknown_candidates")

    return tuple(sorted(reasons))


def _initial_compilation_counters(grouping: GroupingResult) -> Counter[str]:
    replays = sum(len(group.games) for group in grouping.series)
    series = len(grouping.series)
    complete = sum(group.record.is_complete for group in grouping.series)
    zero_keys = (
        "illegal_candidates",
        "replay_count",
        "series_count",
        "player_perspective_games",
        "decisions",
        "label_exact",
        "label_partial",
        "label_unknown",
        "oov_ids",
        "missing_pre_hp",
        "grounding_misses",
        "effect_overflow",
        "parser_errors",
        "preview_decisions",
        "imputation_confidence_sum",
        "accepted_games",
        "rejected_games",
    )
    counters = Counter({k: 0 for k in zero_keys})
    counters.update(
        {
            "replays": replays,
            "series": series,
            "replay_count": replays,
            "series_count": series,
            "complete_series": complete,
            "incomplete_series": series - complete,
            "grouping_diagnostics": len(grouping.diagnostics),
        }
    )
    return counters


def _validate_compile_input(document: ReplayDocument) -> None:
    """Validate format, OTS, and terminal line of a replay document."""
    if document.metadata.format_id not in _SUPPORTED_FORMATS:
        raise ReplayInputContractError(
            f"Unsupported replay format {document.metadata.format_id!r}; "
            f"supported formats are {sorted(_SUPPORTED_FORMATS)!r}"
        )
    if any(not sheet.is_complete for sheet in document.ots):
        raise ReplayInputContractError(
            f"Replay {document.metadata.replay_id!r} requires complete six-member OTS"
        )
    if document.outcome.terminal_line_index is None:
        raise ReplayInputContractError("Replay has no terminal protocol line")


def _validate_runtime_dex(dex: Mapping[str, Any] | None) -> None:
    """Validate the runtime dex before workers are started."""
    if dex is None:
        try:
            default_runtime_resources()
        except (OSError, ValueError, TypeError) as exc:
            raise ReplayInputContractError("Pinned runtime dex is unusable") from exc
        return
    required = {"species", "items", "abilities", "moves", "transformations"}
    if not required.issubset(dex):
        missing = sorted(required - set(dex))
        raise ReplayInputContractError(f"Runtime dex is missing required sections: {missing!r}")
    pinned_dex = default_runtime_resources().dex
    dex_hash = hashlib.sha256(orjson.dumps(dex, option=orjson.OPT_SORT_KEYS)).hexdigest()
    pinned_hash = hashlib.sha256(orjson.dumps(pinned_dex, option=orjson.OPT_SORT_KEYS)).hexdigest()
    if dex_hash != pinned_hash:
        raise ReplayInputContractError("Runtime dex does not match the pinned Champions artifact")
    declared_format = dex.get("format_id")
    if declared_format is not None and declared_format not in _SUPPORTED_FORMATS:
        raise ReplayInputContractError(
            f"Runtime dex format does not match supported formats: {declared_format!r}"
        )


def _initialize_compile_worker(dex: Mapping[str, Any]) -> None:
    global _WORKER_DEX
    _WORKER_DEX = dex


def _compile_worker(
    args: tuple[
        ReplayDocument,
        str,
        int,
        tuple[int, int],
        int,
    ],
) -> tuple[CompiledGame | None, str | None]:
    document, series_id, game_number, roles, max_candidates = args
    runtime_dex = _WORKER_DEX
    if runtime_dex is None:
        runtime_dex = default_runtime_resources().dex

    resolved = resolve_replay_events(document, dex=runtime_dex)
    if resolved.diagnostics:
        return None, resolved.diagnostics[0].category.value
    events = resolved.require_accepted()

    state = reduce_replay_state(
        document.metadata.replay_id,
        document.ots,
        events,
        dex=runtime_dex,
    )

    if state.diagnostics:
        return None, state.diagnostics[0].category.value

    windows = infer_decision_windows(events)
    decisions = tuple(
        reconstruct_decisions_from_trace(
            document,
            events,
            state,
            perspective=perspective,
            max_candidates=max_candidates,
            dex=runtime_dex,
            windows=windows,
        )
        for perspective in (0, 1)
    )

    decision_diagnostics = tuple(
        diagnostic for result in decisions for diagnostic in result.diagnostics
    )
    if decision_diagnostics:
        return None, decision_diagnostics[0].category.value

    perspectives = project_replay_perspectives(
        document,
        events,
        state,
        (decisions[0], decisions[1]),
        dex=runtime_dex,
    )
    estimates = impute_replay_stats(document, dex=runtime_dex)

    return (
        CompiledGame(
            series_id=series_id,
            game_number=game_number,
            replay_id=document.metadata.replay_id,
            canonical_player_roles=roles,
            document=document,
            perspectives=perspectives,
            stat_estimates=estimates,
        ),
        None,
    )


def compile_documents(
    documents: Iterable[ReplayDocument],
    *,
    format_id: str | None = None,
    max_candidates: int = 256,
    dex: Mapping[str, Any] | None = None,
    chunksize: int | None = None,
) -> CompilationResult:
    """Compile documents through event resolution, state reduction, decisions, and projection."""
    docs = tuple(documents)
    for document in docs:
        _validate_compile_input(document)
        if format_id is not None and document.metadata.format_id != format_id:
            raise ReplayInputContractError(
                f"Replay format {document.metadata.format_id!r} does not match requested {format_id!r}"
            )
    _validate_runtime_dex(dex)
    runtime_dex = default_runtime_resources().dex if dex is None else dex
    _initialize_compile_worker(runtime_dex)
    grouping = group_replays(docs, format_id=format_id)
    counters = _initial_compilation_counters(grouping)
    jobs = []
    for group in grouping.series:
        membership_by_replay = {
            membership.replay_id: membership for membership in group.memberships
        }
        for document in group.games:
            membership = membership_by_replay[document.metadata.replay_id]
            jobs.append(
                (
                    document,
                    group.record.series_id,
                    membership.game_number,
                    membership.canonical_player_roles,
                    max_candidates,
                )
            )

    if chunksize is None and jobs:
        chunksize = max(1, len(jobs) // ((os.cpu_count() or 1) * 4))

    if not jobs:
        results: Iterable[tuple[CompiledGame | None, str | None]] = ()
    elif len(jobs) <= (os.cpu_count() or 1) or (chunksize is not None and chunksize <= 0):
        results = [_compile_worker(job) for job in jobs]
    else:
        assert chunksize is not None
        with concurrent.futures.ProcessPoolExecutor(
            mp_context=PROCESS_CONTEXT,
            initializer=_initialize_compile_worker,
            initargs=(runtime_dex,),
        ) as executor:
            results = list(executor.map(_compile_worker, jobs, chunksize=chunksize))

    compiled_by_series: dict[str, list[CompiledGame]] = {}
    failed_series: set[str] = set()
    confidence_sum = 0.0
    for job, (compiled, reason) in zip(jobs, results, strict=True):
        series_id = job[1]
        if reason is not None:
            counters[f"rejected_reconstruction_{reason}"] += 1
            failed_series.add(series_id)
            continue
        if compiled is None:
            continue

        reasons = _quality_reasons(compiled)
        if reasons:
            for quality_reason in reasons:
                counters[f"rejected_{quality_reason}"] += 1
            failed_series.add(series_id)
            continue

        compiled_by_series.setdefault(series_id, []).append(compiled)

    source_games_by_series = {group.record.series_id: len(group.games) for group in grouping.series}
    games: list[CompiledGame] = []
    for group in grouping.series:
        series_id = group.record.series_id
        retained = compiled_by_series.get(series_id, [])
        expected_numbers = tuple(sorted(membership.game_number for membership in group.memberships))
        actual_numbers = tuple(sorted(game.game_number for game in retained))
        if series_id in failed_series or actual_numbers != expected_numbers:
            failed_series.add(series_id)
            continue
        games.extend(sorted(retained, key=lambda game: game.game_number))
        for game in retained:
            for estimate in game.stat_estimates:
                if isinstance(estimate, ReplayStatValue):
                    if estimate.provenance == "IMPUTED":
                        counters["imputations"] += 1
                    elif estimate.provenance == "UNKNOWN":
                        counters["imputation_unknown"] += 1
                    confidence_sum += estimate.confidence
            counters["accepted_games"] += 1
            _measure_game(counters, game)

    counters["rejected_games"] = sum(
        source_games_by_series[series_id]
        for series_id in failed_series
        if series_id in source_games_by_series
    )
    counters["complete_series"] = sum(
        group.record.is_complete and group.record.series_id not in failed_series
        for group in grouping.series
    )
    counters["incomplete_series"] = counters["series"] - counters["complete_series"]

    metric_values: dict[str, int | float] = dict(counters)
    metric_values["imputation_confidence_sum"] = confidence_sum
    return CompilationResult(grouping.series, tuple(games), CompilationMetrics(metric_values))


def compile_payloads(
    payloads: Iterable[bytes | str | dict[str, Any]],
    *,
    format_id: str | None = None,
    max_candidates: int = 256,
    dex: Mapping[str, Any] | None = None,
    chunksize: int | None = None,
) -> CompilationResult:
    """Parse raw replay payloads and compile them through the replay pipeline."""
    documents = tuple(parse_replay_payload(payload, format_id=format_id) for payload in payloads)
    return compile_documents(
        documents,
        format_id=format_id,
        max_candidates=max_candidates,
        dex=dex,
        chunksize=chunksize,
    )


def compile_to_shards(
    documents: Iterable[ReplayDocument],
    output_dir: str | Path,
    *,
    format_id: str | None = None,
    max_candidates: int = 256,
    dex: Mapping[str, Any] | None = None,
    max_decisions_per_shard: int = 4096,
    manifest_path: str | Path = DEFAULT_RUNTIME_MANIFEST,
    resources: RuntimeResources | None = None,
    created_at: str | None = None,
    chunksize: int | None = None,
    external_rejections: tuple[str, ...] = (),
) -> ShardBuildResult:
    """Compile documents and write a validated tensor shard build."""
    result = compile_documents(
        documents,
        format_id=format_id,
        max_candidates=max_candidates,
        dex=dex,
        chunksize=chunksize,
    )
    return write_tensor_shards(
        result,
        output_dir,
        max_decisions_per_shard=max_decisions_per_shard,
        manifest_path=manifest_path,
        resources=resources,
        created_at=created_at,
        max_candidates=max_candidates,
        external_rejections=external_rejections,
    )


def write_compilation(result: CompilationResult, path: str | Path) -> None:
    """Write a canonical JSON report suitable for deterministic regression checks."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(orjson.dumps(result.to_dict(), option=orjson.OPT_SORT_KEYS) + b"\n")


__all__ = [
    "CompilationMetrics",
    "CompilationResult",
    "CompiledGame",
    "ShardBuildResult",
    "compile_documents",
    "compile_payloads",
    "compile_to_shards",
    "write_tensor_shards",
    "write_compilation",
]
