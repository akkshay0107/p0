"""Deterministic offline compiler for the replay reconstruction vertical slice."""

from __future__ import annotations

import concurrent.futures
import hashlib
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

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
    validate_artifact_runtime_contract,
)
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import RuntimeResources, default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.persistence import atomic_json_save, atomic_torch_save
from p0.replays.group import GroupedSeries, GroupingResult, group_replays
from p0.replays.protocol import ReplayDocument, parse_replay_payload
from p0.replays.reconstruct import (
    ReconstructedPerspective,
    impute_stat_points,
    reconstruct_both,
)
from p0.replays.schema import DecisionType, LabelKind
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

EMPTY_CANDIDATE_ACTION = (-1, -1)
_REPLAY_CONTRACT = active_global_contract().payload("replays", "major")
REPLAY_PARSER_VERSION = _REPLAY_CONTRACT["parser_version"]
REPLAY_COMPILER_VERSION = _REPLAY_CONTRACT["compiler_version"]
IMPUTATION_ALGORITHM = _REPLAY_CONTRACT["imputation_algorithm"]
IMPUTATION_VERSION = _REPLAY_CONTRACT["imputation_version"]


@dataclass(frozen=True, slots=True)
class ShardBuildResult:
    manifest_path: Path
    manifest: ShardManifest


def _normalized(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


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
    imputation_seed: int,
    max_decisions_per_shard: int,
    external_rejections: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "parser_version": REPLAY_PARSER_VERSION,
        "replay_ir_version": 1,
        "compiler_version": REPLAY_COMPILER_VERSION,
        # The dataset hash names the output directory, so the shard layout has
        # to be part of it: a schema-only bump must land in a new directory
        # rather than colliding with a build the reader would now reject.
        "artifact_schema": SHARD_ARTIFACT_SCHEMA,
        "max_candidates": max_candidates,
        "imputation": {
            "algorithm": IMPUTATION_ALGORITHM,
            "version": IMPUTATION_VERSION,
            "seed": imputation_seed,
        },
        "max_decisions_per_shard": max_decisions_per_shard,
        "external_rejections": list(sorted(external_rejections)),
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
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Existing dataset artifact failed validation: {path}")

    return ShardBuildResult(root / "manifest.json", manifest)


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
    perspective: ReconstructedPerspective,
    *,
    builder: ObservationBuilder,
    stat_estimates: tuple[Any, ...],
) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    """Convert a single perspective's snapshots into raw python observation and scalar lists."""
    fields = {name: [] for name, _, _ in observation_field_specs()}
    values = _empty_scalar_values()

    estimates = {
        (estimate.side, _normalized(estimate.species)): estimate.precomputed
        for estimate in stat_estimates
        if estimate.precomputed is not None
    }

    winner = game.document.outcome.winner
    outcome = 0.0 if winner < 0 else (1.0 if winner == perspective.player else -1.0)

    for snapshot, decision in zip(perspective.snapshots, perspective.decisions, strict=True):
        snapshot.view.stat_cache = {}
        overrides = {}
        for side, team in ((0, snapshot.view.team), (1, snapshot.view.opponent_team)):
            for pokemon in team.values():
                precomputed = estimates.get(
                    (side if perspective.player == 0 else 1 - side, _normalized(pokemon.species))
                )
                if precomputed is not None:
                    overrides[pokemon] = precomputed
        # ReconstructedSnapshot.events is the request-scoped event window.
        # Keep the handoff explicit so observations cannot silently fall back
        # to FixtureBattleView's default empty event list.
        snapshot.view.events = list(snapshot.events)
        observation = builder.build(snapshot.view, overrides)
        observation.validate(batch_rank=0)
        observation.validate_overflow_contract()

        if any(
            tensor.is_floating_point() and not torch.isfinite(tensor).all()
            for tensor in observation.tensors()
        ):
            raise ValueError("Replay observation contains a non-finite tensor value")

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
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ShardIndexEntry(
        filename=filename,
        sha256=digest,
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
    imputation_seed: int = 0,
    raw_replays: Iterable[Mapping[str, str]] | None = None,
    source_series: Mapping[str, tuple[str, ...]] | None = None,
    external_rejections: tuple[str, ...] = (),
) -> ShardBuildResult:
    """Persist a compiled result as immutable, runtime-bound tensor shards.

    Arguments:
        result: Model-agnostic compilation result to tensorize.
        output_dir: Root directory for runtime-keyed shard output.
        max_decisions_per_shard: Soft decision budget for each shard.
        manifest_path: Runtime contract manifest used to bind the artifacts.
        resources: Optional preloaded runtime resources.
        created_at: Optional deterministic manifest timestamp.
        max_candidates: Maximum number of action candidates per decision.
        imputation_seed: Random seed for stat imputation.
        raw_replays: Optional precomputed identities for raw replay payload.
        source_series: Optional precomputed mappings of source series.
        external_rejections: Input identities rejected before replay parsing.

    Returns:
        The generated shard manifest and its path as a ShardBuildResult.
    """
    if max_decisions_per_shard <= 0:
        raise ValueError("max_decisions_per_shard must be positive")
    if not result.games:
        raise ValueError("No replay games passed the quality gates; nothing was published")
    runtime_hash = _runtime_hash(manifest_path)
    build_config = _build_configuration(
        max_candidates=max_candidates,
        imputation_seed=imputation_seed,
        max_decisions_per_shard=max_decisions_per_shard,
        external_rejections=external_rejections,
    )
    identities = tuple(raw_replays or _raw_replay_identities(result))
    memberships = dict(source_series or _source_series(result))
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
    current_games: list[tuple[CompiledGame, ReconstructedPerspective]] = []
    current_decisions = 0
    shard_index = 0
    failed_games: set[str] = set()

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
            if game.replay_id in failed_games:
                continue

            try:
                fields, values = _perspective_tensors(
                    game,
                    perspective,
                    builder=builder,
                    stat_estimates=game.stat_estimates,
                )
            except (IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
                diagnostics[f"rejected_tensorization_{type(exc).__name__}"] += 1
                if game.replay_id not in failed_games:
                    failed_games.add(game.replay_id)
                    diagnostics["rejected_games"] += 1
                    diagnostics["accepted_games"] -= 1
                continue

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
        os.replace(root, destination)
        return ShardBuildResult(destination / "manifest.json", manifest)
    # Remove the temporary build tree even when cancellation or interruption
    # raises outside the ordinary Exception hierarchy.
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise


def compile_to_shards(
    documents: Iterable[ReplayDocument],
    output_dir: str | Path,
    *,
    format_id: str | None = None,
    max_candidates: int = 256,
    dex: Mapping[str, Any] | None = None,
    imputation_seed: int = 0,
    max_decisions_per_shard: int = 4096,
    manifest_path: str | Path = DEFAULT_RUNTIME_MANIFEST,
    resources: RuntimeResources | None = None,
    created_at: str | None = None,
    chunksize: int | None = None,
    external_rejections: tuple[str, ...] = (),
) -> ShardBuildResult:
    """Compile normalized replay documents and persist their tensor shards.

    Arguments:
        documents: Iterable stream of replay documents to compile.
        output_dir: Directory where shard artifacts are written.
        format_id: Optional exact format filter.
        max_candidates: Maximum number of action candidates per decision.
        dex: Optional stat dex for imputation.
        imputation_seed: Random seed for stat imputation.
        max_decisions_per_shard: Maximum decisions packed into a single shard.
        manifest_path: Path to the runtime manifest for contract validation.
        resources: Optional pre-loaded runtime resources.
        created_at: Optional ISO timestamp stamped into the manifest.
        chunksize: Optional ProcessPoolExecutor chunk size (see ``compile_documents``).
        external_rejections: Input identities rejected before replay parsing.

    Returns:
        A ShardBuildResult containing the manifest path and manifest object.
    """
    result = compile_documents(
        documents,
        format_id=format_id,
        max_candidates=max_candidates,
        dex=dex,
        imputation_seed=imputation_seed,
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
        imputation_seed=imputation_seed,
        external_rejections=external_rejections,
    )


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
    perspectives: tuple[ReconstructedPerspective, ReconstructedPerspective]
    stat_estimates: tuple[Any, ...]

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
                # Precompute the slot-0 legal set and the slot-1 base mask
                # once per decision, then apply per-first-action constraints as
                # cheap boolean array ops instead of calling validate_joint_action
                # per candidate (81x faster — see scratch/bench_legality.py).
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


def _quality_reasons(
    game: CompiledGame,
) -> tuple[str, ...]:
    reasons: set[str] = set()
    if any(
        not ots.raw_payload.strip() or len(ots.revealed_species) < 2 for ots in game.document.ots
    ):
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


def _compile_worker(
    args: tuple[
        ReplayDocument,
        str,
        int,
        tuple[int, int],
        int,
        Mapping[str, Any] | None,
        int,
    ],
) -> tuple[CompiledGame | None, str | None]:
    (
        document,
        series_id,
        game_number,
        canonical_player_roles,
        max_candidates,
        dex,
        imputation_seed,
    ) = args
    estimates = ()

    if dex is not None:
        estimates = impute_stat_points(document, dex=dex, seed=imputation_seed)

    try:
        perspectives = reconstruct_both(document, max_candidates=max_candidates, dex=dex)
    except (IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        return None, type(exc).__name__

    compiled = CompiledGame(
        series_id,
        game_number,
        document.metadata.replay_id,
        canonical_player_roles,
        document,
        perspectives,
        estimates,
    )
    return compiled, None


def compile_documents(
    documents: Iterable[ReplayDocument],
    *,
    format_id: str | None = None,
    max_candidates: int = 256,
    dex: Mapping[str, Any] | None = None,
    imputation_seed: int = 0,
    chunksize: int | None = None,
) -> CompilationResult:
    """Compile a stream of raw ReplayDocuments into state-machine verified CompiledGames.

    This function processes every document twice (once from each player's perspective),
    imputes missing stats, groups games by series, and tracks all diagnostic counters.

    Arguments:
        documents: Iterable stream of replay documents to compile.
        format_id: Optional exact format filter.
        max_candidates: Maximum number of action candidates per decision.
        dex: Optional stat dex for imputation.
        imputation_seed: Random seed for stat imputation.
        chunksize: Optional ProcessPoolExecutor chunk size. When ``None`` (the
            default) a value is derived from the job count and CPU count. For
            small corpora (fewer jobs than workers) compilation runs inline to
            avoid the overhead of spawning a process pool.

    Returns:
        A CompilationResult containing the series, games, and metrics.
    """
    grouping: GroupingResult = group_replays(documents, format_id=format_id)
    counters: Counter[str] = Counter()

    counters["illegal_candidates"] = 0
    for key in (
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
    ):
        counters[key] = 0
    counters["replays"] = sum(len(group.games) for group in grouping.series)
    counters["series"] = len(grouping.series)
    counters["replay_count"] = counters["replays"]
    counters["series_count"] = counters["series"]
    counters["complete_series"] = sum(group.record.is_complete for group in grouping.series)
    counters["incomplete_series"] = counters["series"] - counters["complete_series"]
    counters["grouping_diagnostics"] = len(grouping.diagnostics)

    games: list[CompiledGame] = []
    imputation_confidence_sum = 0.0

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
                    dex,
                    imputation_seed,
                )
            )

    if jobs:
        worker_count = os.cpu_count() or 1
        if chunksize is None:
            # Derive a chunk size that balances pickling overhead against
            # worker utilisation. Each job carries a ReplayDocument with
            # raw_payload bytes, so large chunks can pressure memory.
            chunksize = max(1, len(jobs) // (worker_count * 4))

        if len(jobs) <= worker_count or chunksize <= 0:
            # For small corpora the process-pool startup cost exceeds the
            # benefit; run inline so callers get deterministic single-process
            # behaviour without the multiprocessing fork() deprecation.
            results = (_compile_worker(job) for job in jobs)
        else:
            # Do not inherit the parent process with fork: the compiler can run
            # after PyTorch and other threaded libraries have been initialized,
            # which makes fork unsafe and emits a deprecation warning on Python
            # 3.13+. Use the shared forkserver-on-Linux/spawn-elsewhere context.
            with concurrent.futures.ProcessPoolExecutor(mp_context=PROCESS_CONTEXT) as executor:
                results = executor.map(_compile_worker, jobs, chunksize=chunksize)

        for compiled, exc_name in results:
            if exc_name is not None:
                counters[f"rejected_reconstruction_{exc_name}"] += 1
                counters["rejected_games"] += 1
                continue

            if compiled is None:
                continue

            if dex is not None:
                counters["imputations"] += sum(
                    item.provenance == "IMPUTED" for item in compiled.stat_estimates
                )
                counters["imputation_unknown"] += sum(
                    item.provenance == "UNKNOWN" for item in compiled.stat_estimates
                )
                imputation_confidence_sum += sum(
                    item.confidence for item in compiled.stat_estimates
                )

            reasons = _quality_reasons(compiled)
            if reasons:
                for reason in reasons:
                    counters[f"rejected_{reason}"] += 1
                counters["rejected_games"] += 1
                continue

            counters["accepted_games"] += 1
            games.append(compiled)
            _measure_game(counters, compiled)

    metric_values: dict[str, int | float] = dict(counters)
    metric_values["imputation_confidence_sum"] = imputation_confidence_sum
    return CompilationResult(grouping.series, tuple(games), CompilationMetrics(metric_values))


def compile_payloads(
    payloads: Iterable[bytes | str | dict[str, Any]],
    *,
    format_id: str | None = None,
    max_candidates: int = 256,
    dex: Mapping[str, Any] | None = None,
    imputation_seed: int = 0,
    chunksize: int | None = None,
) -> CompilationResult:
    """Parse raw replay JSON payloads and compile them into verified games."""
    documents = tuple(parse_replay_payload(payload, format_id=format_id) for payload in payloads)

    return compile_documents(
        documents,
        format_id=format_id,
        max_candidates=max_candidates,
        dex=dex,
        imputation_seed=imputation_seed,
        chunksize=chunksize,
    )


def write_compilation(result: CompilationResult, path: str | Path) -> None:
    """Write a canonical JSON report suitable for deterministic regression checks."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(orjson.dumps(result.to_dict(), option=orjson.OPT_SORT_KEYS) + b"\n")


compile_replays = compile_documents


__all__ = [
    "CompilationMetrics",
    "CompilationResult",
    "CompiledGame",
    "ShardBuildResult",
    "compile_documents",
    "compile_payloads",
    "compile_replays",
    "compile_to_shards",
    "write_tensor_shards",
    "write_compilation",
]
