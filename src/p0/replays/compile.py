"""Deterministic offline compiler for the replay reconstruction vertical slice."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from p0.battle.legality import action_mask, validate_joint_action
from p0.battle.series import GameSummary, SideGameSummary
from p0.format_config import (
    DEFAULT_RUNTIME_MANIFEST,
    FORMAT,
    canonical_json_sha256,
    load_active_runtime_manifest,
    validate_artifact_runtime_contract,
)
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import RuntimeResources, default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.persistence import atomic_json_save, atomic_torch_save
from p0.replays.group import GroupedSeries, GroupingResult, group_replays
from p0.replays.identity import canonical_format_id
from p0.replays.protocol import ReplayDocument, parse_replay_payload
from p0.replays.reconstruct import (
    ReconstructedPerspective,
    impute_stat_points,
    reconstruct_both,
)
from p0.replays.schema import LabelKind
from p0.replays.scrape import load_raw_replay, read_fetch_index
from p0.replays.shards import (
    BO1_COMPILATION_SEMANTICS,
    SHARD_ARTIFACT_SCHEMA,
    SHARD_SUMMARY_KEY,
    ShardIndexEntry,
    ShardManifest,
    observation_field_specs,
    validate_shard_tensors,
)

EMPTY_CANDIDATE_ACTION = (-1, -1)
REPLAY_PARSER_VERSION = 1
REPLAY_COMPILER_VERSION = 3
IMPUTATION_ALGORITHM = "causal_stat_point_imputation"
IMPUTATION_VERSION = 1
QUALITY_GATE_VERSION = 1
QUALITY_MANIFEST_NAME = "replay-quality-manifest.json"

DEFAULT_QUALITY_GATE: dict[str, Any] = {
    "require_usable_ots": True,
    "illegal_candidate_tolerance": 0,
    "parser_error_tolerance": 0,
    "state_update_error_tolerance": 0,
    "grounding_miss_tolerance": 0,
    "require_finite_observations": True,
    "quarantine_scope": "replay_game",
}


@dataclass(frozen=True, slots=True)
class ShardBuildResult:
    manifest_path: Path
    manifest: ShardManifest


def _normalized(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _summary_side(document: ReplayDocument, side: int) -> SideGameSummary | None:
    ots = document.ots[side]
    species = tuple(_normalized(value) for value in ots.revealed_species)
    if len(species) < 2:
        return None
    moves_used: dict[str, set[str]] = {}
    mega_species = ""
    switch_count = 0
    for line in document.protocol_lines:
        parts = line.parts
        if len(parts) < 3:
            continue
        if parts[1] in {"switch", "drag"} and parts[2].startswith(f"p{side + 1}"):
            switch_count += 1
        if parts[1] == "move" and parts[2].startswith(f"p{side + 1}") and len(parts) >= 4:
            owner = _normalized(parts[2].split(":", 1)[-1])
            moves_used.setdefault(owner, set()).add(_normalized(parts[3]))
        if parts[1] == "-mega" and parts[2].startswith(f"p{side + 1}"):
            mega_species = _normalized(parts[2].split(":", 1)[-1])
    details = {_normalized(name): payload for name, payload in ots.revealed_details.items()}
    items = {
        _normalized(name): _normalized(str(payload["item"]))
        for name, payload in details.items()
        if payload.get("item")
    }
    abilities = {
        _normalized(name): _normalized(str(payload["ability"]))
        for name, payload in details.items()
        if payload.get("ability")
    }
    return SideGameSummary(
        leads=species[:2],
        brought=species[:4],
        mega_species=mega_species,
        moves_used={name: tuple(sorted(values)) for name, values in sorted(moves_used.items())},
        revealed_items=items,
        revealed_abilities=abilities,
        revealed_formes=(),
        switch_count=switch_count,
        pivot_count=0,
    )


def _game_summary(
    game: CompiledGame,
    *,
    series_score: tuple[int, int],
    canonical_roles: tuple[int, int],
) -> GameSummary | None:
    first_side, second_side = (_summary_side(game.document, side) for side in (0, 1))
    if first_side is None or second_side is None:
        return None
    winner = (
        -1 if game.document.outcome.winner < 0 else canonical_roles[game.document.outcome.winner]
    )
    return GameSummary(
        game_number=game.game_number,
        winner=winner,
        series_score=series_score,
        turns=game.document.outcome.turns,
        sides=(first_side, second_side) if canonical_roles == (0, 1) else (second_side, first_side),
    )


def _runtime_hash(manifest_path: str | Path) -> str:
    return load_active_runtime_manifest(manifest_path).runtime_contract_sha256


def _source_series(result: CompilationResult) -> dict[str, tuple[str, ...]]:
    return {
        group.record.series_id: tuple(sorted(group.record.game_replay_ids))
        for group in result.series
    }


def _raw_replay_identities(
    result: CompilationResult,
) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "replay_id": game.replay_id,
            "content_sha256": hashlib.sha256(game.document.raw_payload).hexdigest(),
        }
        for game in sorted(result.games, key=lambda item: item.replay_id)
    )


def _build_configuration(
    *,
    max_candidates: int,
    imputation_seed: int,
    max_decisions_per_shard: int,
) -> dict[str, Any]:
    return {
        "parser_version": REPLAY_PARSER_VERSION,
        "replay_ir_version": 1,
        "compiler_version": REPLAY_COMPILER_VERSION,
        "max_candidates": max_candidates,
        "imputation": {
            "algorithm": IMPUTATION_ALGORITHM,
            "version": IMPUTATION_VERSION,
            "seed": imputation_seed,
        },
        "quality_gate": dict(DEFAULT_QUALITY_GATE),
        "max_decisions_per_shard": max_decisions_per_shard,
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
        "compilation_semantics": BO1_COMPILATION_SEMANTICS,
        "build_config": dict(build_config),
        "runtime_contract_sha256": runtime_hash,
    }
    return canonical_json_sha256(identity)


def _default_quality_records(result: CompilationResult) -> list[dict[str, Any]]:
    records = []
    for game in sorted(result.games, key=lambda item: item.replay_id):
        diagnostics: Counter[str] = Counter()
        decisions: Counter[str] = Counter()
        for perspective in game.perspectives:
            diagnostics.update(perspective.diagnostics.counters)
            for decision in perspective.decisions:
                decisions[decision.evidence.label_kind.name.lower()] += 1
        records.append(
            {
                "replay_id": game.replay_id,
                "content_sha256": hashlib.sha256(game.document.raw_payload).hexdigest(),
                "source_series_id": game.series_id,
                "source_parent_room": game.document.metadata.parent_room,
                "source_room_id": game.document.metadata.room_id,
                "source_game_number": game.game_number,
                "accepted": True,
                "reason_codes": [],
                "diagnostics": dict(sorted(diagnostics.items())),
                "decision_totals": dict(sorted(decisions.items())),
            }
        )
    return records


def _validate_existing_build(
    root: Path,
    *,
    dataset_hash: str,
    manifest_path: str | Path,
) -> ShardBuildResult:
    try:
        manifest_value = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        manifest = ShardManifest.from_dict(manifest_value)
        validate_artifact_runtime_contract(manifest_value, manifest_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"Existing dataset build is invalid: {root}") from exc
    if manifest.dataset_hash != dataset_hash:
        raise ValueError(f"Dataset directory identity mismatch: {root}")
    expected_artifacts = {
        *(entry.filename for entry in manifest.shards),
        manifest.quality_manifest,
    }
    if set(manifest.artifact_hashes) != expected_artifacts:
        raise ValueError(f"Existing dataset artifact index is incomplete: {root}")
    if any(manifest.artifact_hashes[entry.filename] != entry.sha256 for entry in manifest.shards):
        raise ValueError(f"Existing dataset shard identities are inconsistent: {root}")
    if manifest.artifact_hashes[manifest.quality_manifest] != manifest.quality_manifest_sha256:
        raise ValueError(f"Existing dataset quality identity is inconsistent: {root}")
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
            "runtime_contract_sha256": runtime_hash,
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
    quality_records: Iterable[Mapping[str, Any]] | None = None,
) -> ShardBuildResult:
    """Persist a compiled result as immutable, runtime-bound tensor shards.

    Arguments:
        result: Model-agnostic compilation result to tensorize.
        output_dir: Root directory for runtime-keyed shard output.
        max_decisions_per_shard: Soft decision budget for each shard.
        manifest_path: Runtime contract manifest used to bind the artifacts.
        resources: Optional preloaded runtime resources.
        created_at: Optional deterministic manifest timestamp.

    Returns:
        The generated shard manifest and its path.
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
    current_games: list[tuple[CompiledGame, ReconstructedPerspective]] = []
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
        estimate_cache: dict[str, tuple[Any, ...]] = {}
        for game, perspective in current_games:
            if last_series_id is not None and game.series_id != last_series_id:
                series_offsets.append(game_offsets[-1])
            last_series_id = game.series_id
            if game.replay_id not in estimate_cache:
                estimate_cache[game.replay_id] = impute_stat_points(
                    game.document,
                    dex=builder.resources.dex,
                    seed=imputation_seed,
                )
            fields, values = _perspective_tensors(
                game,
                perspective,
                builder=builder,
                stat_estimates=estimate_cache[game.replay_id],
            )
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
                    "source_replay_id": game.replay_id,
                    "summary": None,
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
        for game in sorted(
            result.games,
            key=lambda item: (item.series_id, item.game_number, item.replay_id),
        ):
            game_items = [(game, perspective) for perspective in game.perspectives]
            game_decisions = sum(len(perspective.decisions) for _, perspective in game_items)
            if current_games and current_decisions + game_decisions > max_decisions_per_shard:
                flush()
            current_games.extend(game_items)
            current_decisions += game_decisions
        flush()
        records = [dict(record) for record in (quality_records or _default_quality_records(result))]
        quality_value = {
            "artifact_schema": "p0.replay_quality.v1",
            "runtime_contract_sha256": runtime_hash,
            "dataset_hash": dataset_hash,
            "source_format_id": source_format_id,
            "records": sorted(records, key=lambda record: str(record["replay_id"])),
        }
        atomic_json_save(root / QUALITY_MANIFEST_NAME, quality_value)
        quality_hash = hashlib.sha256((root / QUALITY_MANIFEST_NAME).read_bytes()).hexdigest()
        artifact_hashes = {entry.filename: entry.sha256 for entry in entries}
        artifact_hashes[QUALITY_MANIFEST_NAME] = quality_hash
        accepted = sum(bool(record.get("accepted")) for record in records)
        source_games = len(records)
        timestamp = created_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
        manifest = ShardManifest(
            runtime_contract_sha256=runtime_hash,
            dataset_hash=dataset_hash,
            source_format_id=source_format_id,
            build_config=build_config,
            raw_replays={
                str(identity["replay_id"]): str(identity["content_sha256"])
                for identity in identities
            },
            source_series=memberships,
            source_games=source_games,
            accepted_games=accepted,
            rejected_games=source_games - accepted,
            quality_manifest=QUALITY_MANIFEST_NAME,
            quality_manifest_sha256=quality_hash,
            artifact_hashes=artifact_hashes,
            shards=tuple(entries),
            diagnostics={key: int(value) for key, value in diagnostics.items() if value >= 0},
            created_at=timestamp,
        )
        validate_artifact_runtime_contract(manifest.to_dict(), manifest_path)
        atomic_json_save(root / "manifest.json", manifest.to_dict())
        os.replace(root, destination)
        return ShardBuildResult(destination / "manifest.json", manifest)
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
) -> ShardBuildResult:
    """Compile normalized replay documents and persist their tensor shards."""
    result = compile_documents(
        documents,
        format_id=format_id,
        max_candidates=max_candidates,
        dex=dex,
        imputation_seed=imputation_seed,
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
    document: ReplayDocument
    perspectives: tuple[ReconstructedPerspective, ReconstructedPerspective]


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
                    "normalized": game.document.to_dict(),
                    "perspectives": [
                        {
                            "player": perspective.player,
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
            if int(decision.decision_type) == 1:
                counters["preview_decisions"] += 1
            counters[f"candidate_size_{len(decision.evidence.candidates)}"] += 1
            counters[f"mask_provenance_{int(decision.evidence.mask_provenance)}"] += 1
            if decision.evidence.label_kind != 3:
                for candidate in decision.evidence.candidates:
                    if not validate_joint_action(snapshot.view.decision, *candidate):
                        counters["illegal_candidates"] += 1
            for tag in decision.evidence.tags:
                counters[f"tag_{tag}"] += 1


def compile_documents(
    documents: Iterable[ReplayDocument],
    *,
    format_id: str | None = None,
    max_candidates: int = 256,
    dex: Mapping[str, Any] | None = None,
    imputation_seed: int = 0,
) -> CompilationResult:
    """Compile documents twice, once from each player-relative perspective."""
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
    for group in grouping.series:
        membership_by_replay = {
            membership.replay_id: membership for membership in group.memberships
        }
        for document in group.games:
            game_number = membership_by_replay[document.metadata.replay_id].game_number
            if dex is not None:
                estimates = impute_stat_points(document, dex=dex, seed=imputation_seed)
                counters["imputations"] += sum(item.provenance == "IMPUTED" for item in estimates)
                counters["imputation_unknown"] += sum(
                    item.provenance == "UNKNOWN" for item in estimates
                )
                imputation_confidence_sum += sum(item.confidence for item in estimates)
            perspectives = reconstruct_both(document, max_candidates=max_candidates, dex=dex)
            compiled = CompiledGame(
                group.record.series_id,
                game_number,
                document.metadata.replay_id,
                document,
                perspectives,
            )
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
) -> CompilationResult:
    documents = tuple(parse_replay_payload(payload, format_id=format_id) for payload in payloads)
    return compile_documents(
        documents,
        format_id=format_id,
        max_candidates=max_candidates,
        dex=dex,
        imputation_seed=imputation_seed,
    )


def compile_raw_cache(
    cache_dir: str | Path,
    *,
    format_id: str,
    max_candidates: int = 256,
) -> CompilationResult:
    root = Path(cache_dir) / format_id
    entries = read_fetch_index(root / "index.jsonl")
    documents = tuple(
        parse_replay_payload(
            load_raw_replay(root / "raw" / f"{entry.replay_id}.json.gz"),
            replay_id=entry.replay_id,
            format_id=format_id,
        )
        for entry in entries
    )
    return compile_documents(documents, format_id=format_id, max_candidates=max_candidates)


def _quality_reasons(
    game: CompiledGame,
    *,
    builder: ObservationBuilder,
    imputation_seed: int,
) -> tuple[tuple[str, ...], dict[str, int], dict[str, int]]:
    reasons: set[str] = set()
    diagnostics: Counter[str] = Counter()
    decision_totals: Counter[str] = Counter()
    if any(
        not ots.raw_payload.strip() or len(ots.revealed_species) < 2 for ots in game.document.ots
    ):
        reasons.add("missing_or_unusable_ots")
    estimates = impute_stat_points(
        game.document,
        dex=builder.resources.dex,
        seed=imputation_seed,
    )
    for perspective in game.perspectives:
        diagnostics.update(perspective.diagnostics.counters)
        for name, reason in (
            ("parser_errors", "parser_error"),
            ("state_update_errors", "state_update_error"),
            ("grounding_misses", "grounding_miss"),
        ):
            if perspective.diagnostics.counters.get(name, 0):
                reasons.add(reason)
        try:
            fields, values = _perspective_tensors(
                game,
                perspective,
                builder=builder,
                stat_estimates=estimates,
            )
            tensors = _tensorize_values(fields, values)
            decisions = int(tensors["loss_mask"].shape[0])
            tensors["game_offsets"] = torch.tensor([0, decisions], dtype=torch.long)
            tensors["series_offsets"] = torch.tensor([0, decisions], dtype=torch.long)
            validate_shard_tensors(tensors)
        except (IndexError, KeyError, RuntimeError, TypeError, ValueError):
            reasons.add("invalid_tensor_contract")
        for snapshot, decision in zip(
            perspective.snapshots,
            perspective.decisions,
            strict=True,
        ):
            evidence = decision.evidence
            decision_totals[evidence.label_kind.name.lower()] += 1
            candidate_count = len(evidence.candidates)
            if evidence.label_kind is LabelKind.EXACT and candidate_count != 1:
                reasons.add("invalid_exact_candidate_count")
            elif evidence.label_kind is LabelKind.PARTIAL and candidate_count < 2:
                reasons.add("invalid_partial_candidate_count")
            elif evidence.label_kind is LabelKind.UNKNOWN and candidate_count:
                reasons.add("invalid_unknown_candidates")
            if any(
                not validate_joint_action(snapshot.view.decision, *candidate)
                for candidate in evidence.candidates
            ):
                reasons.add("illegal_labeled_candidate")
    return (
        tuple(sorted(reasons)),
        {key: int(value) for key, value in sorted(diagnostics.items())},
        {key: int(value) for key, value in sorted(decision_totals.items())},
    )


@dataclass(frozen=True, slots=True)
class _RawReplayMembership:
    replay_id: str
    series_id: str
    game_number: int
    parent_room: str
    room_id: str


def _lightweight_replay_metadata(
    raw: bytes,
    *,
    replay_id: str,
    format_id: str,
) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, Mapping):
        raise ValueError("Replay response root must be an object")
    actual_format = canonical_format_id(value, expected=format_id)
    if actual_format != format_id:
        raise ValueError("Replay response has the wrong format")
    players_value = value.get("players")
    if isinstance(value.get("p1"), str) and isinstance(value.get("p2"), str):
        players = (str(value["p1"]), str(value["p2"]))
    elif isinstance(players_value, list) and len(players_value) == 2:
        players = (str(players_value[0]), str(players_value[1]))
    else:
        raise ValueError("Replay response has no player identity")
    canonical_players = tuple(sorted(_normalized(player) for player in players))
    if len(set(canonical_players)) != 2 or not all(canonical_players):
        raise ValueError("Replay response does not have two distinct players")
    parent_value = value.get("parent", value.get("parentid", value.get("parent_room")))
    if parent_value is not None and not isinstance(parent_value, str):
        raise ValueError("Replay parent room must be a string")
    parent = "" if parent_value is None else parent_value
    room_value = value.get("roomid", value.get("room_id"))
    room_id = room_value if isinstance(room_value, str) and room_value else replay_id
    log = value.get("log")
    lines = log.splitlines() if isinstance(log, str) else log if isinstance(log, list) else ()
    game_number_value = value.get("game_number")
    game_number = None if game_number_value is None else int(game_number_value)
    for line in lines:
        if not isinstance(line, str) or not line.startswith(
            ("|uhtml|bestof|", "|uhtmlchange|bestof|")
        ):
            continue
        game_match = re.search(r"Game\s+(\d+)", line, flags=re.IGNORECASE)
        if game_match:
            game_number = int(game_match.group(1))
        href_match = re.search(r'href=["\']?/([^"\'>]+)', line)
        if href_match:
            parent = href_match.group(1).removeprefix("battle-")
    upload = value.get("uploadtime", value.get("upload_time", 0))
    if isinstance(upload, (int, float)):
        upload_order = float(upload)
    elif isinstance(upload, str) and upload:
        upload_order = datetime.fromisoformat(upload.replace("Z", "+00:00")).timestamp()
    else:
        upload_order = 0.0
    return {
        "replay_id": replay_id,
        "format_id": actual_format,
        "players": canonical_players,
        "parent_room": parent,
        "room_id": room_id,
        "game_number": game_number,
        "upload_order": upload_order,
    }


def _lightweight_memberships(
    metadata: Iterable[Mapping[str, Any]],
    *,
    format_id: str,
) -> tuple[dict[str, _RawReplayMembership], dict[str, tuple[str, ...]]]:
    ordered = sorted(
        metadata,
        key=lambda item: (float(item["upload_order"]), str(item["replay_id"])),
    )
    fallback_state: dict[tuple[str, str], tuple[float, int]] = {}
    buckets: dict[tuple[str, tuple[str, str]], list[Mapping[str, Any]]] = {}
    for item in ordered:
        players = tuple(str(player) for player in item["players"])
        if len(players) != 2:
            raise ValueError("Lightweight replay identity has invalid players")
        parent = str(item["parent_room"]).strip()
        if parent:
            series_key = parent
        else:
            room_match = re.match(r"^(.*?)(?:-game(?:-\d+)?)$", str(item["room_id"]))
            if room_match:
                series_key = room_match.group(1)
            else:
                fallback_key = (players[0], players[1])
                previous = fallback_state.get(fallback_key)
                upload = float(item["upload_order"])
                if previous is None or upload - previous[0] > 24 * 60 * 60:
                    cluster = 0 if previous is None else previous[1] + 1
                else:
                    cluster = previous[1]
                fallback_state[fallback_key] = (upload, cluster)
                series_key = f"fallback:{players[0]}:{players[1]}:{cluster}"
        buckets.setdefault((series_key, (players[0], players[1])), []).append(item)
    memberships: dict[str, _RawReplayMembership] = {}
    source_series: dict[str, tuple[str, ...]] = {}
    for (series_key, players), items in sorted(buckets.items()):
        games = sorted(
            items,
            key=lambda item: (
                item["game_number"] is None,
                int(item["game_number"] or 0),
                float(item["upload_order"]),
                str(item["replay_id"]),
            ),
        )
        series_id = hashlib.sha256(
            "\n".join((format_id, series_key, *players)).encode("utf-8")
        ).hexdigest()[:24]
        replay_ids = tuple(str(item["replay_id"]) for item in games)
        source_series[series_id] = replay_ids
        for inferred_number, item in enumerate(games, 1):
            replay_id = str(item["replay_id"])
            memberships[replay_id] = _RawReplayMembership(
                replay_id=replay_id,
                series_id=series_id,
                game_number=int(item["game_number"] or inferred_number),
                parent_room=str(item["parent_room"]),
                room_id=str(item["room_id"]),
            )
    return memberships, source_series


def build_shards_from_cache(
    cache_dir: str | Path,
    output_dir: str | Path,
    *,
    format_id: str = FORMAT.bo3_format,
    max_candidates: int = 256,
    imputation_seed: int = 0,
    max_decisions_per_shard: int = 4096,
    manifest_path: str | Path = DEFAULT_RUNTIME_MANIFEST,
    resources: RuntimeResources | None = None,
    created_at: str | None = None,
) -> ShardBuildResult:
    """Compile an immutable Bo3 cache into independently trained Bo1 games."""
    if format_id != FORMAT.bo3_format:
        raise ValueError(
            f"Replay shard builds require the Champions Bo3 format {FORMAT.bo3_format}"
        )
    cache_root = Path(cache_dir) / format_id
    entries = read_fetch_index(cache_root / "index.jsonl")
    if not entries:
        raise ValueError(f"Replay cache contains no indexed games: {cache_root}")
    raw_identities: list[dict[str, str]] = []
    records: dict[str, dict[str, Any]] = {}
    lightweight_metadata: list[dict[str, Any]] = []
    for entry in entries:
        raw = load_raw_replay(cache_root / "raw" / f"{entry.replay_id}.json.gz")
        digest = hashlib.sha256(raw).hexdigest()
        if digest != entry.content_sha256:
            raise ValueError(f"Raw replay hash does not match the fetch index: {entry.replay_id}")
        raw_identities.append({"replay_id": entry.replay_id, "content_sha256": digest})
        records[entry.replay_id] = {
            "replay_id": entry.replay_id,
            "content_sha256": digest,
            "source_series_id": f"unparsed:{entry.replay_id}",
            "source_parent_room": None,
            "source_room_id": None,
            "source_game_number": None,
            "accepted": False,
            "reason_codes": [],
            "diagnostics": {},
            "decision_totals": {},
        }
        try:
            metadata = _lightweight_replay_metadata(
                raw,
                replay_id=entry.replay_id,
                format_id=format_id,
            )
        except (TypeError, ValueError) as exc:
            records[entry.replay_id]["reason_codes"] = [
                "parse_error",
                type(exc).__name__,
            ]
            continue
        lightweight_metadata.append(metadata)
    memberships, source_series_map = _lightweight_memberships(
        lightweight_metadata,
        format_id=format_id,
    )
    for membership in memberships.values():
        record = records[membership.replay_id]
        record["source_series_id"] = membership.series_id
        record["source_parent_room"] = membership.parent_room
        record["source_room_id"] = membership.room_id
        record["source_game_number"] = membership.game_number
    for replay_id, record in records.items():
        series_id = str(record["source_series_id"])
        if series_id.startswith("unparsed:"):
            source_series_map[series_id] = (replay_id,)

    runtime_resources = default_runtime_resources() if resources is None else resources
    builder = ObservationBuilder(runtime_resources)
    accepted_games: list[CompiledGame] = []
    counters: Counter[str] = Counter()
    for replay_id, membership in sorted(memberships.items()):
        record = records[replay_id]
        game: CompiledGame | None = None
        try:
            document = parse_replay_payload(
                load_raw_replay(cache_root / "raw" / f"{replay_id}.json.gz"),
                replay_id=replay_id,
                format_id=format_id,
            )
        except (TypeError, ValueError) as exc:
            reasons = ("parse_error", type(exc).__name__)
            diagnostic_counts = {}
            decision_totals = {}
        else:
            try:
                perspectives = reconstruct_both(
                    document,
                    max_candidates=max_candidates,
                    dex=builder.resources.dex,
                )
                game = CompiledGame(
                    series_id=membership.series_id,
                    game_number=membership.game_number,
                    replay_id=replay_id,
                    document=document,
                    perspectives=perspectives,
                )
                reasons, diagnostic_counts, decision_totals = _quality_reasons(
                    game,
                    builder=builder,
                    imputation_seed=imputation_seed,
                )
            except (IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
                reasons = ("reconstruction_error", type(exc).__name__)
                diagnostic_counts = {}
                decision_totals = {}
        record["reason_codes"] = list(reasons)
        record["diagnostics"] = diagnostic_counts
        record["decision_totals"] = decision_totals
        record["accepted"] = not reasons
        if not reasons and game is not None:
            accepted_games.append(game)
            _measure_game(counters, game)
    for record in records.values():
        if not record["accepted"]:
            for reason in record["reason_codes"]:
                counters[f"rejected_{reason}"] += 1
    counters["source_games"] = len(entries)
    counters["accepted_games"] = len(accepted_games)
    counters["rejected_games"] = len(entries) - len(accepted_games)
    counters["source_series"] = len(source_series_map)
    if not accepted_games:
        raise ValueError("No replay games passed the quality gates; nothing was published")
    result = CompilationResult(
        (),
        tuple(accepted_games),
        CompilationMetrics(dict(counters)),
    )
    return write_tensor_shards(
        result,
        output_dir,
        max_decisions_per_shard=max_decisions_per_shard,
        manifest_path=manifest_path,
        resources=runtime_resources,
        created_at=created_at,
        max_candidates=max_candidates,
        imputation_seed=imputation_seed,
        raw_replays=raw_identities,
        source_series=source_series_map,
        quality_records=records.values(),
    )


def write_compilation(result: CompilationResult, path: str | Path) -> None:
    """Write a canonical JSON report suitable for deterministic regression checks."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


compile_replays = compile_documents


__all__ = [
    "CompilationMetrics",
    "CompilationResult",
    "CompiledGame",
    "ShardBuildResult",
    "compile_documents",
    "compile_payloads",
    "compile_raw_cache",
    "compile_replays",
    "compile_to_shards",
    "write_tensor_shards",
    "write_compilation",
]
