"""Deterministic offline compiler and tensor sharder for reconstructed replays."""

from __future__ import annotations

import concurrent.futures
import os
import shutil
import tempfile
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
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
from p0.contracts import RuntimeContract
from p0.format_config import FORMAT, active_runtime_contract
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import RuntimeResources, default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.persistence import atomic_json_save, atomic_output, atomic_torch_save
from p0.replays.group import GroupedSeries, GroupingResult, group_replays
from p0.replays.identity import ReplayMemberId, normalize_showdown_id
from p0.replays.protocol import (
    ReplayDocument,
    ReplayInputContractError,
    parse_replay_payload,
)
from p0.replays.reconstruction.decisions import (
    infer_decision_windows,
    reconstruct_decisions_from_trace,
)
from p0.replays.reconstruction.diagnostics import ReplayRejectionCategory
from p0.replays.reconstruction.events import parse_protocol_event
from p0.replays.reconstruction.projection import (
    ProjectedPerspective,
    ReplayBattleView,
    ReplayStatValue,
    _projection_snapshot_lines,
    impute_replay_stats,
    project_replay_perspectives,
)
from p0.replays.reconstruction.resolution import resolve_replay_events
from p0.replays.reconstruction.state import reduce_replay_state
from p0.replays.schema import DecisionType, LabelKind, _require_iso_timestamp
from p0.replays.shards import (
    FINAL_OBSERVATION_PREFIX,
    SHARD_ARTIFACT_SCHEMA,
    SHARD_SUMMARY_KEY,
    ShardIndexEntry,
    ShardManifest,
    observation_field_specs,
    validate_shard_tensors,
)
from p0.runtime.process_context import PROCESS_CONTEXT

EMPTY_CANDIDATE_ACTION = (-1, -1)

_SUPPORTED_FORMATS = frozenset({FORMAT.battle_format, FORMAT.bo3_format})
_BLOCKING_SERIES_DIAGNOSTICS = frozenset(
    {"game_after_series_won", "team_identity_conflict", "too_many_games"}
)


@dataclass(frozen=True, slots=True)
class ShardBuildResult:
    manifest_path: Path
    manifest: ShardManifest


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
class CompiledSeries:
    group: GroupedSeries
    games: tuple[CompiledGame, ...]

    def __post_init__(self) -> None:
        if (
            not self.games
            or len(self.games) != len(self.group.games)
            or len(self.games) != len(self.group.memberships)
            or len({game.replay_id for game in self.games}) != len(self.games)
        ):
            raise ValueError("Compiled series must contain every source game exactly once")
        for number, (game, member, document) in enumerate(
            zip(self.games, self.group.memberships, self.group.games, strict=True), 1
        ):
            if (
                game.series_id != self.group.record.series_id
                or game.game_number != number
                or member.game_number != number
                or game.replay_id != member.replay_id
                or game.replay_id != document.metadata.replay_id
                or game.replay_id != game.document.metadata.replay_id
                or game.canonical_player_roles != member.canonical_player_roles
            ):
                raise ValueError("Compiled series games must match source membership in order")


@dataclass(frozen=True, slots=True)
class CompilationResult:
    series: tuple[GroupedSeries, ...]
    accepted_series: tuple[CompiledSeries, ...]
    metrics: CompilationMetrics
    rejection_counters: Mapping[str, Mapping[str, int]] = field(default_factory=dict)

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
                for accepted in self.accepted_series
                for game in accepted.games
            ],
            "metrics": self.metrics.to_dict(),
        }


# Per-decision scalar columns in shard order; action masks are stacked separately.
_SCALAR_DTYPES = {
    "mask_provenance": torch.long,
    "label_kind": torch.long,
    "label_confidence": torch.float32,
    "loss_mask": torch.float32,
    "decision_type": torch.long,
    "exact_action": torch.long,
    "candidate_values": torch.long,
    "candidate_offsets": torch.long,
    "outcome": torch.float32,
}


def _empty_scalar_values() -> dict[str, list[Any]]:
    values: dict[str, list[Any]] = {"action_mask": []}
    values.update((name, []) for name in _SCALAR_DTYPES)
    values["candidate_offsets"].append(0)
    return values


def _replay_observation(
    view: ReplayBattleView,
    builder: ObservationBuilder,
    stat_overrides: Mapping[ReplayMemberId, tuple[int, int, int, int, int, int] | None],
) -> StructuredObservation:
    """Build one replay observation with the imputed stats of every identified member."""
    view.stat_cache.clear()
    overrides: dict[Any, tuple[int, int, int, int, int, int] | None] = {}
    for pokemon in (*view.team.values(), *view.opponent_team.values()):
        try:
            overrides[pokemon] = (
                None if pokemon.identity_uncertain else stat_overrides[pokemon.member_id]
            )
        except KeyError as exc:
            raise ValueError("Replay stats are missing a roster member") from exc
    return builder.build(view, overrides)


def _perspective_tensors(
    game: CompiledGame,
    perspective: ProjectedPerspective,
    builder: ObservationBuilder,
) -> tuple[dict[str, list[Any]], dict[str, list[Any]], StructuredObservation]:
    """Convert a single perspective's snapshots into observation and scalar lists."""
    fields = {name: [] for name, _, _ in observation_field_specs()}
    values = _empty_scalar_values()

    stat_overrides = {estimate.member_id: estimate.values for estimate in game.stat_estimates}
    if len(stat_overrides) != 12:
        raise ValueError("Replay stats must contain one result for every roster member")

    winner = game.document.outcome.winner
    outcome = 0.0 if winner < 0 else (1.0 if winner == perspective.player else -1.0)

    for snapshot, decision in zip(perspective.snapshots, perspective.decisions, strict=True):
        observation = _replay_observation(snapshot.view, builder, stat_overrides)

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
    return fields, values, _replay_observation(perspective.final_view, builder, stat_overrides)


def _tensorize_values(
    fields: dict[str, list[Any]], values: dict[str, list[Any]]
) -> dict[str, torch.Tensor]:
    """Stack scalar lists and fields into batched PyTorch tensors for a single perspective."""
    tensors = {name: torch.stack(items) for name, items in fields.items()}
    tensors["action_mask"] = torch.stack(values["action_mask"])
    tensors.update(
        (name, torch.tensor(values[name], dtype=dtype)) for name, dtype in _SCALAR_DTYPES.items()
    )
    tensors["candidate_values"] = tensors["candidate_values"].reshape(-1, 2)
    return tensors


def _save_shard(
    root: Path,
    shard_id: str,
    tensors: dict[str, torch.Tensor],
    summaries: list[dict[str, Any]],
    runtime_major: str,
) -> ShardIndexEntry:
    validate_shard_tensors(tensors)
    filename = f"{shard_id}.pt"
    atomic_torch_save(
        root / filename,
        {
            "artifact_schema": SHARD_ARTIFACT_SCHEMA,
            "runtime_major": runtime_major,
            "shard_id": shard_id,
            "tensors": tensors,
            SHARD_SUMMARY_KEY: summaries,
        },
    )
    return ShardIndexEntry(
        shard_id=shard_id,
        filename=filename,
        series_id=summaries[0]["series_id"],
        replay_ids=tuple(item["source_replay_id"] for item in summaries if item["player"] == 0),
        decisions=tensors["loss_mask"].shape[0],
        games=len(summaries),
    )


def _read_compilation_index(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "compiled.jsonl"
    if not path.exists():
        return {}
    with path.open("rb") as stream:
        rows = [orjson.loads(line) for line in stream if line.strip()]
    return {row["series_id"]: row for row in rows}


def _save_compilation_index(root: Path, rows: Mapping[str, dict[str, Any]]) -> None:
    with atomic_output(root / "compiled.jsonl") as temporary:
        with temporary.open("wb") as stream:
            for series_id in sorted(rows):
                stream.write(orjson.dumps(rows[series_id]) + b"\n")


def _validate_build_inputs(
    groups: tuple[GroupedSeries, ...],
    created_at: str | None,
    external_rejections: Iterable[str],
) -> tuple[str, ...]:
    if created_at is not None:
        _require_iso_timestamp(created_at, "ShardManifest.created_at")
    rejected = tuple(external_rejections)
    if any(not isinstance(replay_id, str) or not replay_id for replay_id in rejected):
        raise ValueError("Rejected replay ids must be non-empty strings")
    if len(set(rejected)) != len(rejected):
        raise ValueError("Rejected replay ids must be unique")
    source_ids = {replay for group in groups for replay in group.record.game_replay_ids}
    if source_ids.intersection(rejected):
        raise ValueError("Rejected replay ids overlap parsed source replay ids")
    return rejected


def _cache_compilation(
    result: CompilationResult,
    root: Path,
    rows: dict[str, dict[str, Any]],
    builder: ObservationBuilder,
    runtime_major: str,
    max_candidates: int,
) -> None:
    accepted = {series.group.record.series_id: series for series in result.accepted_series}
    for group in result.series:
        series_id = group.record.series_id
        series = accepted.get(series_id)
        entry = None
        counters: Counter[str] = Counter(
            replays=len(group.games), accepted_games=0, rejected_games=len(group.games)
        )
        counters.update(result.rejection_counters.get(series_id, {}))
        if series is not None:
            games = [
                (game, perspective) for game in series.games for perspective in game.perspectives
            ]
            entry = _write_shard(root / "tensors", str(uuid.uuid4()), games, builder, runtime_major)
            counters["accepted_games"] = len(series.games)
            counters["rejected_games"] = 0
            for game in series.games:
                _measure_game(counters, game)
        rows[series_id] = {
            "series_id": series_id,
            "replay_ids": list(group.record.game_replay_ids),
            "max_candidates": max_candidates,
            "runtime_major": runtime_major,
            "entry": None if entry is None else entry.to_dict(),
            "diagnostics": dict(counters),
        }


def _publish_dataset(
    groups: tuple[GroupedSeries, ...],
    root: Path,
    rows: Mapping[str, dict[str, Any]],
    runtime_major: str,
    max_candidates: int,
    created_at: str | None,
    external_rejections: Iterable[str],
) -> ShardBuildResult:
    from p0.replays.dataset import assign_series_splits, write_split_manifest

    entries = tuple(
        ShardIndexEntry.from_dict(rows[group.record.series_id]["entry"])
        for group in groups
        if rows[group.record.series_id]["entry"] is not None
    )
    if not entries:
        raise ValueError("No replay games passed the quality gates; nothing was published")
    memberships = {group.record.series_id: group.record.game_replay_ids for group in groups}
    rejected = tuple(external_rejections)
    memberships.update((f"rejected:{replay_id}", (replay_id,)) for replay_id in rejected)
    config = {"max_candidates": max_candidates}
    latest = root / "latest.json"
    if latest.exists():
        latest_id = orjson.loads(latest.read_bytes())["dataset_id"]
        path = root / latest_id / "manifest.json"
        manifest = ShardManifest.from_dict(orjson.loads(path.read_bytes()))
        if (
            manifest.source_series == memberships
            and manifest.shards == entries
            and manifest.runtime_major == runtime_major
            and manifest.build_config == config
        ):
            return ShardBuildResult(path, manifest)
    counters: Counter[str] = Counter()
    for group in groups:
        counters.update(rows[group.record.series_id]["diagnostics"])
    counters["replays"] += len(rejected)
    counters["rejected_games"] += len(rejected)
    dataset_id = str(uuid.uuid4())
    manifest = ShardManifest(
        runtime_major=runtime_major,
        shards=entries,
        diagnostics=dict(counters),
        created_at=created_at or datetime.now(UTC).isoformat(),
        dataset_id=dataset_id,
        source_format_id=groups[0].record.format_id,
        build_config=config,
        source_series=memberships,
        source_games=counters["replays"],
        accepted_games=counters["accepted_games"],
        rejected_games=counters["rejected_games"],
    )
    temporary = Path(tempfile.mkdtemp(prefix=".dataset-", dir=root))
    destination = root / dataset_id
    try:
        for entry in entries:
            os.link(root / "tensors" / entry.filename, temporary / entry.filename)
        split = assign_series_splits((entry.series_id for entry in entries), dataset_id=dataset_id)
        write_split_manifest(split, temporary / "splits.json")
        atomic_json_save(temporary / "manifest.json", manifest.to_dict())
        os.replace(temporary, destination)
        atomic_json_save(latest, {"dataset_id": dataset_id})
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return ShardBuildResult(destination / "manifest.json", manifest)


def write_tensor_shards(
    result: CompilationResult,
    output_dir: str | Path,
    *,
    resources: RuntimeResources | None = None,
    created_at: str | None = None,
    max_candidates: int = 256,
    external_rejections: Iterable[str] = (),
) -> ShardBuildResult:
    """Save reconstructed series and publish a dataset with default splits."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if not result.accepted_series:
        raise ValueError("No replay games passed the quality gates; nothing was published")
    rejected = _validate_build_inputs(result.series, created_at, external_rejections)
    runtime_major = (
        active_runtime_contract()
        if resources is None
        else RuntimeContract.from_resources(resources.vocab, resources.dex)
    ).major_sha256
    resources = default_runtime_resources() if resources is None else resources
    root = Path(output_dir)
    rows = _read_compilation_index(root)
    _cache_compilation(
        result, root, rows, ObservationBuilder(resources), runtime_major, max_candidates
    )
    _save_compilation_index(root, rows)
    return _publish_dataset(
        result.series, root, rows, runtime_major, max_candidates, created_at, rejected
    )


def _write_shard(
    root: Path,
    shard_id: str,
    games: list[tuple[CompiledGame, ProjectedPerspective]],
    builder: ObservationBuilder,
    runtime_major: str,
) -> ShardIndexEntry:
    """Tensorize the given game perspectives and save them as one shard."""
    field_values = {name: [] for name, _, _ in observation_field_specs()}
    final_observations: list[StructuredObservation] = []
    scalar_values = _empty_scalar_values()
    game_offsets = [0]
    summaries: list[dict[str, Any]] = []
    for game, perspective in games:
        fields, values, final_observation = _perspective_tensors(game, perspective, builder)
        final_observations.append(final_observation)

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
    tensors = _tensorize_values(field_values, scalar_values)
    tensors.update(
        {
            f"{FINAL_OBSERVATION_PREFIX}{name}": tensor
            for name, tensor in zip(
                StructuredObservation._FIELD_NAMES,
                StructuredObservation.stack(final_observations).tensors(),
                strict=True,
            )
        }
    )
    tensors["game_offsets"] = torch.tensor(game_offsets, dtype=torch.long)
    tensors["series_offsets"] = torch.tensor([0, game_offsets[-1]], dtype=torch.long)
    return _save_shard(root, shard_id, tensors, summaries, runtime_major)


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

        for decision in perspective.decisions:
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
        "rejected_non_contiguous_game_numbers",
    )
    counters = Counter({key: 0 for key in zero_keys})
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


def _compile_worker(
    args: tuple[
        ReplayDocument,
        str,
        int,
        tuple[int, int],
        int,
    ],
) -> tuple[CompiledGame | None, ReplayRejectionCategory | None]:
    document, series_id, game_number, roles, max_candidates = args
    runtime_dex = default_runtime_resources().dex

    resolved = resolve_replay_events(document, dex=runtime_dex)
    if resolved.diagnostics:
        return None, resolved.diagnostics[0].category
    events = resolved.require_accepted()
    windows = infer_decision_windows(events)

    state = reduce_replay_state(
        document.metadata.replay_id,
        document.ots,
        events,
        dex=runtime_dex,
        snapshot_line_indices=_projection_snapshot_lines(events, windows),
    )

    if state.diagnostics:
        return None, state.diagnostics[0].category
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
        return None, decision_diagnostics[0].category

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
    chunksize: int | None = None,
) -> CompilationResult:
    """
    Compile documents through event resolution, state reduction, decisions, and projection.

    Documents are grouped into Bo3 series and each game is compiled on its own,
    in worker processes when there are more jobs than CPUs. Series are
    all-or-nothing: if a series has a blocking grouping diagnostic, or any of
    its games fails reconstruction or a quality gate, or a game is missing,
    every game in that series is dropped. Dropped games are counted in the
    metrics under rejected_* keys; they do not raise.

    Raises ReplayInputContractError, before any compilation, for duplicate
    replay ids; a replay without two distinct players, complete six-member OTS
    or a terminal line; an unsupported format or one that differs from
    format_id.

    Arguments:
        documents: Parsed replay documents. Replay ids must be unique.
        format_id: Required format for every document; None accepts any supported format.
        max_candidates: Maximum joint-action candidates kept per reconstructed decision.
        chunksize: Jobs per worker task; None picks one from the job and CPU
            counts. A value of zero or less compiles in this process.

    Returns:
        CompilationResult with all source groups, accepted series, and metrics.
    """
    return _compile_groups(_group_documents(documents, format_id), max_candidates, chunksize)


def _group_documents(
    documents: Iterable[ReplayDocument],
    format_id: str | None,
) -> GroupingResult:
    docs = tuple(documents)
    if len({document.metadata.replay_id for document in docs}) != len(docs):
        raise ReplayInputContractError("Compilation input contains duplicate replay ids")
    for document in docs:
        players = tuple(normalize_showdown_id(name) for name in document.metadata.player_names)
        if not all(players) or players[0] == players[1]:
            raise ReplayInputContractError(
                f"Replay {document.metadata.replay_id!r} must name two distinct players"
            )
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
        if format_id is not None and document.metadata.format_id != format_id:
            raise ReplayInputContractError(
                f"Replay format {document.metadata.format_id!r} does not match requested {format_id!r}"
            )
    return group_replays(docs, format_id=format_id)


def _compile_groups(
    grouping: GroupingResult, max_candidates: int, chunksize: int | None
) -> CompilationResult:
    """
    Reconstruct games and retain accepted series.

    Arguments:
        grouping: Validated source groups.
        max_candidates: Candidate limit per decision.
        chunksize: Worker batch size; nonpositive values run serially.

    Returns:
        Source groups, accepted series, and compilation metrics.
    """
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    counters = _initial_compilation_counters(grouping)
    rejection_counters: dict[str, Counter[str]] = {
        group.record.series_id: Counter() for group in grouping.series
    }
    failed_series: set[str] = set()
    for group in grouping.series:
        blocking_codes = {
            d.code for d in group.diagnostics if d.code in _BLOCKING_SERIES_DIAGNOSTICS
        }
        if blocking_codes:
            failed_series.add(group.record.series_id)
            for code in blocking_codes:
                counters[f"rejected_{code}"] += 1
                rejection_counters[group.record.series_id][f"rejected_{code}"] += 1

    jobs = []
    for group in grouping.series:
        if group.record.series_id in failed_series:
            continue
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

    cpu_count = os.cpu_count() or 1
    if chunksize is None:
        chunksize = max(1, len(jobs) // (cpu_count * 4))

    if len(jobs) <= cpu_count or chunksize <= 0:
        results = [_compile_worker(job) for job in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(mp_context=PROCESS_CONTEXT) as executor:
            results = list(executor.map(_compile_worker, jobs, chunksize=chunksize))

    compiled_by_series: dict[str, list[CompiledGame]] = {}
    confidence_sum = 0.0
    for job, (compiled, reason) in zip(jobs, results, strict=True):
        series_id = job[1]
        if reason is not None:
            counters[f"rejected_reconstruction_{reason.value}"] += 1
            rejection_counters[series_id][f"rejected_reconstruction_{reason.value}"] += 1
            failed_series.add(series_id)
            continue
        if compiled is None:
            continue

        reasons = _quality_reasons(compiled)
        if reasons:
            for quality_reason in reasons:
                counters[f"rejected_{quality_reason}"] += 1
                rejection_counters[series_id][f"rejected_{quality_reason}"] += 1
            failed_series.add(series_id)
            continue

        compiled_by_series.setdefault(series_id, []).append(compiled)

    accepted_series: list[CompiledSeries] = []
    for group in grouping.series:
        series_id = group.record.series_id
        retained = compiled_by_series.get(series_id, [])
        retained.sort(key=lambda game: game.game_number)
        expected_numbers = tuple(sorted(membership.game_number for membership in group.memberships))
        actual_numbers = tuple(game.game_number for game in retained)
        if series_id in failed_series or actual_numbers != expected_numbers:
            failed_series.add(series_id)
            continue
        if actual_numbers != tuple(range(1, len(group.games) + 1)):
            failed_series.add(series_id)
            counters["rejected_non_contiguous_game_numbers"] += 1
            rejection_counters[series_id]["rejected_non_contiguous_game_numbers"] += 1
            continue
        accepted_series.append(CompiledSeries(group, tuple(retained)))
        for game in retained:
            for estimate in game.stat_estimates:
                if estimate.provenance == "IMPUTED":
                    counters["imputations"] += 1
                else:
                    counters["imputation_unknown"] += 1
                confidence_sum += estimate.confidence
            counters["accepted_games"] += 1
            _measure_game(counters, game)

    counters["rejected_games"] = sum(
        len(group.games) for group in grouping.series if group.record.series_id in failed_series
    )
    counters["complete_series"] = sum(
        group.record.is_complete and group.record.series_id not in failed_series
        for group in grouping.series
    )
    counters["incomplete_series"] = counters["series"] - counters["complete_series"]

    metric_values: dict[str, int | float] = dict(counters)
    metric_values["imputation_confidence_sum"] = confidence_sum
    return CompilationResult(
        grouping.series,
        tuple(accepted_series),
        CompilationMetrics(metric_values),
        {series_id: dict(values) for series_id, values in rejection_counters.items()},
    )


def compile_payloads(
    payloads: Iterable[bytes | str | dict[str, Any]],
    *,
    format_id: str | None = None,
    max_candidates: int = 256,
    chunksize: int | None = None,
) -> CompilationResult:
    """Parse raw replay payloads and compile them through the replay pipeline."""
    documents = tuple(parse_replay_payload(payload, format_id=format_id) for payload in payloads)
    return compile_documents(
        documents,
        format_id=format_id,
        max_candidates=max_candidates,
        chunksize=chunksize,
    )


def compile_to_shards(
    documents: Iterable[ReplayDocument],
    output_dir: str | Path,
    *,
    format_id: str | None = None,
    max_candidates: int = 256,
    resources: RuntimeResources | None = None,
    created_at: str | None = None,
    chunksize: int | None = None,
    external_rejections: Iterable[str] = (),
    force_reconstruct: bool = False,
) -> ShardBuildResult:
    """Reuse compiled series by replay IDs; force rebuilding after reconstruction edits."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    grouping = _group_documents(documents, format_id)
    rejected = _validate_build_inputs(grouping.series, created_at, external_rejections)
    runtime_major = (
        active_runtime_contract()
        if resources is None
        else RuntimeContract.from_resources(resources.vocab, resources.dex)
    ).major_sha256
    root = Path(output_dir)
    rows = _read_compilation_index(root)
    pending = []
    for group in grouping.series:
        row = rows.get(group.record.series_id)
        if row is not None and not force_reconstruct:
            if row["runtime_major"] != runtime_major:
                raise ValueError(
                    "Cached vocabulary or encoding is incompatible; use --force-reconstruct"
                )
            entry = row["entry"]
            exists = entry is None or (root / "tensors" / entry["filename"]).is_file()
            if (
                tuple(row["replay_ids"]) == group.record.game_replay_ids
                and row["max_candidates"] == max_candidates
                and exists
            ):
                continue
        pending.append(group)
    if pending:
        pending_replays = {replay for group in pending for replay in group.record.game_replay_ids}
        diagnostics = tuple(
            diagnostic
            for diagnostic in grouping.diagnostics
            if pending_replays.intersection(diagnostic.replay_ids)
        )
        result = _compile_groups(
            GroupingResult(tuple(pending), diagnostics), max_candidates, chunksize
        )
        resources = default_runtime_resources() if resources is None else resources
        _cache_compilation(
            result, root, rows, ObservationBuilder(resources), runtime_major, max_candidates
        )
        _save_compilation_index(root, rows)
    return _publish_dataset(
        grouping.series, root, rows, runtime_major, max_candidates, created_at, rejected
    )


__all__ = [
    "CompilationMetrics",
    "CompilationResult",
    "CompiledGame",
    "CompiledSeries",
    "ShardBuildResult",
    "compile_documents",
    "compile_payloads",
    "compile_to_shards",
    "write_tensor_shards",
]
