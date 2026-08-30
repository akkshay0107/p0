"""Orchestrate normalized replays into compiler-facing results."""

from __future__ import annotations

import concurrent.futures
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from p0.format_config import DEFAULT_RUNTIME_MANIFEST
from p0.model.resources import RuntimeResources, default_runtime_resources
from p0.replays.group import group_replays
from p0.replays.protocol import ReplayDocument, parse_replay_payload
from p0.replays.reconstruction.decisions import (
    infer_decision_windows,
    reconstruct_decisions_from_trace,
)
from p0.replays.reconstruction.projection import (
    ReplayStatValue,
    impute_replay_stats,
    project_replay_perspectives,
)
from p0.replays.reconstruction.resolution import resolve_replay_events
from p0.replays.reconstruction.state import reduce_replay_state
from p0.runtime.process_context import PROCESS_CONTEXT

if TYPE_CHECKING:
    from p0.replays.compile import CompilationResult, CompiledGame, ShardBuildResult


def _rejection_name(reason: str) -> str:
    token = reason.partition(":")[0].strip().replace(" ", "_")
    return token if token.replace("_", "").isalnum() else "reconstruction"


def _compile_worker(
    args: tuple[
        ReplayDocument,
        str,
        int,
        tuple[int, int],
        int,
        Mapping[str, Any] | None,
    ],
) -> tuple[CompiledGame | None, str | None]:
    document, series_id, game_number, roles, max_candidates, dex = args
    runtime_dex = default_runtime_resources().dex if dex is None else dex

    try:
        resolved = resolve_replay_events(document, dex=runtime_dex)
        if resolved.diagnostics:
            return None, _rejection_name(resolved.diagnostics[0].reason)
        events = resolved.require_accepted()

        state = reduce_replay_state(
            document.metadata.replay_id,
            document.ots,
            events,
            dex=runtime_dex,
        )
        if state.diagnostics:
            return None, _rejection_name(state.diagnostics[0].reason)
    except (IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        return None, type(exc).__name__

    try:
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
    except (IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        return None, type(exc).__name__

    if any(result.diagnostics for result in decisions):
        return None, _rejection_name(decisions[0].diagnostics[0].reason)

    try:
        perspectives = project_replay_perspectives(
            document,
            events,
            state,
            (decisions[0], decisions[1]),
            dex=runtime_dex,
        )
        estimates = impute_replay_stats(document, dex=runtime_dex)
    except (IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        return None, type(exc).__name__

    from p0.replays.compile import CompiledGame

    return (
        CompiledGame(
            series_id,
            game_number,
            document.metadata.replay_id,
            roles,
            document,
            perspectives,
            estimates,
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
    """Compile documents through the reducer, decisions, projection, and stats path."""
    from p0.replays.compile import (
        CompilationMetrics,
        CompilationResult,
        _initial_compilation_counters,
        _measure_game,
        _quality_reasons,
    )

    grouping = group_replays(documents, format_id=format_id)
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
                    dex,
                )
            )

    if chunksize is None and jobs:
        chunksize = max(1, len(jobs) // ((os.cpu_count() or 1) * 4))

    if not jobs:
        results: Iterable[tuple[CompiledGame | None, str | None]] = ()
    elif len(jobs) <= (os.cpu_count() or 1) or (chunksize is not None and chunksize <= 0):
        results = (_compile_worker(job) for job in jobs)
    else:
        assert chunksize is not None
        with concurrent.futures.ProcessPoolExecutor(mp_context=PROCESS_CONTEXT) as executor:
            results = executor.map(_compile_worker, jobs, chunksize=chunksize)

    games: list[CompiledGame] = []
    confidence_sum = 0.0
    for compiled, reason in results:
        if reason is not None:
            counters[f"rejected_reconstruction_{reason}"] += 1
            counters["rejected_games"] += 1
            continue
        if compiled is None:
            continue

        estimates = tuple(
            value for value in compiled.stat_estimates if isinstance(value, ReplayStatValue)
        )
        counters["imputations"] += sum(value.provenance == "IMPUTED" for value in estimates)
        counters["imputation_unknown"] += sum(value.provenance == "UNKNOWN" for value in estimates)
        confidence_sum += sum(value.confidence for value in estimates)

        reasons = _quality_reasons(compiled)
        if reasons:
            for quality_reason in reasons:
                counters[f"rejected_{quality_reason}"] += 1
            counters["rejected_games"] += 1
            continue

        counters["accepted_games"] += 1
        games.append(compiled)
        _measure_game(counters, compiled)

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
    from p0.replays.compile import write_tensor_shards

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


__all__ = [
    "compile_documents",
    "compile_payloads",
    "compile_to_shards",
]
