"""Deterministic offline compiler for the replay reconstruction vertical slice."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from p0.battle.legality import validate_joint_action
from p0.replays.group import GroupedSeries, GroupingResult, group_replays
from p0.replays.protocol import ReplayDocument, parse_replay_payload
from p0.replays.reconstruct import (
    ReconstructedPerspective,
    impute_stat_points,
    reconstruct_both,
)
from p0.replays.scrape import load_raw_replay, read_fetch_index


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
    counters["grouping_diagnostics"] = len(grouping.diagnostics) + sum(
        len(group.diagnostics) for group in grouping.series
    )
    games: list[CompiledGame] = []
    imputation_confidence_sum = 0.0
    for group in grouping.series:
        for game_number, document in enumerate(group.games, 1):
            if dex is not None:
                estimates = impute_stat_points(document, dex=dex, seed=imputation_seed)
                counters["imputations"] += sum(item.provenance == "IMPUTED" for item in estimates)
                counters["imputation_unknown"] += sum(
                    item.provenance == "UNKNOWN" for item in estimates
                )
                imputation_confidence_sum += sum(item.confidence for item in estimates)
            perspectives = reconstruct_both(document, max_candidates=max_candidates)
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
    "compile_documents",
    "compile_payloads",
    "compile_raw_cache",
    "compile_replays",
    "write_compilation",
]
