"""Build, validate, and audit the immutable team corpus."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from p0.format_config import FORMAT, current_manifest
from p0.model.tokenizer import PokemonTokenizer, Resolution
from p0.paths import DEFAULT_PATHS
from p0.persistence import atomic_json_save
from p0.teams.corpus import (
    CORPUS_MANIFEST_SCHEMA,
    CorpusEntry,
    TeamCorpusManifest,
    corpus_content_hash,
)
from p0.teams.stat_points import STAT_POINT_LIMIT, STAT_POINT_TOTAL_LIMIT
from p0.teams.team import TeamRecord, deduplicate_variants
from p0.teams.validation import AdmissionResult, validate_many


def audit_corpus(
    manifest: TeamCorpusManifest,
    *,
    total_candidates: int | None = None,
    rejections_by_reason: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Compute exact content-coverage metrics across all entries in a manifest."""
    species_set: set[str] = set()
    move_set: set[str] = set()
    item_set: set[str] = set()

    for entry in manifest.entries:
        parts = entry.packed.split("]")
        for part in parts:
            if not part:
                continue
            fields = part.split("|")
            if not fields or not fields[0]:
                continue
            species = fields[1] if len(fields) > 1 and fields[1] else fields[0]
            species_set.add(PokemonTokenizer.normalize_id(species))
            if len(fields) > 2 and fields[2]:
                item_set.add(PokemonTokenizer.normalize_id(fields[2]))
            if len(fields) > 4 and fields[4]:
                for move in fields[4].split(","):
                    if move:
                        move_set.add(PokemonTokenizer.normalize_id(move))

    admitted = len(manifest.entries)
    total = admitted if total_candidates is None else total_candidates
    return {
        "total_candidates": total,
        "admitted_count": admitted,
        "rejected_count": total - admitted,
        "rejections_by_reason": dict(rejections_by_reason or {}),
        "species_coverage": tuple(sorted(species_set)),
        "move_coverage": tuple(sorted(move_set)),
        "item_coverage": tuple(sorted(item_set)),
    }


def _check_vocabulary(tokenizer: PokemonTokenizer, variant: TeamRecord) -> str | None:
    for member in variant.team.members:
        _, status = tokenizer.resolve("species", member.species)
        if status == Resolution.OOV:
            return f"oov_species: {member.species}"
        if member.item:
            _, status = tokenizer.resolve("items", member.item)
            if status == Resolution.OOV:
                return f"oov_item: {member.item}"
        _, status = tokenizer.resolve("abilities", member.ability)
        if status == Resolution.OOV:
            return f"oov_ability: {member.ability}"
        for move in member.moves:
            _, status = tokenizer.resolve("moves", move)
            if status == Resolution.OOV:
                return f"oov_move: {move}"
    return None


def _check_spreads(variant: TeamRecord) -> str | None:
    if variant.spread_provenance not in ("imputed", "exact"):
        return f"illegal_provenance: {variant.spread_provenance}"
    for spread in variant.spreads:
        values = spread.as_tuple()
        if any(value < 0 or value > STAT_POINT_LIMIT for value in values):
            return "illegal_spread_bounds"
        if sum(values) > STAT_POINT_TOTAL_LIMIT:
            return "illegal_spread_total"
    return None


def build_corpus(
    variants: Sequence[TeamRecord],
    *,
    tokenizer: PokemonTokenizer | None = None,
    validator: Callable[..., Sequence[AdmissionResult]] = validate_many,
    global_contract_sha256: str = "",
    format_id: str = FORMAT.battle_format,
    created_at: str | None = None,
) -> tuple[TeamCorpusManifest, dict[str, Any]]:
    """
    Admit, deduplicate, validate, and audit candidate team variants.

    Arguments:
        variants: Candidate teams with metadata.
        tokenizer: Vocabulary used to reject out-of-vocabulary team content.
        validator: Callable that validates the deduplicated candidates.
        global_contract_sha256: Global major-contract identity recorded in the manifest.
        format_id: Battle format associated with the corpus.
        created_at: Optional manifest timestamp.

    Returns:
        The corpus manifest and its coverage/rejection audit.
    """
    if tokenizer is None:
        tokenizer = PokemonTokenizer.from_file(DEFAULT_PATHS.data_root / "vocab.json")
    if not global_contract_sha256:
        global_contract_sha256 = current_manifest().global_sha256

    deduped = deduplicate_variants(variants)
    validation_results = validator(deduped)
    if len(validation_results) != len(deduped):
        raise RuntimeError("Validation result count does not match deduplicated variant count")

    entries: list[CorpusEntry] = []
    rejections: dict[str, int] = {}

    for variant, result in zip(deduped, validation_results, strict=True):
        if not result.valid or not result.packed_team:
            reason = "showdown_invalid"
            if result.problems:
                reason = f"showdown_invalid: {result.problems[0]}"
            rejections[reason] = rejections.get(reason, 0) + 1
            continue

        oov_reason = _check_vocabulary(tokenizer, variant)
        if oov_reason is not None:
            rejections[oov_reason] = rejections.get(oov_reason, 0) + 1
            continue

        spread_reason = _check_spreads(variant)
        if spread_reason is not None:
            rejections[spread_reason] = rejections.get(spread_reason, 0) + 1
            continue

        packed = result.packed_team
        if not isinstance(packed, str):
            raise RuntimeError("Admitted team is missing its packed representation")
        packed_sha256 = hashlib.sha256(packed.encode("utf-8")).hexdigest()

        try:
            entry = CorpusEntry(
                canonical_hash=variant.team.team_hash,
                packed=packed,
                packed_sha256=packed_sha256,
                usage_count=variant.metadata.usage_count,
                spread_provenance=variant.spread_provenance,
            )
        except ValueError as exc:
            reason = f"entry_error: {exc}"
            rejections[reason] = rejections.get(reason, 0) + 1
            continue

        entries.append(entry)

    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()

    ordered_entries = tuple(
        sorted(entries, key=lambda item: (item.canonical_hash, item.packed_sha256))
    )
    manifest = TeamCorpusManifest(
        artifact_schema=CORPUS_MANIFEST_SCHEMA,
        global_contract_sha256=global_contract_sha256,
        format_id=format_id,
        corpus_hash=corpus_content_hash(ordered_entries),
        entries=ordered_entries,
        created_at=created_at,
        sampling_metadata={
            "total_candidates": len(deduped),
            "admitted_count": len(entries),
            "rejected_count": len(deduped) - len(entries),
        },
    )

    audit = audit_corpus(
        manifest,
        total_candidates=len(deduped),
        rejections_by_reason=rejections,
    )
    return manifest, audit


def write_corpus_manifest(manifest: TeamCorpusManifest, output_dir: Path | str) -> Path:
    """Write one corpus manifest into its pool directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "corpus_manifest.json"
    atomic_json_save(manifest_path, manifest.to_dict())
    return manifest_path
