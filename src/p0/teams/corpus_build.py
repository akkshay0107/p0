import hashlib
import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from p0.format_config import FORMAT, current_manifest
from p0.model.tokenizer import PokemonTokenizer, Resolution
from p0.paths import DEFAULT_PATHS
from p0.teams.corpus import (
    CORPUS_MANIFEST_SCHEMA,
    CorpusEntry,
    CorpusSplit,
    TeamCorpusManifest,
    corpus_content_hash,
)
from p0.teams.stat_points import STAT_POINT_LIMIT, STAT_POINT_TOTAL_LIMIT
from p0.teams.team import TeamRecord, deduplicate_variants
from p0.teams.validation import AdmissionResult, validate_many


def assign_split(
    variant: TeamRecord,
    series_to_split: dict[str, CorpusSplit],
    ratio_train: float = 0.8,
    ratio_val: float = 0.1,
    ratio_test: float = 0.1,
    held_out_tags: tuple[str, ...] = (),
) -> CorpusSplit:
    """Deterministic, series-leak-free split assignment."""
    if any(tag in held_out_tags for tag in variant.metadata.archetype_tags):
        return CorpusSplit.HELD_OUT_ARCHETYPE

    for series in variant.metadata.source_series:
        if series in series_to_split:
            return series_to_split[series]

    if variant.metadata.source_series:
        seed_key = ",".join(sorted(variant.metadata.source_series))
    else:
        seed_key = variant.team.team_hash

    digest = hashlib.sha256(seed_key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10000

    train_cutoff = int(ratio_train * 10000)
    val_cutoff = train_cutoff + int(ratio_val * 10000)

    if bucket < train_cutoff:
        split = CorpusSplit.TRAIN
    elif bucket < val_cutoff:
        split = CorpusSplit.VALIDATION
    else:
        split = CorpusSplit.TEST

    for series in variant.metadata.source_series:
        series_to_split[series] = split

    return split


def audit_corpus(manifest: TeamCorpusManifest) -> dict[str, Any]:
    """Compute exact content-coverage metrics across all entries in a manifest."""
    species_set: set[str] = set()
    move_set: set[str] = set()
    item_set: set[str] = set()
    split_counts: dict[str, int] = {}
    archetype_counts: dict[str, int] = {}

    for entry in manifest.entries:
        split_name = entry.split.name
        split_counts[split_name] = split_counts.get(split_name, 0) + 1
        for tag in entry.archetype_tags:
            archetype_counts[tag] = archetype_counts.get(tag, 0) + 1

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

    return {
        "total_candidates": len(manifest.entries),
        "admitted_count": len(manifest.entries),
        "rejected_count": 0,
        "rejections_by_reason": {},
        "species_coverage": tuple(sorted(species_set)),
        "move_coverage": tuple(sorted(move_set)),
        "item_coverage": tuple(sorted(item_set)),
        "split_counts": split_counts,
        "archetype_counts": archetype_counts,
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
    runtime_contract_sha256: str = "",
    format_id: str = FORMAT.battle_format,
    ratio_train: float = 0.8,
    ratio_val: float = 0.1,
    ratio_test: float = 0.1,
    held_out_tags: tuple[str, ...] = (),
    created_at: str | None = None,
) -> tuple[TeamCorpusManifest, dict[str, Any]]:
    """Admit, deduplicate, validate, and audit candidate team variants."""
    if tokenizer is None:
        tokenizer = PokemonTokenizer.from_file(DEFAULT_PATHS.data_root / "vocab.json")
    if not runtime_contract_sha256:
        runtime_contract_sha256 = current_manifest().runtime_contract_sha256

    deduped = deduplicate_variants(variants)
    validation_results = validator(deduped)
    if len(validation_results) != len(deduped):
        raise RuntimeError("Validation result count does not match deduplicated variant count")

    series_to_split: dict[str, CorpusSplit] = {}
    entries: list[CorpusEntry] = []
    rejections: dict[str, int] = {}

    species_set: set[str] = set()
    move_set: set[str] = set()
    item_set: set[str] = set()
    split_counts: dict[str, int] = {}
    archetype_counts: dict[str, int] = {}

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

        split = assign_split(
            variant,
            series_to_split,
            ratio_train=ratio_train,
            ratio_val=ratio_val,
            ratio_test=ratio_test,
            held_out_tags=held_out_tags,
        )
        packed = result.packed_team
        packed_sha256 = hashlib.sha256(packed.encode("utf-8")).hexdigest()

        try:
            entry = CorpusEntry(
                canonical_hash=variant.team.team_hash,
                packed=packed,
                packed_sha256=packed_sha256,
                split=split,
                usage_count=variant.metadata.usage_count,
                archetype_tags=variant.metadata.archetype_tags,
                spread_provenance=variant.spread_provenance,
            )
        except ValueError as exc:
            reason = f"entry_error: {exc}"
            rejections[reason] = rejections.get(reason, 0) + 1
            continue

        entries.append(entry)
        split_name = split.name
        split_counts[split_name] = split_counts.get(split_name, 0) + 1
        for tag in entry.archetype_tags:
            archetype_counts[tag] = archetype_counts.get(tag, 0) + 1

        for member in variant.team.members:
            species_set.add(PokemonTokenizer.normalize_id(member.species))
            if member.item:
                item_set.add(PokemonTokenizer.normalize_id(member.item))
            for move in member.moves:
                move_set.add(PokemonTokenizer.normalize_id(move))

    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()

    ordered_entries = tuple(
        sorted(entries, key=lambda item: (item.canonical_hash, item.packed_sha256))
    )
    manifest = TeamCorpusManifest(
        artifact_schema=CORPUS_MANIFEST_SCHEMA,
        runtime_contract_sha256=runtime_contract_sha256,
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

    audit = {
        "total_candidates": len(deduped),
        "admitted_count": len(entries),
        "rejected_count": len(deduped) - len(entries),
        "rejections_by_reason": rejections,
        "species_coverage": tuple(sorted(species_set)),
        "move_coverage": tuple(sorted(move_set)),
        "item_coverage": tuple(sorted(item_set)),
        "split_counts": split_counts,
        "archetype_counts": archetype_counts,
    }

    return manifest, audit


def populate_pool_directories(
    manifest: TeamCorpusManifest,
    output_root: Path | str,
    reduced_limit: int = 64,
) -> None:
    """Populate durable teams/all and teams/reduced pool directories."""
    if reduced_limit < 1:
        raise ValueError("reduced_limit must be a positive integer")
    root = Path(output_root)
    all_dir = root / "all"
    reduced_dir = root / "reduced"
    all_dir.mkdir(parents=True, exist_ok=True)
    reduced_dir.mkdir(parents=True, exist_ok=True)

    all_manifest_path = all_dir / "corpus_manifest.json"
    all_manifest_path.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    ordered_reduced = tuple(
        sorted(
            manifest.entries,
            key=lambda entry: (-entry.usage_count, entry.canonical_hash, entry.packed_sha256),
        )[:reduced_limit]
    )

    reduced_manifest = TeamCorpusManifest(
        artifact_schema=manifest.artifact_schema,
        runtime_contract_sha256=manifest.runtime_contract_sha256,
        format_id=manifest.format_id,
        corpus_hash=corpus_content_hash(ordered_reduced),
        entries=ordered_reduced,
        created_at=manifest.created_at,
        sampling_metadata={
            **dict(manifest.sampling_metadata),
            "pool_kind": "reduced",
            "reduced_limit": reduced_limit,
        },
    )

    reduced_manifest_path = reduced_dir / "corpus_manifest.json"
    reduced_manifest_path.write_text(
        json.dumps(reduced_manifest.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
