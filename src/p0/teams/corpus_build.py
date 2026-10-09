"""Build, validate, and audit the immutable team corpus."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from p0.format_config import FORMAT
from p0.model.tokenizer import PokemonTokenizer, Resolution
from p0.paths import DEFAULT_PATHS
from p0.persistence import atomic_json_save
from p0.teams.corpus import CORPUS_MANIFEST_NAME, TeamCorpus
from p0.teams.team import TeamRecord, deduplicate_variants
from p0.teams.validation import AdmissionResult, validate_many


def audit_corpus(
    corpus: TeamCorpus,
    *,
    total_candidates: int | None = None,
    rejections_by_reason: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Compute exact content-coverage metrics across all teams in a corpus."""
    species_set: set[str] = set()
    move_set: set[str] = set()
    item_set: set[str] = set()

    packed_teams = [packed for group in corpus.teams for packed in group]
    for packed in packed_teams:
        for part in packed.split("]"):
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

    admitted = len(packed_teams)
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


def build_corpus(
    variants: Sequence[TeamRecord],
    *,
    tokenizer: PokemonTokenizer | None = None,
    validator: Callable[..., Sequence[AdmissionResult]] = validate_many,
    format_id: str = FORMAT.battle_format,
) -> tuple[TeamCorpus, dict[str, Any]]:
    """
    Admit, deduplicate, validate, and audit candidate team variants.

    Arguments:
        variants: Candidate teams.
        tokenizer: Vocabulary used to reject out-of-vocabulary team content.
        validator: Callable that validates the deduplicated candidates.
        format_id: Battle format associated with the corpus.

    Returns:
        The team corpus and its coverage/rejection audit.
    """
    if tokenizer is None:
        tokenizer = PokemonTokenizer.from_file(DEFAULT_PATHS.data_root / "vocab.json")

    deduped = deduplicate_variants(variants)
    validation_results = validator(deduped)
    if len(validation_results) != len(deduped):
        raise RuntimeError("Validation result count does not match deduplicated variant count")

    packed_by_team: dict[str, list[str]] = {}
    rejections: Counter[str] = Counter()

    for variant, result in zip(deduped, validation_results, strict=True):
        packed = result.packed_team
        if not result.valid or not packed:
            rejections[
                f"showdown_invalid: {result.problems[0]}" if result.problems else "showdown_invalid"
            ] += 1
            continue

        reason = _check_vocabulary(tokenizer, variant)
        if reason is not None:
            rejections[reason] += 1
            continue

        packed_by_team.setdefault(variant.team.team_key, []).append(packed)

    if not packed_by_team:
        raise ValueError(f"No teams were admitted to the corpus; rejections: {dict(rejections)}")

    corpus = TeamCorpus(
        format_id=format_id,
        teams=tuple(tuple(sorted(packed_by_team[key])) for key in sorted(packed_by_team)),
    )
    audit = audit_corpus(
        corpus,
        total_candidates=len(deduped),
        rejections_by_reason=rejections,
    )
    return corpus, audit


def write_corpus_manifest(corpus: TeamCorpus, output_dir: Path | str) -> Path:
    """Write one corpus manifest into its pool directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / CORPUS_MANIFEST_NAME
    atomic_json_save(manifest_path, corpus.to_dict())
    return manifest_path
