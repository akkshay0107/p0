"""Build the fresh deterministic vocabulary and its coverage audit."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather

from p0.paths import DEFAULT_PATHS
from p0.persistence import atomic_json_save

ROOT = DEFAULT_PATHS.repository_root
DEFAULT_DEX = ROOT / "data" / "champions_dex.json"
DEFAULT_VOCAB = ROOT / "data" / "vocab.json"
DEFAULT_COVERAGE = ROOT / "data" / "champions_coverage.json"

TABLES = (
    "species",
    "items",
    "abilities",
    "moves",
    "volatiles",
    "fields",
    "status",
    "side_conditions",
    "weathers",
    "trickroom",
    "categories",
    "types",
)
RESERVED_SEMANTICS = ("PAD", "UNKNOWN", "KNOWN_NONE", "OOV")
POKEMON_TYPES = (
    "Normal Fire Water Electric Grass Ice Fighting Poison Ground "
    "Flying Psychic Bug Rock Ghost Dragon Dark Steel Fairy"
).split()
# Protocol-effect family, its vocabulary table, and the poke-env enum that names it.
EFFECT_FAMILIES = {
    "effect": ("volatiles", Effect),
    "field": ("fields", Field),
    "side_condition": ("side_conditions", SideCondition),
    "weather": ("weathers", Weather),
    "status": ("status", Status),
}
NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize(value: str) -> str:
    """Normalize a string by lowercasing and stripping non-alphanumeric characters."""
    return NORMALIZE_RE.sub("", value.lower())


def append_keys(table: dict[str, int], keys: set[str]) -> None:
    """Add new semantic keys to a vocabulary table."""
    next_id = max(table.values(), default=0) + 1
    for key in sorted(keys):
        if key not in table:
            table[key] = next_id
            next_id += 1


def enum_keys(enum: Any) -> set[str]:
    """Extract and normalize keys from an enumeration."""
    return {normalize(member.name) for member in enum if member.name != "UNKNOWN"}


def build(
    dex_path: Path,
    vocab_path: Path,
    coverage_path: Path | None = None,
) -> dict[str, Any]:
    """
    Build the vocab mapping, checking for schema and dataset coverage.

    Arguments:
        dex_path: Champions data file containing legal content and protocol IDs.
        vocab_path: Destination for the atomically written vocabulary.
        coverage_path: Optional destination for the coverage audit JSON.

    Returns:
        The generated coverage audit.
    """
    dex = json.loads(dex_path.read_text(encoding="utf-8"))
    vocab: dict[str, dict[str, int]] = {table: {} for table in TABLES}

    legal = dex.get("legality", {})
    for table in ("species", "items", "abilities", "moves"):
        append_keys(vocab[table], {normalize(identifier) for identifier in legal.get(table, [])})
    for table, enum in EFFECT_FAMILIES.values():
        append_keys(vocab[table], enum_keys(enum))
    append_keys(vocab["trickroom"], {"trickroom"})
    append_keys(vocab["categories"], {"physical", "special", "status"})
    append_keys(vocab["types"], {normalize(entry) for entry in POKEMON_TYPES})

    effect_tables = {family: table for family, (table, _) in EFFECT_FAMILIES.items()}
    legal_effects = dex.get("legalProtocolEffects", {})
    for family, identifiers in legal_effects.items():
        table = effect_tables.get(family)
        if table is None:
            raise ValueError(f"Unknown legal protocol-effect namespace: {family}")
        append_keys(vocab[table], {normalize(value) for value in identifiers})

    for table, values in vocab.items():
        if any(not isinstance(index, int) or index <= 0 for index in values.values()):
            raise ValueError(f"Vocabulary table {table!r} contains a non-positive embedding ID")
        reserved_collisions = sorted(
            set(values) & {normalize(value) for value in RESERVED_SEMANTICS}
        )
        if reserved_collisions:
            raise ValueError(
                f"Vocabulary table {table!r} collides with reserved semantics: {reserved_collisions}"
            )

    missing_content: dict[str, list[str]] = {}
    for table in ("species", "items", "abilities", "moves", "natures"):
        dumped = {normalize(entry.get("id", entry.get("name", ""))) for entry in dex[table]}
        missing = sorted(set(legal.get(table, [])) - dumped)
        if missing:
            missing_content[table] = missing

    known_protocol_ids = set().union(*(enum_keys(enum) for _, enum in EFFECT_FAMILIES.values()))
    known_protocol_ids.update(
        normalize(identifier) for table in effect_tables.values() for identifier in vocab[table]
    )

    protocol_ids = {normalize(value) for value in dex.get("protocolEffects", [])}
    coverage = {
        "missingLegalContent": missing_content,
        # Legal effects were assigned to a known namespace above and added to its table.
        "unmappedLegalEffects": [],
        "unsupportedNonlegalEffects": sorted(
            f"condition:{value}" for value in protocol_ids - known_protocol_ids
        ),
        "vocabularyTables": {name: len(values) for name, values in sorted(vocab.items())},
    }
    if missing_content:
        raise ValueError(f"Champions coverage audit failed: missing={missing_content}")

    atomic_json_save(vocab_path, vocab)
    if coverage_path is not None:
        atomic_json_save(coverage_path, coverage)
    return coverage


def main(argv: list[str] | None = None) -> int:
    """Build vocab CLI entrypoint."""
    argparse.ArgumentParser().parse_args(argv)

    build(DEFAULT_DEX, DEFAULT_VOCAB, DEFAULT_COVERAGE)

    print(
        json.dumps(
            {"vocab": str(DEFAULT_VOCAB), "coverage": str(DEFAULT_COVERAGE)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
