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

from p0.format_config import current_manifest
from p0.paths import DEFAULT_PATHS

ROOT = DEFAULT_PATHS.repository_root
DEFAULT_DEX = ROOT / "data" / "champions_dex.json"
DEFAULT_VOCAB = ROOT / "data" / "vocab.json"
DEFAULT_MANIFEST = ROOT / "data" / "runtime_manifest.json"

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
NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize(value: str) -> str:
    return NORMALIZE_RE.sub("", value.lower())


def append_keys(table: dict[str, int], keys: set[str]) -> None:
    next_id = max(table.values(), default=0) + 1
    for key in sorted(keys):
        if key not in table:
            table[key] = next_id
            next_id += 1


def enum_keys(enum: Any) -> set[str]:
    return {normalize(member.name) for member in enum if member.name != "UNKNOWN"}


def build(
    dex_path: Path,
    vocab_path: Path,
    manifest_path: Path,
    coverage_path: Path | None = None,
) -> dict[str, Any]:
    dex = json.loads(dex_path.read_text(encoding="utf-8"))
    vocab: dict[str, dict[str, int]] = {table: {} for table in TABLES}

    legal = dex.get("legality", {})
    for table in ("species", "items", "abilities", "moves"):
        append_keys(vocab[table], {normalize(identifier) for identifier in legal.get(table, [])})
    append_keys(vocab["volatiles"], enum_keys(Effect))
    append_keys(vocab["fields"], enum_keys(Field))
    append_keys(vocab["status"], enum_keys(Status))
    append_keys(vocab["side_conditions"], enum_keys(SideCondition))
    append_keys(vocab["weathers"], enum_keys(Weather))
    append_keys(vocab["trickroom"], {"trickroom"})
    append_keys(vocab["categories"], {"physical", "special", "status"})
    append_keys(
        vocab["types"],
        {
            normalize(entry)
            for entry in (
                "Normal",
                "Fire",
                "Water",
                "Electric",
                "Grass",
                "Ice",
                "Fighting",
                "Poison",
                "Ground",
                "Flying",
                "Psychic",
                "Bug",
                "Rock",
                "Ghost",
                "Dragon",
                "Dark",
                "Steel",
                "Fairy",
            )
        },
    )

    effect_tables = {
        "effect": "volatiles",
        "field": "fields",
        "side_condition": "side_conditions",
        "weather": "weathers",
        "status": "status",
    }
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

    # Keep the JSON shape consumed by the current runtime. The semantic IDs are
    # recorded in the sidecar manifest until the observation tensor schema grows
    # explicit knownness/provenance fields in Workstream C.
    vocab_path.write_text(json.dumps(vocab, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    missing_content: dict[str, list[str]] = {}
    for table in ("species", "items", "abilities", "moves", "natures"):
        dumped = {normalize(entry.get("id", entry.get("name", ""))) for entry in dex[table]}
        missing = sorted(set(legal.get(table, [])) - dumped)
        if missing:
            missing_content[table] = missing

    enum_families = {
        "effect": enum_keys(Effect),
        "field": enum_keys(Field),
        "side_condition": enum_keys(SideCondition),
        "weather": enum_keys(Weather),
        "status": enum_keys(Status),
    }
    known_protocol_ids = set().union(*enum_families.values())
    known_protocol_ids.update(
        normalize(identifier) for table in effect_tables.values() for identifier in vocab[table]
    )
    protocol_ids = {normalize(value) for value in dex.get("protocolEffects", [])}
    coverage = {
        "schemaVersion": 1,
        "missingLegalContent": missing_content,
        # Legal effects were assigned to a known namespace above and added to its table.
        "unmappedLegalEffects": [],
        "unsupportedNonlegalEffects": sorted(
            f"condition:{value}" for value in protocol_ids - known_protocol_ids
        ),
        "vocabularyTables": {name: len(values) for name, values in sorted(vocab.items())},
    }
    if coverage_path is not None:
        coverage_path.write_text(
            json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if missing_content:
        raise ValueError(f"Champions coverage audit failed: missing={missing_content}")
    manifest = current_manifest(
        vocab_path=vocab_path,
        dex_path=dex_path,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return coverage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dex", type=Path, default=DEFAULT_DEX)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--coverage",
        type=Path,
        help="Optional path for the reproducible coverage audit report",
    )
    args = parser.parse_args(argv)
    build(args.dex, args.vocab, args.manifest, args.coverage)
    print(json.dumps({"vocab": str(args.vocab), "manifest": str(args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
