"""Build the append-only vocabulary from the pinned Champions dex and enums."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from poke_env.battle.effect import Effect
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather
from src.format_config import current_manifest

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEX = ROOT / "data" / "champions_dex.json"
DEFAULT_VOCAB = ROOT / "data" / "vocab.json"
DEFAULT_MANIFEST = ROOT / "data" / "runtime_manifest.json"

TABLES = ("species", "items", "abilities", "moves", "volatiles", "status", "side_conditions", "weathers", "trickroom", "categories", "types")
RESERVED = {"PAD": 0, "UNKNOWN": -1, "KNOWN_NONE": -2, "OOV": -3}
NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize(value: str) -> str:
    return NORMALIZE_RE.sub("", value.lower())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_keys(table: dict[str, int], keys: set[str]) -> None:
    next_id = max(table.values(), default=0) + 1
    for key in sorted(keys):
        if key not in table:
            table[key] = next_id
            next_id += 1


def enum_keys(enum: Any) -> set[str]:
    return {normalize(member.name) for member in enum}


def build(dex_path: Path, vocab_path: Path, manifest_path: Path) -> None:
    dex = json.loads(dex_path.read_text(encoding="utf-8"))
    vocab: dict[str, dict[str, int]] = {
        table: {}
        for table in TABLES
    }

    for table in ("species", "items", "abilities", "moves"):
        append_keys(vocab[table], {normalize(entry["id"]) for entry in dex[table] if entry.get("id")})
    append_keys(vocab["volatiles"], enum_keys(Effect))
    append_keys(vocab["status"], enum_keys(Status))
    append_keys(vocab["side_conditions"], enum_keys(SideCondition))
    append_keys(vocab["weathers"], enum_keys(Weather))
    append_keys(vocab["trickroom"], {"trickroom"})
    append_keys(vocab["categories"], {"physical", "special", "status"})
    append_keys(vocab["types"], {normalize(entry) for entry in (
        "Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison",
        "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy",
    )})

    # Keep the JSON shape consumed by the current runtime. The semantic IDs are
    # recorded in the sidecar manifest until the observation tensor schema grows
    # explicit knownness/provenance fields in Workstream C.
    vocab_path.write_text(json.dumps(vocab, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = current_manifest(
        vocab_path=vocab_path,
        dex_path=dex_path,
    )
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"vocab": str(vocab_path), "manifest": str(manifest_path), "vocab_sha256": manifest.vocab_sha256}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dex", type=Path, default=DEFAULT_DEX)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    build(args.dex, args.vocab, args.manifest)
