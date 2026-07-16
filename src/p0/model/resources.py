"""Validated in-memory resources shared by tokenization, observation, and models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from p0.format_config import load_active_runtime_manifest
from p0.model.tokenizer import PokemonTokenizer, tokenizer
from p0.paths import DEFAULT_PATHS


@dataclass(frozen=True, slots=True)
class RuntimeResources:
    vocab: dict[str, dict[str, int]]
    dex: dict[str, Any]
    tokenizer: PokemonTokenizer
    mega_items: frozenset[str]
    mega_forms: frozenset[str]

    @classmethod
    def from_files(cls, vocab_path: str | Path, dex_path: str | Path) -> RuntimeResources:
        with Path(vocab_path).open("r", encoding="utf-8") as stream:
            vocab = json.load(stream)
        with Path(dex_path).open("r", encoding="utf-8") as stream:
            dex = json.load(stream)
        if not isinstance(vocab, dict) or not isinstance(dex, dict):
            raise ValueError("Runtime vocabulary and dex roots must be objects")
        return cls.from_data(vocab, dex)

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> RuntimeResources:
        path = Path(manifest_path)
        load_active_runtime_manifest(path)
        vocab_path = path.with_name("vocab.json")
        dex_path = path.with_name("champions_dex.json")
        # Dex and Showdown identities are provenance. Updated mechanics remain loadable
        # as long as the resource feature ABI and vocabulary are unchanged.
        return cls.from_files(vocab_path, dex_path)

    @classmethod
    def from_data(
        cls,
        vocab: dict[str, dict[str, int]],
        dex: dict[str, Any],
        *,
        shared_tokenizer: PokemonTokenizer | None = None,
    ) -> RuntimeResources:
        required_vocab = {"species", "items", "abilities", "moves", "types", "categories"}
        required_dex = {"species", "items", "abilities", "moves", "transformations"}
        missing_vocab = sorted(required_vocab - vocab.keys())
        missing_dex = sorted(required_dex - dex.keys())
        if missing_vocab or missing_dex:
            raise ValueError(
                f"Incomplete runtime resources: vocab={missing_vocab}, dex={missing_dex}"
            )
        transformations = dex["transformations"]
        mega_items = frozenset(
            PokemonTokenizer.normalize_id(item)
            for entry in transformations
            if entry.get("isMega")
            for item in entry.get("requiredItems", ())
        )
        mega_forms = frozenset(
            PokemonTokenizer.normalize_id(entry["id"])
            for entry in transformations
            if entry.get("isMega")
        )
        return cls(
            vocab=vocab,
            dex=dex,
            tokenizer=shared_tokenizer or PokemonTokenizer(vocab),
            mega_items=mega_items,
            mega_forms=mega_forms,
        )


@lru_cache(maxsize=1)
def default_runtime_resources() -> RuntimeResources:
    manifest_path = DEFAULT_PATHS.data_root / "runtime_manifest.json"
    load_active_runtime_manifest(manifest_path)
    dex_path = manifest_path.with_name("champions_dex.json")
    with dex_path.open("r", encoding="utf-8") as stream:
        dex = json.load(stream)
    return RuntimeResources.from_data(tokenizer.vocab, dex, shared_tokenizer=tokenizer)
