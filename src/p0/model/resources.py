"""Validated in-memory resources shared by tokenization, observation, and models."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import orjson

from p0.model.tokenizer import PokemonTokenizer, tokenizer
from p0.paths import DEFAULT_PATHS


@dataclass(frozen=True, slots=True)
class RuntimeResources:
    vocab: dict[str, dict[str, int]]
    dex: dict[str, Any]
    tokenizer: PokemonTokenizer
    mega_forms: frozenset[str]
    # Species id to the id of its base species (Vivillon-Sandstorm to vivillon).
    species_bases: dict[str, str]
    # Mega Stone id to the ids of the species that can use it.
    mega_species_by_item: dict[str, frozenset[str]]

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
        mega_forms = frozenset(
            PokemonTokenizer.normalize_id(entry["id"])
            for entry in transformations
            if entry.get("isMega")
        )
        species_bases = {
            entry["id"]: PokemonTokenizer.normalize_id(entry["baseSpecies"])
            for entry in dex["species"]
        }
        mega_species_by_item = {
            PokemonTokenizer.normalize_id(item["id"]): frozenset(
                PokemonTokenizer.normalize_id(species) for species in item["megaStone"]
            )
            for item in dex["items"]
            if item.get("megaStone")
        }
        return cls(
            vocab=vocab,
            dex=dex,
            tokenizer=shared_tokenizer or PokemonTokenizer(vocab),
            mega_forms=mega_forms,
            species_bases=species_bases,
            mega_species_by_item=mega_species_by_item,
        )


@lru_cache(maxsize=1)
def default_runtime_resources() -> RuntimeResources:
    dex_path = DEFAULT_PATHS.data_root / "champions_dex.json"
    dex = orjson.loads(dex_path.read_bytes())
    return RuntimeResources.from_data(tokenizer.vocab, dex, shared_tokenizer=tokenizer)
