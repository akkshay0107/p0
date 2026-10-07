"""Validated in-memory resources shared by tokenization, observation, and models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache, lru_cache, wraps
from typing import Any

import orjson

from p0.format_config import load_active_global_contract
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
    load_active_global_contract()
    dex_path = DEFAULT_PATHS.data_root / "champions_dex.json"
    dex = orjson.loads(dex_path.read_bytes())
    return RuntimeResources.from_data(tokenizer.vocab, dex, shared_tokenizer=tokenizer)


def cache_default_dex[T](
    build: Callable[[Mapping[str, Any]], T],
) -> Callable[[Mapping[str, Any]], T]:
    """Reuse default-dex tables per process; build them read-only, since callers share them."""
    default = cache(lambda: build(default_runtime_resources().dex))

    @wraps(build)
    def lookup(dex: Mapping[str, Any]) -> T:
        # Custom data must not load default resources or reuse their tables.
        if (
            default_runtime_resources.cache_info().currsize
            and dex is default_runtime_resources().dex
        ):
            return default()
        return build(dex)

    return lookup
