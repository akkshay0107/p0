from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from poke_env.battle.effect import Effect
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.status import Status

from src.model.structured_observation import MAX_VOLATILES

CLEAN_ID_RE = re.compile(r"[^a-z0-9]")

STATUS_TO_ID = {
    Status.BRN: "burn",
    Status.FRZ: "freeze",
    Status.PAR: "paralysis",
    Status.PSN: "poison",
    Status.SLP: "sleep",
    Status.TOX: "toxic",
}

EFFECT_TO_ID = {
    Effect.CONFUSION: "confusion",
    Effect.DISABLE: "disable",
    Effect.ENCORE: "encore",
    Effect.LEECH_SEED: "leechseed",
    Effect.SUBSTITUTE: "substitute",
    Effect.TAUNT: "taunt",
}


class PokemonTokenizer:
    """Vocabulary backed tokenizer for structured Pokemon observations."""

    def __init__(self, vocab: dict[str, dict[str, int]]):
        self.vocab = vocab
        self.species = vocab.get("species", {})
        self.items = vocab.get("items", {})
        self.abilities = vocab.get("abilities", {})
        self.moves = vocab.get("moves", {})
        self.volatiles = vocab.get("volatiles", {})
        self.status = vocab.get("status", {})
        self.types = vocab.get("types", {})
        self.categories = vocab.get("categories", {})

    @classmethod
    def from_file(cls, path: str | Path | None = None) -> PokemonTokenizer:
        if path is None:
            path = Path(__file__).resolve().parents[2] / "data" / "vocab.json"
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(json.load(f))

    @staticmethod
    @lru_cache(maxsize=None)
    def _cached_normalize(s: str) -> str:
        return CLEAN_ID_RE.sub("", s.lower())

    @staticmethod
    def normalize_id(name: Any) -> str:
        if name is None:
            return ""
        return PokemonTokenizer._cached_normalize(str(name))

    def id_for(self, table: str, name: Any) -> int:
        vocab_table = self.vocab.get(table, {})
        return vocab_table.get(self.normalize_id(name), 0)

    def status_id(self, status: Status | None) -> int:
        if status is None:
            return 0
        return self.status.get(STATUS_TO_ID.get(status, ""), 0)

    def volatile_ids(self, effects: dict[Any, Any] | None) -> list[int]:
        ids = []
        for effect in (effects or {}).keys():
            name = EFFECT_TO_ID.get(effect, self.normalize_id(getattr(effect, "name", effect)))
            idx = self.volatiles.get(name, 0)
            if idx:
                ids.append(idx)
        ids = sorted(set(ids))[:MAX_VOLATILES]
        return ids + [0] * (MAX_VOLATILES - len(ids))

    def species_id(self, pokemon: Pokemon | None) -> int:
        if pokemon is None:
            return 0
        species = getattr(pokemon, "species", None) or getattr(pokemon, "base_species", None)
        return self.id_for("species", species)

    def ability_id(self, pokemon: Pokemon | None) -> int:
        return 0 if pokemon is None else self.id_for("abilities", getattr(pokemon, "ability", None))

    def item_id(self, pokemon: Pokemon | None) -> int:
        return 0 if pokemon is None else self.id_for("items", getattr(pokemon, "item", None))

    def type_id(self, type_obj: Any) -> int:
        if type_obj is None:
            return 0
        return self.types.get(self.normalize_id(getattr(type_obj, "name", type_obj)), 0)

    def move_id(self, move: Any) -> int:
        move_key = getattr(move, "id", move)
        return self.id_for("moves", move_key)

    def move_type_id(self, move: Any) -> int:
        return self.type_id(getattr(move, "type", None))

    def move_category_id(self, move: Any) -> int:
        if move is None:
            return 0
        category = getattr(move, "category", None)
        return self.categories.get(self.normalize_id(getattr(category, "name", category)), 0)


tokenizer = PokemonTokenizer.from_file()
