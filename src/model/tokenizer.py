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


class PokemonTokenizer:
    """Vocabulary backed tokenizer for structured Pokemon observations."""

    def __init__(self, vocab: dict[str, dict[str, int]]):
        self.vocab = vocab
        self.species = vocab.get("species", {})
        self.items = vocab.get("items", {})
        self.abilities = vocab.get("abilities", {})
        self.moves = vocab.get("moves", {})

        # map enum directly to id
        self._volatiles_str = vocab.get("volatiles", {})
        self.volatiles = {
            effect_enum: self._volatiles_str.get(effect_str, 0)
            for effect_enum, effect_str in {
                Effect.CONFUSION: "confusion",
                Effect.DISABLE: "disable",
                Effect.ENCORE: "encore",
                Effect.LEECH_SEED: "leechseed",
                Effect.SUBSTITUTE: "substitute",
                Effect.TAUNT: "taunt",
            }.items()
        }

        self.status = {
            status_enum: vocab.get("status", {}).get(status_str, 0)
            for status_enum, status_str in {
                Status.BRN: "burn",
                Status.FRZ: "freeze",
                Status.PAR: "paralysis",
                Status.PSN: "poison",
                Status.SLP: "sleep",
                Status.TOX: "toxic",
            }.items()
        }

        self.types = vocab.get("types", {})
        self.categories = vocab.get("categories", {})

        # fast path to avoid computing if no volatile status effect
        self._EMPTY_VOLATILES = [0] * MAX_VOLATILES

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
        return self.status.get(status, 0)

    def volatile_ids(self, effects: dict[Any, Any] | None) -> list[int]:
        if not effects:
            return self._EMPTY_VOLATILES

        ids = []
        for effect in effects.keys():
            idx = self.volatiles.get(effect)
            if idx is None:
                # fallback for unrecognized effects (for now since vocab not finalized)
                name = self.normalize_id(getattr(effect, "name", effect))
                idx = self._volatiles_str.get(name, 0)

            if idx:
                ids.append(idx)

        if not ids:
            return self._EMPTY_VOLATILES

        ids = sorted(set(ids))[:MAX_VOLATILES]
        return ids + [0] * (MAX_VOLATILES - len(ids))

    def species_id(self, pokemon: Pokemon | None) -> int:
        if pokemon is None:
            return 0
        species = pokemon.species or pokemon.base_species
        return self.species.get(self.normalize_id(species), 0)

    def ability_id(self, pokemon: Pokemon | None) -> int:
        if pokemon is None:
            return 0
        return self.abilities.get(self.normalize_id(pokemon.ability), 0)

    def item_id(self, pokemon: Pokemon | None) -> int:
        if pokemon is None:
            return 0
        return self.items.get(self.normalize_id(pokemon.item), 0)

    def type_id(self, type_obj: Any) -> int:
        if type_obj is None:
            return 0
        return self.types.get(self.normalize_id(getattr(type_obj, "name", type_obj)), 0)

    def move_id(self, move: Any) -> int:
        if move is None:
            return 0
        move_key = getattr(move, "id", move)
        return self.moves.get(self.normalize_id(move_key), 0)

    def move_type_id(self, move: Any) -> int:
        if move is None:
            return 0
        return self.types.get(self.normalize_id(getattr(move, "type", None)), 0)

    def move_category_id(self, move: Any) -> int:
        if move is None:
            return 0
        category = getattr(move, "category", None)
        return self.categories.get(self.normalize_id(getattr(category, "name", category)), 0)


tokenizer = PokemonTokenizer.from_file()
