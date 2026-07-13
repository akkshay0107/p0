from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.move import Move
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather

from p0.paths import DEFAULT_PATHS

CLEAN_ID_RE = re.compile(r"[^a-z0-9]")


class Resolution:
    """Knownness for categorical values; ID zero remains structural padding."""

    PAD = "pad"
    UNKNOWN = "unknown"
    KNOWN_NONE = "known_none"
    OOV = "oov"
    KNOWN = "known"


class PokemonTokenizer:
    """Vocabulary backed tokenizer for structured Pokemon observations."""

    def __init__(self, vocab: dict[str, dict[str, int]]):
        self.vocab = vocab
        self.species = vocab.get("species", {})
        self.items = vocab.get("items", {})
        self.abilities = vocab.get("abilities", {})
        self.moves = vocab.get("moves", {})

        # all neutral natures map to serious
        # just present here in case showdown still has it
        self.natures_list = [
            "serious",  # default to serious
            "adamant",
            "bashful",
            "bold",
            "brave",
            "calm",
            "careful",
            "docile",
            "gentle",
            "hardy",
            "hasty",
            "impish",
            "jolly",
            "lax",
            "lonely",
            "mild",
            "modest",
            "naive",
            "naughty",
            "quiet",
            "quirky",
            "rash",
            "relaxed",
            "sassy",
            "timid",
        ]
        self.natures = {nature: idx for idx, nature in enumerate(self.natures_list)}
        for nature in ("serious", "bashful", "docile", "hardy", "quirky"):
            self.natures[nature] = 0

        # Resolve enum members from their normalized names.
        self._volatiles_str = vocab.get("volatiles", {})
        self.volatiles = {
            effect: self._enum_vocab_id(self._volatiles_str, effect, {effect.name: effect.name})
            for effect in Effect
        }

        self._side_conditions_str = vocab.get("side_conditions", {})
        self.side_conditions = {
            condition: self._enum_vocab_id(
                self._side_conditions_str, condition, {condition.name: condition.name}
            )
            for condition in SideCondition
        }
        _weathers_str = vocab.get("weathers", {})
        self.weathers = {
            weather: self._enum_vocab_id(
                _weathers_str,
                weather,
                {
                    Weather.RAINDANCE.name: "rain",
                    Weather.SUNNYDAY.name: "sun",
                    Weather.SANDSTORM.name: "sand",
                    Weather.SNOW.name: "snow",
                },
            )
            for weather in Weather
        }
        _fields_str = vocab.get("fields", {})
        self.fields = {field: self._enum_vocab_id(_fields_str, field) for field in Field}

        _status_str = vocab.get("status", {})
        self.status = {
            status: self._enum_vocab_id(
                _status_str,
                status,
                {
                    Status.BRN.name: "burn",
                    Status.FRZ.name: "freeze",
                    Status.PAR.name: "paralysis",
                    Status.PSN.name: "poison",
                    Status.SLP.name: "sleep",
                    Status.TOX.name: "toxic",
                },
            )
            for status in Status
        }

        _types_str = vocab.get("types", {})
        self.types = {t: _types_str.get(t.name.lower(), 0) for t in PokemonType}

        _categories_str = vocab.get("categories", {})
        self.categories = {c: _categories_str.get(c.name.lower(), 0) for c in MoveCategory}

        # pre-bake the trickroom token ID so _global_field_token never does a runtime vocab lookup
        _trickroom_vocab = vocab.get("trickroom", {})
        self.trickroom_id: int = _trickroom_vocab.get("trickroom", 0)

    @classmethod
    def from_file(cls, path: str | Path | None = None) -> PokemonTokenizer:
        if path is None:
            path = DEFAULT_PATHS.data_root / "vocab.json"
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(json.load(f))

    @staticmethod
    @lru_cache(maxsize=None)
    def _cached_normalize(s: str) -> str:
        return CLEAN_ID_RE.sub("", s.lower())

    @staticmethod
    def normalize_id(name: str | None) -> str:
        if name is None:
            return ""
        return PokemonTokenizer._cached_normalize(name)

    @classmethod
    def _enum_vocab_id(
        cls,
        table: dict[str, int],
        member: object,
        aliases: dict[str, str] | None = None,
    ) -> int:
        name = getattr(member, "name", str(member))
        candidates = []
        if aliases and name in aliases:
            candidates.append(aliases[name])
        candidates.append(name)
        for candidate in candidates:
            value = table.get(cls.normalize_id(candidate))
            if value is not None:
                return value
        return 0

    def id_for(self, table: str, name: str | None) -> int:
        # Keep the legacy hot path allocation-free. Call resolve only when
        # the caller needs knownness/provenance alongside the categorical ID.
        if name is None:
            return 0
        vocab_table = self.vocab.get(table)
        if vocab_table is None:
            return 0
        return vocab_table.get(self.normalize_id(name), 0)

    def effect_id_for(self, table: str, name: str | None) -> int:
        """Resolve protocol effect names without allocating intermediate tables."""
        if name is None:
            return 0
        _, separator, remainder = name.partition(":")
        return self.id_for(table, remainder if separator else name)

    def resolve(self, table: str, name: str | None) -> tuple[int, str]:
        """Resolve a value while exposing why ID zero was returned."""
        if name is None or self.normalize_id(name) == "":
            return 0, Resolution.KNOWN_NONE
        vocab_table = self.vocab.get(table)
        if vocab_table is None:
            return 0, Resolution.UNKNOWN
        key = self.normalize_id(name)
        if key not in vocab_table:
            return 0, Resolution.OOV
        return vocab_table[key], Resolution.KNOWN

    def status_id(self, status: Status | None) -> int:
        if status is None:
            return 0
        return self.status.get(status, 0)

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

    def type_id(self, type_obj: PokemonType | None) -> int:
        if type_obj is None:
            return 0
        return self.types.get(type_obj, 0)

    def move_id(self, move: Move | None) -> int:
        if move is None:
            return 0
        return self.moves.get(move.id, 0)

    def move_type_id(self, move: Move | None) -> int:
        if move is None:
            return 0
        return self.type_id(move.type)

    def move_category_id(self, move: Move | None) -> int:
        if move is None:
            return 0
        return self.categories.get(move.category, 0)

    def nature_id(self, pokemon: Pokemon | None) -> int:
        if pokemon is None or pokemon.nature is None:
            return 0
        return self.natures.get(self.normalize_id(pokemon.nature), 0)


tokenizer = PokemonTokenizer.from_file()
