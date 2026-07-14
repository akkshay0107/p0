from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from p0.battle.views import MoveView, NamedEffectView, PokemonView
from p0.paths import DEFAULT_PATHS

CLEAN_ID_RE = re.compile(r"[^a-z0-9]")


class _EnumIdTable(dict["NamedEffectView | str", int]):
    """Vocabulary IDs cached per enum-like member, resolved from member names.

    Members are resolved lazily so this module never has to import the runtime
    enum classes; after the first resolution a lookup is a plain dict hit,
    identical in cost to a precomputed table.
    """

    def __init__(self, table: dict[str, int], aliases: dict[str, str] | None = None):
        super().__init__()
        self._table = table
        self._aliases = aliases or {}

    def __missing__(self, member: NamedEffectView | str) -> int:
        name = member if isinstance(member, str) else member.name
        alias = self._aliases.get(name)
        value = self._table.get(PokemonTokenizer.normalize_id(alias)) if alias else None
        if value is None:
            value = self._table.get(PokemonTokenizer.normalize_id(name), 0)
        self[member] = value
        return value


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

        # Enum members resolve lazily from their normalized names.
        self.volatiles = _EnumIdTable(vocab.get("volatiles", {}))
        self.side_conditions = _EnumIdTable(vocab.get("side_conditions", {}))
        self.weathers = _EnumIdTable(
            vocab.get("weathers", {}),
            {
                "RAINDANCE": "rain",
                "SUNNYDAY": "sun",
                "SANDSTORM": "sand",
                "SNOW": "snow",
            },
        )
        self.fields = _EnumIdTable(vocab.get("fields", {}))
        self.status = _EnumIdTable(
            vocab.get("status", {}),
            {
                "BRN": "burn",
                "FRZ": "freeze",
                "PAR": "paralysis",
                "PSN": "poison",
                "SLP": "sleep",
                "TOX": "toxic",
            },
        )
        self.types = _EnumIdTable(vocab.get("types", {}))
        self.categories = _EnumIdTable(vocab.get("categories", {}))

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

    def status_id(self, status: NamedEffectView | None) -> int:
        if status is None:
            return 0
        return self.status[status]

    def species_id(self, pokemon: PokemonView | None) -> int:
        if pokemon is None:
            return 0
        species = pokemon.species or pokemon.base_species
        return self.species.get(self.normalize_id(species), 0)

    def ability_id(self, pokemon: PokemonView | None) -> int:
        if pokemon is None:
            return 0
        return self.abilities.get(self.normalize_id(pokemon.ability), 0)

    def item_id(self, pokemon: PokemonView | None) -> int:
        if pokemon is None:
            return 0
        return self.items.get(self.normalize_id(pokemon.item), 0)

    def type_id(self, type_obj: NamedEffectView | None) -> int:
        if type_obj is None:
            return 0
        return self.types[type_obj]

    def move_id(self, move: MoveView | None) -> int:
        if move is None:
            return 0
        return self.moves.get(move.id, 0)

    def move_type_id(self, move: MoveView | None) -> int:
        if move is None:
            return 0
        return self.type_id(move.type)

    def move_category_id(self, move: MoveView | None) -> int:
        if move is None:
            return 0
        return self.categories[move.category]

    def nature_id(self, pokemon: PokemonView | None) -> int:
        if pokemon is None or pokemon.nature is None:
            return 0
        return self.natures.get(self.normalize_id(pokemon.nature), 0)


tokenizer = PokemonTokenizer.from_file()
