"""Battle, team, and Pokémon views used by the battle engine and observation builder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from p0.battle.events import EventRecord
from p0.battle.legality import DecisionView


class NamedEffectView(Protocol):
    """Protocol for named effects like weather, status, and field conditions."""

    @property
    def name(self) -> str: ...


class MoveView(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def type(self) -> Any: ...

    @property
    def category(self) -> Any: ...

    @property
    def current_pp(self) -> int | None: ...

    @property
    def max_pp(self) -> int | None: ...


class PokemonView(Protocol):
    @property
    def species(self) -> str | None: ...

    @property
    def base_species(self) -> str: ...

    @property
    def ability(self) -> str | None: ...

    @property
    def item(self) -> str | None: ...

    @property
    def nature(self) -> str | None: ...

    @property
    def moves(self) -> Mapping[str, Any]: ...

    @property
    def type_1(self) -> Any: ...

    @property
    def type_2(self) -> Any: ...

    @property
    def status(self) -> Any: ...

    @property
    def base_stats(self) -> Mapping[str, int]: ...

    @property
    def stats(self) -> Mapping[str, int | None] | None: ...

    @property
    def boosts(self) -> Mapping[str, int]: ...

    @property
    def current_hp_fraction(self) -> float: ...

    @property
    def protect_counter(self) -> int: ...

    @property
    def first_turn(self) -> bool: ...

    @property
    def weight(self) -> float: ...

    @property
    def fainted(self) -> bool: ...

    @property
    def revealed(self) -> bool: ...

    @property
    def selected_in_teampreview(self) -> bool | None: ...

    @property
    def effects(self) -> Mapping[Any, int]: ...

    @property
    def status_counter(self) -> int: ...

    @property
    def preparing(self) -> Any: ...

    @property
    def last_move(self) -> MoveView | None: ...

    @property
    def level(self) -> int | None: ...

    @property
    def types(self) -> Sequence[Any]: ...


class TransformedPokemonView:
    """Pokemon view overlay representing the target species while retaining user identity."""

    __slots__ = ("_base", "_target", "_original_base_hp")

    def __init__(
        self,
        base: PokemonView,
        target: PokemonView,
        original_base_hp: int | None = None,
    ) -> None:
        self._base = base
        self._target = target
        self._original_base_hp = (
            base.base_stats["hp"] if original_base_hp is None else original_base_hp
        )

    def __hash__(self) -> int:
        return hash(self._base)

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        if isinstance(other, TransformedPokemonView):
            return self._base == other._base
        return self._base == other

    @property
    def species(self) -> str | None:
        return self._target.species

    @property
    def base_species(self) -> str:
        return self._target.base_species

    @property
    def ability(self) -> str | None:
        return self._base.ability

    @property
    def item(self) -> str | None:
        return self._base.item

    @property
    def nature(self) -> str | None:
        return self._base.nature

    @property
    def moves(self) -> Mapping[str, Any]:
        return self._base.moves

    @property
    def type_1(self) -> Any:
        return self._base.type_1

    @property
    def type_2(self) -> Any:
        return self._base.type_2

    @property
    def status(self) -> Any:
        return self._base.status

    @property
    def base_stats(self) -> Mapping[str, int]:
        return {**self._target.base_stats, "hp": self._original_base_hp}

    @property
    def stats(self) -> Mapping[str, int | None] | None:
        target_stats = self._target.stats
        own_stats = self._base.stats
        if target_stats is None:
            return own_stats
        return {**target_stats, "hp": None if own_stats is None else own_stats.get("hp")}

    @property
    def boosts(self) -> Mapping[str, int]:
        return self._base.boosts

    @property
    def current_hp_fraction(self) -> float:
        return self._base.current_hp_fraction

    @property
    def protect_counter(self) -> int:
        return self._base.protect_counter

    @property
    def first_turn(self) -> bool:
        return self._base.first_turn

    @property
    def weight(self) -> float:
        return self._target.weight

    @property
    def fainted(self) -> bool:
        return self._base.fainted

    @property
    def revealed(self) -> bool:
        return self._base.revealed

    @property
    def selected_in_teampreview(self) -> bool | None:
        return self._base.selected_in_teampreview

    @property
    def effects(self) -> Mapping[Any, int]:
        return self._base.effects

    @property
    def status_counter(self) -> int:
        return self._base.status_counter

    @property
    def preparing(self) -> Any:
        return self._base.preparing

    @property
    def last_move(self) -> MoveView | None:
        return self._base.last_move

    @property
    def level(self) -> int | None:
        return self._base.level

    # poke-env reads the next three members when it lists move targets. The format
    # has no Dynamax and no Terastallization.
    is_dynamaxed = False
    is_terastallized = False
    tera_type = None

    @property
    def types(self) -> Sequence[Any]:
        """Expose type changes applied after Transform to move-target logic."""
        return self._base.types


class FieldView(Protocol):
    @property
    def weather(self) -> Mapping[Any, int]: ...

    @property
    def fields(self) -> Mapping[Any, int]: ...

    @property
    def side_conditions(self) -> Mapping[Any, int]: ...

    @property
    def opponent_side_conditions(self) -> Mapping[Any, int]: ...

    @property
    def turn(self) -> int: ...

    @property
    def used_mega_evolve(self) -> bool: ...

    @property
    def opponent_used_mega_evolve(self) -> bool: ...


class BattleView(FieldView, Protocol):
    @property
    def team(self) -> Mapping[str, Any]: ...

    @property
    def opponent_team(self) -> Mapping[str, Any]: ...

    @property
    def active_pokemon(self) -> Sequence[Any | None]: ...

    @property
    def opponent_active_pokemon(self) -> Sequence[Any | None]: ...

    @property
    def available_switches(self) -> Sequence[Sequence[Any]]: ...

    @property
    def can_mega_evolve(self) -> Sequence[bool]: ...

    @property
    def teampreview(self) -> bool: ...

    @property
    def player_role(self) -> str | None: ...

    @property
    def wait(self) -> bool: ...

    @property
    def decision(self) -> DecisionView: ...

    @property
    def stat_cache(self) -> dict[Any, Any]: ...

    @property
    def spatial_events(self) -> Sequence[EventRecord]: ...

    def last_move(self, pokemon: Any) -> str | None: ...
