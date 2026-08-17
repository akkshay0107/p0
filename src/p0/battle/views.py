"""Player-relative structural views consumed by pure battle services."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from p0.battle.events import SpatialSlotRecord
from p0.battle.legality import DecisionView


class NamedEffectView(Protocol):
    """Enum-like protocol value (status, weather, field, volatile, type, ...)."""

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


class TransformedMoveView:
    def __init__(self, base_move: MoveView):
        self._base = base_move

    @property
    def id(self) -> str:
        return self._base.id

    @property
    def type(self) -> Any:
        return self._base.type

    @property
    def category(self) -> Any:
        return self._base.category

    @property
    def current_pp(self) -> int | None:
        return 5

    @property
    def max_pp(self) -> int | None:
        return 5


class TransformedPokemonView:
    def __init__(
        self,
        base: PokemonView,
        target: PokemonView,
        moves: Mapping[str, Any] | None = None,
    ):
        self._base = base
        self._target = target
        source_moves = moves if moves is not None else target.moves
        self._transformed_moves = {k: TransformedMoveView(v) for k, v in source_moves.items()}

    def __hash__(self) -> int:
        return hash(self._base)

    def __eq__(self, other: Any) -> bool:
        return self._base == other or self is other

    @property
    def species(self) -> str | None:
        return self._target.species

    @property
    def base_species(self) -> str:
        return self._target.base_species

    @property
    def ability(self) -> str | None:
        return self._target.ability

    @property
    def item(self) -> str | None:
        return self._base.item

    @property
    def nature(self) -> str | None:
        return self._base.nature

    @property
    def moves(self) -> Mapping[str, Any]:
        return self._transformed_moves

    @property
    def type_1(self) -> Any:
        return self._target.type_1

    @property
    def type_2(self) -> Any:
        return self._target.type_2

    @property
    def status(self) -> Any:
        return self._base.status

    @property
    def base_stats(self) -> Mapping[str, int]:
        return self._target.base_stats

    @property
    def stats(self) -> Mapping[str, int | None] | None:
        return self._target.stats

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
    def available_moves(self) -> Sequence[Sequence[Any]]: ...

    @property
    def available_switches(self) -> Sequence[Sequence[Any]]: ...

    @property
    def can_mega_evolve(self) -> Sequence[bool]: ...

    @property
    def force_switch(self) -> Sequence[bool]: ...

    @property
    def trapped(self) -> Sequence[bool]: ...

    @property
    def maybe_trapped(self) -> Sequence[bool]: ...

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
    def spatial_turn(self) -> Sequence[SpatialSlotRecord]: ...

    def get_pokemon(self, identifier: str) -> Any: ...

    def last_move(self, pokemon: Any) -> str | None: ...


@dataclass(slots=True)
class FixtureBattleView:
    """Small concrete view for replay reconstruction and pure tests."""

    team: Mapping[str, Any]
    opponent_team: Mapping[str, Any]
    active_pokemon: Sequence[Any | None]
    opponent_active_pokemon: Sequence[Any | None]
    available_moves: Sequence[Sequence[Any]]
    available_switches: Sequence[Sequence[Any]]
    can_mega_evolve: Sequence[bool]
    force_switch: Sequence[bool]
    trapped: Sequence[bool]
    maybe_trapped: Sequence[bool]
    teampreview: bool
    player_role: str | None
    wait: bool
    weather: Mapping[Any, int]
    fields: Mapping[Any, int]
    side_conditions: Mapping[Any, int]
    opponent_side_conditions: Mapping[Any, int]
    turn: int
    used_mega_evolve: bool
    opponent_used_mega_evolve: bool
    decision: DecisionView
    identifiers: Mapping[str, Any] = field(default_factory=dict)
    spatial_turn: Sequence[SpatialSlotRecord] = field(
        default_factory=lambda: tuple(SpatialSlotRecord() for _ in range(4))
    )
    stat_cache: dict[Any, Any] = field(default_factory=dict)

    def get_pokemon(self, identifier: str) -> Any:
        return self.identifiers[identifier]

    def last_move(self, pokemon: Any) -> str | None:
        move = pokemon.last_move
        return None if move is None else move.id
