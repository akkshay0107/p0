"""Explicit protocol event classification and argument-shape registry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping


class EventClassification(StrEnum):
    """The disposition of one parsed protocol event."""

    NO_STATE_CHANGE = "no_state_change"
    PUBLIC_STATE = "public_state"
    ACTION_EXECUTION = "action_execution"
    BOUNDARY_SIGNAL = "boundary_signal"
    UNSUPPORTED_STATE = "unsupported_state"
    MALFORMED = "malformed"

    @property
    def rejects_replay(self) -> bool:
        """Return whether this classification rejects the complete replay."""
        return self in {self.UNSUPPORTED_STATE, self.MALFORMED}


@dataclass(frozen=True, slots=True)
class EventRule:
    """Classification and structural requirements for one protocol tag."""

    classification: EventClassification
    minimum_arguments: int
    maximum_arguments: int | None = None
    required_nonempty: tuple[int, ...] = ()
    required_pokemon_refs: tuple[int, ...] = ()
    optional_pokemon_refs: tuple[int, ...] = ()
    effect_argument: int | None = None

    def __post_init__(self) -> None:
        if self.minimum_arguments < 0:
            raise ValueError("EventRule.minimum_arguments must be nonnegative")
        if self.maximum_arguments is not None and self.maximum_arguments < self.minimum_arguments:
            raise ValueError("EventRule.maximum_arguments cannot be below the minimum")
        indices = self.required_nonempty + self.required_pokemon_refs + self.optional_pokemon_refs
        if any(index < 0 for index in indices):
            raise ValueError("EventRule argument indices must be nonnegative")
        if self.effect_argument is not None and self.effect_argument < -1:
            raise ValueError("EventRule.effect_argument must be -1 or nonnegative when present")


def _no_state_change(
    minimum: int,
    maximum: int | None = None,
    *,
    required: tuple[int, ...] = (),
) -> EventRule:
    return EventRule(
        EventClassification.NO_STATE_CHANGE,
        minimum,
        maximum,
        required_nonempty=required,
    )


def _boundary(
    minimum: int,
    maximum: int | None = None,
    *,
    required: tuple[int, ...] = (),
) -> EventRule:
    return EventRule(
        EventClassification.BOUNDARY_SIGNAL,
        minimum,
        maximum,
        required_nonempty=required,
    )


def _state(
    minimum: int,
    maximum: int | None = None,
    *,
    required: tuple[int, ...] = (),
    pokemon_refs: tuple[int, ...] = (),
    optional_pokemon_refs: tuple[int, ...] = (),
    effect: int | None = None,
) -> EventRule:
    return EventRule(
        EventClassification.PUBLIC_STATE,
        minimum,
        maximum,
        required_nonempty=required,
        required_pokemon_refs=pokemon_refs,
        optional_pokemon_refs=optional_pokemon_refs,
        effect_argument=effect,
    )


def _action(
    minimum: int,
    maximum: int | None = None,
    *,
    required: tuple[int, ...] = (),
    pokemon_refs: tuple[int, ...] = (),
    optional_pokemon_refs: tuple[int, ...] = (),
    effect: int | None = None,
) -> EventRule:
    return EventRule(
        EventClassification.ACTION_EXECUTION,
        minimum,
        maximum,
        required_nonempty=required,
        required_pokemon_refs=pokemon_refs,
        optional_pokemon_refs=optional_pokemon_refs,
        effect_argument=effect,
    )


_RULES = {
    # Transport, room, and display messages that do not change battle state.
    "j": _no_state_change(1, 1, required=(0,)),
    "J": _no_state_change(1, 1, required=(0,)),
    "l": _no_state_change(1, 1, required=(0,)),
    "L": _no_state_change(1, 1, required=(0,)),
    "n": _no_state_change(2, 2, required=(0, 1)),
    "html": _no_state_change(1, 1),
    "uhtml": _no_state_change(2, 2, required=(0,)),
    "uhtmlchange": _no_state_change(2, 2, required=(0,)),
    "raw": _no_state_change(1, 1),
    "warning": _no_state_change(1, 1),
    "inactive": _no_state_change(1, 1),
    "inactiveoff": _no_state_change(0, 1),
    "t:": _no_state_change(1, 1, required=(0,)),
    "timestamp": _no_state_change(1, 1, required=(0,)),
    "rated": _no_state_change(0, 1),
    "rule": _no_state_change(1, 1, required=(0,)),
    "tier": _no_state_change(1, 1, required=(0,)),
    "gen": _no_state_change(1, 1, required=(0,)),
    "gametype": _no_state_change(1, 1, required=(0,)),
    "message": _no_state_change(1, 1),
    "-message": _no_state_change(1, 1),
    "-hint": _no_state_change(1, 1),
    "-center": _state(0, 0),
    "-crit": _no_state_change(1, 2, required=(0,)),
    "-supereffective": _no_state_change(1, 2, required=(0,)),
    "-resisted": _no_state_change(1, 2, required=(0,)),
    # Scheduling and phase evidence. A bare separator is handled separately.
    "teampreview": _boundary(0, 1),
    "turn": _boundary(1, 1, required=(0,)),
    "upkeep": _boundary(0, 0),
    "win": _boundary(1, 1, required=(0,)),
    "tie": _boundary(0, 0),
    "forfeit": _boundary(1, 1, required=(0,)),
    # Public initialization and roster facts.
    "player": _state(4, 4, required=(0, 1)),
    "clearpoke": _state(0, 0),
    "poke": _state(2, 3, required=(0, 1)),
    "showteam": _state(2, None, required=(0, 1)),
    "teamsize": _state(2, 2, required=(0, 1)),
    "start": _state(0, 0),
    # Executions and action metadata. Choice inference happens in a later stage.
    "move": _action(3, None, required=(0, 1), pokemon_refs=(0,), optional_pokemon_refs=(2,)),
    "switch": _action(3, None, required=(0, 1, 2), pokemon_refs=(0,)),
    "drag": _action(3, None, required=(0, 1, 2), pokemon_refs=(0,)),
    "replace": _action(3, None, required=(0, 1, 2), pokemon_refs=(0,)),
    "cant": _action(2, None, required=(0, 1), pokemon_refs=(0,), effect=1),
    "-anim": _action(3, None, required=(0, 1), pokemon_refs=(0,), optional_pokemon_refs=(2,)),
    "-prepare": _action(2, None, required=(0, 1), pokemon_refs=(0,), effect=1),
    "-combine": _action(0, None),
    "-waiting": _action(2, None, pokemon_refs=(0, 1)),
    "-miss": _action(1, 2, pokemon_refs=(0,), optional_pokemon_refs=(1,)),
    "-fail": _action(1, None, pokemon_refs=(0,), effect=1),
    "-notarget": _action(0, 1, optional_pokemon_refs=(0,)),
    "-nothing": _action(0, 0),
    # Deterministic public battle-state transitions.
    "faint": _state(1, 1, pokemon_refs=(0,)),
    "detailschange": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "swap": _state(2, None, required=(1,), pokemon_refs=(0,)),
    "-damage": _state(2, None, required=(1,), pokemon_refs=(0,)),
    "-heal": _state(2, None, required=(1,), pokemon_refs=(0,)),
    "-sethp": _state(2, None, required=(1,), pokemon_refs=(0,)),
    "-status": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-curestatus": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-cureteam": _state(1, None, pokemon_refs=(0,)),
    "-boost": _state(3, None, required=(1, 2), pokemon_refs=(0,), effect=1),
    "-unboost": _state(3, None, required=(1, 2), pokemon_refs=(0,), effect=1),
    "-setboost": _state(3, None, required=(1, 2), pokemon_refs=(0,), effect=1),
    "-swapboost": _state(2, None, pokemon_refs=(0, 1)),
    "-copyboost": _state(2, None, pokemon_refs=(0, 1)),
    "-clearboost": _state(1, None, pokemon_refs=(0,)),
    "-clearallboost": _state(0, None),
    "-clearpositiveboost": _state(3, None, pokemon_refs=(0, 1), effect=2),
    "-clearnegativeboost": _state(1, None, pokemon_refs=(0,)),
    "-invertboost": _state(1, None, pokemon_refs=(0,)),
    "-item": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-enditem": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-ability": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-endability": _state(1, None, pokemon_refs=(0,), effect=1),
    "-formechange": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-mega": _state(2, None, required=(1,), pokemon_refs=(0,), effect=-1),
    "-primal": _state(1, None, pokemon_refs=(0,)),
    "-burst": _state(3, None, required=(1, 2), pokemon_refs=(0,), effect=2),
    "-zpower": _state(1, None, pokemon_refs=(0,)),
    "-zbroken": _state(1, None, pokemon_refs=(0,)),
    "-terastallize": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-transform": _state(2, None, pokemon_refs=(0, 1)),
    "-typechange": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-typeadd": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-start": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-end": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-singleturn": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-singlemove": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-weather": _state(1, None, required=(0,), effect=0),
    "-fieldstart": _state(1, None, required=(0,), effect=0),
    "-fieldend": _state(1, None, required=(0,), effect=0),
    "-fieldactivate": _state(1, None, required=(0,), effect=0),
    "-sidestart": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-sideend": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-swapsideconditions": _state(0, None),
    "-activate": _state(1, None, required=(0,), effect=0),
    "-block": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-immune": _state(1, None, pokemon_refs=(0,)),
    "-eat": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-hitcount": _state(2, None, required=(1,), pokemon_refs=(0,)),
    "-mustrecharge": _state(1, None, pokemon_refs=(0,)),
}

CLASSIFICATION_REGISTRY: Mapping[str, EventRule] = MappingProxyType(_RULES)


__all__ = ["CLASSIFICATION_REGISTRY", "EventClassification", "EventRule"]
