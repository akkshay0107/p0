"""Explicit protocol event classification and argument-shape registry."""

from __future__ import annotations

from enum import StrEnum
from functools import partial
from types import MappingProxyType
from typing import Mapping, NamedTuple


class EventClassification(StrEnum):
    """The disposition of one parsed protocol event."""

    NO_STATE_CHANGE = "no_state_change"
    PUBLIC_STATE = "public_state"
    BOUNDARY_SIGNAL = "boundary_signal"
    UNSUPPORTED_STATE = "unsupported_state"
    MALFORMED = "malformed"

    @property
    def rejects_replay(self) -> bool:
        """Return whether this classification rejects the complete replay."""
        return self in {self.UNSUPPORTED_STATE, self.MALFORMED}


class EventRule(NamedTuple):
    """Classification and argument shape for one protocol tag."""

    classification: EventClassification
    minimum_arguments: int
    maximum_arguments: int | None = None
    # Indices of arguments that must be present and nonempty.
    required: tuple[int, ...] = ()
    # Indices of arguments that must be Pokemon references.
    pokemon_refs: tuple[int, ...] = ()
    optional_pokemon_refs: tuple[int, ...] = ()
    # Index of the argument that names the effect; -1 is the last argument.
    effect: int | None = None


# A tag is _state exactly when the reducer has a handler for it; state.py checks this on import.
_no_state_change = partial(EventRule, EventClassification.NO_STATE_CHANGE)
_state = partial(EventRule, EventClassification.PUBLIC_STATE)


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
    "bigerror": _no_state_change(1, 1, required=(0,)),
    # OHKO is emitted as a bare informational marker after the target is
    # reduced to zero HP; the target is intentionally not repeated.
    "-ohko": _no_state_change(0, 0),
    # A hint can report that a delayed move missed, which clears its pending state.
    "-hint": _state(1, 1),
    "-crit": _no_state_change(1, 2, required=(0,)),
    "-supereffective": _no_state_change(1, 2, required=(0,)),
    "-resisted": _no_state_change(1, 2, required=(0,)),
    # Scheduling and phase evidence. A bare separator is handled separately.
    "teampreview": _no_state_change(0, 1),
    "turn": _state(1, 1, required=(0,)),
    "upkeep": _state(0, 0),
    "win": _no_state_change(1, 1, required=(0,)),
    "tie": _no_state_change(0, 0),
    "forfeit": _no_state_change(1, 1, required=(0,)),
    # Public initialization and roster facts. The roster comes from the open team sheets.
    "player": _no_state_change(2, 4, required=(0,)),
    "clearpoke": _no_state_change(0, 0),
    "poke": _no_state_change(2, 3, required=(0, 1)),
    "showteam": _no_state_change(2, None, required=(0, 1)),
    "teamsize": _state(2, 2, required=(0, 1)),
    "start": _no_state_change(0, 0),
    # Executions and action metadata. Choice inference happens in a later stage.
    # A missing target is valid protocol for some moves and is retained as
    # partial evidence by the decision stage.
    "move": _state(2, None, required=(0, 1), pokemon_refs=(0,), optional_pokemon_refs=(2,)),
    "switch": _state(3, None, required=(0, 1, 2), pokemon_refs=(0,)),
    "drag": _state(3, None, required=(0, 1, 2), pokemon_refs=(0,)),
    "replace": _state(2, 2, required=(0, 1), pokemon_refs=(0,)),
    "cant": _state(2, None, required=(0, 1), pokemon_refs=(0,), effect=1),
    "-anim": _no_state_change(
        3, None, required=(0, 1), pokemon_refs=(0,), optional_pokemon_refs=(2,)
    ),
    "-prepare": _state(2, None, required=(0, 1), pokemon_refs=(0,), effect=1),
    "-miss": _no_state_change(1, 2, pokemon_refs=(0,), optional_pokemon_refs=(1,)),
    "-fail": _state(1, None, pokemon_refs=(0,), effect=1),
    # Deterministic public battle-state transitions.
    "faint": _state(1, 1, pokemon_refs=(0,)),
    "detailschange": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "swap": _state(2, None, required=(1,), pokemon_refs=(0,)),
    "-damage": _state(2, None, required=(1,), pokemon_refs=(0,)),
    "-heal": _state(2, None, required=(1,), pokemon_refs=(0,)),
    # One Pokémon/HP pair followed by free-form annotations ([from], [silent]).
    "-sethp": _state(2, None, required=(1,), pokemon_refs=(0,)),
    "-status": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-curestatus": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-boost": _state(3, None, required=(1, 2), pokemon_refs=(0,), effect=1),
    "-unboost": _state(3, None, required=(1, 2), pokemon_refs=(0,), effect=1),
    "-setboost": _state(3, None, required=(1, 2), pokemon_refs=(0,), effect=1),
    "-swapboost": _state(3, None, required=(2,), pokemon_refs=(0, 1)),
    "-copyboost": _state(2, None, pokemon_refs=(0, 1)),
    "-clearboost": _state(1, None, pokemon_refs=(0,)),
    "-clearallboost": _state(0, None),
    "-clearnegativeboost": _state(1, None, pokemon_refs=(0,)),
    "-invertboost": _state(1, None, pokemon_refs=(0,)),
    "-item": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-enditem": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-ability": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-endability": _state(1, None, pokemon_refs=(0,), effect=1),
    "-formechange": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-mega": _state(2, None, required=(1,), pokemon_refs=(0,), effect=-1),
    "-zbroken": _no_state_change(1, None, pokemon_refs=(0,)),
    "-transform": _state(2, None, required=(1,), pokemon_refs=(0, 1)),
    "-start": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-end": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-singleturn": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-singlemove": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-weather": _state(1, None, required=(0,), effect=0),
    "-fieldstart": _state(1, None, required=(0,), effect=0),
    "-fieldend": _state(1, None, required=(0,), effect=0),
    "-fieldactivate": _no_state_change(1, None, required=(0,), effect=0),
    "-sidestart": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-sideend": _state(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-swapsideconditions": _state(0, None),
    "-activate": _state(1, None, required=(0,), effect=0),
    "-block": _no_state_change(2, None, required=(1,), pokemon_refs=(0,), effect=1),
    "-immune": _no_state_change(1, None, pokemon_refs=(0,)),
    "-hitcount": _no_state_change(2, None, required=(1,), pokemon_refs=(0,)),
    "-mustrecharge": _state(1, None, pokemon_refs=(0,)),
}

CLASSIFICATION_REGISTRY: Mapping[str, EventRule] = MappingProxyType(_RULES)


__all__ = [
    "CLASSIFICATION_REGISTRY",
    "EventClassification",
    "EventRule",
]
