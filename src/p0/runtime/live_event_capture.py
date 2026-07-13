"""Battle-scoped live protocol capture for poke-env."""

from __future__ import annotations

from dataclasses import dataclass, field

from p0.model.event_builder import RawBattleEvent

_STATE_ATTRIBUTE = "_p0_live_event_state"
_LAST_MOVE_ATTRIBUTE = "_p0_last_move"


@dataclass(slots=True)
class LiveEventState:
    raw_events: list[RawBattleEvent] = field(default_factory=list)


def state_for(battle: object) -> LiveEventState:
    state = getattr(battle, _STATE_ATTRIBUTE, None)
    if state is None:
        state = LiveEventState()
        setattr(battle, _STATE_ATTRIBUTE, state)
    return state


def set_raw_events(battle: object, raw_events: list[RawBattleEvent]) -> None:
    state_for(battle).raw_events = raw_events


def consume_raw_events(battle: object) -> list[RawBattleEvent]:
    state = state_for(battle)
    events = state.raw_events
    state.raw_events = []
    return events


def last_move(pokemon: object) -> str | None:
    value = getattr(pokemon, _LAST_MOVE_ATTRIBUTE, None)
    return value if isinstance(value, str) else None


def clear_last_move(pokemon: object) -> None:
    if hasattr(pokemon, _LAST_MOVE_ATTRIBUTE):
        delattr(pokemon, _LAST_MOVE_ATTRIBUTE)


def capture_message(battle: object, split_message: list[str]) -> None:
    pre_hp = None
    pokemon = None
    if len(split_message) > 2 and split_message[1] in {"-damage", "-heal", "move"}:
        try:
            pokemon = battle.get_pokemon(split_message[2])  # type: ignore[attr-defined]
        except (AssertionError, IndexError, KeyError, ValueError):
            pokemon = None
    if pokemon is not None and split_message[1] in {"-damage", "-heal"}:
        pre_hp = pokemon.current_hp_fraction
    if pokemon is not None and len(split_message) > 3 and split_message[1] == "move":
        setattr(pokemon, _LAST_MOVE_ATTRIBUTE, split_message[3])
    state_for(battle).raw_events.append(RawBattleEvent(tuple(split_message), pre_hp))
