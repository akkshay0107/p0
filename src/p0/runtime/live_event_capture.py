"""Battle-scoped live protocol capture for poke-env."""

from __future__ import annotations

from dataclasses import dataclass, field

from p0.battle.events import RawBattleEvent

_STATE_ATTRIBUTE = "_p0_live_event_state"


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
    value = getattr(pokemon, "last_move", None)
    move_id = getattr(value, "id", None)
    return move_id if isinstance(move_id, str) else None


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
    state_for(battle).raw_events.append(RawBattleEvent(tuple(split_message), pre_hp))
