"""Battle-scoped live protocol capture for poke-env."""

from __future__ import annotations

from poke_env.battle import DoubleBattle, Pokemon

from p0.battle.events import RawBattleEvent, build_raw_event


def _events_for(battle: DoubleBattle) -> list[RawBattleEvent]:
    try:
        return battle._p0_live_events  # type: ignore[attr-defined]
    except AttributeError:
        battle._p0_live_events = []  # type: ignore[attr-defined]
        return battle._p0_live_events  # type: ignore[attr-defined]


def set_raw_events(battle: DoubleBattle, raw_events: list[RawBattleEvent]) -> None:
    battle._p0_live_events = raw_events  # type: ignore[attr-defined]


def consume_raw_events(battle: DoubleBattle) -> list[RawBattleEvent]:
    events = _events_for(battle)
    battle._p0_live_events = []  # type: ignore[attr-defined]
    return events


def last_move(pokemon: Pokemon) -> str | None:
    move = pokemon.last_move
    return None if move is None else move.id


def capture_message(battle: DoubleBattle, split_message: list[str]) -> None:
    def pre_hp_for(identifier: str) -> float | None:
        try:
            return battle.get_pokemon(identifier).current_hp_fraction
        except (AssertionError, IndexError, KeyError, ValueError):
            return None

    _events_for(battle).append(build_raw_event(split_message, pre_hp_for))
