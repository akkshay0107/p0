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
    """Set the pending raw event buffer on a live battle instance."""
    battle._p0_live_events = raw_events  # type: ignore[attr-defined]


def consume_raw_events(battle: DoubleBattle) -> list[RawBattleEvent]:
    """Drain and return the pending raw event buffer for a live battle instance."""
    events = _events_for(battle)
    battle._p0_live_events = []  # type: ignore[attr-defined]
    return events


def last_move(pokemon: Pokemon) -> str | None:
    """Return the ID of the last move executed by the given Pokemon, or None."""
    move = pokemon.last_move
    return None if move is None else move.id


def capture_message(battle: DoubleBattle, split_message: list[str]) -> None:
    """Capture a raw protocol line from Showdown onto the battle's live event buffer."""
    def pre_hp_for(identifier: str) -> float | None:
        try:
            return battle.get_pokemon(identifier).current_hp_fraction
        except (AssertionError, IndexError, KeyError, ValueError):
            pass

        if ":" in identifier:
            clean_id = identifier.split(":", 1)[-1].strip()
            try:
                return battle.get_pokemon(clean_id).current_hp_fraction
            except (AssertionError, IndexError, KeyError, ValueError):
                pass

        return None

    _events_for(battle).append(build_raw_event(split_message, pre_hp_for))
