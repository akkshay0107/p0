"""Battle-scoped live protocol capture for poke-env."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from functools import partial
from typing import NamedTuple

from poke_env.battle import DoubleBattle, Pokemon

from p0.battle.events import EventRecord, SpatialEventRecorder
from p0.model.tokenizer import tokenizer


class CapturedTransform(NamedTuple):
    target: Pokemon
    original_base_hp: int
    original_height: float
    original_weight: float


def _recorder_for(battle: DoubleBattle) -> SpatialEventRecorder:
    try:
        return battle._p0_spatial_recorder  # type: ignore[attr-defined]
    except AttributeError:
        role = battle.player_role or "p1"
        recorder = SpatialEventRecorder(player_role=role)
        battle._p0_spatial_recorder = recorder  # type: ignore[attr-defined]
        return recorder


def last_move(pokemon: Pokemon) -> str | None:
    """Return the ID of the last move executed by the given Pokemon, or None."""
    move = pokemon.last_move
    return None if move is None else move.id


def capture_message(battle: DoubleBattle, split_message: Sequence[str]) -> None:
    """Update the battle's pending event records from one Showdown protocol message."""
    recorder = _recorder_for(battle)
    if len(split_message) >= 4 and split_message[1] == "-transform":
        base = battle.get_pokemon(split_message[2])
        target = battle.get_pokemon(split_message[3])
        # Keep the copied form independent of later target changes.
        vars(battle).setdefault("_p0_transform_targets", {})[id(base)] = CapturedTransform(
            deepcopy(target), base.base_stats["hp"], base.height, base.weight
        )
    elif len(split_message) >= 3 and split_message[1] in ("switch", "drag"):
        identifier = split_message[2]
        player_role = identifier[:2]
        if player_role in {"p1", "p2"} and battle.player_role in {"p1", "p2"}:
            active = (
                battle.active_pokemon
                if player_role == battle.player_role
                else battle.opponent_active_pokemon
            )
            old_active = active[0 if identifier[2] == "a" else 1]
            if old_active is not None:
                getattr(battle, "_p0_transform_targets", {}).pop(id(old_active), None)
    elif len(split_message) >= 3 and split_message[1] == "faint":
        base = battle.get_pokemon(split_message[2])
        getattr(battle, "_p0_transform_targets", {}).pop(id(base), None)

    role = battle.player_role
    if role and role != recorder.player_role:
        recorder.player_role = role

    recorder.apply_line(split_message, tokenizer, partial(_hp_before, battle))


def _hp_before(battle: DoubleBattle, identifier: str) -> float | None:
    """Return a Pokemon's HP fraction before the current line, or None if unresolved."""
    try:
        return battle.get_pokemon(identifier).current_hp_fraction
    except (AssertionError, IndexError, KeyError, ValueError):
        return None


def pending_events(battle: DoubleBattle) -> tuple[EventRecord, ...]:
    """Return the records of the interval that this player's next decision observes."""
    return _recorder_for(battle).pending()


def consume_events(battle: DoubleBattle) -> None:
    """
    Close the event interval when this player submits a decision.

    Showdown sends no battle events between a request and the player's choice,
    so the records consumed here are exactly those the decision observed.
    """
    _recorder_for(battle).consume()


def is_retry(battle: DoubleBattle) -> bool:
    """
    Return whether a decision requested now retries this player's previous one.

    Showdown answers every accepted choice with battle lines before the next
    request, so no line since the last submission means the choice was rejected.
    """
    return _recorder_for(battle).consumed
