"""Battle-scoped live protocol capture for poke-env."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from functools import partial
from typing import NamedTuple

from poke_env.battle import DoubleBattle, Pokemon

from p0.battle.events import EventRecord, SpatialEventRecorder
from p0.model.tokenizer import tokenizer
from p0.replays.identity import normalize_showdown_id


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


def _transform_target(
    battle: DoubleBattle,
    base: Pokemon,
    target_value: str,
) -> Pokemon:
    """Resolve a Transform target from the protocol reference or species name."""
    try:
        return battle.get_pokemon(target_value)
    except (AssertionError, IndexError, KeyError, ValueError):
        target_species = normalize_showdown_id(target_value)
        candidates = tuple(
            pokemon
            for pokemon in (*battle.active_pokemon, *battle.opponent_active_pokemon)
            if pokemon is not None
            and pokemon is not base
            and target_species
            in {
                normalize_showdown_id(str(pokemon.species)),
                normalize_showdown_id(str(pokemon.base_species)),
            }
        )
        if len(candidates) != 1:
            raise ValueError(
                f"Transform species {target_value!r} resolved to {len(candidates)} active targets"
            )
        return candidates[0]


def transform_target_reference(
    battle: DoubleBattle,
    base: Pokemon,
    target_value: str,
) -> str:
    """Return a standard active-slot reference for a Transform target."""
    target = _transform_target(battle, base, target_value)
    player_role = battle.player_role
    if player_role not in {"p1", "p2"}:
        raise ValueError("Transform target resolution requires a known player role")
    opponent_role = "p2" if player_role == "p1" else "p1"
    for side, active_pokemon in (
        (player_role, battle.active_pokemon),
        (opponent_role, battle.opponent_active_pokemon),
    ):
        for slot, pokemon in enumerate(active_pokemon):
            if pokemon is target:
                return f"{side}{'ab'[slot]}: {target.name}"
    raise ValueError(f"Transform target {target_value!r} is not active")


def capture_message(
    battle: DoubleBattle,
    split_message: Sequence[str],
    *,
    capture_protocol_line: bool = False,
) -> None:
    """Update the battle's pending event records from one Showdown protocol message."""
    if capture_protocol_line:
        vars(battle).setdefault("_p0_protocol_lines", []).append(tuple(split_message))

    recorder = _recorder_for(battle)
    if len(split_message) >= 4 and split_message[1] == "-transform":
        try:
            base = battle.get_pokemon(split_message[2])
            target = _transform_target(battle, base, split_message[3])
            # Keep the copied form independent of later target changes.
            vars(battle).setdefault("_p0_transform_targets", {})[id(base)] = CapturedTransform(
                deepcopy(target), base.base_stats["hp"], base.height, base.weight
            )
        except (AssertionError, IndexError, KeyError, ValueError):
            pass

    elif len(split_message) >= 3 and split_message[1] in ("switch", "drag"):
        try:
            identifier = split_message[2]
            player_role = identifier[:2]
            slot_idx = 0 if len(identifier) > 2 and identifier[2] == "a" else 1

            active_list = None
            if player_role in {"p1", "p2"} and battle.player_role in {"p1", "p2"}:
                active_list = (
                    battle.active_pokemon
                    if player_role == battle.player_role
                    else battle.opponent_active_pokemon
                )

            if active_list and len(active_list) > slot_idx:
                old_active = active_list[slot_idx]
                targets = getattr(battle, "_p0_transform_targets", None)
                if old_active is not None and targets is not None:
                    targets.pop(id(old_active), None)
        except (AssertionError, IndexError, KeyError, ValueError):
            pass
    elif len(split_message) >= 3 and split_message[1] == "faint":
        try:
            base = battle.get_pokemon(split_message[2])
            targets = getattr(battle, "_p0_transform_targets", None)
            if targets is not None:
                targets.pop(id(base), None)
        except (AssertionError, IndexError, KeyError, ValueError):
            pass

    role = battle.player_role
    if role and role != recorder.player_role:
        recorder.player_role = role

    recorder.apply_line(split_message, tokenizer, partial(_hp_before, battle))


def _hp_before(battle: DoubleBattle, identifier: str) -> float | None:
    """Return a Pokemon's HP fraction before the current line, or None if unresolved."""
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


def captured_protocol_lines(battle: DoubleBattle) -> tuple[str, ...]:
    """Return opt-in protocol lines captured for a live battle in arrival order."""
    return tuple("|".join(parts) for parts in getattr(battle, "_p0_protocol_lines", ()))
