"""Battle-scoped live protocol capture for poke-env."""

from __future__ import annotations

from collections.abc import Sequence

from poke_env.battle import DoubleBattle, Pokemon

from p0.battle.events import SpatialTurnRecorder
from p0.model.tokenizer import tokenizer


def _recorder_for(battle: DoubleBattle) -> SpatialTurnRecorder:
    try:
        return battle._p0_spatial_recorder  # type: ignore[attr-defined]
    except AttributeError:
        role = battle.player_role or "p1"
        recorder = SpatialTurnRecorder(player_role=role)
        battle._p0_spatial_recorder = recorder  # type: ignore[attr-defined]
        return recorder


def last_move(pokemon: Pokemon) -> str | None:
    """Return the ID of the last move executed by the given Pokemon, or None."""
    move = pokemon.last_move
    return None if move is None else move.id


def capture_message(battle: DoubleBattle, split_message: Sequence[str]) -> None:
    """Capture a raw protocol line from Showdown onto the battle's live spatial recorder."""
    if len(split_message) >= 2 and split_message[1] == "turn":
        recorder = _recorder_for(battle)
        recorder.reset_turn()

    if len(split_message) >= 4 and split_message[1] == "-transform":
        try:
            base = battle.get_pokemon(split_message[2])
            target = battle.get_pokemon(split_message[3])
            targets = getattr(battle, "_p0_transform_targets", None)
            if targets is None:
                targets = {}
                battle._p0_transform_targets = targets  # type: ignore[attr-defined]
            targets[id(base)] = target
        except (AssertionError, IndexError, KeyError, ValueError):
            pass
            # will run into a bunch of silent fallbacks where the tensor state
            # maps everything to unknown slot, since ditto wont have the moves
            # of the pokemon it copied.

    elif len(split_message) >= 3 and split_message[1] in ("switch", "drag"):
        try:
            # When a pokemon switches in, the one currently in its slot switches out.
            # We clear the transform target of the pokemon switching out.
            identifier = split_message[2]
            player_role = identifier[:2]
            slot_idx = 0 if len(identifier) > 2 and identifier[2] == "a" else 1

            is_p1 = player_role == "p1"
            is_p2 = player_role == "p2"

            active_list = None
            if is_p1 and battle.player_role == "p1" or is_p2 and battle.player_role == "p2":
                active_list = battle.active_pokemon
            elif is_p1 and battle.player_role == "p2" or is_p2 and battle.player_role == "p1":
                active_list = battle.opponent_active_pokemon

            if active_list and len(active_list) > slot_idx:
                old_active = active_list[slot_idx]
                targets = getattr(battle, "_p0_transform_targets", None)
                if old_active is not None and targets is not None:
                    targets.pop(id(old_active), None)
        except (AssertionError, IndexError, KeyError, ValueError):
            pass
    elif len(split_message) >= 3 and split_message[1] == "faint":
        try:
            # Faint refers exactly to the pokemon fainting.
            base = battle.get_pokemon(split_message[2])
            targets = getattr(battle, "_p0_transform_targets", None)
            if targets is not None:
                targets.pop(id(base), None)
        except (AssertionError, IndexError, KeyError, ValueError):
            pass

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

    recorder = _recorder_for(battle)
    role = battle.player_role
    if role and role != recorder.player_role:
        recorder.player_role = role

    recorder.apply_line(split_message, tokenizer, pre_hp_for)
    battle._p0_spatial_turn = recorder.to_records()  # type: ignore[attr-defined]
