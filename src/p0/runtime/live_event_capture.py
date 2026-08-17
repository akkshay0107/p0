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
        role = getattr(battle, "player_role", "p1") or "p1"
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
    role = getattr(battle, "player_role", None)
    if role and role != recorder.player_role:
        recorder.player_role = role

    recorder.apply_line(split_message, tokenizer, pre_hp_for)
    battle._p0_spatial_turn = recorder.to_records()  # type: ignore[attr-defined]
