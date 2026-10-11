"""Translation between the pure action layout and poke-env orders."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from poke_env.battle import DoubleBattle
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
    ForfeitBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)
from poke_env.player.player import Player

from p0.battle.actions import (
    ActionKind,
    decode_action,
    team_selection,
)
from p0.runtime.poke_env_battle_adapter import action_move_slots


def action_to_order(
    action: npt.NDArray[np.int64],
    battle: DoubleBattle,
    fake: bool = False,
    strict: bool = True,
) -> BattleOrder:
    """
    Translate a policy joint action vector into a poke-env BattleOrder.

    PokeEnv.step passes fake and strict on every call. p0 leaves both at
    poke-env's defaults, so every order is validated and an invalid action raises.
    """
    if battle.teampreview:
        selection = team_selection(int(action[0]), int(action[1]), len(battle.team))
        return SingleBattleOrder("/team " + "".join(str(index + 1) for index in selection))

    act0, act1 = int(action[0]), int(action[1])
    if act0 == -2 and act1 == -2:
        return DefaultBattleOrder()

    if act0 == -1 or act1 == -1:
        return ForfeitBattleOrder()

    first = action_to_single_order(act0, battle, fake, 0)
    second = action_to_single_order(act1, battle, fake, 1)
    joined = DoubleBattleOrder.join_orders([first], [second])
    if not joined:
        raise ValueError(
            f"Invalid action {action} from player {battle.player_username} in battle "
            f"{battle.battle_tag}: converted orders {first} and {second} are incompatible"
        )

    return joined[0]


def action_to_single_order(
    action: int, battle: DoubleBattle, fake: bool, position: int
) -> SingleBattleOrder:
    """Translate a single-slot action integer into a SingleBattleOrder."""
    if action == -2:
        return DefaultBattleOrder()

    semantic = decode_action(action)
    if semantic.kind is ActionKind.PASS:
        order: SingleBattleOrder = PassBattleOrder()

    elif semantic.kind is ActionKind.SWITCH:
        try:
            pokemon = tuple(battle.team.values())[semantic.switch_slot]
        except IndexError as error:
            raise ValueError(f"Switch slot {semantic.switch_slot} is outside the team") from error

        order = Player.create_order(pokemon)

    else:
        active = battle.active_pokemon[position]
        if active is None:
            raise ValueError(f"Action {action} specifies a move for empty position {position}")

        if semantic.kind is ActionKind.FORCED_MOVE:
            try:
                move = battle.available_moves[position][0]
            except IndexError as error:
                raise ValueError(f"No forced move is available at position {position}") from error

            # A locked move may list targets; Showdown ignores the one it is sent.
            target = battle.get_possible_showdown_targets(move, active)[0]
            order = Player.create_order(move, move_target=target, mega=semantic.mega)

        else:
            moves = action_move_slots(battle, position)
            try:
                move = moves[semantic.move_slot]
            except IndexError as error:
                raise ValueError(
                    f"Move slot {semantic.move_slot} is outside available moves {moves}"
                ) from error

            order = Player.create_order(
                move,
                move_target=semantic.target,
                mega=semantic.mega,
            )

    if not fake:
        if str(order) not in map(str, battle.valid_orders[position]):
            raise ValueError(
                f"Order {order} for action {action} is not valid for position {position}"
            )

    return order
