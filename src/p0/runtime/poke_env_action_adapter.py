"""Translation between the pure action layout and poke-env orders."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from poke_env.battle import DoubleBattle, Pokemon
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
    ForfeitBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)
from poke_env.player.player import Player

from p0.battle.actions import ActionCodec, ActionKind, SlotAction


class PokeEnvOrderAdapter:
    """Thin live-boundary adapter; policy hot paths continue to use integers."""

    @classmethod
    def action_to_order(
        cls,
        action: npt.NDArray[np.int64],
        battle: DoubleBattle,
        fake: bool = False,
        strict: bool = True,
    ) -> BattleOrder:
        if battle.teampreview:
            selection = ActionCodec.team_selection(int(action[0]), int(action[1]), len(battle.team))
            return SingleBattleOrder("/team " + "".join(str(index + 1) for index in selection))
        if int(action[0]) == -2 and int(action[1]) == -2:
            return DefaultBattleOrder()
        if int(action[0]) == -1 or int(action[1]) == -1:
            return ForfeitBattleOrder()
        try:
            first = cls.action_to_single_order(int(action[0]), battle, fake, 0)
            second = cls.action_to_single_order(int(action[1]), battle, fake, 1)
            joined = DoubleBattleOrder.join_orders([first], [second])
            if joined:
                return joined[0]
            raise ValueError(
                f"Invalid action {action} from player {battle.player_username} in battle "
                f"{battle.battle_tag}: converted orders {first} and {second} are incompatible"
            )
        except ValueError as error:
            if strict:
                raise
            if battle.logger is not None:
                battle.logger.warning("%s; defaulting to a random move", error)
            return Player.choose_random_doubles_move(battle)

    @staticmethod
    def action_to_single_order(
        action: int, battle: DoubleBattle, fake: bool, position: int
    ) -> SingleBattleOrder:
        if action == -2:
            return DefaultBattleOrder()
        semantic = ActionCodec.decode(action)
        if semantic.kind is ActionKind.PASS:
            order: SingleBattleOrder = PassBattleOrder()
        elif semantic.kind is ActionKind.SWITCH:
            try:
                pokemon = tuple(battle.team.values())[semantic.switch_slot]
            except IndexError as error:
                raise ValueError(
                    f"Switch slot {semantic.switch_slot} is outside the team"
                ) from error
            order = Player.create_order(pokemon)
        else:
            active = battle.active_pokemon[position]
            if active is None:
                raise ValueError(f"Action {action} specifies a move for empty position {position}")
            if semantic.kind is ActionKind.FORCED_MOVE:
                try:
                    move = battle.available_moves[position][0]
                except IndexError as error:
                    raise ValueError(
                        f"No forced move is available at position {position}"
                    ) from error
                order = Player.create_order(move, mega=semantic.mega)
            else:
                moves = tuple(active.moves.values())
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
            valid_orders = {str(valid_order) for valid_order in battle.valid_orders[position]}
            if str(order) not in valid_orders:
                raise ValueError(
                    f"Order {order} for action {action} is not in action space {valid_orders}"
                )
        return order

    @classmethod
    def order_to_action(
        cls,
        order: BattleOrder,
        battle: DoubleBattle,
        fake: bool = False,
        strict: bool = True,
    ) -> npt.NDArray[np.int64]:
        if battle.teampreview:
            digits = tuple(int(char) - 1 for char in order.message[6:] if char.isdigit())
            return np.asarray(
                ActionCodec.canonical_team_actions(digits, len(battle.team)), dtype=np.int64
            )
        if isinstance(order, DefaultBattleOrder):
            return np.array([-2, -2], dtype=np.int64)
        if isinstance(order, ForfeitBattleOrder):
            return np.array([-1, -1], dtype=np.int64)
        if not isinstance(order, DoubleBattleOrder):
            raise TypeError(f"Expected a doubles order, got {type(order).__name__}")
        try:
            if not fake and not DoubleBattleOrder.join_orders(
                [order.first_order], [order.second_order]
            ):
                raise ValueError(f"Orders in {order} are incompatible")
            return np.array(
                [
                    cls.single_order_to_action(order.first_order, battle, fake, 0),
                    cls.single_order_to_action(order.second_order, battle, fake, 1),
                ],
                dtype=np.int64,
            )
        except ValueError as error:
            if strict:
                raise
            if battle.logger is not None:
                battle.logger.warning("%s; defaulting to a random move", error)
            return cls.order_to_action(
                Player.choose_random_doubles_move(battle), battle, fake, True
            )

    @staticmethod
    def single_order_to_action(
        order: SingleBattleOrder, battle: DoubleBattle, fake: bool, position: int
    ) -> np.int64:
        if isinstance(order.order, str):
            if isinstance(order, DefaultBattleOrder):
                return np.int64(-2)
            if not isinstance(order, PassBattleOrder):
                raise ValueError(f"Unsupported string order {order}")
            return np.int64(0)
        if not fake:
            valid_orders = {str(valid_order) for valid_order in battle.valid_orders[position]}
            if str(order) not in valid_orders:
                raise ValueError(f"Order {order} is not in action space {valid_orders}")
        if isinstance(order.order, Pokemon):
            species = tuple(pokemon.base_species for pokemon in battle.team.values())
            return np.int64(species.index(order.order.base_species) + 1)
        active = battle.active_pokemon[position]
        if active is None:
            raise ValueError(f"Move order targets empty position {position}")
        available = battle.available_moves[position]
        if len(available) == 1 and available[0].id in {"struggle", "recharge"}:
            return np.int64(ActionCodec.encode(ActionCodec.decode(47 if order.mega else 48)))
        moves: tuple[Any, ...] = tuple(active.moves.values())
        move_slot = tuple(move.id for move in moves).index(order.order.id)
        return np.int64(
            ActionCodec.encode(
                SlotAction(
                    ActionKind.MOVE, move_slot=move_slot, target=order.move_target, mega=order.mega
                )
            )
        )
