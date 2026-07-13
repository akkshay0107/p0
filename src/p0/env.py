import asyncio
import logging
from pathlib import Path
from time import perf_counter
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box, MultiDiscrete
from poke_env.battle import AbstractBattle, DoubleBattle, Pokemon
from poke_env.environment.env import PokeEnv, _EnvPlayer
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
    ForfeitBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)
from poke_env.player.player import Player
from poke_env.ps_client import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    ServerConfiguration,
)
from poke_env.ps_client.ps_client import PSClient
from poke_env.teambuilder import Teambuilder

from p0.format_config import FORMAT
from p0.model import observation_builder
from p0.model.structured_observation import StructuredObservation
from p0.team_picker import load_team_pool

ACT_SIZE = FORMAT.action_size


# patch poke_env's PSClient to avoid login timeouts
async def patched_wait_for_login(self, checking_interval: float = 0.1, wait_for: int = 30):
    start = perf_counter()
    while perf_counter() - start < wait_for:
        await asyncio.sleep(checking_interval)
        if self.logged_in.is_set():
            return
    assert self.logged_in.is_set(), f"Expected {self.username} to be logged in."


PSClient.wait_for_login = patched_wait_for_login


def _build_tp_mask(team_size: int) -> list[int]:
    # since both actions can lie between 0 and ACT_SIZE (47)
    # first action used to decide lead
    # second action used to decide back
    # (p1, p2) [index in team list] choice gets encoded as 6*p1 + p2
    mask = [0] * 2 * ACT_SIZE
    for action in range(36):
        p1 = action // 6 + 1
        p2 = action % 6 + 1
        if p1 < p2:
            mask[action] = 1  # for lead
            mask[ACT_SIZE + action] = 1  # for back
    return mask


_TEAM_PREVIEW_MASK = _build_tp_mask(6)  # never using a team that doesnt have 6 mons


# filter out logging about internal mismatch
_original_handler_handle = logging.Handler.handle


def _patched_handler_handle(self, record):
    if record.msg and "is active, but it's not" in str(record.msg):
        return False
    return _original_handler_handle(self, record)


logging.Handler.handle = _patched_handler_handle


class VGCEnvPlayer(_EnvPlayer):
    async def _handle_battle_request(
        self, battle: AbstractBattle, maybe_default_order: bool = False
    ):
        if battle.teampreview:
            await self.battle_queue.async_put(battle)
            order = await self.order_queue.async_get()
            await self.ps_client.send_message(order.message, battle.battle_tag)
        else:
            await super()._handle_battle_request(battle, maybe_default_order)


# modified from poke-env
# to remove all other gimmicks but mega evolution
class MegaEnv(PokeEnv[npt.NDArray[np.int64]]):
    def __init__(
        self,
        account_configuration1: Optional[AccountConfiguration] = None,
        account_configuration2: Optional[AccountConfiguration] = None,
        avatar: Optional[int] = None,
        battle_format: str = FORMAT.battle_format,
        log_level: Optional[int] = None,
        save_replays: Union[bool, str] = False,
        server_configuration: Optional[ServerConfiguration] = LocalhostServerConfiguration,
        accept_open_team_sheet: Optional[bool] = True,
        start_timer_on_battle_start: bool = False,
        start_listening: bool = True,
        open_timeout: Optional[float] = 10.0,
        ping_interval: Optional[float] = 20.0,
        ping_timeout: Optional[float] = 20.0,
        challenge_timeout: Optional[float] = 60.0,
        team: Optional[Union[str, Teambuilder]] = None,
        fake: bool = False,
        strict: bool = True,
    ):
        super().__init__(
            account_configuration1=account_configuration1,
            account_configuration2=account_configuration2,
            avatar=avatar,
            battle_format=battle_format,
            log_level=log_level,
            save_replays=save_replays,
            server_configuration=server_configuration,
            accept_open_team_sheet=accept_open_team_sheet,
            start_timer_on_battle_start=start_timer_on_battle_start,
            start_listening=start_listening,
            open_timeout=open_timeout,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            challenge_timeout=challenge_timeout,
            team=team,
            choose_on_teampreview=True,
            fake=fake,
            strict=strict,
        )
        # monkey patch
        self.agent1.__class__ = VGCEnvPlayer
        self.agent2.__class__ = VGCEnvPlayer

        self.fake = fake
        self.strict = strict

        self.action_spaces = {
            agent: MultiDiscrete([ACT_SIZE, ACT_SIZE]) for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: Box(low=-np.inf, high=np.inf, shape=(1,)) for agent in self.possible_agents
        }

    @staticmethod
    def single_action_mask(battle: DoubleBattle, pos: int) -> list[int]:
        available_base_species = {p.base_species for p in battle.available_switches[pos]}
        # either ps doesnt track or poke-env doesnt update
        # the trapped variable on the second active pokemon
        # maybe trapped works here since its only shadow tag
        # is the only ability in the vocab that can cause the
        # maybe_trapped to be true (in which case the pokemon)
        # is trapped actually
        if battle.trapped[pos] or battle.maybe_trapped[pos]:
            switch_space = []
        else:
            switch_space = [
                i + 1
                for i, pokemon in enumerate(battle.team.values())
                if pokemon.base_species in available_base_species
            ]

        active_mon = battle.active_pokemon[pos]
        if battle._wait or (any(battle.force_switch) and not battle.force_switch[pos]):
            actions = [0]
        elif all(battle.force_switch) and len(battle.available_switches[pos]) == 1:
            actions = switch_space + [0]
        elif active_mon is None or active_mon.fainted:
            actions = switch_space
        else:
            available_move_ids = {move.id for move in battle.available_moves[pos]}
            move_space = [
                7 + 5 * i + target + 2
                for i, move in enumerate(active_mon.moves.values())
                if move.id in available_move_ids
                for target in battle.get_possible_showdown_targets(move, active_mon)
            ]
            if (
                not move_space
                and len(battle.available_moves[pos]) == 1
                and battle.available_moves[pos][0].id in {"struggle", "recharge"}
            ):
                move_space = [48]
                mega_space = [47] if battle.can_mega_evolve[pos] else []
            else:
                mega_space = (
                    [action + 20 for action in move_space] if battle.can_mega_evolve[pos] else []
                )
            actions = switch_space + move_space + mega_space

        return actions or [0]

    @staticmethod
    def get_action_mask(battle: AbstractBattle) -> list[int]:
        assert isinstance(battle, DoubleBattle)
        if battle.teampreview:
            # mask is always the same
            return _TEAM_PREVIEW_MASK.copy()

        # first half for p1, second half for p2 in battle
        mask = [0] * 2 * ACT_SIZE

        p1_actions = MegaEnv.single_action_mask(battle, 0)
        p2_actions = MegaEnv.single_action_mask(battle, 1)
        for a in p1_actions:
            mask[a] = 1
        for a in p2_actions:
            mask[ACT_SIZE + a] = 1

        return mask

    @staticmethod
    def action_to_order(
        action: npt.NDArray[np.int64],
        battle: DoubleBattle,
        fake: bool = False,
        strict: bool = True,
    ) -> BattleOrder:
        """Convert an action array into a :class:`BattleOrder`.

        The action is a list in doubles, and the individual action mapping is
        as follows, where each 5-long range for a move corresponds to a
        different target (-2, -1, 0, 1, 2).
        -2 = pkm2
        -1 = pkm1
        0 = empty
        1 = opponent1
        2 = opponent2

        element = -2: default
        element = -1: forfeit
        element = 0: pass
        1 <= element <= 6: switch
        7 <= element <= 11: move 1
        12 <= element <= 16: move 2
        17 <= element <= 21: move 3
        22 <= element <= 26: move 4
        27 <= element <= 31: move 1 and mega evolution
        32 <= element <= 36: move 2 and mega evolution
        37 <= element <= 41: move 3 and mega evolution
        42 <= element <= 46: move 4 and mega evolution
        element = 47: mega struggle/recharge
        element = 48: struggle/recharge

        :param action: The action to take.
        :type action: ndarray[int64]
        :param battle: The current battle state
        :type battle: AbstractBattle
        :param fake: If ``True``, return a best-effort order even if it would be
            illegal.
        :type fake: bool
        :param strict: If ``True``, raise an error when the action is illegal;
            otherwise return a default order.
        :type strict: bool

        :return: The battle order for the given action in context of the current battle.
        :rtype: BattleOrder

        """
        if battle.teampreview:
            p1 = action[0] // 6 + 1
            p2 = action[0] % 6 + 1
            p3 = action[1] // 6 + 1
            p4 = action[1] % 6 + 1

            selection = [p1, p2, p3, p4]
            seen = []
            final_selection = []
            for p in selection:
                if p not in seen and 1 <= p <= len(battle.team):
                    final_selection.append(p)
                    seen.append(p)
            for i in range(1, len(battle.team) + 1):
                if i not in seen:
                    final_selection.append(i)
            return SingleBattleOrder("/team " + "".join(map(str, final_selection)))

        if action[0] == -2 and action[1] == -2:
            return DefaultBattleOrder()
        elif action[0] == -1 or action[1] == -1:
            return ForfeitBattleOrder()
        try:
            order1 = MegaEnv._action_to_order_individual(action[0], battle, fake, 0)
        except ValueError as e:
            if strict:
                raise e
            else:
                if battle.logger is not None:
                    battle.logger.warning(str(e) + " Defaulting to random move.")
                order = Player.choose_random_doubles_move(battle)
                order1 = order.first_order if not isinstance(order, DefaultBattleOrder) else order
        try:
            order2 = MegaEnv._action_to_order_individual(action[1], battle, fake, 1)
        except ValueError as e:
            if strict:
                raise e
            else:
                if battle.logger is not None:
                    battle.logger.warning(str(e) + " Defaulting to random move.")
                order = Player.choose_random_doubles_move(battle)
                order2 = order.second_order if not isinstance(order, DefaultBattleOrder) else order
        joined_orders = DoubleBattleOrder.join_orders([order1], [order2])
        if not joined_orders:
            error_msg = (
                f"Invalid action {action} from player {battle.player_username} "
                f"in battle {battle.battle_tag} - converted orders {order1} "
                f"and {order2} are incompatible!"
            )
            if strict:
                raise ValueError(error_msg)
            else:
                if battle.logger is not None:
                    battle.logger.warning(error_msg + " Defaulting to random move.")
                return Player.choose_random_doubles_move(battle)
        else:
            return joined_orders[0]

    @staticmethod
    def _action_to_order_individual(
        action: np.int64, battle: DoubleBattle, fake: bool, pos: int
    ) -> SingleBattleOrder:
        if action == -2:
            return DefaultBattleOrder()
        elif action == 0:
            order: SingleBattleOrder = PassBattleOrder()
        elif action < 7:
            order = Player.create_order(list(battle.team.values())[action - 1])
        else:
            active_mon = battle.active_pokemon[pos]
            if active_mon is None:
                raise ValueError(
                    f"Invalid order from player {battle.player_username} "
                    f"in battle {battle.battle_tag} at position {pos} - action "
                    f"specifies a move, but battle.active_pokemon is None!"
                )
            if action in (47, 48):
                order = Player.create_order(battle.available_moves[pos][0], mega=(action == 47))
            else:
                mvs = list(active_mon.moves.values())
                if (action - 7) % 20 // 5 not in range(len(mvs)):
                    raise ValueError(
                        f"Invalid action {action} from player {battle.player_username} "
                        f"in battle {battle.battle_tag} at position {pos} - action "
                        f"specifies a move but the move index {(action - 7) % 20 // 5} "
                        f"is out of bounds for available moves {mvs}!"
                    )
                order = Player.create_order(
                    mvs[(action - 7) % 20 // 5],
                    move_target=(action.item() - 7) % 5 - 2,
                    mega=(action - 7) // 20 == 1,
                )
        if not fake:
            valid_orders = [str(valid_order) for valid_order in battle.valid_orders[pos]]
            if str(order) not in valid_orders:
                raise ValueError(
                    f"Invalid action {action} from player {battle.player_username} "
                    f"in battle {battle.battle_tag} at position {pos} - order {order} "
                    f"not in action space {valid_orders}!"
                )
        return order

    @staticmethod
    def order_to_action(
        order: BattleOrder,
        battle: DoubleBattle,
        fake: bool = False,
        strict: bool = True,
    ) -> npt.NDArray[np.int64]:
        """Convert a :class:`BattleOrder` into an action array.

        :param order: The order to take.
        :type order: BattleOrder
        :param battle: The current battle state
        :type battle: AbstractBattle
        :param fake: If ``True``, return a best-effort action even if it would be
            illegal.
        :type fake: bool
        :param strict: If ``True``, raise an error when the order is illegal;
            otherwise return default.
        :type strict: bool

        :return: The action for the given battle order in context of the current battle.
        :rtype: ndarray[int64]
        """
        if battle.teampreview:
            msg = order.message[6:]
            msg = "".join(c for c in msg if c.isdigit())
            p1 = int(msg[0]) if len(msg) > 0 else 1
            p2 = int(msg[1]) if len(msg) > 1 else 2
            p3 = int(msg[2]) if len(msg) > 2 else 3
            p4 = int(msg[3]) if len(msg) > 3 else 4

            # Canonicalize: sort pairs to match the p1 < p2 mask
            # For Leads (Action 0)
            l1, l2 = min(p1, p2), max(p1, p2)
            if l1 == l2:
                l2 = (l1 % 6) + 1

            # For Back (Action 1)
            b1, b2 = min(p3, p4), max(p3, p4)
            if b1 == b2:
                b2 = (b1 % 6) + 1

            return np.array([(l1 - 1) * 6 + (l2 - 1), (b1 - 1) * 6 + (b2 - 1)], dtype=np.int64)

        if isinstance(order, DefaultBattleOrder):
            return np.array([-2, -2])
        elif isinstance(order, ForfeitBattleOrder):
            return np.array([-1, -1])
        assert isinstance(order, DoubleBattleOrder)
        joined_orders = DoubleBattleOrder.join_orders([order.first_order], [order.second_order])
        if not fake and not joined_orders:
            error_msg = (
                f"Invalid order {order} from player {battle.player_username} "
                f"in battle {battle.battle_tag} - orders are incompatible!"
            )
            if strict:
                raise ValueError(str(error_msg) + " Defaulting to random move.")
            else:
                if battle.logger is not None:
                    battle.logger.warning(error_msg)
                return MegaEnv.order_to_action(
                    Player.choose_random_doubles_move(battle), battle, fake, strict
                )
        try:
            action1 = MegaEnv._order_to_action_individual(order.first_order, battle, fake, 0)
        except ValueError as e:
            if strict:
                raise e
            else:
                if battle.logger is not None:
                    battle.logger.warning(str(e) + " Defaulting to random move.")
                order = Player.choose_random_doubles_move(battle)
                action1 = MegaEnv._order_to_action_individual(
                    (order.first_order if not isinstance(order, DefaultBattleOrder) else order),
                    battle,
                    fake,
                    0,
                )
        try:
            action2 = MegaEnv._order_to_action_individual(order.second_order, battle, fake, 1)
        except ValueError as e:
            if strict:
                raise e
            else:
                if battle.logger is not None:
                    battle.logger.warning(str(e) + " Defaulting to random move.")
                order = Player.choose_random_doubles_move(battle)
                action2 = MegaEnv._order_to_action_individual(
                    (order.second_order if not isinstance(order, DefaultBattleOrder) else order),
                    battle,
                    fake,
                    1,
                )
        return np.array([action1, action2])

    @staticmethod
    def _order_to_action_individual(
        order: SingleBattleOrder, battle: DoubleBattle, fake: bool, pos: int
    ) -> np.int64:
        if isinstance(order.order, str):
            if isinstance(order, DefaultBattleOrder):
                return np.int64(-2)
            else:
                assert isinstance(order, PassBattleOrder)
                return np.int64(0)
        if not fake:
            valid_orders = [str(valid_order) for valid_order in battle.valid_orders[pos]]
            if str(order) not in valid_orders:
                raise ValueError(
                    f"Invalid order from player {battle.player_username} in battle "
                    f"{battle.battle_tag} at position {pos} - order {order} not in "
                    f"action space {valid_orders}!"
                )
        if isinstance(order.order, Pokemon):
            action = [p.base_species for p in battle.team.values()].index(
                order.order.base_species
            ) + 1
        else:
            active_mon = battle.active_pokemon[pos]
            assert active_mon is not None
            if len(battle.available_moves[pos]) == 1 and battle.available_moves[pos][0].id in [
                "struggle",
                "recharge",
            ]:
                return np.int64(47 if order.mega else 48)

            mvs = list(active_mon.moves.values())
            action = [m.id for m in mvs].index(order.order.id)
            target = order.move_target + 2
            if order.mega:
                gimmick = 1
            else:
                gimmick = 0
            action = 1 + 6 + 5 * action + target + 20 * gimmick
        return np.int64(action)


class SimEnv(MegaEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._observation_targets: dict[str, StructuredObservation] = {}

    def set_observation_targets(
        self,
        agent1_out: StructuredObservation,
        agent2_out: StructuredObservation,
    ) -> None:
        self._observation_targets = {
            self.agent1.username: agent1_out,
            self.agent2.username: agent2_out,
        }

    @classmethod
    def build_env(
        cls,
        env_id: int = 0,
        server_port: int = 8000,
        *,
        team=None,
        team_pool: str = "all",
        opponent_team_pool: str = "all",
        teams_root: str | Path = "teams",
    ):
        if team is None:
            agent_team = load_team_pool(Path(teams_root) / team_pool)
            opponent_team = load_team_pool(Path(teams_root) / opponent_team_pool)
        else:
            agent_team = team
            opponent_team = team
        env = cls(
            account_configuration1=AccountConfiguration(f"TrainAgent_{env_id}", None),
            account_configuration2=AccountConfiguration(f"BestAgent_{env_id}", None),
            server_configuration=ServerConfiguration(
                f"ws://localhost:{server_port}/showdown/websocket",
                "https://play.pokemonshowdown.com/action.php?",
            ),
            battle_format=FORMAT.battle_format,
            accept_open_team_sheet=True,
            start_timer_on_battle_start=False,
            log_level=25,
            # Poke-env accepts one team in its environment constructor. The
            # players own their builders, so replace them after construction
            # to support asymmetric agent/opponent team distributions.
            team=agent_team,
        )
        env.agent1.update_team(agent_team)
        env.agent2.update_team(opponent_team)
        return env

    def calc_reward(self, battle: AbstractBattle) -> float:
        if not battle.finished:
            return 0
        return 1 if battle.won else (-1 if battle.lost else 0)

    def embed_battle(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        out = self._observation_targets.get(battle.player_username)
        if out is None:
            return observation_builder.from_battle(battle)
        observation_builder.from_battle_into(battle, out)
        return out
