import random
from pathlib import Path
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box, MultiDiscrete
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.environment.env import PokeEnv
from poke_env.player.battle_order import BattleOrder, SingleBattleOrder
from poke_env.ps_client import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    ServerConfiguration,
)
from poke_env.teambuilder import Teambuilder

from p0.battle.legality import LegalActionBuilder
from p0.format_config import FORMAT
from p0.model.observation_builder import ObservationBuilder
from p0.model.structured_observation import StructuredObservation
from p0.runtime import poke_env_patches
from p0.runtime.poke_env_action_adapter import PokeEnvOrderAdapter
from p0.runtime.poke_env_battle_adapter import PokeEnvBattleAdapter
from p0.teams.source import TeamSource

ACT_SIZE = FORMAT.action_size


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
        poke_env_patches.install(self.agent1.logger)
        poke_env_patches.install(self.agent2.logger)
        poke_env_patches.enable_environment_team_preview(self.agent1)
        poke_env_patches.enable_environment_team_preview(self.agent2)

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
        return list(
            LegalActionBuilder.legal_actions(PokeEnvBattleAdapter.decision_view(battle), pos)
        )

    @staticmethod
    def get_action_mask(battle: AbstractBattle) -> list[int]:
        assert isinstance(battle, DoubleBattle)
        return (
            LegalActionBuilder.mask(PokeEnvBattleAdapter.current_view(battle).decision)
            .reshape(-1)
            .astype(np.int64)
            .tolist()
        )

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
        return PokeEnvOrderAdapter.action_to_order(action, battle, fake, strict)

    @staticmethod
    def _action_to_order_individual(
        action: np.int64, battle: DoubleBattle, fake: bool, pos: int
    ) -> SingleBattleOrder:
        return PokeEnvOrderAdapter.action_to_single_order(int(action), battle, fake, pos)

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
        return PokeEnvOrderAdapter.order_to_action(order, battle, fake, strict)

    @staticmethod
    def _order_to_action_individual(
        order: SingleBattleOrder, battle: DoubleBattle, fake: bool, pos: int
    ) -> np.int64:
        return PokeEnvOrderAdapter.single_order_to_action(order, battle, fake, pos)


class SimEnv(MegaEnv):
    def __init__(
        self,
        *args,
        observation_builder: ObservationBuilder | None = None,
        battle_adapter: type[PokeEnvBattleAdapter] = PokeEnvBattleAdapter,
        legality_builder: type[LegalActionBuilder] = LegalActionBuilder,
        order_adapter: type[PokeEnvOrderAdapter] = PokeEnvOrderAdapter,
        agent_team_source: TeamSource | None = None,
        opponent_team_source: TeamSource | None = None,
        agent_rng: random.Random | None = None,
        opponent_rng: random.Random | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._observation_targets: dict[str, StructuredObservation] = {}
        self._observation_builder = observation_builder or ObservationBuilder()
        self._battle_adapter = battle_adapter
        self._legality_builder = legality_builder
        self._order_adapter = order_adapter
        self._agent_team_source = agent_team_source
        self._opponent_team_source = opponent_team_source
        self._agent_rng = agent_rng or random.Random()
        self._opponent_rng = opponent_rng or random.Random()

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
        from p0.runtime.composition import build_sim_env
        from p0.teams.source import FileTeamSource, FixedTeamSource

        if team is None:
            agent_source: TeamSource = FileTeamSource(Path(teams_root) / team_pool)
            opponent_source: TeamSource = FileTeamSource(Path(teams_root) / opponent_team_pool)
        else:
            agent_source = opponent_source = FixedTeamSource(team)
        return build_sim_env(
            account_configuration1=AccountConfiguration(f"TrainAgent_{env_id}", None),
            account_configuration2=AccountConfiguration(f"BestAgent_{env_id}", None),
            server_port=server_port,
            agent_team_source=agent_source,
            opponent_team_source=opponent_source,
            env_type=cls,
        )

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self._agent_rng.seed(seed)
            self._opponent_rng.seed(seed + 1)
        if self._agent_team_source is not None:
            self.agent1.update_team(self._agent_team_source.sample(self._agent_rng).packed)
        if self._opponent_team_source is not None:
            self.agent2.update_team(self._opponent_team_source.sample(self._opponent_rng).packed)
        return super().reset(seed=seed, options=options)

    def calc_reward(self, battle: AbstractBattle) -> float:
        if not battle.finished:
            return 0
        return 1 if battle.won else (-1 if battle.lost else 0)

    def embed_battle(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        adapter = getattr(self, "_battle_adapter", PokeEnvBattleAdapter)
        view = adapter.view(battle)
        _ = view.decision
        builder = getattr(self, "_observation_builder", None)
        if builder is None:
            builder = self._observation_builder = ObservationBuilder()
        out = self._observation_targets.get(battle.player_username)
        if out is None:
            return builder.build(view)
        builder.build_into(view, out)
        return out
