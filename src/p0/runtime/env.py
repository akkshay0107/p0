import random
from collections.abc import Callable
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box, MultiDiscrete
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.environment.env import PokeEnv
from poke_env.ps_client import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    ServerConfiguration,
)
from poke_env.teambuilder import Teambuilder

from p0.battle.legality import action_mask
from p0.battle.views import BattleView
from p0.format_config import FORMAT
from p0.model.observation_builder import ObservationBuilder
from p0.model.structured_observation import StructuredObservation
from p0.runtime import poke_env_patches
from p0.runtime.poke_env_action_adapter import action_to_order
from p0.runtime.poke_env_battle_adapter import battle_view, current_battle_view
from p0.teams.source import TeamSource

ACT_SIZE = FORMAT.action_size


def get_action_mask(battle: AbstractBattle) -> list[int]:
    if not isinstance(battle, DoubleBattle):
        raise TypeError(f"Expected DoubleBattle, got {type(battle).__name__}")
    return action_mask(battle_view(battle).decision).reshape(-1).astype(np.int64).tolist()


def _get_current_action_mask(battle: AbstractBattle) -> list[int]:
    """Build the mask from the view refreshed by ``SimEnv.embed_battle``."""
    if not isinstance(battle, DoubleBattle):
        raise TypeError(f"Expected DoubleBattle, got {type(battle).__name__}")
    return action_mask(current_battle_view(battle).decision).reshape(-1).astype(np.int64).tolist()


# modified from poke-env
# to remove all other gimmicks but mega evolution
class MegaEnv(PokeEnv[npt.NDArray[np.int64]]):
    action_to_order = staticmethod(action_to_order)
    get_action_mask = staticmethod(get_action_mask)

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


class SimEnv(MegaEnv):
    get_action_mask = staticmethod(_get_current_action_mask)

    def __init__(
        self,
        *args,
        observation_builder: ObservationBuilder,
        battle_view_factory: Callable[[DoubleBattle], BattleView],
        agent_team_source: TeamSource,
        opponent_team_source: TeamSource,
        agent_rng: random.Random,
        opponent_rng: random.Random,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._observation_targets: dict[str, StructuredObservation] = {}
        self._observation_builder = observation_builder
        self._battle_view_factory = battle_view_factory
        self._agent_team_source = agent_team_source
        self._opponent_team_source = opponent_team_source
        self._agent_rng = agent_rng
        self._opponent_rng = opponent_rng

    def set_observation_targets(
        self,
        agent1_out: StructuredObservation,
        agent2_out: StructuredObservation,
    ) -> None:
        self._observation_builder.validate_output(agent1_out)
        self._observation_builder.validate_output(agent2_out)
        self._observation_targets = {
            self.agent1.username: agent1_out,
            self.agent2.username: agent2_out,
        }

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self._agent_rng.seed(seed)
            self._opponent_rng.seed(seed + 1)
        self.agent1.update_team(self._agent_team_source.sample(self._agent_rng).packed)
        self.agent2.update_team(self._opponent_team_source.sample(self._opponent_rng).packed)
        return super().reset(seed=seed, options=options)

    def calc_reward(self, battle: AbstractBattle) -> float:
        if not battle.finished:
            return 0
        return 1 if battle.won else (-1 if battle.lost else 0)

    def embed_battle(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        view = self._battle_view_factory(battle)
        out = self._observation_targets.get(battle.player_username)
        if out is None:
            return self._observation_builder.build(view)
        self._observation_builder.build_into_prevalidated(view, out)
        return out
