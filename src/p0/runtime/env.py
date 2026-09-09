"""Gymnasium environment wrappers for poke-env simulations."""

from __future__ import annotations

import random
import uuid
from collections.abc import Callable, Mapping

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box, MultiDiscrete
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.environment.env import PokeEnv
from poke_env.player import Player
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
    """Compute flat integer action mask list for a double battle."""
    if not isinstance(battle, DoubleBattle):
        raise TypeError(f"Expected DoubleBattle, got {type(battle).__name__}")

    return action_mask(battle_view(battle).decision).reshape(-1).astype(np.int64).tolist()


def _get_current_action_mask(battle: AbstractBattle) -> list[int]:
    """Build the mask from the view refreshed by SimEnv.embed_battle."""
    if not isinstance(battle, DoubleBattle):
        raise TypeError(f"Expected DoubleBattle, got {type(battle).__name__}")

    return action_mask(current_battle_view(battle).decision).reshape(-1).astype(np.int64).tolist()


class MegaEnv(PokeEnv[npt.NDArray[np.int64]]):
    """Custom PokeEnv subclass supporting doubles action spaces and mega evolution."""

    action_to_order = staticmethod(action_to_order)
    get_action_mask = staticmethod(get_action_mask)

    def __init__(
        self,
        account_configuration1: AccountConfiguration | None = None,
        account_configuration2: AccountConfiguration | None = None,
        avatar: int | None = None,
        battle_format: str = FORMAT.battle_format,
        log_level: int | None = None,
        save_replays: bool | str = False,
        server_configuration: ServerConfiguration | None = LocalhostServerConfiguration,
        accept_open_team_sheet: bool | None = True,
        start_timer_on_battle_start: bool = False,
        start_listening: bool = True,
        open_timeout: float | None = 10.0,
        ping_interval: float | None = 20.0,
        ping_timeout: float | None = 20.0,
        challenge_timeout: float | None = 60.0,
        team: str | Teambuilder | None = None,
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
    """High-performance simulation environment supporting observation target pre-allocation."""

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
        self._series_scores = [0, 0]
        self._series_games_played = 0
        self._decision_steps = 0
        self._resume_reset_pending = False
        self.series_id = str(uuid.uuid4())

    @property
    def series_scores(self) -> list[int]:
        """Best-of-three win counts for (agent, opponent) in the current series."""
        return self._series_scores

    @property
    def series_games_played(self) -> int:
        """Number of games played so far in the current best-of-three series."""
        return self._series_games_played

    def training_state(self) -> dict[str, object]:
        """Capture stochastic and series state needed for deterministic PPO resume."""
        return {
            "agent_rng": self._agent_rng.getstate(),
            "opponent_rng": self._opponent_rng.getstate(),
            "agent_team": self._current_team_packed(self.agent1),
            "opponent_team": self._current_team_packed(self.agent2),
            "series_scores": tuple(self._series_scores),
            "series_games_played": self._series_games_played,
            "series_id": self.series_id,
        }

    def restore_training_state(self, state: Mapping[str, object]) -> None:
        """Restore a state previously returned by the training_state method."""
        agent_rng = state.get("agent_rng")
        opponent_rng = state.get("opponent_rng")
        agent_team = state.get("agent_team")
        opponent_team = state.get("opponent_team")
        scores = state.get("series_scores")
        games_played = state.get("series_games_played")
        series_id = state.get("series_id")
        if not isinstance(scores, (tuple, list)) or len(scores) != 2:
            raise ValueError("Invalid SimEnv series_scores training state")
        if (
            not isinstance(games_played, int)
            or not isinstance(series_id, str)
            or not isinstance(agent_team, str)
            or not isinstance(opponent_team, str)
            or not agent_team.strip()
            or not opponent_team.strip()
        ):
            raise ValueError("Invalid SimEnv series metadata training state")
        try:
            self._agent_rng.setstate(agent_rng)  # type: ignore[arg-type]
            self._opponent_rng.setstate(opponent_rng)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid SimEnv random training state") from exc
        self.agent1.update_team(agent_team)
        self.agent2.update_team(opponent_team)
        self._series_scores = [int(scores[0]), int(scores[1])]
        self._series_games_played = games_played
        self.series_id = series_id
        self._resume_reset_pending = True

    @staticmethod
    def _current_team_packed(player: Player) -> str:
        """Read the packed team from poke-env's current team builder."""
        if player._team is None:
            raise RuntimeError("Simulation player has no serializable active team")
        team = player._team.yield_team()
        if not isinstance(team, str) or not team.strip():
            raise RuntimeError("Simulation player has an invalid active team")
        return team

    def prepare_for_checkpoint(self) -> None:
        """Mark the next reset as a clean replacement for the active game."""
        self._resume_reset_pending = True

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

        is_new_series = sum(self._series_scores) == 0 and self._series_games_played == 0
        is_completed_series = max(self._series_scores) >= 2 or self._series_games_played >= 3
        preserve_restored_game = self._resume_reset_pending
        self._resume_reset_pending = False

        if is_completed_series or is_new_series:
            self._series_scores = [0, 0]
            self._series_games_played = 0
            self.series_id = str(uuid.uuid4())
            self.agent1.update_team(self._agent_team_source.sample(self._agent_rng).packed)
            self.agent2.update_team(self._opponent_team_source.sample(self._opponent_rng).packed)

        self._decision_steps = 0
        # preemptively add the game that will be played
        if not preserve_restored_game or is_new_series or is_completed_series:
            self._series_games_played += 1
        return super().reset(seed=seed, options=options)

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Score one finished game from the passed battle's own perspective.

        poke-env calls this once per agent per step, with that agent's battle
        object, so it must stay free of side effects. The series score is
        credited once per game by the record_game_result method.
        """
        if not battle.finished:
            return 0.0

        if battle.won:
            return 1.0
        if battle.lost:
            return -1.0
        return 0.0

    def _record_game_result(self, battle: AbstractBattle) -> None:
        """Credit one finished game to the best-of-three score."""
        if battle.won:
            self._series_scores[0] += 1
        elif battle.lost:
            self._series_scores[1] += 1

    def step(self, actions):
        self._decision_steps += 1
        obs, rewards, terminated, truncated, info = super().step(actions)

        # poke-env reports termination only when exactly one side is wiped out,
        # so ties, forfeits and timer losses arrive as truncations. Any finished
        # battle is terminal here; truncation is reserved for the step cap below,
        # which is the only case whose value target should bootstrap.
        battle = self.battle1
        if battle is not None and battle.finished:
            for agent in terminated:
                terminated[agent] = True
            for agent in truncated:
                truncated[agent] = False
            self._record_game_result(battle)
        elif self._decision_steps >= 198 and not any(truncated.values()):
            for agent in truncated:
                truncated[agent] = True
            for agent in rewards:
                rewards[agent] = 0.0

        return obs, rewards, terminated, truncated, info

    def embed_battle(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        view = self._battle_view_factory(battle)
        out = self._observation_targets.get(battle.player_username)

        if out is None:
            return self._observation_builder.build(view)

        self._observation_builder.build_into_prevalidated(view, out)
        return out
