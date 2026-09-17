"""
Live policy player and battle memory management.

This module provides the core RLPlayer agent integrating neural network policy inference,
history token management, series token persistence, and team sampling for live Pokemon
Showdown battles.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import DefaultBattleOrder, Player

from p0.battle.legality import action_mask
from p0.format_config import FORMAT
from p0.model.architecture_contract import HISTORY_WINDOW
from p0.model.cls_reducer import pack_history_tokens
from p0.model.config import ModelConfig
from p0.model.factory import build_policy, compile_policy
from p0.model.observation_builder import ObservationBuilder
from p0.model.policy import MemoryInputs, PolicyNet
from p0.model.resources import default_runtime_resources
from p0.model.token_store import SeriesTokenStore
from p0.runtime import poke_env_patches
from p0.runtime.poke_env_action_adapter import action_to_order
from p0.runtime.poke_env_battle_adapter import battle_view
from p0.teams.source import TeamSource
from p0.training.checkpoint import DEFAULT_CHECKPOINT_STORE, CheckpointStore

DEFAULT_BATTLE_FORMAT = FORMAT.bo3_format


@dataclass(slots=True)
class _LiveBattleHistory:
    """Track decision tokens for a live battle."""

    tokens: list[torch.Tensor] = field(default_factory=list)

    def append(self, token: torch.Tensor, device: torch.device) -> None:
        self.tokens.append(token.detach().to(device=device, dtype=torch.float32))

    def recent_values(self, empty: torch.Tensor) -> torch.Tensor:
        if not self.tokens:
            return empty
        return torch.stack(self.tokens[-HISTORY_WINDOW:]).unsqueeze(0)

    def complete_values(self, device: torch.device) -> torch.Tensor:
        return torch.stack(self.tokens).to(device).unsqueeze(0)


@dataclass(slots=True)
class _LiveSeriesState:
    """Track games and score for one Best-of-3 series."""

    key: str
    opponent_id: str
    games_played: int = 0
    wins: int = 0
    losses: int = 0

    @property
    def complete(self) -> bool:
        return self.wins >= 2 or self.losses >= 2 or self.games_played >= 3


class TeamPlayerMixin:
    """Mixin adding team sampling and resampling from a TeamSource to any Player."""

    team_source: TeamSource | None
    team_rng: random.Random
    current_team_packed: str | None

    def __init__(
        self,
        *args,
        team_rng: random.Random,
        team_source: TeamSource | None = None,
        **kwargs,
    ):
        self.team_source = team_source
        self.team_rng = team_rng

        if team_source is not None:
            if "team" in kwargs:
                raise ValueError("Pass either team or team_source, not both")
            self.current_team_packed = team_source.sample(team_rng).packed
            kwargs["team"] = self.current_team_packed
        else:
            self.current_team_packed = kwargs.get("team")

        super().__init__(*args, **kwargs)

    def update_team(self, team):
        super().update_team(team)  # pyright: ignore[reportAttributeAccessIssue]

        if isinstance(team, str):
            self.current_team_packed = team
        elif hasattr(team, "yield_team"):
            self.current_team_packed = team.yield_team()

    def _finish_team_battle(self, battle: AbstractBattle, *, resample_team: bool) -> None:
        """Finish the underlying battle and optionally prepare the next team."""
        super()._battle_finished_callback(battle)  # pyright: ignore[reportAttributeAccessIssue]

        if resample_team and self.team_source is not None:
            self.update_team(self.team_source.sample(self.team_rng).packed)

        battle_id = getattr(battle, "battle_tag", None)
        battles = getattr(self, "_battles", None)
        if battle_id and battles is not None:
            battles.pop(battle_id, None)

    def _should_resample_team_after_battle(self, battle: AbstractBattle) -> bool:
        del battle
        return True

    def _battle_finished_callback(self, battle: AbstractBattle):
        self._finish_team_battle(
            battle,
            resample_team=self._should_resample_team_after_battle(battle),
        )


class RLPlayer(TeamPlayerMixin, Player):
    """Class that plays moves as per the trained policy net."""

    def __init__(
        self,
        policy: PolicyNet,
        *,
        observation_builder: ObservationBuilder,
        team_rng: random.Random,
        team_source: TeamSource | None = None,
        top_p: float = 0.9,
        battle_format: str = DEFAULT_BATTLE_FORMAT,
        **kwargs,
    ):
        if battle_format != DEFAULT_BATTLE_FORMAT:
            raise ValueError(
                f"RLPlayer only supports the configured Bo3 format {DEFAULT_BATTLE_FORMAT!r}; "
                f"got {battle_format!r}"
            )

        kwargs["battle_format"] = DEFAULT_BATTLE_FORMAT
        # The Bo3 format forces open sheets server-side; poke-env must neither
        # negotiate nor wait for a negotiation response.
        kwargs["accept_open_team_sheet"] = False
        super().__init__(team_rng=team_rng, team_source=team_source, **kwargs)
        poke_env_patches.install(self.logger)
        # Showdown's configured Bo3 format uses Force Open Team Sheets, so there
        # is no accept/reject command to send during battle creation.
        setattr(self.ps_client, "_p0_force_open_team_sheet", True)
        self.policy = policy
        self.observation_builder = observation_builder

        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}.")

        self.top_p = top_p
        self._memory_model_id = id(policy)
        self._empty_history_tensor = torch.zeros((1, 0, policy.d_model), device=policy.device)
        self._battle_histories: dict[str, _LiveBattleHistory] = {}
        self._series_store = SeriesTokenStore(policy.d_model)
        self._series_by_opponent: dict[str, _LiveSeriesState] = {}
        self._series_by_battle: dict[str, _LiveSeriesState] = {}
        self._series_sequence = 0

    @staticmethod
    def _battle_key(battle: DoubleBattle) -> str:
        key = getattr(battle, "battle_tag", None)
        if not key:
            raise ValueError("Live battle has no stable battle identifier")
        return str(key)

    @staticmethod
    def _opponent_id(battle: DoubleBattle) -> str:
        opponent = battle.opponent_username
        if not opponent or not opponent.strip():
            raise ValueError("Live Bo3 battle has no opponent identity")
        return opponent.strip().casefold()

    def _series_for_battle(self, battle: DoubleBattle) -> _LiveSeriesState:
        """Return the Bo3 series state associated with a battle."""
        battle_key = self._battle_key(battle)
        state = self._series_by_battle.get(battle_key)
        if state is not None:
            return state

        opponent_id = self._opponent_id(battle)
        state = self._series_by_opponent.get(opponent_id)
        if state is None:
            self._series_sequence += 1
            state = _LiveSeriesState(
                key=f"bo3:{opponent_id}:{self._series_sequence}",
                opponent_id=opponent_id,
            )
            self._series_by_opponent[opponent_id] = state

        self._series_by_battle[battle_key] = state
        return state

    def _drop_series(self, state: _LiveSeriesState) -> None:
        self._series_store.drop(state.key)
        current = self._series_by_opponent.get(state.opponent_id)
        if current is state:
            self._series_by_opponent.pop(state.opponent_id, None)

    def invalidate_memory_for_model_reload(self) -> None:
        """Drop per-battle memory when a policy artifact is replaced."""
        self._battle_histories.clear()
        self._series_store.clear()
        self._series_by_opponent.clear()
        self._series_by_battle.clear()
        self._memory_model_id = id(self.policy)
        self._empty_history_tensor = torch.zeros(
            (1, 0, self.policy.d_model), device=self.policy.device
        )

    def _memory_inputs(self, battle: DoubleBattle) -> MemoryInputs:
        if (
            id(self.policy) != self._memory_model_id
            or self._empty_history_tensor.device != self.policy.device
            or self._empty_history_tensor.size(-1) != self.policy.d_model
        ):
            self.invalidate_memory_for_model_reload()

        key = self._battle_key(battle)
        history = self._battle_histories.get(key)
        values = (
            self._empty_history_tensor
            if history is None
            else history.recent_values(self._empty_history_tensor)
        )

        history_tokens, history_mask = pack_history_tokens(values)

        series_key = self._series_for_battle(battle).key
        series_tokens, series_mask = self._series_store.get_tokens(
            [series_key], device=self.policy.device
        )
        return MemoryInputs(
            series_tokens=series_tokens,
            series_mask=series_mask,
            history_tokens=history_tokens,
            history_mask=history_mask,
        )

    def _append_history(self, battle: DoubleBattle, token: torch.Tensor) -> None:
        # Store detached summary token for battle memory.
        key = self._battle_key(battle)
        history = self._battle_histories.get(key)
        if history is None:
            history = _LiveBattleHistory()
            self._battle_histories[key] = history
        history.append(token, self.policy.device)

    def _get_action(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        view = battle_view(battle)
        obs = self.observation_builder.build(view)
        mask = torch.from_numpy(action_mask(view.decision))

        obs = obs.unsqueeze(0).to(self.policy.device)
        mask = mask.unsqueeze(0).to(self.policy.device)

        with torch.no_grad():
            encoded = self.policy.encode(obs, mask)
            prepared = self.policy.prepare(encoded, self._memory_inputs(battle))
            out = self.policy.act(prepared, mask, top_p=self.top_p)

        # Waiting requests return before action selection; every token appended here
        # therefore corresponds to an actual policy decision.
        self._append_history(battle, out.history_token[0])
        return out.actions[0].cpu().numpy()

    def choose_move(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        view = battle_view(battle)
        # Forced-open Bo3 team sheets arrive as a showteam notification before
        # the actual teampreview request; there are no active slots to choose for
        # that intermediate callback.
        if view.wait or (not battle.teampreview and not any(battle.active_pokemon)):
            return DefaultBattleOrder()
        return action_to_order(self._get_action(battle), battle)

    def get_observation(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        return self.observation_builder.build(battle_view(battle))

    def teampreview(self, battle: AbstractBattle) -> str:
        assert isinstance(battle, DoubleBattle)
        key = self._battle_key(battle)
        self._battle_histories.pop(key, None)
        action = self._get_action(battle)
        order = action_to_order(action, battle)
        return order.message

    def _battle_finished_callback(self, battle: AbstractBattle):
        if not isinstance(battle, DoubleBattle):
            raise TypeError(f"RLPlayer requires DoubleBattle, got {type(battle).__name__}")

        battle_key = self._battle_key(battle)
        state = self._series_by_battle.pop(battle_key, None)
        if state is None:
            raise RuntimeError(f"No Bo3 series state exists for battle {battle_key!r}")
        if not battle.finished:
            raise RuntimeError(f"Bo3 callback received unfinished battle {battle_key!r}")

        history = self._battle_histories.pop(battle_key, None)
        if history is not None:
            values = history.complete_values(self.policy.device)
            with torch.no_grad():
                values_mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
                new_tokens = self.policy.series(values, values_mask)[0]
            # SeriesTokenStore synchronously detaches the completed summary to
            # CPU, establishing a strict device-memory boundary between games.
            self._series_store.append(state.key, new_tokens)

        state.games_played += 1
        if battle.won:
            state.wins += 1
        elif battle.lost:
            state.losses += 1

        series_complete = state.complete
        if series_complete:
            self._drop_series(state)

        TeamPlayerMixin._finish_team_battle(
            self,
            battle,
            resample_team=series_complete,
        )


LOGGER = logging.getLogger(__name__)


def load_player_policy(
    checkpoint_path: Path | None,
    allow_random_init: bool = False,
    policy_store: CheckpointStore = DEFAULT_CHECKPOINT_STORE,
) -> PolicyNet:
    """
    Load and prepare a policy network for live player inference.

    Arguments:
        checkpoint_path: Path to the model checkpoint, or None for random weights.
        allow_random_init: Whether to allow random initialization when no checkpoint is given.
        policy_store: CheckpointStore used to load checkpoint weights.

    Returns:
        Evaluated, compiled policy network ready for inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        if not allow_random_init:
            raise ValueError("A checkpoint is required unless random init is explicitly allowed.")

        LOGGER.warning("Starting bot with randomly initialized policy weights.")
        resources = default_runtime_resources()
        policy = build_policy(ModelConfig.baseline(), resources).to(device)
        policy.eval()
        return policy

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    policy = policy_store.load_policy(checkpoint_path, device)
    episode = policy_store.load_training(checkpoint_path, policy)
    LOGGER.info(
        "Loaded checkpoint from %s (episode %d)",
        checkpoint_path,
        episode,
    )
    LOGGER.info("Running inference on device: %s", device)
    policy = compile_policy(policy, enable=device.type == "cuda")
    policy.eval()
    return policy
