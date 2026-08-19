"""Live policy player, command-line configuration, and Showdown listener lifecycle.

This module provides the core RLPlayer agent integrating neural network policy inference,
history token management, series token persistence, team sampling, and CLI configuration
for live Pokemon Showdown battles.
"""

import argparse
import asyncio
import logging
import os
import random
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
from poke_env import AccountConfiguration, LocalhostServerConfiguration, ServerConfiguration
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
from p0.teams.factory import build_team_source
from p0.teams.source import FileTeamSource, TeamSource
from p0.training.checkpoint import DEFAULT_POLICY_STORE, PolicyStore
from p0.training.config import load_config

_LIVE_HISTORY_CAPACITY = 2 * HISTORY_WINDOW
DEFAULT_BATTLE_FORMAT = FORMAT.bo3_format


@dataclass(slots=True)
class _LiveBattleHistory:
    """Own one battle's resident history and optional CPU spill archive."""

    spill_to_cpu: bool
    decision_count: int = 0
    cpu_chunks: list[torch.Tensor] = field(default_factory=list)
    resident_tokens: list[torch.Tensor] = field(default_factory=list)

    def append(self, token: torch.Tensor, device: torch.device) -> None:
        if self.spill_to_cpu and len(self.resident_tokens) == _LIVE_HISTORY_CAPACITY:
            chunk = torch.stack(self.resident_tokens[:HISTORY_WINDOW]).cpu()
            self.cpu_chunks.append(chunk)
            del self.resident_tokens[:HISTORY_WINDOW]

        self.resident_tokens.append(token.detach().to(device=device, dtype=torch.float32))
        self.decision_count += 1

    def recent_values(self, empty: torch.Tensor) -> torch.Tensor:
        if not self.resident_tokens:
            return empty
        return torch.stack(self.resident_tokens[-HISTORY_WINDOW:]).unsqueeze(0)

    def complete_values(self, device: torch.device) -> torch.Tensor:
        resident_values = torch.stack(self.resident_tokens)
        if not self.cpu_chunks:
            values = resident_values
        else:
            cpu_prefix = torch.cat(self.cpu_chunks).to(device)
            values = torch.cat((cpu_prefix, resident_values))

        if values.size(0) != self.decision_count:
            raise RuntimeError("Live battle history does not match its decision count")
        return values.unsqueeze(0)


@dataclass(slots=True)
class _LiveSeriesState:
    """Local identity and score for one live Showdown Bo3 series.

    Showdown creates a parent BestOfGame room and a separate child battle room
    for each game. poke-env only exposes the child battle tag, so the live player
    keeps the parent-equivalent identity locally for the lifetime of the series.
    """

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

        battle_id = getattr(battle, "battle_tag", None) or getattr(battle, "tag", None)
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
        key = getattr(battle, "battle_tag", None) or getattr(battle, "tag", None)
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
        """Return the parent-series state associated with a child battle.

        Showdown's child battle room IDs are intentionally game-scoped.  The
        opponent identity is stable across the child rooms in a Bo3, while the
        local sequence disambiguates a later Bo3 against the same opponent.
        """
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

        history_tokens, history_mask, history_age_ids = pack_history_tokens(values)

        series_key = self._series_for_battle(battle).key
        series_tokens, series_mask = self._series_store.get_tokens(
            [series_key], device=self.policy.device
        )
        return MemoryInputs(
            series_tokens=series_tokens,
            series_mask=series_mask,
            history_tokens=history_tokens,
            history_mask=history_mask,
            history_age_ids=history_age_ids,
        )

    def _append_history(self, battle: DoubleBattle, token: torch.Tensor) -> None:
        # token is the reducer pre-memory local summary, not the
        # post-memory cls readout used for the action/value decision. Store
        # it detached so the live battle cache is a snapshot of this completed
        # decision rather than a cross-turn autograd graph.
        key = self._battle_key(battle)
        history = self._battle_histories.get(key)
        if history is None:
            history = _LiveBattleHistory(spill_to_cpu=self.policy.device.type != "cpu")
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
                new_tokens = self.policy.series.resample_single_game(values)[0]
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
DEFAULT_CHALLENGE_LIMIT = 1_000_000
DEFAULT_CHECKPOINT_CANDIDATES = (Path("artifacts/checkpoints/ppo_checkpoint.pt"),)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(root_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root_dir / path
    return path.resolve()


def _resolve_path_list(root_dir: Path, values: Iterable[str]) -> list[Path]:
    resolved_paths = []
    for value in values:
        path = _resolve_path(root_dir, value)
        if path is not None:
            resolved_paths.append(path)
    return resolved_paths


def _split_server_urls(server: str) -> tuple[str, str]:
    websocket_url, separator, authentication_url = server.partition(",")
    if not separator:
        raise ValueError(
            "--server must be '<websocket_url>,<authentication_url>' when provided as a "
            "single value."
        )
    return websocket_url.strip(), authentication_url.strip()


def _build_server_configuration(
    websocket_url: str | None,
    authentication_url: str | None,
    server: str | None,
) -> ServerConfiguration:
    if server:
        websocket_url, authentication_url = _split_server_urls(server)

    if websocket_url and authentication_url:
        return ServerConfiguration(websocket_url, authentication_url)

    if websocket_url or authentication_url:
        raise ValueError("Both websocket and authentication URLs must be provided together.")

    return LocalhostServerConfiguration


def _resolve_checkpoint_path(root_dir: Path, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        if checkpoint.exists():
            return checkpoint
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

    for candidate in DEFAULT_CHECKPOINT_CANDIDATES:
        path = root_dir / candidate
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(
        "No checkpoint file found. Set SHOWDOWN_CHECKPOINT or pass --checkpoint."
    )


def _load_policy(
    checkpoint_path: Path | None,
    allow_random_init: bool,
    policy_store: PolicyStore = DEFAULT_POLICY_STORE,
) -> PolicyNet:
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
    episode = policy_store.load_training_state(checkpoint_path, policy)
    LOGGER.info(
        "Loaded checkpoint from %s (episode %d)",
        checkpoint_path,
        episode,
    )
    LOGGER.info("Running inference on device: %s", device)
    policy = compile_policy(policy, enable=device.type == "cuda")
    policy.eval()
    return policy


@dataclass(frozen=True, slots=True)
class RLBotConfig:
    username: str
    password: str | None
    battle_format: str
    websocket_url: str
    authentication_url: str
    checkpoint_path: Path | None
    team_files: list[Path]
    team_pool: str
    top_p: float
    max_concurrent_battles: int
    challenge_limit: int
    opponent: str | None
    allow_random_init: bool
    log_level: str


def parse_args(argv: list[str] | None = None) -> RLBotConfig:
    """Parse command line arguments and return structured bot configuration.

    Arguments:
      argv: list of command line argument strings or None to parse sys.argv

    Returns:
      RLBotConfig dataclass holding parsed runtime configuration values
    """
    app_defaults = load_config()
    root_dir = app_defaults.paths.repository_root
    bot_defaults = app_defaults.bot

    env_team_files = os.getenv("SHOWDOWN_TEAM_FILES", "")
    configured_team_files = [str(path) for path in bot_defaults.team_files]

    parser = argparse.ArgumentParser(description="Run the VGC RL Showdown bot.")
    parser.add_argument(
        "--server",
        default=os.getenv("SHOWDOWN_SERVER"),
        help="Combined showdown server config as '<websocket_url>,<authentication_url>'.",
    )
    parser.add_argument(
        "--websocket-url",
        default=os.getenv("SHOWDOWN_WS_URL", bot_defaults.websocket_url),
        help="Showdown websocket URL.",
    )
    parser.add_argument(
        "--authentication-url",
        default=os.getenv("SHOWDOWN_AUTH_URL", bot_defaults.authentication_url),
        help="Showdown authentication URL.",
    )
    parser.add_argument(
        "--username",
        default=os.getenv("SHOWDOWN_USERNAME", bot_defaults.username),
        help="Account username used for Showdown login.",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("SHOWDOWN_PASSWORD")
        or os.getenv("SHOWDOWN_BOT_PASSWORD")
        or bot_defaults.password,
        help="Account password used for Showdown login.",
    )
    parser.add_argument(
        "--format",
        dest="battle_format",
        default=os.getenv("SHOWDOWN_BATTLE_FORMAT", bot_defaults.battle_format),
        help="Bo3 format to queue for and accept challenges in.",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.getenv(
            "SHOWDOWN_CHECKPOINT",
            str(bot_defaults.checkpoint_path) if bot_defaults.checkpoint_path else None,
        ),
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--team-file",
        action="append",
        default=env_team_files.split(os.pathsep) if env_team_files else configured_team_files,
        help="Specific team file to include. Can be repeated.",
    )
    parser.add_argument(
        "--team-pool",
        choices=("all", "reduced"),
        default=os.getenv("SHOWDOWN_TEAM_POOL", "all"),
        help="Named team pool under the teams directory.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=float(os.getenv("SHOWDOWN_TOP_P", str(bot_defaults.top_p))),
        help="Top-p sampling threshold used by the policy.",
    )
    parser.add_argument(
        "--max-concurrent-battles",
        type=int,
        default=int(
            os.getenv("SHOWDOWN_MAX_CONCURRENT_BATTLES", str(bot_defaults.max_concurrent_battles))
        ),
        help="Maximum simultaneous battles.",
    )
    parser.add_argument(
        "--challenge-limit",
        type=int,
        default=int(os.getenv("SHOWDOWN_CHALLENGE_LIMIT", str(bot_defaults.challenge_limit))),
        help="How many incoming challenges to accept before exiting.",
    )
    parser.add_argument(
        "--opponent",
        default=os.getenv("SHOWDOWN_ACCEPT_OPPONENT", bot_defaults.opponent),
        help="Only accept challenges from this opponent username.",
    )
    parser.add_argument(
        "--allow-random-init",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("SHOWDOWN_ALLOW_RANDOM_INIT", bot_defaults.allow_random_init),
        help="Allow booting without a checkpoint.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("SHOWDOWN_LOG_LEVEL", bot_defaults.log_level),
        help="Python logging level.",
    )

    args = parser.parse_args(argv)

    server_configuration = _build_server_configuration(
        websocket_url=args.websocket_url,
        authentication_url=args.authentication_url,
        server=args.server,
    )
    team_files = _resolve_path_list(root_dir, args.team_file)
    checkpoint_path = _resolve_path(root_dir, args.checkpoint)

    if checkpoint_path is None and not args.allow_random_init:
        checkpoint_path = _resolve_checkpoint_path(root_dir, checkpoint_path)

    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0.0, 1.0].")

    if args.battle_format != DEFAULT_BATTLE_FORMAT:
        raise ValueError(f"--format must match the RLPlayer Bo3 format {DEFAULT_BATTLE_FORMAT!r}.")

    if args.max_concurrent_battles < 1:
        raise ValueError("--max-concurrent-battles must be at least 1.")

    if args.challenge_limit < 1:
        raise ValueError("--challenge-limit must be at least 1.")

    return RLBotConfig(
        username=args.username,
        password=args.password,
        battle_format=args.battle_format,
        websocket_url=server_configuration.websocket_url,
        authentication_url=server_configuration.authentication_url,
        checkpoint_path=checkpoint_path,
        team_files=team_files,
        team_pool=args.team_pool,
        top_p=args.top_p,
        max_concurrent_battles=args.max_concurrent_battles,
        challenge_limit=args.challenge_limit,
        opponent=args.opponent,
        allow_random_init=args.allow_random_init,
        log_level=args.log_level.upper(),
    )


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=logging.getLevelNamesMapping().get(level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


async def run_bot(
    config: RLBotConfig,
    policy_store: PolicyStore = DEFAULT_POLICY_STORE,
) -> None:
    """Boot and run the RL bot Showdown listener process.

    Arguments:
      config: RLBotConfig containing connection, policy, and team options
      policy_store: PolicyStore implementation for checkpoint loading

    Returns:
      None
    """
    poke_env_patches.install()
    app_config = load_config()

    team_source = (
        FileTeamSource.from_files(config.team_files)
        if config.team_files
        else build_team_source(
            app_config.teams.all if config.team_pool == "all" else app_config.teams.reduced
        )
    )
    checkpoint_path = config.checkpoint_path

    policy = _load_policy(
        checkpoint_path,
        allow_random_init=config.allow_random_init,
        policy_store=policy_store,
    )
    server_configuration = ServerConfiguration(
        config.websocket_url,
        config.authentication_url,
    )
    account_configuration = AccountConfiguration(config.username, config.password)

    bot_player = RLPlayer(
        policy=policy,
        top_p=config.top_p,
        observation_builder=ObservationBuilder(policy.resources),
        team_rng=random.Random(),
        account_configuration=account_configuration,
        battle_format=config.battle_format,
        server_configuration=server_configuration,
        team_source=team_source,
        max_concurrent_battles=config.max_concurrent_battles,
    )

    LOGGER.info(
        "Starting RL bot as '%s' against %s using %s",
        config.username,
        config.websocket_url,
        checkpoint_path if checkpoint_path is not None else "random-init policy",
    )

    if config.opponent:
        LOGGER.info("Accepting challenges only from '%s'", config.opponent)
    else:
        LOGGER.info("Accepting challenges from any opponent")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    accept_task = asyncio.create_task(
        bot_player.accept_challenges(config.opponent, config.challenge_limit)
    )
    stop_task = asyncio.create_task(stop_event.wait())

    try:
        done, pending = await asyncio.wait(
            {accept_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

        if stop_task in done and stop_event.is_set():
            LOGGER.info("Shutdown signal received, stopping bot listener.")
            accept_task.cancel()
            await asyncio.gather(accept_task, return_exceptions=True)
        else:
            await accept_task
    finally:
        await bot_player.ps_client.stop_listening()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for running the RL bot process."""
    try:
        config = parse_args(argv)
        _configure_logging(config.log_level)
        asyncio.run(run_bot(config))
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
