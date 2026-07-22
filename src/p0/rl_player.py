import argparse
import asyncio
import logging
import os
import random
import signal
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from poke_env import AccountConfiguration, LocalhostServerConfiguration, ServerConfiguration
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import DefaultBattleOrder, Player

from p0.battle.legality import action_mask
from p0.battle.series import MAX_PRIOR_GAMES
from p0.format_config import FORMAT
from p0.model.architecture_contract import HISTORY_WINDOW, SERIES_SLOTS
from p0.model.cls_reducer import pack_history_tokens
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.observation_builder import ObservationBuilder
from p0.model.policy import PolicyNet
from p0.model.resources import default_runtime_resources
from p0.runtime import poke_env_patches
from p0.runtime.poke_env_action_adapter import action_to_order
from p0.runtime.poke_env_battle_adapter import battle_view
from p0.teams.source import FileTeamSource, TeamSource
from p0.training.checkpoint import DEFAULT_POLICY_STORE, PolicyStore
from p0.training.config import load_config


class RLPlayer(Player):
    """
    Class that plays moves as per the trained policy net.
    """

    def __init__(
        self,
        policy: PolicyNet,
        top_p: float = 0.9,
        *args,
        observation_builder: ObservationBuilder,
        team_rng: random.Random,
        team_source: TeamSource | None = None,
        **kwargs,
    ):
        self.team_source = team_source
        self.team_rng = team_rng
        if team_source is not None:
            if "team" in kwargs:
                raise ValueError("Pass either team or team_source, not both")
            kwargs["team"] = team_source.sample(self.team_rng).packed
        super().__init__(*args, **kwargs)
        poke_env_patches.install(self.logger)
        self.policy = policy
        self.observation_builder = observation_builder

        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}.")
        self.top_p = top_p
        self.top_p = top_p
        self._memory_model_id = id(policy)
        self._battle_history: dict[str, list[torch.Tensor]] = {}
        self._series_tokens: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._prior_game_histories: dict[str, list[torch.Tensor]] = {}

    @staticmethod
    def _battle_key(battle: DoubleBattle) -> str:
        key = getattr(battle, "battle_tag", None) or getattr(battle, "tag", None)
        if not key:
            raise ValueError("Live battle has no stable battle identifier")
        return str(key)

    @staticmethod
    def _series_key(battle: DoubleBattle) -> str:
        return str(getattr(battle, "_p0_series_id", RLPlayer._battle_key(battle).split("-game")[0]))

    def set_series_game_histories(
        self,
        series_id: str,
        game_histories: Sequence[torch.Tensor],
    ) -> None:
        """Resample completed game turn histories into series tokens for subsequent games."""
        if len(game_histories) > MAX_PRIOR_GAMES:
            raise ValueError(
                f"At most {MAX_PRIOR_GAMES} completed game histories can condition a game"
            )
        prepared_histories = []
        for hist in game_histories:
            if hist.dim() == 2:
                hist = hist.unsqueeze(0)
            prepared_histories.append(hist.to(self.policy.device))
        tokens, mask = self.policy.encode_series(prepared_histories)
        self._series_tokens[series_id] = (tokens.detach(), mask.detach())

    def invalidate_memory_for_model_reload(self) -> None:
        """Drop orchestration tensors when a policy artifact is replaced."""
        self._battle_history.clear()
        self._series_tokens.clear()
        self._prior_game_histories.clear()
        self._memory_model_id = id(self.policy)

    def _memory_inputs(self, battle: DoubleBattle):
        if id(self.policy) != self._memory_model_id:
            self.invalidate_memory_for_model_reload()
        key = self._battle_key(battle)
        history = self._battle_history.get(key, [])
        if history:
            values = torch.stack(history[-HISTORY_WINDOW:]).unsqueeze(0).to(self.policy.device)
        else:
            values = torch.zeros((1, 0, self.policy.d_model), device=self.policy.device)
        history_tokens, history_mask, history_age_ids = pack_history_tokens(values)
        series = self._series_tokens.get(self._series_key(battle))
        if series is None:
            series_tokens = torch.zeros(
                (1, SERIES_SLOTS, self.policy.d_model), device=self.policy.device
            )
            series_mask = torch.zeros(
                (1, SERIES_SLOTS), dtype=torch.bool, device=self.policy.device
            )
        else:
            series_tokens, series_mask = series
            series_tokens = series_tokens.to(self.policy.device)
            series_mask = series_mask.to(self.policy.device)
        return series_tokens, series_mask, history_tokens, history_mask, history_age_ids

    def _append_history(self, battle: DoubleBattle, token: torch.Tensor) -> None:
        key = self._battle_key(battle)
        entries = self._battle_history.setdefault(key, [])
        entries.append(token.detach().to(torch.float32).cpu())

    def _get_action(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        view = battle_view(battle)
        obs = self.observation_builder.build(view)
        mask = torch.from_numpy(action_mask(view.decision))

        obs = obs.unsqueeze(0).to(self.policy.device)
        mask = mask.unsqueeze(0).to(self.policy.device)
        with torch.no_grad():
            out = self.policy.act_obs(obs, mask, *self._memory_inputs(battle), top_p=self.top_p)
        self._append_history(battle, out.history_token[0])
        return out.actions[0].cpu().numpy()

    def choose_move(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        if battle._wait:
            return DefaultBattleOrder()
        return action_to_order(self._get_action(battle), battle)

    def get_observation(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        return self.observation_builder.build(battle_view(battle))

    def teampreview(self, battle: AbstractBattle) -> str:
        assert isinstance(battle, DoubleBattle)
        self._battle_history.pop(self._battle_key(battle), None)
        action = self._get_action(battle)
        order = action_to_order(action, battle)
        return order.message

    def _battle_finished_callback(self, battle: AbstractBattle):
        if not isinstance(battle, DoubleBattle):
            return
        key = self._battle_key(battle)
        series_key = self._series_key(battle)
        history = self._battle_history.pop(key, None)
        full_game_history = (
            torch.stack(history).unsqueeze(0)
            if history
            else torch.zeros((1, 0, self.policy.d_model))
        )
        prior = self._prior_game_histories.get(series_key, [])
        updated = [*prior, full_game_history]
        self._prior_game_histories[series_key] = updated
        if len(updated) <= MAX_PRIOR_GAMES:
            self.set_series_game_histories(series_key, updated)

        if battle.finished and getattr(battle, "_p0_series_complete", True):
            self._series_tokens.pop(series_key, None)
            self._prior_game_histories.pop(series_key, None)
        if self.team_source is not None:
            self.update_team(self.team_source.sample(self.team_rng).packed)


LOGGER = logging.getLogger(__name__)
DEFAULT_BATTLE_FORMAT = FORMAT.battle_format
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
    policy.eval()
    return policy


@dataclass(slots=True)
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
    accept_open_team_sheet: bool
    allow_random_init: bool
    log_level: str


def parse_args(argv: list[str] | None = None) -> RLBotConfig:
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
        help="Battle format to queue for and accept challenges in.",
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
        default=os.getenv(
            "SHOWDOWN_TEAM_POOL", str(app_defaults.environment.agent_team_source.path)
        ),
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
        "--accept-open-team-sheet",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("SHOWDOWN_ACCEPT_OPEN_TEAM_SHEET", bot_defaults.accept_open_team_sheet),
        help="Whether the bot accepts open team sheet battles.",
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
        accept_open_team_sheet=args.accept_open_team_sheet,
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
    poke_env_patches.install()
    root_dir = load_config().paths.repository_root
    team_source = (
        FileTeamSource.from_files(config.team_files)
        if config.team_files
        else FileTeamSource(root_dir / "teams" / config.team_pool)
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
        accept_open_team_sheet=config.accept_open_team_sheet,
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
