import argparse
import asyncio
import logging
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from poke_env import AccountConfiguration, LocalhostServerConfiguration, ServerConfiguration
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import DefaultBattleOrder, Player
from torch.distributions import Categorical

from src.env import MegaEnv
from src.lookups import ACT_SIZE
from src.model import observation_builder
from src.model.policy import PolicyNet
from src.team_picker import RandomTeamFromPool
from src.train.utils import initial_state, load_checkpoint


class RLPlayer(Player):
    """
    Class that plays moves as per the trained policy net.
    """

    def __init__(
        self,
        policy: PolicyNet,
        p: float = 0.9,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.policy = policy

        assert 0.0 <= p <= 1.0
        self.p = p
        self.state = None

    def _apply_top_p(self, logits: torch.Tensor) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > self.p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        sorted_logits[sorted_indices_to_remove] = float("-inf")

        return sorted_logits.scatter(-1, sorted_indices, sorted_logits)

    def _top_p(self, obs, action_mask):
        if self.state is None:
            batch_size = 1
            device = self.policy.device
            self.state = initial_state(self.policy, batch_size, device)

        with torch.no_grad():
            tokens, aux = self.policy.encoder(obs, aux=True)
            numerical = obs.numerical
            if numerical.dim() == 2:
                numerical = numerical.unsqueeze(0)

            logits, _, _, self.state, z = self.policy.actor(
                tokens, aux, numerical, state=self.state, action_mask=action_mask, sample=False
            )

            logits1 = logits[:, 0]
            if action_mask is not None:
                logits1 = logits1.masked_fill(action_mask[:, 0] == 0, float("-inf"))

            p1_logits_top_p = self._apply_top_p(logits1)
            cat1 = Categorical(logits=p1_logits_top_p)
            action1 = cat1.sample()  # (B,)

            a1_emb = self.policy.actor._build_action_context(action1, tokens, aux, numerical)
            logits2 = self.policy.actor.head2(torch.cat([z, a1_emb], dim=-1))

            # Combine and apply sequential masks
            logits = torch.stack([logits1, logits2], dim=1)
            if action_mask is not None:
                is_tp_t = (numerical[:, 25, 2] > 0.5).bool()
                logits = self.policy.actor._apply_sequential_masks(
                    logits, action1, action_mask, is_tp_t
                )

            p2_logits_top_p = self._apply_top_p(logits[:, 1])
            cat2 = Categorical(logits=p2_logits_top_p)
            action2 = cat2.sample()  # (B,)

            return torch.stack([action1, action2], dim=-1)

    def _get_action(self, battle: AbstractBattle, is_tp: bool):
        obs = self.get_observation(battle)
        action_mask_list = MegaEnv.get_action_mask(battle)
        action_mask = torch.tensor([action_mask_list[:ACT_SIZE], action_mask_list[ACT_SIZE:]])

        obs = obs.unsqueeze(0).to(self.policy.device)
        with torch.no_grad():
            actions = self._top_p(
                obs,
                action_mask.unsqueeze(0).to(self.policy.device),
            )
        return actions[0].cpu().numpy()

    def choose_move(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        if battle._wait:
            return DefaultBattleOrder()
        return MegaEnv.action_to_order(self._get_action(battle, False), battle)

    def get_observation(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        return observation_builder.from_battle(battle)

    def teampreview(self, battle: AbstractBattle) -> str:
        assert isinstance(battle, DoubleBattle)
        # Reset state at the beginning of each battle's Team Preview
        self.state = None
        action = self._get_action(battle, True)
        order = MegaEnv.action_to_order(action, battle)
        return order.message

    def _battle_finished_callback(self, battle: AbstractBattle):
        # Reset state to prevent leaks or state carry-over across battles
        self.state = None


LOGGER = logging.getLogger(__name__)
DEFAULT_BATTLE_FORMAT = "gen9championsvgc2026regma"
DEFAULT_CHALLENGE_LIMIT = 1_000_000
DEFAULT_CHECKPOINT_CANDIDATES = (Path("checkpoints/ppo_checkpoint.pt"),)


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


def _load_team_pool(team_paths: Iterable[Path]) -> RandomTeamFromPool:
    teams = []
    for path in sorted(team_paths):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        teams.append(path.read_text(encoding="utf-8").strip())

    teams = [team for team in teams if team]
    if not teams:
        raise FileNotFoundError("No usable team files were found for the bot.")

    return RandomTeamFromPool(teams)


def _resolve_team_paths(
    root_dir: Path, teams_dir: Path | None, team_files: list[Path]
) -> list[Path]:
    if team_files:
        return team_files

    resolved_dir = teams_dir or (root_dir / "teams")
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Teams directory not found: {resolved_dir}")

    return sorted(path for path in resolved_dir.iterdir() if path.is_file())


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


def _load_policy(checkpoint_path: Path | None, allow_random_init: bool) -> PolicyNet:
    # Use recommended dimensions if unspecified
    policy = PolicyNet()

    if checkpoint_path is None:
        if not allow_random_init:
            raise ValueError("A checkpoint is required unless random init is explicitly allowed.")
        LOGGER.warning("Starting bot with randomly initialized policy weights.")
        policy.eval()
        return policy

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    episode = load_checkpoint(checkpoint_path, policy)
    LOGGER.info(
        "Loaded checkpoint from %s%s",
        checkpoint_path,
        f" (episode {episode})" if episode is not None else "",
    )
    policy.eval()
    return policy


@dataclass(frozen=True)
class RLBotConfig:
    username: str
    password: str | None
    battle_format: str
    websocket_url: str
    authentication_url: str
    checkpoint_path: Path | None
    teams_dir: Path | None
    team_files: list[Path]
    top_p: float
    max_concurrent_battles: int
    challenge_limit: int
    opponent: str | None
    accept_open_team_sheet: bool
    allow_random_init: bool
    log_level: str


def parse_args(argv: list[str] | None = None) -> RLBotConfig:
    root_dir = Path(__file__).resolve().parent.parent
    env_team_files = os.getenv("SHOWDOWN_TEAM_FILES", "")
    parser = argparse.ArgumentParser(description="Run the VGC RL Showdown bot.")
    parser.add_argument(
        "--server",
        default=os.getenv("SHOWDOWN_SERVER"),
        help="Combined showdown server config as '<websocket_url>,<authentication_url>'.",
    )
    parser.add_argument(
        "--websocket-url",
        default=os.getenv("SHOWDOWN_WS_URL"),
        help="Showdown websocket URL.",
    )
    parser.add_argument(
        "--authentication-url",
        default=os.getenv("SHOWDOWN_AUTH_URL"),
        help="Showdown authentication URL.",
    )
    parser.add_argument(
        "--username",
        default=os.getenv("SHOWDOWN_USERNAME", "Bot"),
        help="Account username used for Showdown login.",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("SHOWDOWN_PASSWORD") or os.getenv("SHOWDOWN_BOT_PASSWORD"),
        help="Account password used for Showdown login.",
    )
    parser.add_argument(
        "--format",
        dest="battle_format",
        default=os.getenv("SHOWDOWN_BATTLE_FORMAT", DEFAULT_BATTLE_FORMAT),
        help="Battle format to queue for and accept challenges in.",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.getenv("SHOWDOWN_CHECKPOINT"),
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--teams-dir",
        default=os.getenv("SHOWDOWN_TEAMS_DIR"),
        help="Directory containing Showdown export team files.",
    )
    parser.add_argument(
        "--team-file",
        action="append",
        default=env_team_files.split(os.pathsep) if env_team_files else [],
        help="Specific team file to include. Can be repeated.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=float(os.getenv("SHOWDOWN_TOP_P", "0.9")),
        help="Top-p sampling threshold used by the policy.",
    )
    parser.add_argument(
        "--max-concurrent-battles",
        type=int,
        default=int(os.getenv("SHOWDOWN_MAX_CONCURRENT_BATTLES", "10")),
        help="Maximum simultaneous battles.",
    )
    parser.add_argument(
        "--challenge-limit",
        type=int,
        default=int(os.getenv("SHOWDOWN_CHALLENGE_LIMIT", str(DEFAULT_CHALLENGE_LIMIT))),
        help="How many incoming challenges to accept before exiting.",
    )
    parser.add_argument(
        "--opponent",
        default=os.getenv("SHOWDOWN_ACCEPT_OPPONENT"),
        help="Only accept challenges from this opponent username.",
    )
    parser.add_argument(
        "--accept-open-team-sheet",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("SHOWDOWN_ACCEPT_OPEN_TEAM_SHEET", True),
        help="Whether the bot accepts open team sheet battles.",
    )
    parser.add_argument(
        "--allow-random-init",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("SHOWDOWN_ALLOW_RANDOM_INIT", False),
        help="Allow booting without a checkpoint.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("SHOWDOWN_LOG_LEVEL", "INFO"),
        help="Python logging level.",
    )
    args = parser.parse_args(argv)

    server_configuration = _build_server_configuration(
        websocket_url=args.websocket_url,
        authentication_url=args.authentication_url,
        server=args.server,
    )
    teams_dir = _resolve_path(root_dir, args.teams_dir)
    team_files = _resolve_path_list(root_dir, args.team_file)
    checkpoint_path = _resolve_path(root_dir, args.checkpoint)
    if checkpoint_path is None and not args.allow_random_init:
        checkpoint_path = _resolve_checkpoint_path(root_dir, checkpoint_path)
    if args.top_p < 0.0 or args.top_p > 1.0:
        raise ValueError("--top-p must be between 0.0 and 1.0.")
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
        teams_dir=teams_dir,
        team_files=team_files,
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
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


async def run_bot(config: RLBotConfig) -> None:
    root_dir = Path(__file__).resolve().parent.parent
    team_paths = _resolve_team_paths(root_dir, config.teams_dir, config.team_files)
    team = _load_team_pool(team_paths)
    checkpoint_path = config.checkpoint_path
    policy = _load_policy(checkpoint_path, allow_random_init=config.allow_random_init)
    server_configuration = ServerConfiguration(
        config.websocket_url,
        config.authentication_url,
    )
    account_configuration = AccountConfiguration(config.username, config.password)
    bot_player = RLPlayer(
        policy=policy,
        p=config.top_p,
        account_configuration=account_configuration,
        battle_format=config.battle_format,
        server_configuration=server_configuration,
        team=team,
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
