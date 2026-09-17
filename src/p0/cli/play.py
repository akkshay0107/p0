"""Command-line interface for running the Showdown RL bot."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
from pathlib import Path

from poke_env import AccountConfiguration, LocalhostServerConfiguration, ServerConfiguration

from p0.model.observation_builder import ObservationBuilder
from p0.rl_player import DEFAULT_BATTLE_FORMAT, RLPlayer, load_player_policy
from p0.runtime import poke_env_patches
from p0.teams.factory import build_team_source
from p0.teams.source import FileTeamSource
from p0.training.checkpoint import DEFAULT_CHECKPOINT_STORE, CheckpointStore
from p0.training.config import load_config

logger = logging.getLogger("p0.cli.play")


def _build_parser() -> argparse.ArgumentParser:
    defaults = load_config()
    bot = defaults.bot

    parser = argparse.ArgumentParser(prog="p0-play", description="Run the Pokemon Showdown RL bot.")
    parser.add_argument(
        "--server",
        default=os.getenv("SHOWDOWN_SERVER"),
        help="Server as '<websocket_url>,<authentication_url>'.",
    )
    parser.add_argument(
        "--websocket-url",
        default=os.getenv("SHOWDOWN_WS_URL", bot.websocket_url),
        help="Showdown websocket URL.",
    )
    parser.add_argument(
        "--authentication-url",
        default=os.getenv("SHOWDOWN_AUTH_URL", bot.authentication_url),
        help="Showdown authentication URL.",
    )
    parser.add_argument(
        "--username",
        default=os.getenv("SHOWDOWN_USERNAME", bot.username),
        help="Showdown account username.",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("SHOWDOWN_PASSWORD")
        or os.getenv("SHOWDOWN_BOT_PASSWORD")
        or bot.password,
        help="Showdown account password.",
    )
    parser.add_argument(
        "--format",
        dest="battle_format",
        default=os.getenv("SHOWDOWN_BATTLE_FORMAT", bot.battle_format),
        help="Battle format to queue and play.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=os.getenv(
            "SHOWDOWN_CHECKPOINT",
            str(bot.checkpoint_path) if bot.checkpoint_path else None,
        ),
        help="Path to the policy checkpoint.",
    )
    parser.add_argument(
        "--team-file",
        action="append",
        type=Path,
        default=None,
        help="Team file to include. Can be repeated.",
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
        default=float(os.getenv("SHOWDOWN_TOP_P", str(bot.top_p))),
        help="Top-p sampling threshold.",
    )
    parser.add_argument(
        "--challenge-limit",
        type=int,
        default=int(os.getenv("SHOWDOWN_CHALLENGE_LIMIT", str(bot.challenge_limit))),
        help="Number of challenges to accept before exiting.",
    )
    parser.add_argument(
        "--opponent",
        default=os.getenv("SHOWDOWN_ACCEPT_OPPONENT", bot.opponent),
        help="Only accept challenges from this opponent username.",
    )
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        default=os.getenv("SHOWDOWN_ALLOW_RANDOM_INIT", "").lower() in {"1", "true", "yes"}
        or bot.allow_random_init,
        help="Allow running with randomly initialized weights if no checkpoint exists.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("SHOWDOWN_LOG_LEVEL", bot.log_level),
        help="Logging level.",
    )
    return parser


def _build_server_configuration(args: argparse.Namespace) -> ServerConfiguration:
    if args.server:
        parts = [p.strip() for p in args.server.split(",", 1)]
        if len(parts) == 2:
            return ServerConfiguration(parts[0], parts[1])
        raise ValueError("--server must be '<websocket_url>,<authentication_url>'.")
    if args.websocket_url and args.authentication_url:
        return ServerConfiguration(args.websocket_url, args.authentication_url)
    return LocalhostServerConfiguration


async def run_bot(
    args: argparse.Namespace,
    policy_store: CheckpointStore = DEFAULT_CHECKPOINT_STORE,
) -> None:
    """Boot and run the RL bot Showdown listener process."""
    poke_env_patches.install()
    app_config = load_config()

    if args.battle_format != DEFAULT_BATTLE_FORMAT:
        raise ValueError(f"--format must match the RLPlayer Bo3 format {DEFAULT_BATTLE_FORMAT!r}.")

    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0.0, 1.0].")

    team_files = args.team_file or [Path(p) for p in app_config.bot.team_files if Path(p).exists()]
    team_source = (
        FileTeamSource.from_files(team_files)
        if team_files
        else build_team_source(
            app_config.teams.all if args.team_pool == "all" else app_config.teams.reduced,
            expected_format_id=args.battle_format,
        )
    )

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    policy = load_player_policy(
        checkpoint_path,
        allow_random_init=args.allow_random_init,
        policy_store=policy_store,
    )
    server_configuration = _build_server_configuration(args)
    account_configuration = AccountConfiguration(args.username, args.password)

    bot_player = RLPlayer(
        policy=policy,
        top_p=args.top_p,
        observation_builder=ObservationBuilder(policy.resources),
        team_rng=random.Random(),
        account_configuration=account_configuration,
        battle_format=args.battle_format,
        server_configuration=server_configuration,
        team_source=team_source,
        max_concurrent_battles=1,
    )

    logger.info(
        "Starting RL bot as '%s' against %s (checkpoint: %s)",
        args.username,
        server_configuration.websocket_url,
        checkpoint_path or "random-init",
    )

    try:
        await bot_player.accept_challenges(args.opponent, args.challenge_limit)
    except asyncio.CancelledError:
        logger.info("Bot listener received cancellation.")
    finally:
        await bot_player.ps_client.stop_listening()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for running the RL bot process."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        asyncio.run(run_bot(args))
    except (KeyboardInterrupt, SystemExit):
        logger.info("Exiting on user interrupt.")
        return 0
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Run error: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
