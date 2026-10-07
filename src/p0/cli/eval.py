"""Command-line interface for policy evaluation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from p0.cli import LOG_FORMAT
from p0.evaluation.harness import EvaluationHarness, MatchupResult
from p0.format_config import load_active_global_contract
from p0.model.policy import PolicyNet
from p0.persistence import atomic_json_save
from p0.runtime.showdown import local_server_configuration, start_showdown_servers
from p0.teams.source import TeamSource
from p0.training.checkpoint import DEFAULT_CHECKPOINT_STORE
from p0.training.config import load_config
from p0.training.utils import default_device

logger = logging.getLogger("p0.cli.eval")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="p0-eval", description="Evaluate a trained policy.")
    parser.add_argument("--checkpoint", type=Path, help="Path to policy checkpoint to evaluate.")
    opponent_choice = parser.add_mutually_exclusive_group()
    opponent_choice.add_argument(
        "--opponent",
        choices=("random", "max_power", "simple_heuristics"),
        help="Named baseline opponent (default: random).",
    )
    opponent_choice.add_argument(
        "--opponent-checkpoint",
        type=Path,
        help="Opponent policy checkpoint.",
    )
    parser.add_argument("--teams-path", type=Path, default=None, help="Team pool directory.")
    parser.add_argument("--episodes", type=int, help="Number of episodes per matchup.")
    parser.add_argument("--seed", type=int, help="Random seed for evaluations.")
    parser.add_argument("--report-dir", type=Path, help="Directory to save evaluation reports.")
    parser.add_argument("--config", type=Path, help="Path to global YAML configuration file.")
    return parser


async def _run_matchup(
    harness: EvaluationHarness,
    source: TeamSource,
    policy_a: PolicyNet | None,
    opponent_name: str,
    opponent: PolicyNet | str,
) -> MatchupResult:
    """Start one local Showdown server and play the configured matchup on it."""
    with start_showdown_servers(1) as servers:
        server_configuration = local_server_configuration(servers[0].port)
        return await harness.run_matchup(
            name_a="PlayerCheckpoint" if policy_a else "RandomA",
            policy_a=policy_a,
            name_b=opponent_name,
            policy_b=opponent,
            team_source=source,
            server_configuration=server_configuration,
        )


def main(argv: list[str] | None = None) -> int:
    """Run evaluation against baseline opponents or checkpoints and persist report."""
    args = _parser().parse_args(argv)

    try:
        config = load_config(args.config)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Error loading configuration: {exc}", file=sys.stderr)
        return 1

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
    )

    episodes = config.evaluation.episodes_per_matchup if args.episodes is None else args.episodes
    seed = args.seed if args.seed is not None else config.evaluation.seed
    report_dir = config.evaluation.report_dir if args.report_dir is None else args.report_dir
    if episodes <= 0:
        print("Error: --episodes must be a positive integer.", file=sys.stderr)
        return 1
    if seed < 0:
        print("Error: --seed must be a non-negative integer.", file=sys.stderr)
        return 1

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = default_device()

    try:
        manifest = load_active_global_contract()
    except (OSError, KeyError, TypeError, ValueError) as exc:
        logger.error("Could not validate runtime contract: %s", exc)
        return 1

    policy_a = None
    if args.checkpoint is not None:
        try:
            policy_a = DEFAULT_CHECKPOINT_STORE.load_policy(args.checkpoint, device)
            policy_a.eval()
        except (OSError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            logger.error("Failed to load policy checkpoint A: %s", exc)
            return 1

    policy_b = None
    opponent_name = args.opponent or "random"
    if args.opponent_checkpoint is not None:
        try:
            policy_b = DEFAULT_CHECKPOINT_STORE.load_policy(args.opponent_checkpoint, device)
            policy_b.eval()
            opponent_name = "OpponentCheckpoint"
        except (OSError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            logger.error("Failed to load opponent checkpoint: %s", exc)
            return 1

    teams_path = config.teams.all if args.teams_path is None else args.teams_path

    harness = EvaluationHarness(
        teams_path=teams_path,
        episodes_per_matchup=episodes,
        seed=seed,
    )

    try:
        source = harness.build_team_source()
    except (FileNotFoundError, OSError, ValueError) as exc:
        logger.error("Evaluation team source is unavailable: %s", exc)
        return 1

    try:
        matchup_result = asyncio.run(
            _run_matchup(
                harness,
                source,
                policy_a,
                opponent_name,
                policy_b if policy_b is not None else opponent_name,
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.exception("Evaluation execution failed: %s", exc)
        return 1

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "evaluation_report.json"
    matchup = matchup_result.to_dict()
    report = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "episodes": episodes,
        "seed": seed,
        "global_contract_sha256": manifest.global_sha256,
        "policy_a": str(args.checkpoint) if args.checkpoint else "Random",
        "policy_b": (
            str(args.opponent_checkpoint)
            if args.opponent_checkpoint
            else f"Baseline:{opponent_name}"
        ),
        "teams_path": str(teams_path.resolve()) if teams_path else None,
        "checkpoints": {
            "policy_a_sha256": (
                hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
                if args.checkpoint is not None
                else None
            ),
            "policy_b_sha256": (
                hashlib.sha256(args.opponent_checkpoint.read_bytes()).hexdigest()
                if args.opponent_checkpoint is not None
                else None
            ),
        },
        "matchup": matchup,
        "matchups": [matchup],
    }

    try:
        atomic_json_save(report_path, report)
        logger.info("Evaluation report written to: %s", report_path)
    except (OSError, TypeError, ValueError) as exc:
        logger.error("Failed to save evaluation report: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
