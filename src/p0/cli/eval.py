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

from p0.evaluation.harness import EvaluationHarness, MatchupResult
from p0.format_config import load_active_global_contract
from p0.persistence import atomic_json_save
from p0.runtime.showdown import start_showdown_servers
from p0.teams.corpus import CorpusSplit
from p0.training.checkpoint import DEFAULT_CHECKPOINT_STORE
from p0.training.config import load_config

logger = logging.getLogger("p0.cli.eval")

_VALID_SPLITS = {
    "train": CorpusSplit.TRAIN,
    "val": CorpusSplit.VALIDATION,
    "validation": CorpusSplit.VALIDATION,
    "test": CorpusSplit.TEST,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="p0-eval", description="Evaluate a trained policy.")
    parser.add_argument("--checkpoint", type=Path, help="Path to policy checkpoint to evaluate.")
    parser.add_argument(
        "--opponent",
        choices=("random", "max_power", "simple_heuristics"),
        default="random",
        help="Named baseline opponent (default: random).",
    )
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        help="Opponent policy checkpoint (overrides --opponent).",
    )
    parser.add_argument("--teams-path", type=Path, default=None, help="Team pool directory.")
    parser.add_argument(
        "--split",
        choices=("train", "val", "validation", "test"),
        default=None,
        help="Corpus split to evaluate against.",
    )
    parser.add_argument("--episodes", type=int, help="Number of episodes per matchup.")
    parser.add_argument("--seed", type=int, help="Random seed for evaluations.")
    parser.add_argument("--report-dir", type=Path, help="Directory to save evaluation reports.")
    parser.add_argument("--port", type=int, default=8120, help="Showdown port to use.")
    parser.add_argument("--config", type=Path, help="Path to global YAML configuration file.")
    return parser


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
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    episodes = args.episodes or config.evaluation.episodes_per_matchup
    seed = args.seed if args.seed is not None else config.evaluation.seed
    report_dir = args.report_dir or Path(config.evaluation.report_dir)

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        manifest = load_active_global_contract(config.paths.data_root / "runtime_manifest.json")
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
    opponent_name = args.opponent
    if args.opponent_checkpoint is not None:
        try:
            policy_b = DEFAULT_CHECKPOINT_STORE.load_policy(args.opponent_checkpoint, device)
            policy_b.eval()
            opponent_name = "OpponentCheckpoint"
        except (OSError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            logger.error("Failed to load opponent checkpoint: %s", exc)
            return 1

    teams_path = args.teams_path or config.teams.all
    split_filter = _VALID_SPLITS.get(args.split) if args.split else None

    harness = EvaluationHarness(
        teams_path=teams_path,
        format_id=config.bot.battle_format,
        episodes_per_matchup=episodes,
        seed=seed,
        port=args.port,
        split=split_filter,
    )

    try:
        source = harness.build_team_source()
    except (FileNotFoundError, OSError, ValueError) as exc:
        logger.error("Evaluation team source is unavailable: %s", exc)
        return 1

    async def run() -> MatchupResult:
        logger.info("Starting local Showdown server on port %d...", args.port)
        with start_showdown_servers(1, ports=(args.port,)) as servers:
            return await harness.run_matchup(
                name_a="PlayerCheckpoint" if policy_a else "RandomA",
                policy_a=policy_a,
                name_b=opponent_name,
                policy_b=policy_b if policy_b is not None else args.opponent,
                team_source=source,
                server_configuration=servers[0],
            )

    try:
        matchup_result = asyncio.run(run())
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.exception("Evaluation execution failed: %s", exc)
        return 1

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "evaluation_report.json"
    report = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "episodes": episodes,
        "seed": seed,
        "global_contract_sha256": manifest.global_sha256,
        "policy_a": str(args.checkpoint) if args.checkpoint else "Random",
        "policy_b": (
            str(args.opponent_checkpoint)
            if args.opponent_checkpoint
            else f"Baseline:{args.opponent}"
        ),
        "teams_path": str(teams_path.resolve()) if teams_path else None,
        "split": args.split or "default",
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
        "matchup": matchup_result.to_dict(),
        "matchups": [matchup_result.to_dict()],
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
