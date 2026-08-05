"""Command-line interface for policy evaluation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from p0.evaluation.harness import EvaluationHarness
from p0.format_config import load_active_runtime_manifest
from p0.persistence import atomic_json_save
from p0.training.checkpoint import DEFAULT_POLICY_STORE
from p0.training.config import load_config

logger = logging.getLogger("p0.cli.eval")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="p0-eval", description="Evaluate a trained policy.")
    parser.add_argument("--checkpoint", type=Path, help="Path to policy checkpoint to evaluate.")
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        help="Optional opponent policy checkpoint (omitted runs against RandomPlayer).",
    )
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=None,
        help="Path to the team corpus manifest.",
    )
    parser.add_argument("--episodes", type=int, help="Number of episodes per matchup.")
    parser.add_argument("--seed", type=int, help="Random seed for evaluations.")
    parser.add_argument("--report-dir", type=Path, help="Directory to save evaluation reports.")
    parser.add_argument(
        "--port", type=int, default=8120, help="Local Pokémon Showdown port to use."
    )
    parser.add_argument("--config", type=Path, help="Path to global YAML configuration file.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Permit fixed-team fallback and label the report as non-evidence.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the complete evaluation category set and persist its report.

    Arguments:
        argv: Optional command-line arguments supplied by an embedding caller.

    Returns:
        Zero on success, or one when configuration, corpus, evaluation, or
        report persistence fails.
    """
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

    episodes = (
        args.episodes if args.episodes is not None else config.evaluation.episodes_per_matchup
    )
    seed = args.seed if args.seed is not None else config.evaluation.seed
    report_dir = (
        args.report_dir if args.report_dir is not None else Path(config.evaluation.report_dir)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        runtime_manifest = load_active_runtime_manifest(
            config.paths.data_root / "runtime_manifest.json"
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        logger.error("Could not validate the active runtime contract: %s", exc)
        return 1

    policy_a = None
    if args.checkpoint is not None:
        logger.info("Loading policy under evaluation from: %s", args.checkpoint)
        try:
            policy_a = DEFAULT_POLICY_STORE.load_policy(args.checkpoint, device)
            policy_a.eval()
        except (OSError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            logger.error("Failed to load policy checkpoint A: %s", exc)
            return 1

    policy_b = None
    if args.opponent_checkpoint is not None:
        logger.info("Loading opponent policy from: %s", args.opponent_checkpoint)
        try:
            policy_b = DEFAULT_POLICY_STORE.load_policy(args.opponent_checkpoint, device)
            policy_b.eval()
        except (OSError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            logger.error("Failed to load policy checkpoint B: %s", exc)
            return 1

    corpus_manifest = (
        args.corpus_manifest if args.corpus_manifest is not None else config.corpus.manifest_path
    )
    corpus_hash = ""
    format_id = config.bot.battle_format
    if corpus_manifest.is_file():
        try:
            raw = json.loads(corpus_manifest.read_text(encoding="utf-8"))
            corpus_hash = raw.get("corpus_hash", "")
            format_id = raw.get("format_id", format_id)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.error("Could not read corpus manifest headers: %s", exc)
            return 1

    harness = EvaluationHarness(
        corpus_path=corpus_manifest if corpus_manifest.is_file() else None,
        corpus_hash=corpus_hash,
        format_id=format_id,
        episodes_per_matchup=episodes,
        seed=seed,
        port=args.port,
        smoke_test=args.smoke_test,
    )

    try:
        sources = harness.build_team_sources()
    except ValueError as exc:
        logger.error("Evaluation corpus is incomplete: %s", exc)
        return 1
    matchup_results = []

    async def run_all() -> None:
        from p0.runtime.showdown import start_showdown_servers

        logger.info("Starting local Showdown server on port %d...", args.port)
        with start_showdown_servers(1, ports=(args.port,)) as servers:
            server_config = servers[0]

            for category, source in sources.items():
                result = await harness.run_matchup(
                    name_a="PlayerCheckpoint" if policy_a else "RandomA",
                    policy_a=policy_a,
                    name_b="OpponentCheckpoint" if policy_b else "RandomB",
                    policy_b=policy_b,
                    team_category=category,
                    team_source=source,
                    server_configuration=server_config,
                )
                matchup_results.append(result.to_dict())

    try:
        asyncio.run(run_all())
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.exception("Evaluation execution encountered an unhandled error: %s", exc)
        return 1

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "evaluation_report.json"

    report = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "episodes_per_matchup": episodes,
        "seed": seed,
        "runtime_contract_sha256": runtime_manifest.runtime_contract_sha256,
        "policy_a": str(args.checkpoint) if args.checkpoint else "Random",
        "policy_b": str(args.opponent_checkpoint) if args.opponent_checkpoint else "Random",
        "fallback": any(
            bool(metadata.get("fallback")) for metadata in harness.category_metadata.values()
        ),
        "corpus": {
            "path": str(corpus_manifest.resolve()),
            "hash": corpus_hash,
            "categories": harness.category_metadata,
        },
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
        "matchups": matchup_results,
    }

    try:
        atomic_json_save(report_path, report)
        logger.info("Evaluation report successfully written to: %s", report_path)
    except (OSError, TypeError, ValueError) as exc:
        logger.error("Failed to save evaluation report: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
