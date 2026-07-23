"""Operational replay acquisition, Bo1 shard compilation, and split commands."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from p0.format_config import DEFAULT_RUNTIME_MANIFEST, FORMAT
from p0.replays.compile import build_shards_from_cache
from p0.replays.dataset import assign_series_splits, write_split_manifest
from p0.replays.group import group_replays
from p0.replays.protocol import parse_replay_payload
from p0.replays.scrape import ReplayFetcher, ScrapeConfig, load_raw_replay
from p0.replays.shards import load_shard_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="p0-replays")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scrape = subparsers.add_parser("scrape")
    scrape.add_argument("--cache-dir", type=Path, default=Path("artifacts/replays"))
    scrape.add_argument("--limit-games", type=int, default=50)
    scrape.add_argument("--max-pages", type=int, default=100)
    scrape.add_argument("--concurrency", type=int, default=4)
    scrape.add_argument("--replay-id", action="append", default=None)

    build = subparsers.add_parser("build-shards")
    build.add_argument("--cache-dir", type=Path, default=Path("artifacts/replays"))
    build.add_argument("--output-dir", type=Path, default=Path("artifacts/shards"))
    build.add_argument("--max-candidates", type=int, default=256)
    build.add_argument("--imputation-seed", type=int, default=0)
    build.add_argument("--max-decisions-per-shard", type=int, default=4096)
    build.add_argument("--runtime-manifest", type=Path, default=DEFAULT_RUNTIME_MANIFEST)

    splits = subparsers.add_parser("create-splits")
    splits.add_argument("--shard-manifest", type=Path, required=True)
    splits.add_argument("--output", type=Path, default=None)
    splits.add_argument("--seed", type=int, default=0)
    splits.add_argument("--validation-fraction", type=float, default=0.1)
    splits.add_argument("--test-fraction", type=float, default=0.1)
    splits.add_argument("--runtime-manifest", type=Path, default=DEFAULT_RUNTIME_MANIFEST)
    return parser


def _accepted_series(
    manifest_path: Path,
    quality_filename: str,
    expected_sha256: str,
) -> tuple[str, ...]:
    quality_path = manifest_path.parent / quality_filename
    if hashlib.sha256(quality_path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(f"Replay quality manifest hash mismatch: {quality_path}")
    value = json.loads(quality_path.read_text(encoding="utf-8"))
    records = value.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Malformed replay quality manifest: {quality_path}")
    return tuple(
        sorted(
            {
                str(record["source_series_id"])
                for record in records
                if isinstance(record, dict) and record.get("accepted")
            }
        )
    )


def _scrape(args: argparse.Namespace) -> dict[str, Any]:
    config = ScrapeConfig(
        format_id=FORMAT.bo3_format,
        cache_dir=args.cache_dir,
        max_pages=args.max_pages,
        limit_games=args.limit_games,
        concurrency=args.concurrency,
    )
    entries = ReplayFetcher(config).acquire(args.replay_id)
    documents = []
    unparsed = 0
    for entry in entries:
        try:
            documents.append(
                parse_replay_payload(
                    load_raw_replay(
                        args.cache_dir / FORMAT.bo3_format / "raw" / f"{entry.replay_id}.json.gz"
                    ),
                    replay_id=entry.replay_id,
                    format_id=FORMAT.bo3_format,
                )
            )
        except (TypeError, ValueError):
            unparsed += 1
    source_series = len(group_replays(documents, format_id=FORMAT.bo3_format).series) + unparsed
    return {
        "format_id": FORMAT.bo3_format,
        "index_path": str(ReplayFetcher(config).index_path.resolve()),
        "source_games": len(entries),
        "source_series": source_series,
        "accepted_games": None,
        "rejected_games": None,
        "dataset_hash": None,
        "limit_games": args.limit_games,
    }


def _build(args: argparse.Namespace) -> dict[str, Any]:
    built = build_shards_from_cache(
        args.cache_dir,
        args.output_dir,
        format_id=FORMAT.bo3_format,
        max_candidates=args.max_candidates,
        imputation_seed=args.imputation_seed,
        max_decisions_per_shard=args.max_decisions_per_shard,
        manifest_path=args.runtime_manifest,
    )
    manifest = built.manifest
    return {
        "manifest_path": str(built.manifest_path.resolve()),
        "quality_manifest_path": str(
            (built.manifest_path.parent / manifest.quality_manifest).resolve()
        ),
        "dataset_hash": manifest.dataset_hash,
        "runtime_hash": manifest.runtime_contract_sha256,
        "source_games": manifest.source_games,
        "accepted_games": manifest.accepted_games,
        "rejected_games": manifest.rejected_games,
        "source_series": len(manifest.source_series),
        "accepted_series": len(
            _accepted_series(
                built.manifest_path,
                manifest.quality_manifest,
                manifest.quality_manifest_sha256,
            )
        ),
    }


def _create_splits(args: argparse.Namespace) -> dict[str, Any]:
    value = json.loads(args.shard_manifest.read_text(encoding="utf-8"))
    manifest = load_shard_manifest(value, args.runtime_manifest)
    series_ids = _accepted_series(
        args.shard_manifest,
        manifest.quality_manifest,
        manifest.quality_manifest_sha256,
    )
    split = assign_series_splits(
        series_ids,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        runtime_contract_sha256=manifest.runtime_contract_sha256,
        dataset_hash=manifest.dataset_hash,
    )
    output = args.output or args.shard_manifest.parent / "splits.json"
    write_split_manifest(split, output)
    counts = {
        name: sum(assigned == name for assigned in split.assignments.values())
        for name in ("train", "validation", "test")
    }
    return {
        "split_manifest_path": str(output.resolve()),
        "shard_manifest_path": str(args.shard_manifest.resolve()),
        "dataset_hash": manifest.dataset_hash,
        "runtime_hash": manifest.runtime_contract_sha256,
        "source_series": len(series_ids),
        "source_games": manifest.source_games,
        "accepted_games": manifest.accepted_games,
        "rejected_games": manifest.rejected_games,
        "split_series": counts,
    }


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "scrape":
        result = _scrape(args)
    elif args.command == "build-shards":
        result = _build(args)
    else:
        result = _create_splits(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
