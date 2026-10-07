"""Operational replay acquisition, Bo1 shard compilation, and split commands."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from p0.format_config import FORMAT
from p0.replays.compile import compile_to_shards
from p0.replays.dataset import LazyReplayDataset, assign_series_splits, write_split_manifest
from p0.replays.group import group_replays
from p0.replays.protocol import ReplayDocument, parse_replay_payload
from p0.replays.scrape import ReplayFetcher, ScrapeConfig, load_raw_replay


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser for replay operations."""
    parser = argparse.ArgumentParser(prog="p0-replays")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scrape = subparsers.add_parser("scrape")
    scrape.add_argument("--cache-dir", type=Path, default=Path("artifacts/replays"))
    scrape.add_argument("--limit-games", type=int, default=50)
    scrape.add_argument("--replay-id", action="append", default=None)

    build = subparsers.add_parser("build-shards")
    build.add_argument("--cache-dir", type=Path, default=Path("artifacts/replays"))
    build.add_argument("--output-dir", type=Path, default=Path("artifacts/shards"))

    splits = subparsers.add_parser("create-splits")
    splits.add_argument("--shard-manifest", type=Path, required=True)

    return parser


def _scrape(
    cache_dir: Path,
    limit_games: int,
    replay_ids: list[str] | None,
) -> dict[str, Any]:
    """Scrape Pokemon Showdown replays up to the selected game limit."""
    config = ScrapeConfig(
        format_id=FORMAT.bo3_format,
        cache_dir=cache_dir,
        limit_games=limit_games,
    )

    fetcher = ReplayFetcher(config)
    entries = fetcher.acquire(replay_ids)
    documents = []
    unparsed = 0

    for entry in entries:
        try:
            documents.append(
                parse_replay_payload(
                    load_raw_replay(
                        cache_dir / FORMAT.bo3_format / "raw" / f"{entry.replay_id}.json.gz"
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
        "index_path": str(fetcher.index_path.resolve()),
        "source_games": len(entries),
        "source_series": source_series,
        "accepted_games": None,
        "rejected_games": None,
        "dataset_hash": None,
        "limit_games": limit_games,
    }


def _yield_documents(cache_dir: Path) -> Iterator[ReplayDocument]:
    """Yield parsed replay documents from the cache."""
    for path in sorted((cache_dir / FORMAT.bo3_format / "raw").glob("*.json.gz")):
        identity = path.name.removesuffix(".json.gz")
        try:
            payload = load_raw_replay(path)
            yield parse_replay_payload(
                payload,
                replay_id=identity,
                format_id=FORMAT.bo3_format,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed cached replay: {path}") from exc


def _build(cache_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Compile a folder of scraped raw replays into PyTorch tensor shards."""
    built = compile_to_shards(
        documents=_yield_documents(cache_dir),
        output_dir=output_dir,
        format_id=FORMAT.bo3_format,
    )

    manifest = built.manifest

    return {
        "manifest_path": str(built.manifest_path.resolve()),
        "dataset_hash": manifest.dataset_hash,
        "global_hash": manifest.global_contract_sha256,
        "source_games": manifest.source_games,
        "accepted_games": manifest.accepted_games,
        "rejected_games": manifest.rejected_games,
        "source_series": len(manifest.source_series),
    }


def _create_splits(shard_manifest: Path) -> dict[str, Any]:
    """Assign validation and test splits uniformly across all compiled series."""
    dataset = LazyReplayDataset(shard_manifest)
    manifest = dataset.manifest
    series_ids = dataset.accepted_series_ids()
    split = assign_series_splits(
        series_ids,
        global_contract_sha256=manifest.global_contract_sha256,
        dataset_hash=manifest.dataset_hash,
    )

    output = shard_manifest.parent / "splits.json"
    write_split_manifest(split, output)

    counts = {
        name: sum(assigned == name for assigned in split.assignments.values())
        for name in ("train", "validation", "test")
    }

    return {
        "split_manifest_path": str(output.resolve()),
        "shard_manifest_path": str(shard_manifest.resolve()),
        "dataset_hash": manifest.dataset_hash,
        "global_hash": manifest.global_contract_sha256,
        "source_series": len(series_ids),
        "source_games": manifest.source_games,
        "accepted_games": manifest.accepted_games,
        "rejected_games": manifest.rejected_games,
        "split_series": counts,
    }


def main(argv: list[str] | None = None) -> None:
    """Replay CLI entrypoint."""
    args = _parser().parse_args(argv)

    if args.command == "scrape":
        result = _scrape(args.cache_dir, args.limit_games, args.replay_id)
    elif args.command == "build-shards":
        result = _build(args.cache_dir, args.output_dir)
    else:
        result = _create_splits(args.shard_manifest)

    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
