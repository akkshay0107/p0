"""Fetch, filter, and reconstruct replays using JSONL indexes and cached series."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from p0.format_config import FORMAT
from p0.replays.compile import compile_to_shards
from p0.replays.dataset import LazyReplayDataset, assign_series_splits, write_split_manifest
from p0.replays.group import group_replays
from p0.replays.protocol import ReplayInputContractError, parse_replay_payload
from p0.replays.scrape import (
    ReplayFetcher,
    ScrapeConfig,
    load_raw_replay,
    read_fetch_index,
    select_replays,
)

DEFAULT_REPLAY_CACHE = Path("artifacts/replays")
DEFAULT_TENSOR_CACHE = Path("artifacts/datasets")
LOGGER = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="p0-replays")
    commands = parser.add_subparsers(dest="command", required=True)
    scrape = commands.add_parser("scrape")
    scrape.add_argument("--cache-dir", type=Path, default=DEFAULT_REPLAY_CACHE)
    scrape.add_argument("--format", default=FORMAT.bo3_format)
    scrape.add_argument(
        "--limit-games",
        type=int,
        default=50,
        help="Soft cap for new games; fetching linked siblings can exceed it",
    )
    scrape.add_argument("--replay-id", action="append")
    scrape.add_argument("--after", help="Earliest upload date, ISO-8601")

    for name in ("list", "build-shards"):
        command = commands.add_parser(name)
        command.add_argument("--cache-dir", type=Path, default=DEFAULT_REPLAY_CACHE)
        command.add_argument("--format", default=FORMAT.bo3_format)
        command.add_argument("--min-elo", type=int)
        command.add_argument("--max-elo", type=int)
        command.add_argument("--after", help="Earliest upload date, ISO-8601")
        command.add_argument("--before", help="Latest upload date, ISO-8601")
        command.add_argument("--player")
        if name == "build-shards":
            command.add_argument("--output-dir", type=Path, default=DEFAULT_TENSOR_CACHE)
            command.add_argument(
                "--force-reconstruct",
                action="store_true",
                help="Rebuild cached tensors after reconstruction or resource changes",
            )

    splits = commands.add_parser("create-splits")
    splits.add_argument("--shard-manifest", type=Path, required=True)
    splits.add_argument("--seed", type=int, default=0)
    return parser


def _scrape(args: argparse.Namespace) -> dict[str, Any]:
    fetcher = ReplayFetcher(
        ScrapeConfig(
            format_id=args.format,
            cache_dir=args.cache_dir,
            limit_games=args.limit_games,
            cutoff=args.after,
        )
    )
    entries = fetcher.acquire(args.replay_id)
    return {"index_path": str(fetcher.index_path.resolve()), "cached_games": len(entries)}


def _build(args: argparse.Namespace, selected_ids: set[str]) -> dict[str, Any]:
    index = args.cache_dir / args.format / "index.jsonl"
    entries = read_fetch_index(index)
    selected_parents = {
        entry.parent_room
        for entry in entries
        if entry.replay_id in selected_ids and entry.parent_room
    }
    documents = []
    rejected_ids: set[str] = set()
    rejected_parents: set[str] = set()
    for entry in entries:
        try:
            document = parse_replay_payload(
                load_raw_replay(index.parent / entry.raw_path),
                replay_id=entry.replay_id,
                format_id=args.format,
            )
            if any(not sheet.is_complete for sheet in document.ots):
                raise ReplayInputContractError("Replay requires complete six-member OTS")
            if document.outcome.terminal_line_index is None:
                raise ReplayInputContractError("Replay has no terminal protocol line")
            documents.append(document)
        except (OSError, TypeError, ValueError) as exc:
            LOGGER.warning("Skipping unusable replay %s: %s", entry.replay_id, exc)
            if entry.replay_id in selected_ids or entry.parent_room in selected_parents:
                rejected_ids.add(entry.replay_id)
                if entry.parent_room:
                    rejected_parents.add(entry.parent_room)
    # Selecting one game selects its complete cached series, preserving BC history.
    groups = group_replays(documents, format_id=args.format).series
    selected = []
    for group in groups:
        parent = group.games[0].metadata.parent_room
        if not selected_ids.intersection(group.record.game_replay_ids) and (
            not parent or parent not in selected_parents
        ):
            continue
        if parent and parent in rejected_parents:
            rejected_ids.update(group.record.game_replay_ids)
        else:
            selected.extend(group.games)
    built = compile_to_shards(
        selected,
        args.output_dir,
        format_id=args.format,
        force_reconstruct=args.force_reconstruct,
        external_rejections=sorted(rejected_ids),
    )
    return {
        "manifest_path": str(built.manifest_path.resolve()),
        "dataset_id": built.manifest.dataset_id,
        "source_games": built.manifest.source_games,
        "accepted_games": built.manifest.accepted_games,
        "rejected_games": built.manifest.rejected_games,
    }


def _create_splits(shard_manifest: Path, seed: int) -> dict[str, Any]:
    dataset = LazyReplayDataset(shard_manifest)
    split = assign_series_splits(
        dataset.accepted_series_ids(),
        dataset_id=dataset.manifest.dataset_id,
        seed=seed,
    )
    path = shard_manifest.parent / "splits.json"
    write_split_manifest(split, path)
    return {"split_manifest_path": str(path.resolve()), "split_id": split.split_id}


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "scrape":
        result = _scrape(args)
    elif args.command == "create-splits":
        result = _create_splits(args.shard_manifest, args.seed)
    else:
        entries = select_replays(
            read_fetch_index(args.cache_dir / args.format / "index.jsonl"),
            format_id=args.format,
            min_rating=args.min_elo,
            max_rating=args.max_elo,
            after=args.after,
            before=args.before,
            player=args.player,
        )
        result = (
            [entry.to_dict() for entry in entries]
            if args.command == "list"
            else _build(args, {entry.replay_id for entry in entries})
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
