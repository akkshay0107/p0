"""Replay catalog selection through the public CLI."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from p0.cli.replays import main
from p0.format_config import FORMAT
from p0.replays.dataset import LazyReplayDataset
from p0.replays.schema import FetchIndexEntry
from p0.replays.scrape import write_fetch_index
from tests.unit.replay_fixtures import sample_replay_payload


class TestReplayCLI:
    def test_filtered_build_includes_siblings_and_reuses_snapshot(
        self, tmp_path: Path, capsys
    ) -> None:
        root = tmp_path / "replays" / FORMAT.bo3_format
        (root / "raw").mkdir(parents=True)
        rows = []
        for replay_id, parent, number, rating in (
            ("a1", "a", 1, 1800),
            ("a2", "a", 2, 1400),
            ("b1", "b", 1, 1200),
        ):
            body = sample_replay_payload(replay_id, parent=parent, game_number=number)
            path = f"raw/{replay_id}.json.gz"
            (root / path).write_bytes(gzip.compress(json.dumps(body).encode()))
            rows.append(
                FetchIndexEntry(
                    replay_id,
                    FORMAT.bo3_format,
                    f"https://example.com/{replay_id}",
                    "2026-01-01T00:00:00Z",
                    200,
                    path,
                    "2026-01-01T00:00:00Z",
                    rating,
                    ("Alice", "Bob"),
                    parent,
                )
            )
        write_fetch_index(root / "index.jsonl", rows)
        args = [
            "build-shards",
            "--cache-dir",
            str(tmp_path / "replays"),
            "--output-dir",
            str(tmp_path / "tensors"),
            "--min-elo",
            "1700",
        ]
        main(args)
        first = json.loads(capsys.readouterr().out)
        dataset = LazyReplayDataset(first["manifest_path"])
        assert dataset.manifest.accepted_games == 2
        assert dataset.manifest.shards[0].replay_ids == ("a1", "a2")
        assert len(list(dataset)) == 4
        assert (Path(first["manifest_path"]).parent / "splits.json").is_file()
        main(args)
        second = json.loads(capsys.readouterr().out)
        assert second["dataset_id"] == first["dataset_id"]
