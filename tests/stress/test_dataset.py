from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from p0.replays.compile import compile_payloads, write_tensor_shards
from p0.replays.dataset import LazyReplayDataset
from tests.stress._helpers import stress_count
from tests.stress.replay_fixtures import golden_replay_payload


def _build_dataset(tmp_path, count: int):
    payloads = tuple(
        golden_replay_payload(f"dataset-{index}", series_id=f"dataset-series-{index}")
        for index in range(count)
    )
    result = compile_payloads(payloads, format_id=payloads[0]["formatid"])
    return write_tensor_shards(
        result,
        tmp_path / "dataset",
        max_decisions_per_shard=1,
        created_at="2026-01-01T00:00:00Z",
    )


@pytest.mark.stress
@pytest.mark.parametrize("num_workers", (0, 2))
def test_dataset_workers_yield_each_golden_perspective_once(tmp_path, num_workers: int) -> None:
    count = stress_count("P0_STRESS_DATASET_REPLAYS", 3)
    built = _build_dataset(tmp_path, count)
    expected = _expected_chunks(built)
    loader = DataLoader(
        LazyReplayDataset(built.manifest_path, verify_hashes=True),
        batch_size=None,
        num_workers=num_workers,
    )

    observed = {
        (chunk.series_id, chunk.game_number, chunk.player, chunk.canonical_player)
        for chunk in loader
    }

    assert observed == expected


@pytest.mark.stress
def test_dataset_rejects_tampered_golden_shard(tmp_path) -> None:
    built = _build_dataset(tmp_path, 1)
    shard_path = built.manifest_path.parent / built.manifest.shards[0].filename
    shard_path.write_bytes(shard_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="hash mismatch"):
        next(iter(LazyReplayDataset(built.manifest_path, verify_hashes=True)))


def _expected_chunks(build) -> set[tuple[str, int, int, int]]:
    expected = set()
    for shard in build.manifest.shards:
        artifact = torch.load(
            build.manifest_path.parent / shard.filename,
            map_location="cpu",
            weights_only=True,
        )
        expected.update(
            (
                str(summary["series_id"]),
                int(summary["game_number"]),
                int(summary["player"]),
                int(summary["canonical_player"]),
            )
            for summary in artifact["series_summaries"]
        )
    return expected
