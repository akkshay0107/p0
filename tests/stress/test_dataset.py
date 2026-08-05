from __future__ import annotations

import pytest
from torch.utils.data import DataLoader

from p0.replays.compile import compile_payloads, write_tensor_shards
from p0.replays.dataset import LazyReplayDataset, SeriesSplitManifest
from tests.stress._helpers import stress_count
from tests.stress.replay_fixtures import golden_replay_payload, golden_series_id


def _build_dataset_from_payloads(tmp_path, payloads):
    result = compile_payloads(payloads, format_id=payloads[0]["formatid"])
    return write_tensor_shards(
        result,
        tmp_path / "dataset",
        max_decisions_per_shard=1,
        created_at="2026-01-01T00:00:00Z",
    )


def _build_dataset(tmp_path, count: int):
    payloads = tuple(
        golden_replay_payload(f"dataset-{index}", series_id=f"dataset-series-{index}")
        for index in range(count)
    )
    return _build_dataset_from_payloads(tmp_path, payloads)


@pytest.mark.stress
@pytest.mark.parametrize("num_workers", (0, 1, 2, 4))
def test_dataset_workers_yield_each_golden_perspective_once(tmp_path, num_workers: int) -> None:
    count = stress_count("P0_STRESS_DATASET_REPLAYS", 3)
    built = _build_dataset(tmp_path, count)
    expected = {
        (golden_series_id(f"dataset-series-{index}"), 1, player, player)
        for index in range(count)
        for player in (0, 1)
    }
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
def test_dataset_prefetch_and_repeated_iteration_are_stable(tmp_path) -> None:
    built = _build_dataset(tmp_path, 3)
    loader = DataLoader(
        LazyReplayDataset(built.manifest_path, verify_hashes=True),
        batch_size=None,
        num_workers=2,
        prefetch_factor=2,
    )

    def identity_rows():
        return [
            (chunk.series_id, chunk.game_number, chunk.player, chunk.canonical_player)
            for chunk in loader
        ]

    first = identity_rows()
    second = identity_rows()
    assert first == second


@pytest.mark.stress
def test_dataset_split_filter_and_series_end_use_explicit_source_records(tmp_path) -> None:
    payloads = (
        golden_replay_payload("train-game", series_id="train-series"),
        golden_replay_payload("test-game", series_id="test-series"),
    )
    built = _build_dataset_from_payloads(tmp_path, payloads)
    split = SeriesSplitManifest(
        runtime_contract_sha256=built.manifest.runtime_contract_sha256,
        seed=7,
        assignments={
            golden_series_id("train-series"): "train",
            golden_series_id("test-series"): "test",
        },
        dataset_hash=built.manifest.dataset_hash,
    )

    train = list(LazyReplayDataset(built.manifest_path, split="train", split_manifest=split))
    test = list(LazyReplayDataset(built.manifest_path, split="test", split_manifest=split))
    assert {(chunk.series_id, chunk.player) for chunk in train} == {
        (golden_series_id("train-series"), 0),
        (golden_series_id("train-series"), 1),
    }
    assert {(chunk.series_id, chunk.player) for chunk in test} == {
        (golden_series_id("test-series"), 0),
        (golden_series_id("test-series"), 1),
    }


@pytest.mark.stress
def test_dataset_marks_the_last_game_of_a_series_explicitly(tmp_path) -> None:
    payloads = (
        golden_replay_payload("bo3-game-1", series_id="bo3-series", game_number=1),
        golden_replay_payload("bo3-game-2", series_id="bo3-series", game_number=2, winner="Bob"),
    )
    built = _build_dataset_from_payloads(tmp_path, payloads)
    chunks = list(LazyReplayDataset(built.manifest_path, verify_hashes=True))
    assert [(chunk.game_number, chunk.player, chunk.is_series_end) for chunk in chunks] == [
        (1, 0, False),
        (1, 1, False),
        (2, 0, True),
        (2, 1, True),
    ]
