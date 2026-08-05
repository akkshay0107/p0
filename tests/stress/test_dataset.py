from __future__ import annotations

from typing import Any

import pytest
from torch.utils.data import DataLoader

from p0.replays.compile import compile_payloads, write_tensor_shards
from p0.replays.dataset import LazyReplayDataset, SeriesSplitManifest
from tests.stress._helpers import stress_count, stress_rng
from tests.stress.replay_fixtures import golden_replay_payload, golden_series_id


def _dataset_identity(chunk):
    return (chunk.series_id, chunk.game_number, chunk.player, chunk.canonical_player)


def _build_dataset_from_payloads(tmp_path, payloads):
    result = compile_payloads(payloads, format_id=payloads[0]["formatid"])
    return write_tensor_shards(
        result,
        tmp_path / "dataset",
        max_decisions_per_shard=1,
        created_at="2026-01-01T00:00:00Z",
    )


def _build_dataset(tmp_path, count: int):
    payloads = list(
        golden_replay_payload(f"dataset-{index}", series_id=f"dataset-series-{index}")
        for index in range(count)
    )
    stress_rng().shuffle(payloads)
    return _build_dataset_from_payloads(tmp_path, payloads)


@pytest.mark.stress
@pytest.mark.parametrize("num_workers", (0, 1, 2, 4))
def test_dataset_workers_yield_each_golden_perspective_once(tmp_path, num_workers: int) -> None:
    count = stress_count("P0_STRESS_DATASET_REPLAYS", 128)
    built = _build_dataset(tmp_path, count)
    expected = {
        (golden_series_id(f"dataset-series-{index}"), 1, player, player)
        for index in range(count)
        for player in (0, 1)
    }
    loader_kwargs: dict[str, Any] = {"multiprocessing_context": "spawn"} if num_workers else {}
    loader = DataLoader(
        LazyReplayDataset(built.manifest_path, verify_hashes=True),
        batch_size=None,
        num_workers=num_workers,
        collate_fn=_dataset_identity,
        **loader_kwargs,
    )
    observed = set(loader)

    assert observed == expected


@pytest.mark.stress
def test_dataset_prefetch_and_repeated_iteration_are_stable(tmp_path) -> None:
    built = _build_dataset(tmp_path, stress_count("P0_STRESS_DATASET_PREFETCH_REPLAYS", 64))
    loader = DataLoader(
        LazyReplayDataset(built.manifest_path, verify_hashes=True),
        batch_size=None,
        num_workers=2,
        collate_fn=_dataset_identity,
        prefetch_factor=4,
        persistent_workers=True,
        multiprocessing_context="spawn",
    )

    def identity_rows():
        return list(loader)

    expected = identity_rows()
    for _ in range(stress_count("P0_STRESS_DATASET_ITERATIONS", 64)):
        assert identity_rows() == expected


@pytest.mark.stress
def test_dataset_split_filter_and_series_end_use_explicit_source_records(tmp_path) -> None:
    series_count = stress_count("P0_STRESS_DATASET_SPLIT_SERIES", 64)
    payloads = tuple(
        payload
        for index in range(series_count)
        for payload in (
            golden_replay_payload(f"train-game-{index}", series_id=f"train-series-{index}"),
            golden_replay_payload(f"test-game-{index}", series_id=f"test-series-{index}"),
        )
    )
    built = _build_dataset_from_payloads(tmp_path, payloads)
    split = SeriesSplitManifest(
        runtime_contract_sha256=built.manifest.runtime_contract_sha256,
        seed=7,
        assignments={
            golden_series_id(f"train-series-{index}"): "train" for index in range(series_count)
        }
        | {golden_series_id(f"test-series-{index}"): "test" for index in range(series_count)},
        dataset_hash=built.manifest.dataset_hash,
    )

    train = list(LazyReplayDataset(built.manifest_path, split="train", split_manifest=split))
    test = list(LazyReplayDataset(built.manifest_path, split="test", split_manifest=split))
    assert {(chunk.series_id, chunk.player) for chunk in train} == {
        (golden_series_id(f"train-series-{index}"), player)
        for index in range(series_count)
        for player in (0, 1)
    }
    assert {(chunk.series_id, chunk.player) for chunk in test} == {
        (golden_series_id(f"test-series-{index}"), player)
        for index in range(series_count)
        for player in (0, 1)
    }


@pytest.mark.stress
def test_dataset_marks_the_last_game_of_a_series_explicitly(tmp_path) -> None:
    series_count = stress_count("P0_STRESS_DATASET_BO3_SERIES", 64)
    payloads = tuple(
        payload
        for index in range(series_count)
        for payload in (
            golden_replay_payload(
                f"bo3-{index}-game-1", series_id=f"bo3-series-{index}", game_number=1
            ),
            golden_replay_payload(
                f"bo3-{index}-game-2",
                series_id=f"bo3-series-{index}",
                game_number=2,
                winner="Bob",
            ),
        )
    )
    built = _build_dataset_from_payloads(tmp_path, payloads)
    chunks = list(LazyReplayDataset(built.manifest_path, verify_hashes=True))
    observed = {}
    for chunk in chunks:
        observed.setdefault(chunk.series_id, []).append(
            (chunk.game_number, chunk.player, chunk.is_series_end)
        )
    assert set(observed) == {
        golden_series_id(f"bo3-series-{index}") for index in range(series_count)
    }
    assert all(
        rows == [(1, 0, False), (1, 1, False), (2, 0, True), (2, 1, True)]
        for rows in observed.values()
    )
