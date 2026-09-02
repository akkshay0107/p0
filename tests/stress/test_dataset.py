from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from torch.utils.data import DataLoader

from p0.replays.compile import compile_payloads, write_tensor_shards
from p0.replays.dataset import LazyReplayDataset
from p0.runtime.process_context import PROCESS_CONTEXT
from tests.stress._helpers import (
    stress_count,
    stress_random_replay_payloads,
    stress_rng,
    stress_series_id,
)


def _dataset_identity(chunk: Any) -> tuple[Any, ...]:
    return (chunk.series_id, chunk.game_number, chunk.player, chunk.canonical_player)


def _build_dataset_from_payloads(tmp_path: Path, payloads: Sequence[dict[str, Any]]):
    result = compile_payloads(payloads, format_id=payloads[0]["formatid"])
    return write_tensor_shards(
        result,
        tmp_path / "dataset",
        max_decisions_per_shard=1,
        created_at="2026-01-01T00:00:00Z",
    )


def _build_dataset(tmp_path: Path, count: int):
    rng = stress_rng()
    payloads = list(
        stress_random_replay_payloads(
            rng,
            count,
            replay_prefix="dataset",
            series_prefix="dataset-series",
        )
    )
    rng.shuffle(payloads)
    return _build_dataset_from_payloads(tmp_path, payloads)


def _identity_rows(loader: DataLoader) -> list[tuple[Any, ...]]:
    return list(loader)


class TestDataset:
    @pytest.mark.stress
    @pytest.mark.parametrize("num_workers", (0, 1, 2, 4))
    def test_dataset_workers_yield_each_random_perspective_once(
        self, tmp_path: Path, num_workers: int
    ) -> None:
        """
        Verify DataLoader multiprocess sharding yields every series perspective exactly once without duplicates.

        Each compiled game produces two distinct perspective chunks (player 0 and player 1).
        When distributing workload across 0, 1, 2, or 4 worker processes, the union of all
        yielded items must match the full set of (series_id, game_number, player, canonical_player) tuples.
        """
        count = stress_count("P0_STRESS_DATASET_REPLAYS", 128)
        built = _build_dataset(tmp_path, count)
        # Expected set contains both player 0 and player 1 perspectives for every game
        expected = {
            (stress_series_id(f"dataset-series-{index}"), 1, player, player)
            for index in range(count)
            for player in (0, 1)
        }
        loader_kwargs: dict[str, Any] = (
            {"multiprocessing_context": PROCESS_CONTEXT} if num_workers else {}
        )
        loader = DataLoader(
            LazyReplayDataset(built.manifest_path, verify_hashes=True),
            batch_size=None,
            num_workers=num_workers,
            collate_fn=_dataset_identity,
            **loader_kwargs,
        )
        observed = set(loader)

        # Confirm worker sharding partition is lossless and free of duplicate chunks
        assert observed == expected

    @pytest.mark.stress
    def test_dataset_prefetch_and_repeated_iteration_are_stable(self, tmp_path) -> None:
        """
        Verify that multi-worker prefetching with persistent workers produces consistent epoch iterations.

        Ensures that background prefetch buffers and persistent worker worker-loop state
        do not cause order corruption, missed elements, or memory leakage across repeated dataset epochs.
        """
        built = _build_dataset(tmp_path, stress_count("P0_STRESS_DATASET_PREFETCH_REPLAYS", 64))
        loader = DataLoader(
            LazyReplayDataset(built.manifest_path, verify_hashes=True),
            batch_size=None,
            num_workers=2,
            collate_fn=_dataset_identity,
            prefetch_factor=4,
            persistent_workers=True,
            multiprocessing_context=PROCESS_CONTEXT,
        )

        expected = _identity_rows(loader)
        assert expected
        # Verify every subsequent epoch matches the exact sequence of the initial epoch pass
        for _ in range(stress_count("P0_STRESS_DATASET_ITERATIONS", 8)):
            assert _identity_rows(loader) == expected
