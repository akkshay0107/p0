from __future__ import annotations

from typing import Any

import pytest
import torch

from p0.replays.compile import compile_payloads, write_tensor_shards
from p0.replays.shards import validate_shard_tensors
from tests.stress._helpers import (
    stress_count,
    stress_random_bo3_payloads,
    stress_random_replay_payloads,
    stress_rng,
    stress_series_id,
)


class TestReplayCompile:
    @pytest.mark.stress
    def test_compiler_preserves_random_replay_identity_at_scale(self, tmp_path) -> None:
        """
        Verify that replay compilation and tensor sharding accurately preserve all games and metadata at scale.

        Checks that:
        1. 128+ randomized replays compile without drops or ID corruption.
        2. Shard writer creates valid tensor files satisfying PyTorch schema and dimension constraints.
        3. Manifest captures all source replay IDs and summary statistics accurately.
        """
        count = stress_count("P0_STRESS_REPLAY_COUNT", 128)
        rng = stress_rng()
        payloads = list(
            stress_random_replay_payloads(
                rng,
                count,
                replay_prefix="stress",
                series_prefix="series",
            )
        )
        # Shuffle input payloads to ensure compiler does not rely on pre-sorted input streams
        rng.shuffle(payloads)
        payloads = tuple(payloads)
        result = compile_payloads(payloads, format_id=payloads[0]["formatid"])

        assert result.metrics.counters["replays"] == count
        assert result.metrics.counters["accepted_games"] == count
        assert {game.replay_id for game in result.games} == {
            f"stress-{index}" for index in range(count)
        }

        # Write each decision into separate shard files to test boundary splitting logic
        built = write_tensor_shards(
            result,
            tmp_path / "shards",
            max_decisions_per_shard=1,
            created_at="2026-01-01T00:00:00Z",
        )
        assert built.manifest.source_games == count
        assert built.manifest.accepted_games == count
        assert set(built.manifest.raw_replays) == {f"stress-{index}" for index in range(count)}
        assert len(built.manifest.shards) == count
        assert {
            summary["source_replay_id"]
            for shard in built.manifest.shards
            for summary in _summaries(built, shard.filename)
        } == {f"stress-{index}" for index in range(count)}

        # Validate tensor dtype, finiteness, and dimension contracts for each written shard file
        for shard in built.manifest.shards:
            artifact = torch.load(
                built.manifest_path.parent / shard.filename,
                map_location="cpu",
                weights_only=True,
            )
            validate_shard_tensors(artifact["tensors"])

    @pytest.mark.stress
    def test_compiler_keeps_series_together_across_shard_boundaries(self, tmp_path) -> None:
        """
        Verify that games belonging to the same Best-of-3 series are never split across shard files.

        Even when max_decisions_per_shard is set to 1, series atomicity requires all games within
        a single match series to reside within the same shard file for correct recurrent training context.
        """
        series_count = stress_count("P0_STRESS_COMPILE_SERIES", 64)
        rng = stress_rng()
        payloads = tuple(
            reversed(
                stress_random_bo3_payloads(
                    rng,
                    series_count,
                    replay_prefix="series",
                    series_prefix="series",
                )
            )
        )
        result = compile_payloads(payloads, format_id=payloads[0]["formatid"])
        assert len(result.games) == len(payloads)

        built = write_tensor_shards(
            result,
            tmp_path / "series-boundaries",
            max_decisions_per_shard=1,
            created_at="2026-01-01T00:00:00Z",
        )
        # Extract unique series IDs present within each generated shard
        shard_series = [
            {str(summary["series_id"]) for summary in _summaries(built, shard.filename)}
            for shard in built.manifest.shards
        ]
        # Each shard must contain exactly one atomic series (no fragmentation across shards)
        assert all(len(series_ids) == 1 for series_ids in shard_series)
        # The set of all shard series must match the expected series hash IDs
        assert {next(iter(series_ids)) for series_ids in shard_series} == {
            stress_series_id(f"series-{index}") for index in range(series_count)
        }


def _summaries(build, filename: str) -> list[dict[str, Any]]:
    """Helper to inspect series summary metadata from a serialized PyTorch tensor shard."""
    artifact = torch.load(
        build.manifest_path.parent / filename,
        map_location="cpu",
        weights_only=True,
    )
    return artifact["series_summaries"]
