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


@pytest.mark.stress
def test_compiler_preserves_random_replay_identity_at_scale(tmp_path) -> None:
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
    rng.shuffle(payloads)
    payloads = tuple(payloads)
    result = compile_payloads(payloads, format_id=payloads[0]["formatid"])

    assert result.metrics.counters["replays"] == count
    assert result.metrics.counters["accepted_games"] == count
    assert {game.replay_id for game in result.games} == {
        f"stress-{index}" for index in range(count)
    }

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

    for shard in built.manifest.shards:
        artifact = torch.load(
            built.manifest_path.parent / shard.filename,
            map_location="cpu",
            weights_only=True,
        )
        validate_shard_tensors(artifact["tensors"])


@pytest.mark.stress
def test_compiler_is_deterministic_for_random_replays(tmp_path) -> None:
    rng = stress_rng()
    payloads_list = list(
        stress_random_replay_payloads(
            rng,
            stress_count("P0_STRESS_DETERMINISTIC_REPLAYS", 64),
            replay_prefix="stress",
            series_prefix="series",
        )
    )
    rng.shuffle(payloads_list)
    payloads = tuple(payloads_list)
    first = compile_payloads(payloads, format_id=payloads[0]["formatid"])
    second = compile_payloads(tuple(reversed(payloads)), format_id=payloads[0]["formatid"])

    first_build = write_tensor_shards(
        first,
        tmp_path / "first",
        max_decisions_per_shard=1,
        created_at="2026-01-01T00:00:00Z",
    )
    second_build = write_tensor_shards(
        second,
        tmp_path / "second",
        max_decisions_per_shard=1,
        created_at="2026-01-01T00:00:00Z",
    )

    assert first_build.manifest.to_dict() == second_build.manifest.to_dict()


@pytest.mark.stress
def test_compiler_keeps_series_together_across_shard_boundaries(tmp_path) -> None:
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
    shard_series = [
        {str(summary["series_id"]) for summary in _summaries(built, shard.filename)}
        for shard in built.manifest.shards
    ]
    assert all(len(series_ids) == 1 for series_ids in shard_series)
    assert {next(iter(series_ids)) for series_ids in shard_series} == {
        stress_series_id(f"series-{index}") for index in range(series_count)
    }


def _summaries(build, filename: str) -> list[dict[str, Any]]:
    artifact = torch.load(
        build.manifest_path.parent / filename,
        map_location="cpu",
        weights_only=True,
    )
    return artifact["series_summaries"]
