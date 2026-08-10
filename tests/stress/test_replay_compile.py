from __future__ import annotations

from typing import Any

import pytest
import torch

from p0.replays.compile import compile_payloads, write_tensor_shards
from p0.replays.shards import validate_shard_tensors
from tests.stress._helpers import stress_count, stress_rng
from tests.unit.replay_fixtures import golden_replay_payload, golden_series_id


def _compile_payloads(count: int) -> tuple[dict[str, Any], ...]:
    payloads = [
        golden_replay_payload(
            f"golden-{index}",
            series_id=f"series-{index}",
        )
        for index in range(count)
    ]
    stress_rng().shuffle(payloads)
    return tuple(payloads)


@pytest.mark.stress
def test_compiler_preserves_golden_replay_identity_at_scale(tmp_path) -> None:
    count = stress_count("P0_STRESS_REPLAY_COUNT", 128)
    payloads = _compile_payloads(count)
    result = compile_payloads(payloads, format_id=payloads[0]["formatid"])

    assert result.metrics.counters["replays"] == count
    assert result.metrics.counters["accepted_games"] == count
    assert {game.replay_id for game in result.games} == {
        f"golden-{index}" for index in range(count)
    }

    built = write_tensor_shards(
        result,
        tmp_path / "shards",
        max_decisions_per_shard=1,
        created_at="2026-01-01T00:00:00Z",
    )
    assert built.manifest.source_games == count
    assert built.manifest.accepted_games == count
    assert set(built.manifest.raw_replays) == {f"golden-{index}" for index in range(count)}
    assert len(built.manifest.shards) == count
    assert {
        summary["source_replay_id"]
        for shard in built.manifest.shards
        for summary in _summaries(built, shard.filename)
    } == {f"golden-{index}" for index in range(count)}

    for shard in built.manifest.shards:
        artifact = torch.load(
            built.manifest_path.parent / shard.filename,
            map_location="cpu",
            weights_only=True,
        )
        validate_shard_tensors(artifact["tensors"])


@pytest.mark.stress
def test_compiler_is_deterministic_for_golden_replays(tmp_path) -> None:
    payloads = _compile_payloads(stress_count("P0_STRESS_DETERMINISTIC_REPLAYS", 64))
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
    payloads = tuple(
        payload
        for index in range(series_count)
        for payload in (
            golden_replay_payload(f"series-{index}-1", series_id=f"series-{index}", game_number=1),
            golden_replay_payload(
                f"series-{index}-2",
                series_id=f"series-{index}",
                game_number=2,
                winner="Bob",
            ),
        )
    )
    payloads = tuple(reversed(payloads))
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
        golden_series_id(f"series-{index}") for index in range(series_count)
    }


def _summaries(build, filename: str) -> list[dict[str, Any]]:
    artifact = torch.load(
        build.manifest_path.parent / filename,
        map_location="cpu",
        weights_only=True,
    )
    return artifact["series_summaries"]
