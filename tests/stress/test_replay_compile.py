from __future__ import annotations

from typing import Any

import pytest
import torch

from p0.replays import compile as compile_module
from p0.replays.compile import compile_payloads, write_tensor_shards
from p0.replays.shards import validate_shard_tensors
from tests.stress._helpers import stress_count
from tests.stress.replay_fixtures import golden_replay_payload, golden_series_id


def _compile_payloads(count: int) -> tuple[dict[str, Any], ...]:
    return tuple(
        golden_replay_payload(
            f"golden-{index}",
            series_id=f"series-{index}",
        )
        for index in range(count)
    )


@pytest.mark.stress
def test_compiler_preserves_golden_replay_identity_at_scale(tmp_path) -> None:
    count = stress_count("P0_STRESS_REPLAY_COUNT", 4)
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
    payloads = _compile_payloads(stress_count("P0_STRESS_DETERMINISTIC_REPLAYS", 2))
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
def test_compiler_retains_exact_partial_unknown_and_rejected_labels() -> None:
    exact = golden_replay_payload("exact", series_id="label-series")
    partial = golden_replay_payload(
        "partial",
        series_id="label-series-2",
        first_move_target=None,
    )
    result = compile_payloads((exact, partial), format_id=exact["formatid"])

    counters = result.metrics.counters
    assert counters["accepted_games"] == 2
    assert counters["label_exact"] == 3
    assert counters["label_partial"] == 1
    assert counters["label_unknown"] == 4

    capped = compile_payloads((partial,), format_id=partial["formatid"], max_candidates=1)
    assert capped.metrics.counters["label_partial"] == 0
    assert capped.metrics.counters["label_exact"] == 1
    assert capped.metrics.counters["label_unknown"] == 3

    rejected = golden_replay_payload("rejected", series_id="rejected-series")
    rejected["log"] = "\n".join(
        line for line in str(rejected["log"]).splitlines() if "|showteam|" not in line
    )
    rejected_result = compile_payloads((rejected,), format_id=rejected["formatid"])
    assert not rejected_result.games
    assert rejected_result.metrics.counters["rejected_games"] == 1


@pytest.mark.stress
def test_compiler_keeps_series_together_across_shard_boundaries(tmp_path) -> None:
    payloads = (
        golden_replay_payload("series-a-1", series_id="series-a", game_number=1),
        golden_replay_payload("series-a-2", series_id="series-a", game_number=2, winner="Bob"),
        golden_replay_payload("series-b-1", series_id="series-b", game_number=1),
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
    assert shard_series == [
        {golden_series_id("series-a")},
        {golden_series_id("series-b")},
    ]


@pytest.mark.stress
def test_compiler_closes_process_pool_after_worker_failure(monkeypatch) -> None:
    events: list[str] = []

    class FailingPool:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback
            events.append("exit")

        def map(self, *args, **kwargs):
            del args, kwargs
            events.append("map")
            raise RuntimeError("injected worker failure")

    monkeypatch.setattr(compile_module.concurrent.futures, "ProcessPoolExecutor", FailingPool)
    payload = golden_replay_payload("worker-failure", series_id="worker-failure-series")
    with pytest.raises(RuntimeError, match="injected worker failure"):
        compile_payloads((payload,), format_id=payload["formatid"])
    assert events == ["enter", "map", "exit"]


def _summaries(build, filename: str) -> list[dict[str, Any]]:
    artifact = torch.load(
        build.manifest_path.parent / filename,
        map_location="cpu",
        weights_only=True,
    )
    return artifact["series_summaries"]
