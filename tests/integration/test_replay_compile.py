from __future__ import annotations

from collections import Counter
from typing import Any

import pytest
import torch

from p0.replays.compile import compile_payloads, write_tensor_shards
from tests.stress._helpers import (
    stress_count,
    stress_random_bo3_payloads,
    stress_random_replay_payloads,
    stress_rng,
    stress_series_id,
)


class TestReplayCompile:
    @pytest.mark.heavy
    @pytest.mark.integration
    def test_compiler_preserves_random_replay_identity_at_scale(self, tmp_path) -> None:
        """Preserve every source ID and both perspective summaries through real compilation and disk shards."""
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
        expected_ids = Counter({f"stress-{index}": 1 for index in range(count)})
        assert (
            Counter(game.replay_id for series in result.accepted_series for game in series.games)
            == expected_ids
        )

        # Write each decision into separate shard files to test boundary splitting logic
        built = write_tensor_shards(
            result,
            tmp_path / "shards",
            max_decisions_per_shard=1,
            created_at="2026-01-01T00:00:00Z",
        )
        assert built.manifest.source_games == count
        assert built.manifest.accepted_games == count
        assert Counter(tuple(built.manifest.raw_replays)) == expected_ids
        assert len(built.manifest.shards) == count
        assert Counter(
            (summary["source_replay_id"], summary["player"])
            for shard in built.manifest.shards
            for summary in _summaries(built, shard.filename)
        ) == Counter((f"stress-{index}", player) for index in range(count) for player in (0, 1))

    @pytest.mark.heavy
    @pytest.mark.integration
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
        assert len(result.accepted_series) == series_count

        built = write_tensor_shards(
            result,
            tmp_path / "series-boundaries",
            max_decisions_per_shard=1,
            created_at="2026-01-01T00:00:00Z",
        )
        summaries = [_summaries(built, shard.filename) for shard in built.manifest.shards]
        assert len(summaries) == series_count
        assert all(len(rows) == 4 for rows in summaries)
        owners = Counter()
        for rows in summaries:
            series_ids = {str(row["series_id"]) for row in rows}
            assert len(series_ids) == 1
            owners.update(series_ids)
            assert Counter((row["game_number"], row["player"]) for row in rows) == Counter(
                (game, player) for game in (1, 2) for player in (0, 1)
            )
        assert owners == Counter(
            {stress_series_id(f"series-{index}"): 1 for index in range(series_count)}
        )


def _summaries(build, filename: str) -> list[dict[str, Any]]:
    """Helper to inspect series summary metadata from a serialized PyTorch tensor shard."""
    artifact = torch.load(
        build.manifest_path.parent / filename,
        map_location="cpu",
        weights_only=True,
    )
    return artifact["series_summaries"]
