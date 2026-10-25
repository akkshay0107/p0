from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import torch

from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import (
    CAT_IDX_STAT_PROVENANCE,
    NUM_IDX_LEVEL_STATS,
    StatProvenance,
)
from p0.replays.compile import compile_payloads, write_tensor_shards
from p0.replays.dataset import LazyReplayDataset
from tests.stress._helpers import (
    stress_count,
    stress_random_bo3_payloads,
    stress_random_replay_payloads,
    stress_rng,
    stress_series_id,
)
from tests.unit.replay_fixtures import decision_payload, payload_with_ots_natures

_GOLDEN_REPLAY_DIRECTORY = (
    Path(__file__).parents[2] / "src/p0/replays/reconstruction/golden_replays"
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
            created_at="2026-01-01T00:00:00Z",
        )
        assert built.manifest.source_games == count
        assert built.manifest.accepted_games == count
        assert (
            Counter(
                tuple(
                    replay
                    for members in built.manifest.source_series.values()
                    for replay in members
                )
            )
            == expected_ids
        )
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

        Series atomicity requires all games within
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


@pytest.mark.heavy
@pytest.mark.integration
class TestReplayProjectionArtifacts:
    @pytest.mark.parametrize("with_natures", (False, True), ids=("unknown", "imputed"))
    def test_shards_preserve_stat_estimates_for_both_perspectives_and_final_boards(
        self, tmp_path: Path, with_natures: bool
    ) -> None:
        payload = payload_with_ots_natures("imputed") if with_natures else decision_payload()
        result = compile_payloads((payload,))
        built = write_tensor_shards(
            result,
            tmp_path,
            resources=default_runtime_resources(),
            created_at="2026-01-01T00:00:00Z",
        )

        chunks = tuple(LazyReplayDataset(built.manifest_path))
        assert {chunk.player for chunk in chunks} == {0, 1}
        expected_stats = (
            [111 / 300, 107 / 300, 61 / 300, 63 / 300, 71 / 300, 155 / 300]
            if with_natures
            else [0.0] * 6
        )
        expected_kind = StatProvenance.IMPUTED if with_natures else StatProvenance.UNKNOWN
        if with_natures:
            assert result.metrics.counters["imputations"] > 0
            assert result.metrics.counters["imputation_confidence_sum"] > 0.0
        for chunk in chunks:
            row = 0 if chunk.player == 0 else 6
            for observation in (chunk.observations, chunk.final_observation):
                assert (
                    observation.categorical[:, row, CAT_IDX_STAT_PROVENANCE] == expected_kind
                ).all()
                stats = observation.numerical[:, row, NUM_IDX_LEVEL_STATS : NUM_IDX_LEVEL_STATS + 6]
                torch.testing.assert_close(
                    stats, stats.new_tensor(expected_stats).expand_as(stats), rtol=1e-5, atol=1e-6
                )
                if not with_natures:
                    assert (
                        observation.categorical[:, :12, CAT_IDX_STAT_PROVENANCE]
                        == StatProvenance.UNKNOWN
                    ).all()

    @pytest.mark.skipif(
        not tuple(_GOLDEN_REPLAY_DIRECTORY.glob("*.json")),
        reason="local golden replay corpus is not present",
    )
    def test_local_golden_replays_are_all_accepted(self) -> None:
        paths = tuple(sorted(_GOLDEN_REPLAY_DIRECTORY.glob("*.json")))

        result = compile_payloads(path.read_bytes() for path in paths)

        assert result.metrics.counters["rejected_games"] == 0
        assert result.metrics.counters["accepted_games"] == len(paths)


def _summaries(build, filename: str) -> list[dict[str, Any]]:
    """Helper to inspect series summary metadata from a serialized PyTorch tensor shard."""
    artifact = torch.load(
        build.manifest_path.parent / filename,
        map_location="cpu",
        weights_only=True,
    )
    return artifact["series_summaries"]
