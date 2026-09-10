"""Tests for replay compilation and shard production."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from p0.replays.compile import (
    compile_payloads,
    write_tensor_shards,
)
from p0.replays.dataset import LazyReplayDataset
from p0.replays.protocol import ReplayInputContractError
from tests.unit.replay_fixtures import (
    golden_replay_payload,
    payload_with_ots_natures,
    sample_replay_payload,
    write_dataset_replay_dataset,
)


class TestReplayCompiler:
    def test_canonical_player_identity_survives_replay_side_swap(self, tmp_path: Path) -> None:
        """Verify canonical player IDs track original human players even when Showdown swaps p1/p2 sides between games."""
        first = sample_replay_payload("game-1")
        first["game_number"] = 1
        second = sample_replay_payload("game-2")
        second["game_number"] = 2
        second["p1"] = "Bob"
        second["p2"] = "Alice"

        built = write_dataset_replay_dataset(tmp_path, (first, second))
        chunks = list(LazyReplayDataset(built.manifest_path))

        # Game 2 swaps sides (p1=Bob, p2=Alice), so canonical_player maps (p1->1, p2->0)
        assert [(chunk.game_number, chunk.player, chunk.canonical_player) for chunk in chunks] == [
            (1, 0, 0),
            (1, 1, 1),
            (2, 0, 1),
            (2, 1, 0),
        ]

    def test_downstream_shards_reject_noncontiguous_source_game_numbers(
        self,
        tmp_path: Path,
    ) -> None:
        """Reject shard publication when a series has a missing first game."""
        second = sample_replay_payload("game-2")
        second["game_number"] = 2
        third = sample_replay_payload("game-3")
        third["game_number"] = 3

        with pytest.raises(ValueError, match="incomplete chronological games"):
            write_dataset_replay_dataset(tmp_path, (third, second))

    def test_compiler_retains_exact_partial_unknown_and_rejected_labels(self) -> None:
        """Verify compiler tracks counters for exact, partial, unknown, and rejected labels based on evidence and OTS validity."""
        exact = golden_replay_payload("exact", series_id="label-series")
        partial = golden_replay_payload(
            "partial", series_id="label-series-2", first_move_target=None
        )
        partial["log"] = str(partial["log"]).replace(
            "|move|p1a: Pikachu|Protect\n", "|move|p1a: Pikachu|Tackle\n"
        )
        result = compile_payloads((exact, partial), format_id=exact["formatid"])
        counters = result.metrics.counters
        assert counters["accepted_games"] == 2
        assert counters["label_exact"] == 0
        assert counters["label_partial"] == 8
        assert counters["label_unknown"] == 0

        capped = compile_payloads((partial,), format_id=partial["formatid"], max_candidates=1)
        assert capped.metrics.counters["label_partial"] == 0
        assert capped.metrics.counters["label_exact"] == 0
        assert capped.metrics.counters["label_unknown"] == 4

        rejected = golden_replay_payload("rejected", series_id="rejected-series")
        rejected["log"] = "\n".join(
            line for line in str(rejected["log"]).splitlines() if "|showteam|" not in line
        )
        with pytest.raises(ReplayInputContractError) as error:
            compile_payloads((rejected,), format_id=rejected["formatid"])
        assert error.value.category == "INVALID_INPUT_CONTRACT"

    def test_compiler_is_stable_across_worker_chunk_sizes(self) -> None:
        """Verify public compilation produces the same replay result for serial and parallel chunks."""
        payloads = (
            golden_replay_payload("chunk-a", series_id="chunk-series-a"),
            golden_replay_payload("chunk-b", series_id="chunk-series-b"),
        )

        serial = compile_payloads(payloads, chunksize=0)
        parallel = compile_payloads(payloads, chunksize=1)

        assert serial.to_dict() == parallel.to_dict()

    def test_reconstructed_views_carry_open_team_sheet_nature(self) -> None:
        """Verify open team sheet natures are preserved for both projected sides."""
        payload = payload_with_ots_natures("ots-nature")
        result = compile_payloads((payload,))
        assert result.games

        own: set[str | None] = set()
        opponent: set[str | None] = set()
        for game in result.games:
            for perspective in game.perspectives:
                for snapshot in perspective.snapshots:
                    own.update(mon.nature for mon in snapshot.view.team.values())
                    opponent.update(mon.nature for mon in snapshot.view.opponent_team.values())

        # Every projected roster member comes from the same open team sheet facts.
        natures = {"Jolly", "Adamant", "Bold", "Timid", "Serious"}
        assert opponent and None not in opponent
        assert opponent <= natures
        assert own and None not in own
        assert own <= natures

    def test_write_tensor_shards_evaluates_release_gates_and_writes_report(
        self, tmp_path: Path
    ) -> None:
        """Verify write_tensor_shards creates release_gate.json and populates gate_report."""
        payload = golden_replay_payload("gate-test", series_id="gate-series")
        result = compile_payloads((payload,))
        build_result = write_tensor_shards(
            result,
            tmp_path / "dataset",
            created_at="2026-01-01T00:00:00Z",
        )
        assert build_result.gate_report is not None
        assert build_result.gate_report.checks["label_loss_contract"] is True
        assert build_result.gate_report.checks["rejection_categories"] is True

        gate_path = build_result.manifest_path.parent / "release_gate.json"
        assert gate_path.is_file()
        gate_data = json.loads(gate_path.read_text(encoding="utf-8"))
        assert gate_data == build_result.gate_report.to_dict()
        assert gate_data["checks"]["label_loss_contract"] is True
