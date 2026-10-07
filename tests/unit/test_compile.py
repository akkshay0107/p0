"""Tests for replay compilation and shard production."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import pytest

from p0.replays.compile import (
    compile_payloads,
    compile_to_shards,
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
        second["log"] = (
            str(second["log"]).replace("p1", "TMP").replace("p2", "p1").replace("TMP", "p2")
        )

        built = write_dataset_replay_dataset(tmp_path, (first, second))
        chunks = list(LazyReplayDataset(built.manifest_path))

        # Game 2 swaps sides (p1=Bob, p2=Alice), so canonical_player maps (p1->1, p2->0)
        assert [(chunk.game_number, chunk.player, chunk.canonical_player) for chunk in chunks] == [
            (1, 0, 0),
            (1, 1, 1),
            (2, 0, 1),
            (2, 1, 0),
        ]

    def test_series_missing_its_first_game_is_rejected_without_blocking_publication(
        self,
        tmp_path: Path,
    ) -> None:
        """A series without game 1 cannot seed series memory, so only that series is dropped."""
        second = sample_replay_payload("game-2", parent="missing-first", game_number=2)
        third = sample_replay_payload("game-3", parent="missing-first", game_number=3)
        complete = sample_replay_payload("game-1", parent="complete", game_number=1)

        compiled = compile_payloads((third, second, complete), chunksize=0)
        built = write_tensor_shards(compiled, tmp_path, created_at="2026-01-01T00:00:00Z")

        assert compiled.metrics.counters["accepted_games"] == 1
        assert compiled.metrics.counters["rejected_games"] == 2
        assert compiled.metrics.counters["decisions"] == 4
        assert compiled.metrics.counters["rejected_non_contiguous_game_numbers"] == 1
        assert built.manifest.accepted_games == 1
        assert built.manifest.rejected_games == 2
        assert built.manifest.diagnostics["rejected_non_contiguous_game_numbers"] == 1
        chunks = list(LazyReplayDataset(built.manifest_path))
        assert {chunk.game_number for chunk in chunks} == {1}

    def test_compiled_series_rejects_missing_or_reordered_members(self) -> None:
        first = sample_replay_payload("member-1", parent="members", game_number=1)
        second = sample_replay_payload("member-2", parent="members", game_number=2)
        accepted = compile_payloads((second, first), chunksize=0).accepted_series[0]

        with pytest.raises(ValueError, match="every source game"):
            replace(accepted, games=accepted.games[:1])
        with pytest.raises(ValueError, match="source membership in order"):
            replace(accepted, games=tuple(reversed(accepted.games)))

    def test_shard_limit_uses_whole_incoming_series(self, tmp_path: Path) -> None:
        single = sample_replay_payload("single-1", parent="single", game_number=1)
        first = sample_replay_payload("valid-1", parent="valid", game_number=1)
        second = sample_replay_payload("valid-2", parent="valid", game_number=2)
        result = compile_payloads((first, second, single), chunksize=0)

        built = write_tensor_shards(
            result,
            tmp_path,
            max_decisions_per_shard=9,
            created_at="2026-01-01T00:00:00Z",
        )

        assert built.manifest.accepted_games == 3
        assert sorted(shard.decisions for shard in built.manifest.shards) == [4, 8]

    def test_source_series_preserves_game_number_order(self, tmp_path: Path) -> None:
        first = sample_replay_payload("z-game", game_number=1)
        second = sample_replay_payload("a-game", game_number=2)

        built = write_dataset_replay_dataset(tmp_path, (second, first))
        chunks = list(LazyReplayDataset(built.manifest_path))

        series_id = chunks[0].series_id
        assert built.manifest.source_series[series_id] == ("z-game", "a-game")
        assert [chunk.game_number for chunk in chunks] == [1, 1, 2, 2]

    def test_compiler_retains_exact_partial_unknown_and_rejected_labels(self) -> None:
        """Verify compiler tracks counters for exact, partial, and unknown labels based on evidence."""
        exact = golden_replay_payload("exact", series_id="label-series")
        exact["log"] = str(exact["log"]).replace(
            '["Tackle","Helping Hand"]', '["Protect","Tackle"]'
        )
        exact["log"] = str(exact["log"]).replace(
            "|move|p1b: Eevee|Tackle|p2b: Charmander", "|move|p1b: Eevee|Protect"
        )
        exact["log"] = str(exact["log"]).replace(
            "|move|p2b: Charmander|Tackle|p1b: Eevee", "|move|p2b: Charmander|Protect"
        )
        partial = golden_replay_payload(
            "partial", series_id="label-series-2", first_move_target=None
        )
        partial["log"] = str(partial["log"]).replace(
            "|move|p1a: Pikachu|Protect\n", "|move|p1a: Pikachu|Tackle\n"
        )
        result = compile_payloads((exact, partial), format_id=exact["formatid"])
        counters = result.metrics.counters
        assert counters["accepted_games"] == 2
        assert counters["label_exact"] == 2
        assert counters["label_partial"] == 6
        assert counters["label_unknown"] == 0

        capped = compile_payloads((partial,), format_id=partial["formatid"], max_candidates=1)
        assert capped.metrics.counters["label_partial"] == 0
        assert capped.metrics.counters["label_exact"] == 0
        assert capped.metrics.counters["label_unknown"] == 4

    @pytest.mark.heavy
    def test_compilation_is_input_order_invariant_and_candidates_hold_executed_orders(
        self,
    ) -> None:
        """Verify compile order does not change results and each executed order is a candidate."""
        first = sample_replay_payload("g1")
        second = sample_replay_payload("g2", winner="Bob")

        result = compile_payloads((first, second))

        assert result.to_dict() == compile_payloads((second, first)).to_dict()
        left, right = result.accepted_series[0].games[0].perspectives
        assert (9, 11) in left.decisions[1].evidence.candidates
        assert (9, 11) in right.decisions[1].evidence.candidates
        assert left.snapshots[1].pre_line_index < left.snapshots[1].post_line_index
        assert result.metrics.counters["illegal_candidates"] == 0

    def test_compiler_matches_serial_results_with_two_workers(self, tmp_path: Path) -> None:
        script = tmp_path / "compare_compilers.py"
        script.write_text(
            textwrap.dedent("""            import os
            from p0.replays.compile import compile_payloads
            from tests.unit.replay_fixtures import golden_replay_payload

            if __name__ == "__main__":
                assert os.cpu_count() == 2
                payloads = tuple(
                    golden_replay_payload(f"game-{i}", series_id=f"series-{i}")
                    for i in range(3)
                )
                serial = compile_payloads(payloads, chunksize=0)
                assert len(serial.accepted_series) == 3
                # Three jobs exceed the two CPUs, so positive chunks use the process pool.
                for chunksize in (1, 2):
                    parallel = compile_payloads(payloads, chunksize=chunksize)
                    assert parallel.to_dict() == serial.to_dict()
            """),
            encoding="utf-8",
        )
        root = Path(__file__).resolve().parents[2]
        environment = {
            **os.environ,
            "PYTHON_CPU_COUNT": "2",
            "PYTHONPATH": os.pathsep.join((str(root), str(root / "src"))),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }

        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, result.stdout + result.stderr

    def test_reconstructed_views_carry_open_team_sheet_nature(self) -> None:
        """Verify open team sheet natures match literal sheet values for both projected sides."""
        payload = payload_with_ots_natures("ots-nature")
        result = compile_payloads((payload,))
        assert result.accepted_series

        expected_team_0 = {
            "Pikachu": "Jolly",
            "Eevee": "Adamant",
            "Raichu": "Serious",
            "Jolteon": "Serious",
            "Vaporeon": "Serious",
            "Flareon": "Serious",
        }
        expected_team_1 = {
            "Bulbasaur": "Bold",
            "Charmander": "Timid",
            "Squirtle": "Serious",
            "Ivysaur": "Serious",
            "Charmeleon": "Serious",
            "Wartortle": "Serious",
        }
        for game in result.accepted_series[0].games:
            for perspective in game.perspectives:
                expected_own = expected_team_0 if perspective.player == 0 else expected_team_1
                expected_opp = expected_team_1 if perspective.player == 0 else expected_team_0
                for snapshot in perspective.snapshots:
                    assert {
                        mon.species: mon.nature for mon in snapshot.view.team.values()
                    } == expected_own
                    assert {
                        mon.species: mon.nature for mon in snapshot.view.opponent_team.values()
                    } == expected_opp

    def test_external_rejections_keep_raw_identity_and_do_not_enter_dataset(
        self,
        tmp_path: Path,
    ) -> None:
        payload = golden_replay_payload("accepted", series_id="accepted-series")
        result = compile_payloads((payload,), chunksize=0)
        built = write_tensor_shards(
            result,
            tmp_path,
            created_at="2026-01-01T00:00:00Z",
            external_rejections={"malformed": "a" * 64},
        )
        dataset = LazyReplayDataset(built.manifest_path)

        assert built.manifest.source_games == 2
        assert built.manifest.accepted_games == 1
        assert built.manifest.rejected_games == 1
        assert set(built.manifest.raw_replays) == {"accepted", "malformed"}
        assert dataset.accepted_series_ids() == ("72673574c974979d25baec4b",)
        assert dataset.manifest.raw_replays["malformed"] == "a" * 64
        chunks = tuple(dataset)
        assert len(chunks) == 2
        assert all(chunk.series_id == "72673574c974979d25baec4b" for chunk in chunks)

    def test_existing_dataset_ignores_legacy_release_report(self, tmp_path: Path) -> None:
        payload = golden_replay_payload("legacy-report", series_id="legacy-series")
        result = compile_payloads((payload,), chunksize=0)
        first = write_tensor_shards(
            result,
            tmp_path,
            created_at="2026-01-01T00:00:00Z",
        )
        (first.manifest_path.parent / "release_gate.json").write_text("not-json", encoding="utf-8")

        document = result.series[0].games[0]
        second = compile_to_shards(iter((document,)), tmp_path)

        assert second.manifest_path == first.manifest_path
        with pytest.raises(ValueError, match="max_candidates must be positive"):
            write_tensor_shards(result, tmp_path, max_candidates=0)
        with pytest.raises(ValueError, match="max_candidates must be positive"):
            compile_to_shards((document,), tmp_path, max_candidates=0)
        with pytest.raises(ReplayInputContractError, match="duplicate"):
            compile_to_shards((document, document), tmp_path)
        invalid = replace(document, outcome=replace(document.outcome, terminal_line_index=None))
        with pytest.raises(ReplayInputContractError, match="terminal"):
            compile_to_shards((invalid,), tmp_path)
        (first.manifest_path.parent / first.manifest.shards[0].filename).write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="artifact failed validation"):
            compile_to_shards((document,), tmp_path)

    def test_extra_game_after_series_won_rejected(self) -> None:
        """Verify that a 2-0 series followed by a 3rd game is rejected under precision-first policy."""
        games = tuple(
            sample_replay_payload(f"g{i}", game_number=i, winner="Alice") for i in (1, 2, 3)
        )
        result = compile_payloads(games)
        assert not result.accepted_series
        assert result.metrics.counters["accepted_games"] == 0
        assert result.metrics.counters["rejected_games"] == 3
        assert result.metrics.counters["rejected_game_after_series_won"] == 1
        assert result.series[0].record.game_replay_ids == ("g1", "g2", "g3")

    def test_four_game_series_accounting(self, tmp_path: Path) -> None:
        """Verify four same-parent inputs retain all four replay identities in source accounting while being rejected."""
        valid = [
            sample_replay_payload(f"v{i}", parent="p-valid", game_number=i, winner="Alice")
            for i in (1, 2)
        ]
        four = [
            sample_replay_payload(f"f{i}", parent="p-four", game_number=i, winner="Alice")
            for i in (1, 2, 3, 4)
        ]
        result = compile_payloads((*valid, *four))

        assert result.metrics.counters["accepted_games"] == 2
        assert result.metrics.counters["rejected_games"] == 4
        assert result.metrics.counters["rejected_too_many_games"] >= 1
        assert {game.replay_id for game in result.accepted_series[0].games} == {"v1", "v2"}

        built = write_tensor_shards(result, tmp_path / "shards")
        assert built.manifest.source_games == 6
        assert built.manifest.accepted_games == 2
        assert built.manifest.rejected_games == 4
        assert {"f1", "f2", "f3", "f4"} <= set(built.manifest.raw_replays)
