"""Tests for causal projection, replay stats, and the production compiler path."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
from poke_env.battle import DoubleBattle

from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import (
    CAT_IDX_STAT_PROVENANCE,
    StatProvenance,
)
from p0.replays.compile import compile_documents, compile_payloads, write_tensor_shards
from p0.replays.protocol import parse_replay_payload
from p0.replays.reconstruction.projection import impute_replay_stats
from tests.unit.replay_fixtures import decision_payload

_GOLDEN_REPLAY_DIRECTORY = (
    Path(__file__).parents[2] / "src/p0/replays/reconstruction/golden_replays"
)


class TestReconstructionProjection:
    def test_production_compiler_projects_both_perspectives_from_one_compilation(self) -> None:
        result = compile_payloads((decision_payload(),), chunksize=0)

        assert len(result.games) == 1
        first, second = result.games[0].perspectives
        assert first.player == 0
        assert second.player == 1
        assert first.snapshots[0].view.teampreview
        assert second.snapshots[0].view.teampreview
        assert first.snapshots[1].view.team != second.snapshots[1].view.team
        assert first.snapshots[1].view.opponent_team != second.snapshots[1].view.opponent_team
        first_active = first.snapshots[1].view.active_pokemon[0]
        second_active = second.snapshots[1].view.active_pokemon[0]
        assert first_active is not None
        assert second_active is not None
        assert first_active.member_id.side.value == "p1"
        assert second_active.member_id.side.value == "p2"

    def test_replay_stats_are_explicit_for_both_sides_and_never_known(self) -> None:
        document = parse_replay_payload(decision_payload())
        estimates = impute_replay_stats(document, dex=default_runtime_resources().dex)

        assert len(estimates) == 12
        assert {estimate.member_id for estimate in estimates} == {
            member.member_id for sheet in document.ots for member in sheet.members
        }
        assert all(estimate.provenance in {"IMPUTED", "UNKNOWN"} for estimate in estimates)
        assert all(estimate.provenance != "KNOWN" for estimate in estimates)

    def test_shards_preserve_explicit_unknown_stat_provenance(self, tmp_path: Path) -> None:
        result = compile_payloads((decision_payload(),), chunksize=0)
        built = write_tensor_shards(
            result,
            tmp_path,
            resources=default_runtime_resources(),
            created_at="2026-01-01T00:00:00Z",
        )

        assert "compiler_backend" not in built.manifest.build_config
        artifact = torch.load(
            built.manifest_path.parent / built.manifest.shards[0].filename,
            map_location="cpu",
            weights_only=True,
        )
        provenance = artifact["tensors"]["categorical"][:, :12, CAT_IDX_STAT_PROVENANCE]
        assert torch.all(provenance == int(StatProvenance.UNKNOWN))

    def test_golden_corpus_retains_only_resolved_illusion_histories(self) -> None:
        paths = tuple(sorted(_GOLDEN_REPLAY_DIRECTORY.glob("*.json")))
        documents = tuple(parse_replay_payload(path.read_bytes()) for path in paths)
        result = compile_documents(documents, chunksize=0)

        assert len(paths) == 51
        assert result.metrics.counters["accepted_games"] == 40
        assert result.metrics.counters["rejected_games"] == 11
        assert result.metrics.counters["rejected_reconstruction_unresolved_illusion"] == 11
        assert len(result.games) == 40
        assert all(
            len(perspective.snapshots) == len(perspective.decisions)
            for game in result.games
            for perspective in game.perspectives
        )

    def test_state_matches_independent_poke_env_cursors(self) -> None:
        path = sorted(_GOLDEN_REPLAY_DIRECTORY.glob("*.json"))[0]
        document = parse_replay_payload(path.read_bytes())
        result = compile_payloads((document.raw_payload,), chunksize=0)
        assert len(result.games) == 1
        perspective = result.games[0].perspectives[0]
        boundaries = {snapshot.pre_line_index: snapshot for snapshot in perspective.snapshots}
        oracle = DoubleBattle(
            document.metadata.replay_id,
            document.metadata.player_names[0],
            logging.getLogger("p0.test.oracle"),
            gen=9,
        )
        skipped = frozenset({"", "t:", "expire", "uhtmlchange", "showteam", "win", "tie"})

        for line in document.protocol_lines:
            snapshot = boundaries.get(line.index)
            if (
                snapshot is not None
                and not snapshot.view.teampreview
                and oracle.player_role is not None
            ):
                live_active = oracle.active_pokemon
                for slot, live in enumerate(live_active):
                    projected = snapshot.view.active_pokemon[slot]
                    assert (live is None) == (projected is None)
                    if live is None or projected is None:
                        continue
                    assert live.fainted == projected.fainted
                    assert live.current_hp_fraction == pytest.approx(
                        projected.current_hp_fraction,
                        abs=0.02,
                    )
                    assert dict(live.boosts) == dict(projected.boosts)

            if line.parts[1] in skipped:
                continue
            try:
                oracle.parse_message(list(line.parts))
            except (AssertionError, IndexError, KeyError, NotImplementedError, ValueError) as exc:
                pytest.fail(f"poke-env rejected {line.raw!r}: {exc}")
