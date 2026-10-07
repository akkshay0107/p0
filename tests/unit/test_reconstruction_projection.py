"""Tests for causal projection, replay stats, and the production compiler path."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from poke_env.battle import DoubleBattle

from p0.battle.events import EventKind
from p0.battle.legality import GAME_END_DECISION
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import (
    CAT_IDX_STAT_PROVENANCE,
    NUM_IDX_CAN_MEGA,
    NUM_IDX_LEGALITY_UNKNOWN,
    StatProvenance,
)
from p0.replays.compile import compile_payloads, write_tensor_shards
from p0.replays.group import group_replays
from p0.replays.protocol import parse_replay_payload
from p0.replays.reconstruction.decisions import reconstruct_decisions_from_trace
from p0.replays.reconstruction.projection import (
    impute_replay_stats,
    project_battle_view,
    project_replay_perspectives,
)
from p0.replays.reconstruction.resolution import resolve_replay_events
from p0.replays.reconstruction.state import AbilityState, reduce_replay_state
from tests.unit.replay_fixtures import decision_payload, golden_replay_payload

_GOLDEN_REPLAY_DIRECTORY = (
    Path(__file__).parents[2] / "src/p0/replays/reconstruction/golden_replays"
)
_REPLAY_FIXTURE_DIRECTORY = Path(__file__).parents[1] / "fixtures" / "replays"
_ILLUSION_REPLAY = _REPLAY_FIXTURE_DIRECTORY / "gen9championsvgc2026regmbbo3-2670656566.json"
_STATE_REPLAY = _REPLAY_FIXTURE_DIRECTORY / "gen9championsvgc2026regmbbo3-2641278886.json"


class TestReconstructionProjection:
    def test_later_illusion_identity_cannot_change_prior_observation(self) -> None:
        path = _ILLUSION_REPLAY
        document = parse_replay_payload(path.read_bytes())
        resources = default_runtime_resources()
        events = resolve_replay_events(document, dex=resources.dex).require_accepted()
        original = reduce_replay_state(
            document.metadata.replay_id,
            document.ots,
            events,
            dex=resources.dex,
            snapshot_line_indices=(111,),
        ).require_accepted()[0]
        hidden_id = original.sides[1].active[0]
        assert hidden_id is not None
        hidden = original.member(hidden_id)
        altered = replace(
            hidden,
            original_species="Zoroark",
            current_form="Zoroark",
            nature="Bold",
            item="Choice Scarf",
            ability=AbilityState("mystery"),
            base_types=("Dark",),
            current_types=("Dark",),
            base_stats=tuple((name, 1) for name, _ in hidden.base_stats),
            weight=1000.0,
            moves=(),
        )
        changed = replace(
            original,
            members=tuple(
                altered if member.member_id == hidden_id else member for member in original.members
            ),
        )
        dex = default_runtime_resources().dex
        replay_events = resolve_replay_events(document, dex=dex).require_accepted()
        replay_state = reduce_replay_state(
            document.metadata.replay_id, document.ots, replay_events, dex=dex
        )
        decisions = (
            reconstruct_decisions_from_trace(
                document, replay_events, replay_state, perspective=0, dex=dex
            ),
            reconstruct_decisions_from_trace(
                document, replay_events, replay_state, perspective=1, dex=dex
            ),
        )
        perspectives = project_replay_perspectives(
            document, replay_events, replay_state, decisions, dex=dex
        )
        decision = next(
            item.view.decision for item in perspectives[0].snapshots if item.pre_line_index == 112
        )
        first_view = project_battle_view(
            original, document.ots, perspective=0, decision=decision, dex=resources.dex
        )
        second_view = project_battle_view(
            changed, document.ots, perspective=0, decision=decision, dex=resources.dex
        )
        builder = ObservationBuilder(resources)
        first = builder.build(first_view)
        second = builder.build(second_view)
        for left, right in zip(first.tensors(), second.tensors(), strict=True):
            torch.testing.assert_close(left, right)

    def test_unrevealed_illusion_uses_displayed_public_fields(self) -> None:
        path = _ILLUSION_REPLAY
        document = parse_replay_payload(path.read_bytes())
        dex = default_runtime_resources().dex
        replay_events = resolve_replay_events(document, dex=dex).require_accepted()
        replay_state = reduce_replay_state(
            document.metadata.replay_id, document.ots, replay_events, dex=dex
        )
        decisions = (
            reconstruct_decisions_from_trace(
                document, replay_events, replay_state, perspective=0, dex=dex
            ),
            reconstruct_decisions_from_trace(
                document, replay_events, replay_state, perspective=1, dex=dex
            ),
        )
        perspectives = project_replay_perspectives(
            document, replay_events, replay_state, decisions, dex=dex
        )
        perspective = perspectives[0]
        snapshot = next(
            snapshot
            for snapshot in perspective.snapshots
            if any(
                mon is not None and mon.identity_uncertain
                for mon in snapshot.view.opponent_active_pokemon
            )
        )
        disguised = snapshot.view.opponent_active_pokemon[0]
        assert disguised is not None
        assert disguised.species == "Venusaur"
        assert disguised.base_species == "Venusaur"
        assert disguised.ability == "unknown"
        assert tuple(disguised.moves) == ()
        assert tuple(value.name for value in disguised.types) == ("Grass", "Poison")
        own_snapshot = next(
            item
            for item in perspectives[1].snapshots
            if item.pre_line_index == snapshot.pre_line_index
        )
        own_active = own_snapshot.view.active_pokemon[0]
        assert own_active is not None
        assert own_active.species == "Zoroark-Hisui"

        observation = ObservationBuilder(default_runtime_resources()).build(snapshot.view)
        assert observation.numerical[6, 6].item() == pytest.approx(80 / 160)
        assert observation.numerical[6, 26].item() == 0.0
        assert observation.numerical[6, NUM_IDX_CAN_MEGA].item() == 0.0
        assert observation.numerical[6, NUM_IDX_LEGALITY_UNKNOWN].item() == 1.0
        assert observation.categorical[6, 5:9].count_nonzero().item() == 0

    def test_illusion_does_not_replace_a_known_bench_member(self) -> None:
        path = _ILLUSION_REPLAY
        document = parse_replay_payload(path.read_bytes())
        resources = default_runtime_resources()
        events = resolve_replay_events(document, dex=resources.dex).require_accepted()
        state = reduce_replay_state(
            document.metadata.replay_id,
            document.ots,
            events,
            dex=resources.dex,
            snapshot_line_indices=(111,),
        ).require_accepted()[0]
        known_venusaur = state.sides[1].active[1]
        assert known_venusaur is not None
        bench_side = replace(state.sides[1], active=(state.sides[1].active[0], None))
        bench_state = replace(state, sides=(state.sides[0], bench_side))
        dex = default_runtime_resources().dex
        replay_events = resolve_replay_events(document, dex=dex).require_accepted()
        replay_state = reduce_replay_state(
            document.metadata.replay_id, document.ots, replay_events, dex=dex
        )
        decisions = (
            reconstruct_decisions_from_trace(
                document, replay_events, replay_state, perspective=0, dex=dex
            ),
            reconstruct_decisions_from_trace(
                document, replay_events, replay_state, perspective=1, dex=dex
            ),
        )
        perspectives = project_replay_perspectives(
            document, replay_events, replay_state, decisions, dex=dex
        )
        decision = next(
            item.view.decision for item in perspectives[0].snapshots if item.pre_line_index == 112
        )

        view = project_battle_view(
            bench_state,
            document.ots,
            perspective=0,
            decision=decision,
            dex=resources.dex,
        )

        displayed = view.opponent_active_pokemon[0]
        assert displayed is not None and displayed.identity_uncertain
        assert displayed.species == "Venusaur"
        bench = next(
            member for member in view.opponent_team.values() if member.member_id == known_venusaur
        )
        assert bench.species == "Venusaur"
        assert bench.revealed
        assert not bench.identity_uncertain

        observation = ObservationBuilder(resources).build(view)
        displayed_species = observation.categorical[6, 0]
        assert (observation.categorical[6:12, 0] == displayed_species).sum().item() == 1

    def test_compiled_game_projects_opposing_perspectives_with_correct_roster_and_sides(
        self,
    ) -> None:
        """Verify opposing perspectives mirror each other with correct roster states, side roles, and flags."""
        result = compile_payloads((decision_payload(),), chunksize=0)

        first, second = result.accepted_series[0].games[0].perspectives
        assert first.player == 0
        assert second.player == 1
        assert first.snapshots[0].view.teampreview
        assert second.snapshots[0].view.teampreview

        # Cross-perspective roster states mirror each other
        first_team = first.snapshots[1].view.team
        first_opp = first.snapshots[1].view.opponent_team
        second_team = second.snapshots[1].view.team
        second_opp = second.snapshots[1].view.opponent_team

        assert {k: mon.state for k, mon in first_team.items()} == {
            k: mon.state for k, mon in second_opp.items()
        }
        assert {k: mon.state for k, mon in first_opp.items()} == {
            k: mon.state for k, mon in second_team.items()
        }
        assert all(not mon.opponent for mon in first_team.values())
        assert all(mon.opponent for mon in first_opp.values())
        assert all(not mon.opponent for mon in second_team.values())
        assert all(mon.opponent for mon in second_opp.values())

        first_active = first.snapshots[1].view.active_pokemon[0]
        second_active = second.snapshots[1].view.active_pokemon[0]
        assert first_active is not None
        assert second_active is not None
        assert first_active.member_id.side.value == "p1"
        assert second_active.member_id.side.value == "p2"
        first_opp_active = first.snapshots[1].view.opponent_active_pokemon[0]
        second_opp_active = second.snapshots[1].view.opponent_active_pokemon[0]
        assert first_opp_active is not None
        assert second_opp_active is not None
        assert first_active.state == second_opp_active.state
        assert second_active.state == first_opp_active.state

    def test_preview_observation_excludes_executed_lead_choices(self) -> None:
        first_payload = golden_replay_payload("preview-first")
        second_payload = golden_replay_payload("preview-second")
        second_payload["log"] = (
            str(second_payload["log"])
            .replace(
                "|switch|p1a: Pikachu|Pikachu, L50|100/100",
                "|switch|p1a: Raichu|Raichu, L50|100/100",
            )
            .replace(
                "|move|p1a: Pikachu|Protect|p1a: Pikachu",
                "|move|p1a: Raichu|Protect|p1a: Raichu",
            )
        )

        first_result = compile_payloads((first_payload,), chunksize=0)
        second_result = compile_payloads((second_payload,), chunksize=0)
        first = first_result.accepted_series[0].games[0].perspectives[0]
        second = second_result.accepted_series[0].games[0].perspectives[0]
        first_preview = first.snapshots[0].view
        second_preview = second.snapshots[0].view

        assert first_preview.active_pokemon == second_preview.active_pokemon == (None, None)
        assert (
            tuple(member.selected_in_teampreview for member in first_preview.team.values())
            == (None,) * 6
        )
        assert (
            tuple(member.selected_in_teampreview for member in second_preview.team.values())
            == (None,) * 6
        )

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

    @pytest.mark.skipif(
        not tuple(_GOLDEN_REPLAY_DIRECTORY.glob("*.json")),
        reason="local golden replay corpus is not present",
    )
    def test_golden_corpus_retains_only_resolved_illusion_histories(self) -> None:
        paths = tuple(sorted(_GOLDEN_REPLAY_DIRECTORY.glob("*.json")))
        documents = tuple(parse_replay_payload(path.read_bytes()) for path in paths)
        grouping = group_replays(documents)
        dex = default_runtime_resources().dex
        accepted_ids: set[str] = set()
        rejections: list[str] = []
        for document in documents:
            resolved = resolve_replay_events(document, dex=dex)
            if resolved.diagnostics:
                rejections.append(resolved.diagnostics[0].category.value)
                continue
            events = resolved.require_accepted()
            state = reduce_replay_state(document.metadata.replay_id, document.ots, events, dex=dex)
            if state.diagnostics:
                rejections.append(state.diagnostics[0].category.value)
                continue
            decisions = (
                reconstruct_decisions_from_trace(document, events, state, perspective=0, dex=dex),
                reconstruct_decisions_from_trace(document, events, state, perspective=1, dex=dex),
            )
            diagnostics = tuple(item for result in decisions for item in result.diagnostics)
            if diagnostics:
                rejections.append(diagnostics[0].category.value)
                continue
            perspectives = project_replay_perspectives(document, events, state, decisions, dex=dex)
            assert all(len(item.snapshots) == len(item.decisions) for item in perspectives)
            accepted_ids.add(document.metadata.replay_id)

        assert len(paths) == 51
        assert len(accepted_ids) == 40
        assert len(rejections) == 11
        accepted_groups = tuple(
            group
            for group in grouping.series
            if not any(item.code == "non_contiguous_game_numbers" for item in group.diagnostics)
            and all(game.metadata.replay_id in accepted_ids for game in group.games)
        )
        assert sum(len(group.games) for group in accepted_groups) == 33
        assert sum(group.record.is_complete for group in accepted_groups) == 11
        assert rejections.count("AMBIGUOUS_IDENTITY") == 11
        assert sum(item.code == "non_contiguous_game_numbers" for item in grouping.diagnostics) == 1

    def test_state_matches_independent_poke_env_cursors(self) -> None:
        path = _STATE_REPLAY
        document = parse_replay_payload(path.read_bytes())
        dex = default_runtime_resources().dex
        replay_events = resolve_replay_events(document, dex=dex).require_accepted()
        replay_state = reduce_replay_state(
            document.metadata.replay_id, document.ots, replay_events, dex=dex
        )
        decisions = (
            reconstruct_decisions_from_trace(
                document, replay_events, replay_state, perspective=0, dex=dex
            ),
            reconstruct_decisions_from_trace(
                document, replay_events, replay_state, perspective=1, dex=dex
            ),
        )
        perspectives = project_replay_perspectives(
            document, replay_events, replay_state, decisions, dex=dex
        )
        perspective = perspectives[0]
        boundaries = {snapshot.pre_line_index: snapshot for snapshot in perspective.snapshots}
        oracle = DoubleBattle(
            document.metadata.replay_id,
            document.metadata.player_names[0],
            logging.getLogger("p0.test.oracle"),
            gen=9,
        )
        skipped = frozenset({"", "t:", "expire", "uhtmlchange", "showteam", "win", "tie"})
        compared_boundaries = 0
        compared_members = 0

        for line in document.protocol_lines:
            snapshot = boundaries.get(line.index)
            if (
                snapshot is not None
                and not snapshot.view.teampreview
                and oracle.player_role is not None
            ):
                compared_boundaries += 1
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
                    compared_members += 1

            if line.parts[1] in skipped:
                continue
            try:
                oracle.parse_message(list(line.parts))
            except (AssertionError, IndexError, KeyError, NotImplementedError, ValueError) as exc:
                pytest.fail(f"poke-env rejected {line.raw!r}: {exc}")

        assert compared_boundaries == 6
        assert compared_members == 11


class TestSpatialEventIntervals:
    def test_each_player_sees_every_move_since_its_own_previous_decision(self) -> None:
        """A waiting player's interval spans the opponent's decision instead of resetting."""
        payload = _STATE_REPLAY.read_bytes()
        raw_lines = [line.raw for line in parse_replay_payload(payload).protocol_lines]
        document = parse_replay_payload(payload)
        dex = default_runtime_resources().dex
        replay_events = resolve_replay_events(document, dex=dex).require_accepted()
        replay_state = reduce_replay_state(
            document.metadata.replay_id, document.ots, replay_events, dex=dex
        )
        decisions = (
            reconstruct_decisions_from_trace(
                document, replay_events, replay_state, perspective=0, dex=dex
            ),
            reconstruct_decisions_from_trace(
                document, replay_events, replay_state, perspective=1, dex=dex
            ),
        )
        perspectives = project_replay_perspectives(
            document, replay_events, replay_state, decisions, dex=dex
        )
        starts = [[snapshot.pre_line_index for snapshot in item.snapshots] for item in perspectives]

        spans_opponent_decision = 0
        for perspective in (0, 1):
            opponent_starts = set(starts[1 - perspective])
            previous = 0
            for snapshot in perspectives[perspective].snapshots:
                start = snapshot.pre_line_index
                interval = raw_lines[previous:start]
                recorded = [record.kind for record in snapshot.view.spatial_events]

                assert recorded.count(EventKind.MOVE) == sum(
                    line.startswith("|move|") for line in interval
                )
                if any(previous < line < start for line in opponent_starts):
                    spans_opponent_decision += 1
                previous = start

        assert spans_opponent_decision > 0

    def test_final_view_holds_the_exchange_after_each_players_last_decision(self) -> None:
        """The last exchange has no request; the final view keeps it with no choice."""
        payload = _STATE_REPLAY.read_bytes()
        raw_lines = [line.raw for line in parse_replay_payload(payload).protocol_lines]
        document = parse_replay_payload(payload)
        dex = default_runtime_resources().dex
        replay_events = resolve_replay_events(document, dex=dex).require_accepted()
        replay_state = reduce_replay_state(
            document.metadata.replay_id, document.ots, replay_events, dex=dex
        )
        decisions = (
            reconstruct_decisions_from_trace(
                document, replay_events, replay_state, perspective=0, dex=dex
            ),
            reconstruct_decisions_from_trace(
                document, replay_events, replay_state, perspective=1, dex=dex
            ),
        )
        perspectives = project_replay_perspectives(
            document, replay_events, replay_state, decisions, dex=dex
        )

        for perspective in perspectives:
            final_interval = raw_lines[perspective.snapshots[-1].pre_line_index :]
            recorded = [record.kind for record in perspective.final_view.spatial_events]

            assert perspective.final_view.decision == GAME_END_DECISION
            assert recorded.count(EventKind.MOVE) == sum(
                line.startswith("|move|") for line in final_interval
            )
            assert recorded.count(EventKind.FAINT) == sum(
                line.startswith("|faint|") for line in final_interval
            )
            assert EventKind.MOVE in recorded
            damage_events = [
                r for r in perspective.final_view.spatial_events if r.kind == EventKind.DAMAGE
            ]
            assert damage_events
            assert all(r.amount_known == 1.0 and r.amount < 0.0 for r in damage_events)
