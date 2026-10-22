"""Tests for causal projection, replay stats, and the production compiler path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from p0.battle.events import EventKind
from p0.battle.legality import GAME_END_DECISION
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import (
    CAT_IDX_PRESENCE_STATUS,
    NUM_IDX_CAN_MEGA,
    NUM_IDX_LEGALITY_UNKNOWN,
    NUM_IDX_ORIG_IDX_RATIO,
    PresenceStatus,
)
from p0.replays.compile import compile_payloads
from p0.replays.protocol import parse_replay_payload
from p0.replays.reconstruction.decisions import (
    infer_decision_windows,
    reconstruct_decisions_from_trace,
)
from p0.replays.reconstruction.projection import (
    project_replay_perspectives,
)
from p0.replays.reconstruction.resolution import resolve_replay_events
from p0.replays.reconstruction.state import reduce_replay_state
from tests.unit.replay_fixtures import decision_payload, golden_replay_payload

_REPLAY_FIXTURE_DIRECTORY = Path(__file__).parents[1] / "fixtures" / "replays"
_ILLUSION_REPLAY = _REPLAY_FIXTURE_DIRECTORY / "gen9championsvgc2026regmbbo3-2670656566.json"
_STATE_REPLAY = _REPLAY_FIXTURE_DIRECTORY / "gen9championsvgc2026regmbbo3-2641278886.json"


class TestReconstructionProjection:
    def test_unrevealed_illusion_uses_displayed_public_fields(self) -> None:
        path = _ILLUSION_REPLAY
        document = parse_replay_payload(path.read_bytes())
        replay_events = resolve_replay_events(document).require_accepted()
        replay_state = reduce_replay_state(document.metadata.replay_id, document.ots, replay_events)
        windows = infer_decision_windows(replay_events)
        decisions = (
            reconstruct_decisions_from_trace(
                document, replay_events, replay_state, perspective=0, windows=windows
            ),
            reconstruct_decisions_from_trace(
                document, replay_events, replay_state, perspective=1, windows=windows
            ),
        )
        perspectives = project_replay_perspectives(document, replay_state, decisions)
        perspective = perspectives[0]
        snapshot = next(item for item in perspective.snapshots if item.pre_line_index == 112)
        disguised = snapshot.view.opponent_active_pokemon[0]
        assert disguised is not None
        assert disguised.species == "Venusaur"
        assert disguised.base_species == "Venusaur"
        assert disguised.ability == "chlorophyll"
        assert tuple(disguised.moves) == ("sleeppowder", "gigadrain", "toxic", "solarbeam")
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
        # The disguise entered after the real Venusaur, so it takes Venusaur's team entry,
        # as live tracking does, and the real Venusaur's row carries no roster index.
        venusaur = next(
            member.member_id.roster_index
            for member in document.ots[1].members
            if member.species == "Venusaur"
        )
        assert observation.numerical[6, 26].item() == pytest.approx((venusaur + 1) / 6)
        assert observation.numerical[7, 26].item() == 0.0
        assert observation.numerical[6, NUM_IDX_CAN_MEGA].item() == 0.0
        assert observation.numerical[6, NUM_IDX_LEGALITY_UNKNOWN].item() == 0.0
        assert observation.categorical[6, 5:9].count_nonzero().item() == 4

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

        first_result = compile_payloads((first_payload,))
        second_result = compile_payloads((second_payload,))
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


class TestSpatialEventIntervals:
    def test_intervals_close_at_each_players_decision_and_keep_the_final_exchange(self) -> None:
        """A waiting player's interval spans the opponent's decision instead of resetting."""
        payload = _STATE_REPLAY.read_bytes()
        document = parse_replay_payload(payload)
        raw_lines = [line.raw for line in document.protocol_lines]
        replay_events = resolve_replay_events(document).require_accepted()
        replay_state = reduce_replay_state(document.metadata.replay_id, document.ots, replay_events)
        windows = infer_decision_windows(replay_events)
        decisions = (
            reconstruct_decisions_from_trace(
                document, replay_events, replay_state, perspective=0, windows=windows
            ),
            reconstruct_decisions_from_trace(
                document, replay_events, replay_state, perspective=1, windows=windows
            ),
        )
        perspectives = project_replay_perspectives(document, replay_state, decisions)
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


class TestDisguisedFaint:
    def test_unrevealed_faint_shows_on_the_disguise_until_that_pokemon_enters(self) -> None:
        # Zoroark leads looking like Wartortle and faints to Life Orb recoil, which does not
        # reveal it. Pikachu faints in the same turn, so p1 is asked for a replacement before
        # the real Wartortle enters.
        payload = decision_payload()
        payload["log"] = (
            str(payload["log"])
            .replace(
                '"species":"Charmander","name":"Charmander","ability":"Overgrow"',
                '"species":"Zoroark","name":"Zoroark","ability":"Illusion"',
            )
            .replace(
                "|switch|p2b: Charmander|Charmander, L50|100/100",
                "|switch|p2b: Wartortle|Wartortle, L50|100/100",
            )
            .replace(
                "|move|p1b: Eevee|Tackle|p2b: Charmander\n"
                "|move|p2a: Bulbasaur|Protect|p2a: Bulbasaur\n"
                "|move|p2b: Charmander|Tackle|p1b: Eevee\n"
                "|\n",
                "|move|p1b: Eevee|Protect|p1b: Eevee\n"
                "|move|p2a: Bulbasaur|Protect|p2a: Bulbasaur\n"
                "|move|p2b: Wartortle|Tackle|p1a: Pikachu\n"
                "|-damage|p1a: Pikachu|0 fnt\n"
                "|faint|p1a: Pikachu\n"
                "|-damage|p2b: Wartortle|0 fnt|[from] item: Life Orb\n"
                "|faint|p2b: Wartortle\n"
                "|\n"
                "|upkeep\n"
                "|\n"
                "|switch|p1a: Raichu|Raichu, L50|100/100\n"
                "|switch|p2b: Wartortle|Wartortle, L50|100/100\n"
                "|turn|2\n"
                "|\n"
                "|move|p1a: Raichu|Protect|p1a: Raichu\n"
                "|move|p1b: Eevee|Protect|p1b: Eevee\n"
                "|\n",
            )
        )

        lines = str(payload["log"]).splitlines()
        sheet_index = next(
            index for index, line in enumerate(lines) if line.startswith("|showteam|p2|")
        )
        entries = json.loads(lines[sheet_index].split("|", 3)[3])
        entries[1]["item"] = "Life Orb"
        lines[sheet_index] = "|showteam|p2|" + json.dumps(entries, separators=(",", ":"))
        payload["log"] = "\n".join(lines)

        opponent, owner = compile_payloads((payload,)).accepted_series[0].games[0].perspectives

        _, _, replacement, next_turn = opponent.snapshots
        seen_at_replacement = {
            mon.species: mon.fainted for mon in replacement.view.opponent_team.values()
        }
        assert seen_at_replacement["Wartortle"] is True
        assert seen_at_replacement["Zoroark"] is False
        seen_next_turn = {mon.species: mon.fainted for mon in next_turn.view.opponent_team.values()}
        assert seen_next_turn["Wartortle"] is False
        assert seen_next_turn["Zoroark"] is True
        # Its owner always knows which Pokemon fainted.
        own_replacement = owner.snapshots[2]
        known = {mon.species: mon.fainted for mon in own_replacement.view.team.values()}
        assert known["Zoroark"] is True
        assert known["Wartortle"] is False


def _illusion_payload(p2_members: tuple[tuple[str, str], ...], body: str) -> dict[str, Any]:
    """Return the decision payload with a p2 sheet of (species, ability) and a new log body."""
    payload = decision_payload()
    lines = str(payload["log"]).splitlines()
    preview = lines[: next(i for i, line in enumerate(lines) if line.startswith("|showteam|p2|"))]
    entries = [
        {
            "species": species,
            "name": species,
            "ability": ability,
            "moves": ["Protect", "Night Daze" if ability == "Illusion" else "Tackle"],
            "item": "Life Orb" if ability == "Illusion" else "Leftovers",
        }
        for species, ability in p2_members
    ]
    payload["log"] = "\n".join(
        (*preview, f"|showteam|p2|{json.dumps(entries, separators=(',', ':'))}", *body.splitlines())
    )
    return payload


_ZOROARK_TEAM = (
    ("Bulbasaur", "Overgrow"),
    ("Zoroark", "Illusion"),
    ("Squirtle", "Torrent"),
    ("Ivysaur", "Overgrow"),
    ("Charmeleon", "Blaze"),
    ("Wartortle", "Torrent"),
)
_LEADS = "|\n|switch|p1a: Pikachu|Pikachu, L50|100/100\n|switch|p1b: Eevee|Eevee, L50|100/100\n"
_PROTECT_TURN = "|\n|move|p1a: Pikachu|Protect|p1a: Pikachu\n|move|p1b: Eevee|Protect|p1b: Eevee\n"


def _turn_view(perspective: Any, turn: int) -> Any:
    return next(item.view for item in perspective.snapshots if item.view.turn == turn)


class TestIllusionRoster:
    def test_owner_knows_its_disguise_does_not_have_the_copied_pressure(self) -> None:
        payload = _illusion_payload(
            _ZOROARK_TEAM,
            "|switch|p1a: Weavile|Weavile, L50|100/100\n"
            "|switch|p1b: Eevee|Eevee, L50|100/100\n"
            "|switch|p2a: Bulbasaur|Bulbasaur, L50|100/100\n"
            "|switch|p2b: Wartortle|Wartortle, L50|100/100\n"
            "|turn|1\n|\n"
            "|move|p1a: Weavile|Night Daze|p2a: Bulbasaur\n"
            "|move|p1b: Eevee|Protect|p1b: Eevee\n"
            "|move|p2a: Bulbasaur|Tackle|p1a: Weavile|[miss]\n"
            "|move|p2b: Wartortle|Night Daze|p1b: Eevee\n"
            "|\n|upkeep\n|turn|2\n|\n"
            "|move|p1a: Weavile|Protect|p1a: Weavile\n"
            "|move|p1b: Eevee|Protect|p1b: Eevee\n"
            "|move|p2a: Bulbasaur|Protect|p2a: Bulbasaur\n"
            "|move|p2b: Wartortle|Protect|p2b: Wartortle\n|\n|win|Alice",
        )
        lines = str(payload["log"]).splitlines()
        sheet_index = next(
            index for index, line in enumerate(lines) if line.startswith("|showteam|p1|")
        )
        members = (
            ("Weavile", "Pressure"),
            ("Zoroark", "Illusion"),
            ("Eevee", "Run Away"),
            ("Raichu", "Static"),
            ("Vaporeon", "Water Absorb"),
            ("Flareon", "Flash Fire"),
        )
        entries = [
            {
                "species": name,
                "name": name,
                "ability": ability,
                "item": "Leftovers",
                "moves": ["Protect", "Night Daze" if ability == "Illusion" else "Tackle"],
            }
            for name, ability in members
        ]
        lines[sheet_index] = "|showteam|p1|" + json.dumps(entries, separators=(",", ":"))
        payload["log"] = "\n".join(lines)

        opponent = compile_payloads((payload,)).accepted_series[0].games[0].perspectives[0]

        bulbasaur = _turn_view(opponent, 2).opponent_active_pokemon[0]
        assert bulbasaur is not None
        tackle = bulbasaur.moves["tackle"]
        assert tackle.current_pp == tackle.max_pp - 1

    def test_distinct_later_identities_have_identical_opponent_prefixes(self) -> None:
        prefix = (
            _LEADS + "|switch|p2a: Bulbasaur|Bulbasaur, L50|100/100\n"
            "|switch|p2b: Wartortle|Wartortle, L50|100/100\n"
            "|turn|1\n" + _PROTECT_TURN + "|move|p2a: Bulbasaur|Tackle|p1a: Pikachu\n"
        )
        actual = _illusion_payload(
            _ZOROARK_TEAM,
            prefix + "|move|p2b: Wartortle|Tackle|p1a: Pikachu\n|\n|win|Alice",
        )
        disguised = _illusion_payload(
            _ZOROARK_TEAM,
            prefix + "|move|p2b: Wartortle|Night Daze|p1a: Pikachu\n|\n|win|Alice",
        )

        real_game = compile_payloads((actual,)).accepted_series[0].games[0]
        disguised_game = compile_payloads((disguised,)).accepted_series[0].games[0]

        builder = ObservationBuilder(default_runtime_resources())
        for turn in (0, 1):
            first = builder.build(_turn_view(real_game.perspectives[0], turn))
            second = builder.build(_turn_view(disguised_game.perspectives[0], turn))
            for left, right in zip(first.tensors(), second.tensors(), strict=True):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
        real_owner = _turn_view(real_game.perspectives[1], 1).active_pokemon[1]
        disguised_owner = _turn_view(disguised_game.perspectives[1], 1).active_pokemon[1]
        assert real_owner is not None and real_owner.species == "Wartortle"
        assert disguised_owner is not None and disguised_owner.species == "Zoroark"

    def test_a_copied_pokemon_counts_as_seen_and_the_owner_sees_its_illusion_user(self) -> None:
        # Zoroark leads looking like Wartortle and uses Night Daze, which Wartortle lacks.
        payload = _illusion_payload(
            _ZOROARK_TEAM,
            _LEADS + "|switch|p2a: Bulbasaur|Bulbasaur, L50|100/100\n"
            "|switch|p2b: Wartortle|Wartortle, L50|100/100\n"
            "|turn|1\n" + _PROTECT_TURN + "|move|p2a: Bulbasaur|Protect|p2a: Bulbasaur\n"
            "|move|p2b: Wartortle|Night Daze|p1a: Pikachu\n"
            "|-activate|p1a: Pikachu|move: Protect\n"
            "|\n"
            "|upkeep\n"
            "|turn|2\n"
            "|\n"
            "|switch|p2b: Squirtle|Squirtle, L50|100/100\n"
            "|move|p1a: Pikachu|Tackle|p2a: Bulbasaur\n"
            "|-damage|p2a: Bulbasaur|90/100\n"
            "|move|p1b: Eevee|Tackle|p2a: Bulbasaur\n"
            "|-damage|p2a: Bulbasaur|80/100\n"
            "|move|p2a: Bulbasaur|Tackle|p1a: Pikachu\n"
            "|-damage|p1a: Pikachu|90/100\n"
            "|\n"
            "|upkeep\n"
            "|turn|3\n" + _PROTECT_TURN + "|\n"
            "|win|Alice",
        )

        opponent, owner = compile_payloads((payload,)).accepted_series[0].games[0].perspectives

        # Live tracking files the disguise under Wartortle, so Wartortle counts as seen.
        seen = {mon.species: mon.revealed for mon in _turn_view(opponent, 3).opponent_team.values()}
        assert seen["Wartortle"] is True
        assert seen["Zoroark"] is False
        known = {mon.species: mon.revealed for mon in _turn_view(owner, 3).team.values()}
        assert known["Zoroark"] is True

    def test_a_type_change_on_a_disguise_is_public(self) -> None:
        # Zoroark leads looking like Charmeleon, uses Night Daze, and is soaked.
        payload = _illusion_payload(
            _ZOROARK_TEAM,
            _LEADS + "|switch|p2a: Bulbasaur|Bulbasaur, L50|100/100\n"
            "|switch|p2b: Charmeleon|Charmeleon, L50|100/100\n"
            "|turn|1\n" + _PROTECT_TURN + "|move|p2a: Bulbasaur|Protect|p2a: Bulbasaur\n"
            "|move|p2b: Charmeleon|Night Daze|p1a: Pikachu\n"
            "|-activate|p1a: Pikachu|move: Protect\n"
            "|-start|p2b: Charmeleon|typechange|Water\n"
            "|\n"
            "|upkeep\n"
            "|turn|2\n" + _PROTECT_TURN + "|\n"
            "|win|Alice",
        )

        opponent, owner = compile_payloads((payload,)).accepted_series[0].games[0].perspectives

        disguise = _turn_view(opponent, 2).opponent_active_pokemon[1]
        assert disguise is not None and disguise.species == "Charmeleon"
        assert [value.name for value in disguise.types] == ["Water"]
        zoroark = _turn_view(owner, 2).active_pokemon[1]
        assert zoroark is not None and zoroark.species == "Zoroark"
        assert [value.name for value in zoroark.types] == ["Water"]

    def test_a_type_added_to_a_disguise_joins_its_shown_types(self) -> None:
        # Zoroark leads looking like Charmeleon, uses Night Daze, and gains Grass.
        payload = _illusion_payload(
            _ZOROARK_TEAM,
            _LEADS + "|switch|p2a: Bulbasaur|Bulbasaur, L50|100/100\n"
            "|switch|p2b: Charmeleon|Charmeleon, L50|100/100\n"
            "|turn|1\n" + _PROTECT_TURN + "|move|p2a: Bulbasaur|Protect|p2a: Bulbasaur\n"
            "|move|p2b: Charmeleon|Night Daze|p1a: Pikachu\n"
            "|-activate|p1a: Pikachu|move: Protect\n"
            "|-start|p2b: Charmeleon|typeadd|Grass|[from] move: Forest's Curse\n"
            "|\n"
            "|upkeep\n"
            "|turn|2\n" + _PROTECT_TURN + "|\n"
            "|win|Alice",
        )

        opponent, owner = compile_payloads((payload,)).accepted_series[0].games[0].perspectives

        disguise = _turn_view(opponent, 2).opponent_active_pokemon[1]
        assert disguise is not None and disguise.species == "Charmeleon"
        assert [value.name for value in disguise.types] == ["Fire", "Grass"]
        zoroark = _turn_view(owner, 2).active_pokemon[1]
        assert zoroark is not None and zoroark.species == "Zoroark"
        assert [value.name for value in zoroark.types] == ["Dark", "Grass"]

    def test_a_revealed_illusion_user_stays_seen_when_it_returns_disguised(self) -> None:
        payload = _illusion_payload(
            _ZOROARK_TEAM,
            _LEADS + "|switch|p2a: Bulbasaur|Bulbasaur, L50|100/100\n"
            "|switch|p2b: Wartortle|Wartortle, L50|100/100\n"
            "|turn|1\n"
            "|\n"
            "|move|p1a: Pikachu|Tackle|p2b: Wartortle\n"
            "|-damage|p2b: Wartortle|90/100\n"
            "|replace|p2b: Zoroark|Zoroark, L50\n"
            "|-end|p2b: Zoroark|Illusion\n"
            "|move|p1b: Eevee|Protect|p1b: Eevee\n"
            "|move|p2a: Bulbasaur|Protect|p2a: Bulbasaur\n"
            "|move|p2b: Zoroark|Protect|p2b: Zoroark\n"
            "|\n"
            "|upkeep\n"
            "|turn|2\n"
            "|\n"
            "|switch|p2b: Squirtle|Squirtle, L50|100/100\n"
            "|move|p1a: Pikachu|Protect|p1a: Pikachu\n"
            "|move|p1b: Eevee|Protect|p1b: Eevee\n"
            "|move|p2a: Bulbasaur|Protect|p2a: Bulbasaur\n"
            "|\n"
            "|upkeep\n"
            "|turn|3\n"
            "|\n"
            "|switch|p2b: Wartortle|Wartortle, L50|90/100\n"
            "|move|p1a: Pikachu|Protect|p1a: Pikachu\n"
            "|move|p1b: Eevee|Protect|p1b: Eevee\n"
            "|move|p2a: Bulbasaur|Protect|p2a: Bulbasaur\n"
            "|\n"
            "|upkeep\n"
            "|turn|4\n" + _PROTECT_TURN + "|\n"
            "|win|Alice",
        )

        opponent, _ = compile_payloads((payload,)).accepted_series[0].games[0].perspectives

        view = _turn_view(opponent, 4)
        disguise = view.opponent_active_pokemon[1]
        assert disguise is not None and disguise.species == "Wartortle"
        zoroark = next(mon for mon in view.opponent_team.values() if mon.species == "Zoroark")
        assert zoroark.revealed
        observation = ObservationBuilder(default_runtime_resources()).build(view)
        # Bench rows in sheet order: Zoroark and Squirtle were seen, Ivysaur and Charmeleon not.
        assert observation.categorical[6:12, CAT_IDX_PRESENCE_STATUS].tolist() == [
            PresenceStatus.ACTIVE,
            PresenceStatus.ACTIVE,
            PresenceStatus.BENCH_REVEALED,
            PresenceStatus.BENCH_REVEALED,
            PresenceStatus.RESERVE_UNCONFIRMED,
            PresenceStatus.RESERVE_UNCONFIRMED,
        ]

    def test_a_disguise_fainting_beside_the_pokemon_it_copies_stays_unseen(self) -> None:
        # The second Wartortle uses Night Daze, so it is the disguise, and Life Orb recoil
        # makes it faint without a reveal while the real Wartortle stands beside it.
        payload = _illusion_payload(
            _ZOROARK_TEAM,
            _LEADS + "|switch|p2a: Wartortle|Wartortle, L50|100/100\n"
            "|switch|p2b: Bulbasaur|Bulbasaur, L50|100/100\n"
            "|turn|1\n"
            "|\n"
            "|switch|p2b: Wartortle|Wartortle, L50|100/100\n"
            "|move|p1a: Pikachu|Protect|p1a: Pikachu\n"
            "|move|p1b: Eevee|Protect|p1b: Eevee\n"
            "|move|p2a: Wartortle|Protect|p2a: Wartortle\n"
            "|move|p2b: Wartortle|Night Daze|p1a: Pikachu\n"
            "|-activate|p1a: Pikachu|move: Protect\n"
            "|-damage|p2b: Wartortle|0 fnt|[from] item: Life Orb\n"
            "|faint|p2b: Wartortle\n"
            "|\n"
            "|upkeep\n"
            "|\n"
            "|switch|p2b: Bulbasaur|Bulbasaur, L50|100/100\n"
            "|turn|2\n"
            "|\n"
            "|switch|p2a: Squirtle|Squirtle, L50|100/100\n"
            "|move|p1a: Pikachu|Tackle|p2b: Bulbasaur\n"
            "|-damage|p2b: Bulbasaur|90/100\n"
            "|move|p1b: Eevee|Tackle|p2b: Bulbasaur\n"
            "|-damage|p2b: Bulbasaur|80/100\n"
            "|move|p2b: Bulbasaur|Protect|p2b: Bulbasaur\n"
            "|\n"
            "|upkeep\n"
            "|turn|3\n"
            "|\n"
            "|switch|p2a: Wartortle|Wartortle, L50|100/100\n"
            "|move|p1a: Pikachu|Tackle|p2b: Bulbasaur\n"
            "|-damage|p2b: Bulbasaur|70/100\n"
            "|move|p1b: Eevee|Tackle|p2b: Bulbasaur\n"
            "|-damage|p2b: Bulbasaur|60/100\n"
            "|move|p2b: Bulbasaur|Protect|p2b: Bulbasaur\n"
            "|\n"
            "|upkeep\n"
            "|turn|4\n" + _PROTECT_TURN + "|\n"
            "|win|Alice",
        )

        opponent, owner = compile_payloads((payload,)).accepted_series[0].games[0].perspectives

        standing = _turn_view(opponent, 2).opponent_active_pokemon[0]
        assert standing is not None and standing.species == "Wartortle"
        assert not standing.fainted
        # Live tracking never moves the faint onto Zoroark, even after Wartortle returns.
        for turn in (2, 4):
            fainted = {
                mon.species: mon.fainted
                for mon in _turn_view(opponent, turn).opponent_team.values()
            }
            assert fainted == dict.fromkeys(fainted, False)
        known = {mon.species: mon.fainted for mon in _turn_view(owner, 4).team.values()}
        assert known["Zoroark"] is True

    def test_a_later_disguise_beside_a_changed_form_takes_the_team_entry(self) -> None:
        # Aegislash turns to its Blade form, then a disguise of that form enters beside it.
        payload = _illusion_payload(
            (
                ("Aegislash", "Stance Change"),
                ("Zoroark", "Illusion"),
                ("Squirtle", "Torrent"),
                ("Ivysaur", "Overgrow"),
                ("Charmeleon", "Blaze"),
                ("Wartortle", "Torrent"),
            ),
            _LEADS + "|switch|p2a: Aegislash|Aegislash, L50|100/100\n"
            "|switch|p2b: Squirtle|Squirtle, L50|100/100\n"
            "|turn|1\n" + _PROTECT_TURN + "|-formechange|p2a: Aegislash|Aegislash-Blade|"
            "[from] ability: Stance Change\n"
            "|move|p2a: Aegislash|Tackle|p1a: Pikachu\n"
            "|-activate|p1a: Pikachu|move: Protect\n"
            "|move|p2b: Squirtle|Protect|p2b: Squirtle\n"
            "|\n"
            "|upkeep\n"
            "|turn|2\n"
            "|\n"
            "|switch|p2b: Aegislash|Aegislash-Blade, L50|100/100\n"
            "|move|p1a: Pikachu|Protect|p1a: Pikachu\n"
            "|move|p1b: Eevee|Protect|p1b: Eevee\n"
            "|move|p2a: Aegislash|Protect|p2a: Aegislash\n"
            "|move|p2b: Aegislash|Night Daze|p1a: Pikachu\n"
            "|-activate|p1a: Pikachu|move: Protect\n"
            "|\n"
            "|upkeep\n"
            "|turn|3\n" + _PROTECT_TURN + "|\n"
            "|win|Alice",
        )

        opponent, _ = compile_payloads((payload,)).accepted_series[0].games[0].perspectives

        view = _turn_view(opponent, 3)
        observation = ObservationBuilder(default_runtime_resources()).build(view)
        # Live tracking gives the team entry to the later of two displays, so the
        # disguise in the second slot carries Aegislash's roster index.
        assert observation.numerical[6, NUM_IDX_ORIG_IDX_RATIO].item() == 0.0
        assert observation.numerical[7, NUM_IDX_ORIG_IDX_RATIO].item() == pytest.approx(1 / 6)
