"""Tests for replay protocol parsing and grouping."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from p0.format_config import (
    DEFAULT_RUNTIME_MANIFEST,
)
from p0.replays.compile import (
    compile_documents,
    compile_payloads,
    write_tensor_shards,
)
from p0.replays.group import group_replays, individual_games, validated_bo3_series
from p0.replays.identity import ReplayMemberId, ReplaySide, linked_replay_ids
from p0.replays.protocol import ReplayInputContractError, ReplayParseError, parse_replay_payload
from p0.replays.reconstruction.projection import impute_replay_stats
from p0.replays.schema import (
    FetchMetadata,
    GameEndReason,
    OTSData,
    OTSMember,
    ProtocolLine,
    ReplayMetadata,
    ReplayOutcome,
)
from p0.replays.shards import (
    load_shard_manifest,
    validate_shard_tensors,
)
from tests.unit.replay_fixtures import sample_replay_payload


class TestReplayProtocolAndGrouping:
    def test_protocol_records_are_strict_and_ordered(self) -> None:
        """Verify protocol and ordered OTS records retain their exact sequence."""
        document = parse_replay_payload(sample_replay_payload("g1"))
        assert [line.index for line in document.protocol_lines] == list(
            range(len(document.protocol_lines))
        )
        assert tuple(member.species for member in document.ots[0].members[:2]) == (
            "Pikachu",
            "Eevee",
        )
        assert document.ots[0].members[0].moves == ("Protect", "Tackle")
        assert document.ots[0].members[0].member_id == ReplayMemberId(ReplaySide.P1, 0)
        assert document.ots[0].is_complete
        assert document.outcome.winner == 0
        assert ReplayMetadata.from_dict(document.metadata.to_dict()) == document.metadata
        assert (
            ProtocolLine.from_dict(document.protocol_lines[2].to_dict())
            == document.protocol_lines[2]
        )
        with pytest.raises(ValueError, match="unknown"):
            ProtocolLine.from_dict({**document.protocol_lines[0].to_dict(), "unknown": 1})
        with pytest.raises(ReplayParseError, match="Malformed protocol line"):
            parse_replay_payload({**sample_replay_payload("bad"), "log": "not a protocol line"})


class TestReplayInputContract:
    @pytest.mark.parametrize(
        "chat_lines",
        (
            "|c|Alice|hello\nroom text",
            "|chat|Alice|hello\nroom text",
            "|c:|Alice|hello\nroom text",
            "|chatmsg|Alice|hello\nroom text",
            (
                "|c|☆Alice|!dt sharp break\n"
                "'sharp break' has no exact match. Approximate match:\n"
                "|c|☆Alice|/raw <ul>Sharp Beak</ul>\nroom text"
            ),
        ),
    )
    def test_chat_aliases_and_multiline_responses_leave_exact_battle_stream(
        self, chat_lines: str
    ) -> None:
        """Verify chat aliases and their unframed responses are stripped from the exact battle stream."""
        payload = sample_replay_payload("chat-response")
        payload["log"] = str(payload["log"]).replace(
            "|win|Alice", f"{chat_lines}\n||MESSAGE\n|win|Alice"
        )

        document = parse_replay_payload(payload)

        assert tuple(line.raw for line in document.protocol_lines) == (
            "|start",
            "|teampreview",
            '|showteam|p1|[{"species":"Pikachu","ability":"Static","moves":["Protect","Tackle"]},{"species":"Eevee","ability":"Run Away","moves":["Tackle","Helping Hand"]},{"species":"Raichu","ability":"Static","moves":["Protect","Thunderbolt"]},{"species":"Jolteon","ability":"Volt Absorb","moves":["Protect","Thunderbolt"]},{"species":"Vaporeon","ability":"Water Absorb","moves":["Protect","Surf"]},{"species":"Flareon","ability":"Flash Fire","moves":["Protect","Flare Blitz"]}]',
            '|showteam|p2|[{"species":"Bulbasaur","ability":"Overgrow","moves":["Protect","Tackle"]},{"species":"Charmander","ability":"Blaze","moves":["Tackle","Helping Hand"]},{"species":"Squirtle","ability":"Torrent","moves":["Protect","Water Gun"]},{"species":"Ivysaur","ability":"Overgrow","moves":["Protect","Tackle"]},{"species":"Charmeleon","ability":"Blaze","moves":["Protect","Ember"]},{"species":"Wartortle","ability":"Torrent","moves":["Protect","Water Gun"]}]',
            "|",
            "|switch|p1a: Pikachu|Pikachu, L50|100/100",
            "|switch|p1b: Eevee|Eevee, L50|100/100",
            "|switch|p2a: Bulbasaur|Bulbasaur, L50|100/100",
            "|switch|p2b: Charmander|Charmander, L50|100/100",
            "|turn|1",
            "|",
            "|move|p1a: Pikachu|Protect|p1a: Pikachu",
            "|move|p1b: Eevee|Tackle|p2b: Charmander",
            "|move|p2a: Bulbasaur|Protect|p2a: Bulbasaur",
            "|move|p2b: Charmander|Tackle|p1b: Eevee",
            "|",
            "||MESSAGE",
            "|win|Alice",
        )
        assert document.outcome == ReplayOutcome(0, GameEndReason.NORMAL, 1, 17)

    @pytest.mark.parametrize(
        "ending, winner, reason, terminal_line_index",
        (
            ("|tie", -1, GameEndReason.NORMAL, 16),
            (
                "|-message|Alice| lost due to inactivity.\n|win|Bob",
                1,
                GameEndReason.TIMEOUT,
                17,
            ),
            (
                "|-message|Alice| forfeited.\n|win|Bob",
                1,
                GameEndReason.FORFEIT,
                17,
            ),
            (
                "|-message|All players are inactive.\n|tie",
                -1,
                GameEndReason.TIMEOUT,
                17,
            ),
        ),
    )
    def test_terminal_forms_preserve_terminal_index_and_reason(
        self,
        ending: str,
        winner: int,
        reason: GameEndReason,
        terminal_line_index: int,
    ) -> None:
        payload = sample_replay_payload("input-terminal")
        document = parse_replay_payload(
            {**payload, "log": str(payload["log"]).replace("|win|Alice", ending)}
        )
        assert document.outcome == ReplayOutcome(
            winner, reason, turns=1, terminal_line_index=terminal_line_index
        )

    def test_terminal_result_rejects_a_second_result_or_battle_event(self) -> None:
        payload = sample_replay_payload("input-conflicting-terminal")
        for suffix in ("|win|Bob", "|tie", "|win|Alice", "|turn|99"):
            invalid = {**payload, "log": f"{payload['log']}\n{suffix}"}
            with pytest.raises(ReplayInputContractError, match="after terminal result"):
                compile_payloads((invalid,), chunksize=0)

        trailing_display = {
            **payload,
            "log": f"{payload['log']}\n|t:|1700000000\n|message|Battle ended.",
        }
        assert len(compile_payloads((trailing_display,), chunksize=0).accepted_series[0].games) == 1

    def test_player_names_that_normalize_to_same_identity_are_rejected(self) -> None:
        invalid = sample_replay_payload("input-colliding-players", players=("Alice", "A lice"))
        with pytest.raises(ReplayInputContractError, match="distinct players"):
            parse_replay_payload(invalid)

    def test_malformed_metadata_and_complete_compile_contract_raise_named_errors(self) -> None:
        payload = sample_replay_payload("input-invalid")
        with pytest.raises(ReplayInputContractError, match="game_number"):
            parse_replay_payload({**payload, "game_number": "unknown"})

        document = parse_replay_payload(payload)
        incomplete = replace(document, ots=(OTSData(ReplaySide.P1, "", ()), document.ots[1]))
        with pytest.raises(ReplayInputContractError, match="complete six-member OTS"):
            compile_documents((incomplete,))

        with pytest.raises(ReplayInputContractError, match="does not match requested"):
            compile_documents((document,), format_id="unsupported-format")

    def test_ots_preserves_duplicate_species_as_distinct_ordered_members(self) -> None:
        payload = sample_replay_payload("duplicate-species")
        duplicate_roster = [
            {"name": "First", "species": "Pikachu", "moves": ["Protect"]},
            {"name": "Second", "species": "Pikachu", "moves": ["Tackle"]},
        ]
        lines = [
            (
                f"|showteam|p1|{json.dumps(duplicate_roster, separators=(',', ':'))}"
                if line.startswith("|showteam|p1|")
                else line
            )
            for line in str(payload["log"]).splitlines()
        ]
        payload["log"] = "\n".join(lines)

        document = parse_replay_payload(payload)
        members = document.ots[0].members

        assert tuple(member.species for member in members) == ("Pikachu", "Pikachu")
        assert tuple(member.nickname for member in members) == ("First", "Second")
        assert tuple(member.member_id for member in members) == (
            ReplayMemberId(ReplaySide.P1, 0),
            ReplayMemberId(ReplaySide.P1, 1),
        )
        assert OTSData.from_dict(document.ots[0].to_dict()) == document.ots[0]

    def test_ots_rejects_noncontiguous_and_cross_side_member_ids(self) -> None:
        member = parse_replay_payload(sample_replay_payload("member-ids")).ots[0].members[0]

        with pytest.raises(ValueError, match="contiguous ordered IDs"):
            OTSData(
                ReplaySide.P1,
                "payload",
                (replace(member, member_id=ReplayMemberId(ReplaySide.P1, 1)),),
            )
        with pytest.raises(ValueError, match="contiguous ordered IDs"):
            OTSData(
                ReplaySide.P1,
                "payload",
                (replace(member, member_id=ReplayMemberId(ReplaySide.P2, 0)),),
            )

    def test_protocol_rejects_repeated_showteam_payloads(self) -> None:
        payload = sample_replay_payload("repeated-showteam")
        p1_showteam = next(
            line for line in str(payload["log"]).splitlines() if line.startswith("|showteam|p1|")
        )
        payload["log"] = f"{payload['log']}\n{p1_showteam}"

        with pytest.raises(ReplayParseError, match="repeated showteam payloads for p1"):
            parse_replay_payload(payload)

    def test_public_bo3_metadata_and_empty_protocol_commands_are_preserved(self) -> None:
        """Verify public Showdown best-of-3 HTML headers parse parent room IDs and game numbers correctly."""
        payload = sample_replay_payload("gen9championsvgc2026regmcbo3-100")
        payload.pop("p1")
        payload.pop("p2")
        payload.pop("parent")
        payload.pop("game_number")
        payload["players"] = ["Alice", "Bob"]
        payload["format"] = "[Gen 9 Champions] VGC 2026 Reg M-C (Bo3)"
        payload["formatid"] = "gen9championsvgc2026regmcbo3"
        payload["log"] = "\n".join(
            [
                "|uhtml|bestof|<h2><strong>Game 1</strong> of "
                '<a href="/game-bestof3-gen9championsvgc2026regmcbo3-99">a best-of-3</a></h2>',
                "|",
                "||Alice is ready for game 2.",
                str(payload["log"]),
                "|uhtml|next|Next: "
                '<a href="/battle-gen9championsvgc2026regmcbo3-101">'
                "<strong>Game 2 of 3</strong></a>",
            ]
        )

        document = parse_replay_payload(
            payload,
            format_id="gen9championsvgc2026regmcbo3",
        )

        assert document.metadata.format_id == "gen9championsvgc2026regmcbo3"
        assert document.metadata.parent_room == ("game-bestof3-gen9championsvgc2026regmcbo3-99")
        assert document.metadata.game_number == 1
        assert [line.raw for line in document.protocol_lines[:3]] == [
            str(payload["log"]).splitlines()[0],
            "|",
            "||Alice is ready for game 2.",
        ]

    def test_null_parent_is_an_orphan_instead_of_a_literal_series_id(self) -> None:
        """Verify replays with parent=None fall back to player pairing rather than creating a string 'None' series."""
        payload = sample_replay_payload("orphan")
        payload["parent"] = None

        document = parse_replay_payload(payload)

        assert document.metadata.parent_room == ""
        assert group_replays((document,)).series[0].record.grouping_method.name == (
            "FALLBACK_SAME_PLAYERS"
        )

    def test_link_extraction_is_same_format_and_model_agnostic(self) -> None:
        """Verify linked_replay_ids only extracts next-game hyperlinks matching the active battle format."""
        payload = sample_replay_payload("gen9championsvgc2026regmcbo3-100")
        payload["log"] = "\n".join(
            [
                '|uhtml|bestof|<a href="/game-bestof3-gen9championsvgc2026regmcbo3-99">series</a>',
                '|uhtml|next|<a href="/battle-gen9championsvgc2026regmcbo3-101">Game 2</a>',
                '|uhtml|other|<a href="/battle-gen9otherformat-5">other</a>',
            ]
        )
        body = json.dumps(payload).encode()

        assert linked_replay_ids(
            body,
            format_id="gen9championsvgc2026regmcbo3",
        ) == ("gen9championsvgc2026regmcbo3-101",)

    def test_packed_open_team_sheet_imputation_is_deterministic(self) -> None:
        """Verify open team sheet EV/stat imputation is deterministic across repeated runs."""
        payload = sample_replay_payload("packed")
        payload["log"] = "\n".join(
            [
                "|start",
                "|teampreview",
                "|showteam|p1|Pikachu|||static|protect,tackle|Jolly|||||50",
                "|showteam|p2|Bulbasaur|||overgrow|protect,tackle|Bold|||||50",
                "|switch|p1a: Pikachu|Pikachu, L50",
                "|switch|p1b: Pikachu|Pikachu, L50",
                "|switch|p2a: Bulbasaur|Bulbasaur, L50",
                "|switch|p2b: Bulbasaur|Bulbasaur, L50",
                "|turn|1",
                "|move|p1a: Pikachu|Protect|p2a: Bulbasaur",
            ]
        )
        document = parse_replay_payload(payload)
        assert document.ots[0].members[0].moves == ("protect", "tackle")
        dex = {
            "species": [
                {
                    "id": "pikachu",
                    "baseStats": {"hp": 35, "atk": 55, "def": 40, "spa": 50, "spd": 50, "spe": 90},
                },
                {
                    "id": "bulbasaur",
                    "baseStats": {"hp": 45, "atk": 49, "def": 49, "spa": 65, "spd": 65, "spe": 45},
                },
            ],
            "moves": [
                {"id": "protect", "category": "Status"},
                {"id": "tackle", "category": "Physical"},
            ],
        }
        first = impute_replay_stats(document, dex=dex)
        second = impute_replay_stats(document, dex=dex)
        assert first == second

        # Pikachu is covered by the usage priors; Bulbasaur is not, and its sheet reveals
        # only one physical and one status move, so no fallback category reaches two.
        by_member = {item.member_id: item for item in first}
        pikachu_stat = by_member[document.ots[0].members[0].member_id]
        assert pikachu_stat.values == (111, 107, 61, 63, 71, 155)
        assert pikachu_stat.provenance == "IMPUTED"
        assert pikachu_stat.confidence == pytest.approx(0.470793, rel=1e-4)

        bulbasaur_stat = by_member[document.ots[1].members[0].member_id]
        assert bulbasaur_stat.values is None
        assert bulbasaur_stat.provenance == "UNKNOWN"
        assert bulbasaur_stat.confidence == 0.0

    def test_new_schema_records_round_trip(self) -> None:
        """Verify FetchMetadata, OTSData, and ReplayOutcome serialize and deserialize without loss."""
        fetch = FetchMetadata(
            source_url="https://example.invalid/replay.json",
            fetched_at="2026-07-19T00:00:00Z",
            http_status=200,
            attempt=2,
            retry_count=1,
            elapsed_ms=12,
        )
        assert FetchMetadata.from_dict(fetch.to_dict()) == fetch
        member = OTSMember(
            member_id=ReplayMemberId(ReplaySide.P1, 0),
            nickname="Pikachu",
            species="Pikachu",
            item="Light Ball",
            ability="Static",
            moves=("Protect", "Thunderbolt"),
            nature="Timid",
            gender="M",
            level=50,
            evs="0,0,0,252,4,252",
            raw_packed_set="Pikachu|Pikachu|LightBall|Static|Protect,Thunderbolt",
        )
        ots = OTSData(ReplaySide.P1, member.raw_packed_set, (member,))
        assert OTSData.from_dict(ots.to_dict()) == ots
        malformed_member = member.to_dict()
        malformed_member["item"] = 1
        with pytest.raises(ValueError, match="string fields"):
            OTSMember.from_dict(malformed_member)
        malformed_ots = ots.to_dict()
        malformed_ots["raw_payload"] = {"team": []}
        with pytest.raises(ValueError, match="raw_payload"):
            OTSData.from_dict(malformed_ots)
        outcome = ReplayOutcome(0, GameEndReason.NORMAL, 1, 4)
        assert ReplayOutcome.from_dict(outcome.to_dict()) == outcome

    def test_grouping_parent_and_fallback_are_deterministic(self) -> None:
        """Verify parent and fallback grouping pin ordered members, score and completion in either input order."""
        first = parse_replay_payload(sample_replay_payload("g1", parent="series-1"))
        second = parse_replay_payload(sample_replay_payload("g2", parent="series-1", winner="Bob"))
        fallback_first = parse_replay_payload(sample_replay_payload("fallback-game-1", parent=""))
        fallback_second = parse_replay_payload(sample_replay_payload("fallback-game-2", parent=""))

        for ordered in ((first, second), (second, first)):
            parent_result = group_replays(ordered, format_id=first.metadata.format_id)
            assert len(parent_result.series) == 1
            parent = parent_result.series[0]
            assert parent.record.game_replay_ids == ("g1", "g2")
            assert parent.record.score == (1, 1)
            assert not parent.record.is_complete
            assert parent.record.grouping_method.name == "PARENT_ROOM"

        for ordered in ((fallback_first, fallback_second), (fallback_second, fallback_first)):
            fallback_result = group_replays(ordered)
            assert len(fallback_result.series) == 1
            fallback = fallback_result.series[0]
            assert fallback.record.game_replay_ids == (
                "fallback-game-1",
                "fallback-game-2",
            )
            assert fallback.record.score == (2, 0)
            assert fallback.record.is_complete
            assert fallback.record.grouping_method.name == "FALLBACK_SAME_PLAYERS"

    def test_grouping_preserves_authoritative_numbers_and_stable_series_id(self) -> None:
        """Verify series grouping maintains stable series hashes as additional games in the match are discovered."""
        first = parse_replay_payload(sample_replay_payload("g1", game_number=1))
        second = parse_replay_payload(sample_replay_payload("g2", game_number=2))

        incomplete = group_replays((first,)).series[0]
        complete = group_replays((second, first)).series[0]

        assert incomplete.record.series_id == complete.record.series_id
        assert [membership.game_number for membership in complete.memberships] == [1, 2]
        assert complete.record.is_complete

    def test_grouping_quarantines_missing_and_duplicate_game_numbers(self) -> None:
        """Verify series grouping marks series with non-contiguous or duplicate game numbers as incomplete with diagnostics."""
        second = parse_replay_payload(sample_replay_payload("g2", game_number=2))
        third = parse_replay_payload(sample_replay_payload("g3", game_number=3))
        missing = group_replays((third, second)).series[0]

        duplicate_a = parse_replay_payload(sample_replay_payload("dup-a", game_number=1))
        duplicate_b = parse_replay_payload(sample_replay_payload("dup-b", game_number=1))
        duplicate = group_replays((duplicate_a, duplicate_b)).series[0]

        assert [membership.game_number for membership in missing.memberships] == [2, 3]
        assert not missing.record.is_complete
        assert "non_contiguous_game_numbers" in {
            diagnostic.code for diagnostic in missing.diagnostics
        }
        assert [membership.game_number for membership in duplicate.memberships] == [1, 1]
        assert not duplicate.record.is_complete
        assert "duplicate_game_number" in {diagnostic.code for diagnostic in duplicate.diagnostics}

    def test_grouping_quarantines_games_after_series_won(self) -> None:
        """Verify matches with extra games played after a 2-0 win are quarantined with game_after_series_won diagnostic."""
        games = tuple(
            parse_replay_payload(sample_replay_payload(f"g{number}", game_number=number))
            for number in (1, 2, 3)
        )

        group = group_replays(games).series[0]

        assert group.record.score == (2, 0)
        assert not group.record.is_complete
        assert "game_after_series_won" in {diagnostic.code for diagnostic in group.diagnostics}
        assert validated_bo3_series(games) == ()

    def test_grouping_quarantines_missing_outcomes_and_team_conflicts(self) -> None:
        """Verify series grouping flags incomplete outcomes and team roster changes across games in a BO3."""
        unresolved_payload = sample_replay_payload(
            "unresolved", game_number=1, parent="unresolved-series"
        )
        unresolved_payload["log"] = "\n".join(
            line
            for line in str(unresolved_payload["log"]).splitlines()
            if not line.startswith("|win|")
        )
        unresolved = group_replays((parse_replay_payload(unresolved_payload),)).series[0]

        first = parse_replay_payload(
            sample_replay_payload("team-1", game_number=1, parent="team-series")
        )
        changed_payload = sample_replay_payload("team-2", game_number=2, parent="team-series")
        changed_payload["log"] = str(changed_payload["log"]).replace("Pikachu", "Raichu")
        conflicted = group_replays((first, parse_replay_payload(changed_payload))).series[0]

        assert not unresolved.record.is_complete
        assert "missing_outcome" in {diagnostic.code for diagnostic in unresolved.diagnostics}
        assert not conflicted.record.is_complete
        assert "team_identity_conflict" in {
            diagnostic.code for diagnostic in conflicted.diagnostics
        }
        assert all(
            "team_identity_conflict" in membership.diagnostics
            for membership in conflicted.memberships
        )

    def test_side_roles_are_canonical_and_bo1_bo3_views_share_games(self) -> None:
        """Verify game_player_roles maps perspectives canonical to the match winner/loser across side swaps."""
        first = parse_replay_payload(sample_replay_payload("g1", game_number=1))
        second = parse_replay_payload(
            sample_replay_payload(
                "g2",
                game_number=2,
                winner="Alice",
                players=("Bob", "Alice"),
            )
        )
        grouping = group_replays((second, first)).series[0]

        assert grouping.record.game_player_roles == ((0, 1), (1, 0))
        assert grouping.record.score == (2, 0)
        assert tuple(game.metadata.replay_id for game in individual_games((second, first))) == (
            "g1",
            "g2",
        )
        assert validated_bo3_series((first,)) == ()
        inferred_first = parse_replay_payload(sample_replay_payload("inferred-1"))
        inferred_second = parse_replay_payload(sample_replay_payload("inferred-2"))
        assert validated_bo3_series((inferred_first, inferred_second)) == ()
        same_side_second = parse_replay_payload(sample_replay_payload("g2", game_number=2))
        assert tuple(
            game.metadata.replay_id
            for game in validated_bo3_series((same_side_second, first))[0].games
        ) == ("g1", "g2")

    def test_replay_fixture_compiles_to_runtime_bound_schema_v5_shard(self, tmp_path: Path) -> None:
        """Verify compile_payloads serializes into PyTorch schema v5 shards with valid action masks and CSR offsets."""
        result = compile_payloads((sample_replay_payload("shard-fixture"),))
        built = write_tensor_shards(result, tmp_path, created_at="2026-01-01T00:00:00Z")

        manifest = load_shard_manifest(
            json.loads(built.manifest_path.read_text(encoding="utf-8")), DEFAULT_RUNTIME_MANIFEST
        )
        assert manifest.decisions == 4
        assert manifest.games == 2
        assert manifest.series == 1
        assert manifest.diagnostics["label_unknown"] == 0

        shard_path = built.manifest_path.parent / manifest.shards[0].filename
        payload = torch.load(shard_path, weights_only=True, map_location="cpu")
        tensors = payload["tensors"]
        assert all(
            torch.isfinite(tensor).all()
            for tensor in tensors.values()
            if tensor.is_floating_point()
        )
        assert tensors["categorical"].shape[0] == manifest.decisions
        assert tensors["action_mask"].shape == (4, 2, 49)
        # Candidate counts use the pinned Showdown target classes: normal/any
        # exclude the user, while non-choosable targets contribute target 0.
        assert tensors["candidate_offsets"].tolist() == [0, 12, 15, 27, 30]
        assert tensors["game_offsets"].tolist() == [0, 2, 4]
        assert tensors["series_offsets"].tolist() == [0, 4]
        assert len(payload["series_summaries"]) == manifest.games
        assert [item["canonical_player"] for item in payload["series_summaries"]] == [0, 1]
        assert torch.count_nonzero(tensors["spatial_cat"]) > 0

        stale_provenance = tensors["mask_provenance"].clone()
        stale_provenance[0] = 2
        tensors["mask_provenance"] = stale_provenance
        with pytest.raises(ValueError, match="unsupported value"):
            validate_shard_tensors(tensors)

    def test_shard_bytes_are_deterministic_for_fixed_inputs(self, tmp_path: Path) -> None:
        """Verify write_tensor_shards produces byte-identical files and manifests across independent invocations with identical inputs."""
        result = compile_payloads((sample_replay_payload("shard-fixture"),))
        first = write_tensor_shards(result, tmp_path / "first", created_at="2026-01-01T00:00:00Z")
        second = write_tensor_shards(result, tmp_path / "second", created_at="2026-01-01T00:00:00Z")
        first_path = first.manifest_path.parent / first.manifest.shards[0].filename
        second_path = second.manifest_path.parent / second.manifest.shards[0].filename
        assert first_path.read_bytes() == second_path.read_bytes()
        assert first.manifest.to_dict() == second.manifest.to_dict()
