"""Tests for replay protocol parsing and grouping."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from p0.format_config import (
    DEFAULT_RUNTIME_MANIFEST,
    load_runtime_manifest,
)
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.replays.compile import (
    CompilationResult,
    ShardBuildResult,
    compile_documents,
    compile_payloads,
    write_tensor_shards,
)
from p0.replays.group import group_replays, individual_games, validated_bo3_series
from p0.replays.identity import ReplayMemberId, ReplaySide, linked_replay_ids
from p0.replays.protocol import ReplayInputContractError, ReplayParseError, parse_replay_payload
from p0.replays.reconstruction.projection import impute_replay_stats
from p0.replays.schema import (
    ActionEvidence,
    FetchMetadata,
    GameEndReason,
    GroupingMethod,
    LabelKind,
    MaskProvenance,
    OTSData,
    OTSMember,
    ProtocolLine,
    ReplayMetadata,
    ReplayOutcome,
    SeriesRecord,
)
from p0.replays.scrape import (
    HttpResponse,
    ReplayFetcher,
    ScrapeConfig,
    load_raw_replay,
)
from p0.replays.shards import (
    ShardIndexEntry,
    ShardManifest,
    load_shard_manifest,
    validate_shard_tensors,
)
from tests.unit.replay_fixtures import golden_replay_payload, sample_replay_payload


def _write_dataset_replay_dataset(
    tmp_path: Path, payloads: tuple[dict[str, object], ...]
) -> ShardBuildResult:
    result = compile_payloads(payloads)
    return write_tensor_shards(result, tmp_path, created_at="2026-01-01T00:00:00Z")


def torch_summaries(built) -> list[dict[str, object]]:
    payload_path = built.manifest_path.parent / built.manifest.shards[0].filename
    payload = torch.load(payload_path, weights_only=True, map_location="cpu")
    return payload["series_summaries"]


def _build_dataset_from_payloads(tmp_path, payloads):
    result = compile_payloads(payloads, format_id=payloads[0]["formatid"])
    return write_tensor_shards(
        result,
        tmp_path / "dataset",
        max_decisions_per_shard=1,
        created_at="2026-01-01T00:00:00Z",
    )


def _build_dataset(tmp_path, count: int):
    payloads = tuple(
        golden_replay_payload(f"dataset-{index}", series_id=f"dataset-series-{index}")
        for index in range(count)
    )
    return _build_dataset_from_payloads(tmp_path, payloads)


def _payload_with_ots_natures(replay_id: str) -> dict[str, object]:
    """A pipeline payload whose open team sheets declare natures, as real replays do."""
    natures = {"Pikachu": "Jolly", "Eevee": "Adamant", "Bulbasaur": "Bold", "Charmander": "Timid"}
    payload = sample_replay_payload(replay_id)
    lines = []
    for line in str(payload["log"]).splitlines():
        if line.startswith("|showteam|"):
            head, _, body = line.rpartition("|")
            roster = json.loads(body)
            for mon in roster:
                mon["nature"] = natures.get(mon["species"], "Serious")
            line = f"{head}|{json.dumps(roster, separators=(',', ':'))}"
        lines.append(line)
    payload["log"] = "\n".join(lines)
    return payload


def _numerical_rows(result: CompilationResult) -> list[tuple[float, ...]]:
    builder = ObservationBuilder(default_runtime_resources())
    rows: list[tuple[float, ...]] = []
    for game in result.games:
        for perspective in game.perspectives:
            for snapshot in perspective.snapshots:
                observation = builder.build(snapshot.view)
                rows.extend(
                    tuple(float(value) for value in token) for token in observation.numerical
                )
    return rows


def _evidence(kind: LabelKind) -> ActionEvidence:
    candidates = {
        LabelKind.EXACT: ((7, 1),),
        LabelKind.PARTIAL: ((7, 1), (8, 1)),
        LabelKind.UNKNOWN: (),
    }[kind]
    return ActionEvidence(
        label_kind=kind,
        candidates=candidates,
        confidence=0.5 if kind is not LabelKind.UNKNOWN else 0.0,
        mask_provenance=MaskProvenance.CONSERVATIVE_RECONSTRUCTED,
        tags=("fixture",),
    )


def _series_record() -> SeriesRecord:
    return SeriesRecord(
        series_id="s1",
        format_id="gen9championsvgc2026regmbbo3",
        players=("alice", "bob"),
        game_replay_ids=("r1", "r2"),
        game_player_roles=((0, 1), (1, 0)),
        team_hashes=("a" * 64, "b" * 64),
        is_complete=True,
        score=(2, 0),
        grouping_method=GroupingMethod.PARENT_ROOM,
        grouping_confidence=1.0,
    )


def _shard_manifest_fixture_unit() -> ShardManifest:
    active_contract = load_runtime_manifest().global_sha256
    entry = ShardIndexEntry(
        filename="shard-000.pt", sha256="c" * 64, decisions=10, games=2, series=1, byte_size=1024
    )
    return ShardManifest(
        global_contract_sha256=active_contract,
        shards=(entry,),
        diagnostics={"oov_ids": 0},
        created_at="2026-07-17T00:00:00Z",
        dataset_hash="d" * 64,
        source_format_id="gen9championsvgc2026regmbbo3",
        build_config={"max_candidates": 256},
        raw_replays={"game-1": "f" * 64},
        source_series={"series-1": ("game-1",)},
        source_games=1,
        accepted_games=1,
        rejected_games=0,
        artifact_hashes={
            "shard-000.pt": "c" * 64,
        },
    )


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
    def test_chat_variants_bare_room_message_and_double_bar_message_are_neutral(self) -> None:
        payload = sample_replay_payload("input-chat")
        for chat in ("c", "chat", "c:", "chatmsg"):
            candidate = {
                **payload,
                "log": str(payload["log"]).replace(
                    "|win|Alice", f"|{chat}|Alice|hello\nroom text\n|win|Alice"
                ),
            }
            document = parse_replay_payload(candidate)
            assert document.outcome.winner == 0

        candidate = {
            **payload,
            "log": str(payload["log"]).replace("|win|Alice", "||MESSAGE\n|win|Alice"),
        }
        document = parse_replay_payload(candidate)
        assert document.outcome.winner == 0

    def test_terminal_forms_preserve_terminal_index_and_reason(self) -> None:
        payload = sample_replay_payload("input-terminal")
        for ending, reason in (
            ("|tie", GameEndReason.NORMAL),
            ("|-message|Alice| lost due to inactivity.\n|win|Bob", GameEndReason.TIMEOUT),
            ("|-message|Alice| forfeited.\n|win|Bob", GameEndReason.FORFEIT),
        ):
            candidate = {**payload, "log": str(payload["log"]).replace("|win|Alice", ending)}
            document = parse_replay_payload(candidate)
            assert document.outcome.terminal_line_index is not None
            assert document.outcome.end_reason is reason
        assert (
            parse_replay_payload(
                {**payload, "log": str(payload["log"]).replace("|win|Alice", "|tie")}
            ).outcome.winner
            == -1
        )

    def test_inactivity_tie_is_a_timeout(self) -> None:
        payload = sample_replay_payload("input-inactivity-tie")
        payload["log"] = str(payload["log"]).replace(
            "|win|Alice", "|-message|All players are inactive.\n|tie"
        )

        assert parse_replay_payload(payload).outcome.end_reason is GameEndReason.TIMEOUT

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

    def test_changed_runtime_dex_is_rejected(self) -> None:
        payload = sample_replay_payload("input-dex")
        document = parse_replay_payload(payload)
        dex = json.loads(json.dumps(default_runtime_resources().dex))
        dex["moves"][0]["name"] = "Changed move name"
        with pytest.raises(ReplayInputContractError, match="pinned Champions artifact"):
            compile_documents((document,), dex=dex)

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

    def test_protocol_ignores_chat_and_multiline_chat_responses(self) -> None:
        """Verify chat commands (|c|, |chatmsg|, bot command output) are stripped from parsed battle protocol lines."""
        payload = sample_replay_payload("chat-response")
        payload["log"] = "\n".join(
            [
                str(payload["log"]),
                "|c|☆Alice|!dt sharp break",
                "'sharp break' has no exact match. Approximate match:",
                "|c|☆Alice|/raw <ul>Sharp Beak</ul>",
                "|turn|2",
            ]
        )

        document = parse_replay_payload(payload)

        assert all(line.parts[1] not in {"c", "chatmsg"} for line in document.protocol_lines)
        assert all("sharp break" not in line.raw for line in document.protocol_lines)

    def test_public_bo3_metadata_and_empty_protocol_commands_are_preserved(self) -> None:
        """Verify public Showdown best-of-3 HTML headers parse parent room IDs and game numbers correctly."""
        payload = sample_replay_payload("gen9championsvgc2026regmbbo3-100")
        payload.pop("p1")
        payload.pop("p2")
        payload.pop("parent")
        payload.pop("game_number")
        payload["players"] = ["Alice", "Bob"]
        payload["format"] = "[Gen 9 Champions] VGC 2026 Reg M-B (Bo3)"
        payload["formatid"] = "gen9championsvgc2026regmbbo3"
        payload["log"] = "\n".join(
            [
                "|uhtml|bestof|<h2><strong>Game 1</strong> of "
                '<a href="/game-bestof3-gen9championsvgc2026regmbbo3-99">a best-of-3</a></h2>',
                "|",
                "||Alice is ready for game 2.",
                str(payload["log"]),
                "|uhtml|next|Next: "
                '<a href="/battle-gen9championsvgc2026regmbbo3-101">'
                "<strong>Game 2 of 3</strong></a>",
            ]
        )

        document = parse_replay_payload(
            payload,
            format_id="gen9championsvgc2026regmbbo3",
        )

        assert document.metadata.format_id == "gen9championsvgc2026regmbbo3"
        assert document.metadata.parent_room == ("game-bestof3-gen9championsvgc2026regmbbo3-99")
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
        payload = sample_replay_payload("gen9championsvgc2026regmbbo3-100")
        payload["log"] = "\n".join(
            [
                '|uhtml|bestof|<a href="/game-bestof3-gen9championsvgc2026regmbbo3-99">series</a>',
                '|uhtml|next|<a href="/battle-gen9championsvgc2026regmbbo3-101">Game 2</a>',
                '|uhtml|other|<a href="/battle-gen9otherformat-5">other</a>',
            ]
        )
        body = json.dumps(payload).encode()

        assert linked_replay_ids(
            body,
            format_id="gen9championsvgc2026regmbbo3",
        ) == ("gen9championsvgc2026regmbbo3-101",)

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
        assert by_member[document.ots[0].members[0].member_id].provenance == "IMPUTED"
        assert by_member[document.ots[0].members[0].member_id].confidence > 0.0
        assert by_member[document.ots[1].members[0].member_id].provenance == "UNKNOWN"
        assert by_member[document.ots[1].members[0].member_id].values is None

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
        """Verify series grouping by parent room or player fallback resolves identical series records regardless of order."""
        first = parse_replay_payload(sample_replay_payload("g1", parent="series-1"))
        second = parse_replay_payload(sample_replay_payload("g2", parent="series-1", winner="Bob"))
        parent_result = group_replays((second, first), format_id=first.metadata.format_id)
        assert len(parent_result.series) == 1
        assert parent_result.series[0].record.game_replay_ids == ("g1", "g2")
        assert parent_result.series[0].record.score == (1, 1)
        fallback_first = parse_replay_payload(sample_replay_payload("fallback-game-1", parent=""))
        fallback_second = parse_replay_payload(sample_replay_payload("fallback-game-2", parent=""))
        fallback = group_replays((fallback_second, fallback_first))
        assert len(fallback.series) == 1
        assert fallback.series[0].record.grouping_method.name == "FALLBACK_SAME_PLAYERS"

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

    def test_grouping_quarantines_games_after_a_series_clinch(self) -> None:
        """Verify matches with extra games played after a 2-0 clinch are quarantined with game_after_series_clinch diagnostic."""
        games = tuple(
            parse_replay_payload(sample_replay_payload(f"g{number}", game_number=number))
            for number in (1, 2, 3)
        )

        group = group_replays(games).series[0]

        assert group.record.score == (2, 0)
        assert not group.record.is_complete
        assert "game_after_series_clinch" in {diagnostic.code for diagnostic in group.diagnostics}
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

    def test_reconstruction_is_causal_symmetric_and_compilable(self) -> None:
        """Verify battle reconstruction is causal (pre_line < post_line) and produces candidate actions containing executed orders."""
        first = sample_replay_payload("g1")
        second = sample_replay_payload("g2", winner="Bob")
        result = compile_payloads((first, second))
        assert result.to_dict() == compile_payloads((second, first)).to_dict()
        game = result.games[0]
        left, right = game.perspectives
        assert left.player == 0 and right.player == 1
        assert left.decisions[0].evidence.label_kind is LabelKind.PARTIAL
        assert left.decisions[1].evidence.label_kind is LabelKind.PARTIAL
        assert right.decisions[1].evidence.label_kind is LabelKind.PARTIAL
        assert (9, 11) in left.decisions[1].evidence.candidates
        assert (9, 11) in right.decisions[1].evidence.candidates
        assert left.snapshots[1].pre_line_index < left.snapshots[1].post_line_index
        assert result.metrics.counters["illegal_candidates"] == 0

    def test_fetcher_retries_and_writes_immutable_raw_cache(self, tmp_path) -> None:
        """Verify ReplayFetcher retries on HTTP 503 errors and writes raw gzip JSON replay logs to the disk cache."""
        payload = sample_replay_payload("g1")
        body = json.dumps(payload).encode()
        calls: list[str] = []
        failures = {"https://search.invalid?format=f&page=1": 1}

        def transport(url: str, timeout: float) -> HttpResponse:
            calls.append(url)
            if failures.get(url, 0):
                failures[url] -= 1
                return HttpResponse(503, b"retry")
            if "search.invalid" in url:
                return HttpResponse(
                    200, json.dumps([{"id": "g1", "uploadtime": 1_750_000_000}]).encode()
                )
            return HttpResponse(200, body)

        config = ScrapeConfig(
            format_id="f",
            cache_dir=tmp_path,
            search_url="https://search.invalid",
            replay_url_template="https://replay.invalid/{replay_id}.json",
            retries=2,
            backoff_seconds=0,
            rate_limit_per_second=0,
        )
        entries = ReplayFetcher(config, transport=transport).acquire()
        assert entries[0].replay_id == "g1"
        assert calls.count("https://search.invalid?format=f&page=1") == 2
        raw_path = tmp_path / "f" / "raw" / "g1.json.gz"
        assert load_raw_replay(raw_path) == body
        assert ReplayFetcher(config, transport=transport).acquire() == entries

    def test_fetcher_accepts_display_formats_and_follows_sibling_links(self, tmp_path) -> None:
        """Verify ReplayFetcher discovers search replay IDs and automatically fetches linked sibling games in a BO3."""
        format_id = "gen9championsvgc2026regmbbo3"
        first_id = f"{format_id}-100"
        second_id = f"{format_id}-101"
        first = sample_replay_payload(first_id, game_number=1)
        first["format"] = "[Gen 9 Champions] VGC 2026 Reg M-B (Bo3)"
        first["formatid"] = format_id
        first["log"] = (
            f"|uhtml|bestof|<strong>Game 1</strong> of "
            f'<a href="/game-bestof3-{format_id}-99">a best-of-3</a>\n'
            f'|uhtml|next|<a href="/battle-{second_id}">Game 2 of 3</a>\n'
            f"{first['log']}"
        )
        second = sample_replay_payload(second_id, game_number=2)
        second["format"] = "[Gen 9 Champions] VGC 2026 Reg M-B (Bo3)"
        second["formatid"] = format_id
        bodies = {
            f"https://replay.invalid/{first_id}.json": json.dumps(first).encode(),
            f"https://replay.invalid/{second_id}.json": json.dumps(second).encode(),
        }

        def transport(url: str, timeout: float) -> HttpResponse:
            del timeout
            if "search.invalid" in url:
                return HttpResponse(
                    200,
                    json.dumps(
                        [
                            {
                                "id": first_id,
                                "format": "[Gen 9 Champions] VGC 2026 Reg M-B (Bo3)",
                            }
                        ]
                    ).encode(),
                )
            return HttpResponse(200, bodies[url])

        config = ScrapeConfig(
            format_id=format_id,
            cache_dir=tmp_path,
            search_url="https://search.invalid",
            replay_url_template="https://replay.invalid/{replay_id}.json",
            rate_limit_per_second=0,
        )
        fetcher = ReplayFetcher(config, transport=transport)

        assert fetcher.discover_ids() == (first_id,)
        assert tuple(entry.replay_id for entry in fetcher.acquire((first_id,))) == (
            first_id,
            second_id,
        )

    def test_fetcher_preserves_malformed_replay_json_for_compilation_audit(self, tmp_path) -> None:
        """Verify non-JSON replay responses are stored in the raw cache for compilation auditing rather than silently dropped."""
        config = ScrapeConfig(
            format_id="f",
            cache_dir=tmp_path,
            replay_url_template="https://replay.invalid/{replay_id}.json",
            rate_limit_per_second=0,
        )

        def transport(url: str, timeout: float) -> HttpResponse:
            del url, timeout
            return HttpResponse(200, b"not-json")

        entries = ReplayFetcher(config, transport=transport).acquire(("f-1",))

        assert [entry.replay_id for entry in entries] == ["f-1"]
        assert load_raw_replay(tmp_path / "f" / "raw" / "f-1.json.gz") == b"not-json"

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

    def test_shard_manifest_rejects_runtime_contract_mismatch(self, tmp_path: Path) -> None:
        """Verify load_shard_manifest raises ValueError if the global contract SHA-256 does not match active runtime."""
        result = compile_payloads((sample_replay_payload("shard-fixture"),))
        built = write_tensor_shards(result, tmp_path, created_at="2026-01-01T00:00:00Z")
        value = json.loads(built.manifest_path.read_text(encoding="utf-8"))
        value["global_contract_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="global contract"):
            load_shard_manifest(value, DEFAULT_RUNTIME_MANIFEST)
