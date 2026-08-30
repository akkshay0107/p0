from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from p0.battle.legality import DecisionView, SlotDecision
from p0.format_config import (
    DEFAULT_RUNTIME_MANIFEST,
    FORMAT,
    current_manifest,
    load_active_runtime_manifest,
    load_runtime_manifest,
)
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import (
    StructuredObservation,
)
from p0.paths import DEFAULT_PATHS
from p0.replays.compile import (
    CompilationResult,
    ShardBuildResult,
    _perspective_tensors,
    compile_payloads,
    compile_to_shards,
    write_tensor_shards,
)
from p0.replays.dataset import (
    LazyReplayDataset,
    SeriesSplitManifest,
    assign_series_splits,
    load_split_manifest,
    write_split_manifest,
)
from p0.replays.evidence import EvidenceRequest, ObservedAction, extract_action_evidence
from p0.replays.group import group_replays, individual_games, validated_bo3_series
from p0.replays.identity import ReplayMemberId, ReplaySide, linked_replay_ids
from p0.replays.protocol import ReplayParseError, parse_replay_payload
from p0.replays.reconstruction import pipeline as reconstruction_pipeline
from p0.replays.reconstruction.projection import impute_replay_stats
from p0.replays.schema import (
    ActionEvidence,
    FetchIndexEntry,
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
    ReplayFetchError,
    ReplayUnavailableError,
    ScrapeConfig,
    load_raw_replay,
    read_fetch_index,
)
from p0.replays.shards import (
    SHARD_TENSOR_SPECS,
    ShardIndexEntry,
    ShardManifest,
    load_shard_manifest,
    observation_field_specs,
    validate_shard_tensors,
)
from tests.unit.replay_fixtures import golden_replay_payload


def _sample_replay_payload(
    replay_id: str,
    parent: str = "series-1",
    *,
    winner: str = "Alice",
    game_number: int | None = None,
    players: tuple[str, str] = ("Alice", "Bob"),
) -> dict[str, object]:
    ots = {
        "p1": [
            {"species": "Pikachu", "ability": "Static", "moves": ["Protect", "Tackle"]},
            {"species": "Eevee", "ability": "Run Away", "moves": ["Tackle", "Helping Hand"]},
            {"species": "Raichu", "ability": "Static", "moves": ["Protect", "Thunderbolt"]},
            {"species": "Jolteon", "ability": "Volt Absorb", "moves": ["Protect", "Thunderbolt"]},
            {"species": "Vaporeon", "ability": "Water Absorb", "moves": ["Protect", "Surf"]},
            {"species": "Flareon", "ability": "Flash Fire", "moves": ["Protect", "Flare Blitz"]},
        ],
        "p2": [
            {"species": "Bulbasaur", "ability": "Overgrow", "moves": ["Protect", "Tackle"]},
            {"species": "Charmander", "ability": "Blaze", "moves": ["Tackle", "Helping Hand"]},
            {"species": "Squirtle", "ability": "Torrent", "moves": ["Protect", "Water Gun"]},
            {"species": "Ivysaur", "ability": "Overgrow", "moves": ["Protect", "Tackle"]},
            {"species": "Charmeleon", "ability": "Blaze", "moves": ["Protect", "Ember"]},
            {"species": "Wartortle", "ability": "Torrent", "moves": ["Protect", "Water Gun"]},
        ],
    }
    lines = [
        "|start",
        "|teampreview",
        f"|showteam|p1|{json.dumps(ots['p1'], separators=(',', ':'))}",
        f"|showteam|p2|{json.dumps(ots['p2'], separators=(',', ':'))}",
        # a bare vertical bar is Showdown's update-block separator: the simulator writes one
        # every time it resumes, so each separator opens exactly one answered request
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
        f"|win|{winner}",
    ]
    return {
        "id": replay_id,
        "format": "gen9championsvgc2026regmbbo3",
        "p1": players[0],
        "p2": players[1],
        "uploadtime": 1_750_000_000,
        "roomid": replay_id,
        "parent": parent,
        "game_number": game_number,
        "log": "\n".join(lines),
    }


def _write_dataset_replay_dataset(
    tmp_path: Path, payloads: tuple[dict[str, object], ...]
) -> ShardBuildResult:
    result = compile_payloads(payloads)
    return write_tensor_shards(result, tmp_path, created_at="2026-01-01T00:00:00Z")


def test_split_assignment_is_order_independent_and_round_trips(tmp_path: Path) -> None:
    """Verify series split hashing produces deterministic train/val/test splits regardless of input series ID ordering."""
    global_hash = load_active_runtime_manifest(DEFAULT_RUNTIME_MANIFEST).global_sha256
    first = assign_series_splits(
        ("series-b", "series-a"),
        seed=17,
        validation_fraction=0.2,
        test_fraction=0.2,
        global_contract_sha256=global_hash,
        dataset_hash="a" * 64,
    )
    second = assign_series_splits(
        ("series-a", "series-b"),
        seed=17,
        validation_fraction=0.2,
        test_fraction=0.2,
        global_contract_sha256=global_hash,
        dataset_hash="a" * 64,
    )
    assert first.to_dict() == second.to_dict()
    path = tmp_path / "splits.json"
    write_split_manifest(first, path)
    assert load_split_manifest(path).to_dict() == first.to_dict()


def test_lazy_dataset_yields_canonical_bo3_game_perspectives(
    tmp_path: Path,
) -> None:
    """Verify LazyReplayDataset yields 2 perspectives per game with canonical player tracking and terminal series flags."""
    built = _write_dataset_replay_dataset(
        tmp_path, (_sample_replay_payload("game-1"), _sample_replay_payload("game-2"))
    )
    chunks = list(LazyReplayDataset(built.manifest_path))

    # 2 games * 2 perspectives = 4 chunks in sequential order
    assert [(chunk.game_number, chunk.player) for chunk in chunks] == [
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    ]
    assert [chunk.canonical_player for chunk in chunks] == [0, 1, 0, 1]
    assert [chunk.is_series_end for chunk in chunks] == [False, False, True, True]
    assert all(chunk.length == 2 for chunk in chunks)
    assert chunks[2].candidate_offsets.tolist() == [0, 12, 16]


def test_canonical_player_identity_survives_replay_side_swap(tmp_path: Path) -> None:
    """Verify canonical player IDs track original human players even when Showdown swaps p1/p2 sides between games."""
    first = _sample_replay_payload("game-1")
    first["game_number"] = 1
    second = _sample_replay_payload("game-2")
    second["game_number"] = 2
    second["p1"] = "Bob"
    second["p2"] = "Alice"

    built = _write_dataset_replay_dataset(tmp_path, (first, second))
    chunks = list(LazyReplayDataset(built.manifest_path))

    # Game 2 swaps sides (p1=Bob, p2=Alice), so canonical_player maps (p1->1, p2->0)
    assert [(chunk.game_number, chunk.player, chunk.canonical_player) for chunk in chunks] == [
        (1, 0, 0),
        (1, 1, 1),
        (2, 0, 1),
        (2, 1, 0),
    ]


def test_downstream_shards_preserve_noncontiguous_source_game_numbers(
    tmp_path: Path,
) -> None:
    """Verify LazyReplayDataset preserves source game numbers even when non-contiguous (e.g. games 2 and 3)."""
    second = _sample_replay_payload("game-2")
    second["game_number"] = 2
    third = _sample_replay_payload("game-3")
    third["game_number"] = 3

    built = _write_dataset_replay_dataset(tmp_path, (third, second))
    chunks = list(LazyReplayDataset(built.manifest_path))

    assert [(chunk.game_number, chunk.player) for chunk in chunks] == [
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
    ]


def test_split_dataset_keeps_series_together(tmp_path: Path) -> None:
    """Verify series split partitioning assigns all games of a series to the same split, avoiding data leakage."""
    built = _write_dataset_replay_dataset(
        tmp_path,
        (
            _sample_replay_payload("game-1", "series-1"),
            _sample_replay_payload("game-2", "series-2"),
        ),
    )
    series_ids = sorted({str(summary["series_id"]) for summary in torch_summaries(built)})
    global_hash = built.manifest.global_contract_sha256
    split = SeriesSplitManifest(
        global_hash,
        0,
        {series_ids[0]: "train", series_ids[1]: "test"},
        dataset_hash=built.manifest.dataset_hash,
    )
    split_path = tmp_path / "splits.json"
    write_split_manifest(split, split_path)
    train = list(LazyReplayDataset(built.manifest_path, split="train", split_manifest=split_path))
    test = list(LazyReplayDataset(built.manifest_path, split="test", split_manifest=split_path))
    assert {chunk.series_id for chunk in train} == {series_ids[0]}
    assert {chunk.series_id for chunk in test} == {series_ids[1]}


def test_dataset_rejects_tampered_shard(tmp_path: Path) -> None:
    """Verify LazyReplayDataset raises ValueError when shard bytes do not match the manifest SHA-256 hash."""
    built = _write_dataset_replay_dataset(tmp_path, (_sample_replay_payload("game-1"),))
    shard_path = built.manifest_path.parent / built.manifest.shards[0].filename
    shard_path.write_bytes(shard_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        next(iter(LazyReplayDataset(built.manifest_path, verify_hashes=True)))


def torch_summaries(built) -> list[dict[str, object]]:
    payload_path = built.manifest_path.parent / built.manifest.shards[0].filename
    payload = torch.load(payload_path, weights_only=True, map_location="cpu")
    return payload["series_summaries"]


def test_protocol_records_are_strict_and_ordered() -> None:
    """Verify protocol and ordered OTS records retain their exact sequence."""
    document = parse_replay_payload(_sample_replay_payload("g1"))
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
        ProtocolLine.from_dict(document.protocol_lines[2].to_dict()) == document.protocol_lines[2]
    )
    with pytest.raises(ValueError, match="unknown"):
        ProtocolLine.from_dict({**document.protocol_lines[0].to_dict(), "unknown": 1})
    with pytest.raises(ReplayParseError, match="Malformed protocol line"):
        parse_replay_payload({**_sample_replay_payload("bad"), "log": "not a protocol line"})


def test_ots_preserves_duplicate_species_as_distinct_ordered_members() -> None:
    payload = _sample_replay_payload("duplicate-species")
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


def test_ots_rejects_noncontiguous_and_cross_side_member_ids() -> None:
    member = parse_replay_payload(_sample_replay_payload("member-ids")).ots[0].members[0]

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


def test_protocol_rejects_repeated_showteam_payloads() -> None:
    payload = _sample_replay_payload("repeated-showteam")
    p1_showteam = next(
        line for line in str(payload["log"]).splitlines() if line.startswith("|showteam|p1|")
    )
    payload["log"] = f"{payload['log']}\n{p1_showteam}"

    with pytest.raises(ReplayParseError, match="repeated showteam payloads for p1"):
        parse_replay_payload(payload)


def test_protocol_ignores_chat_and_multiline_chat_responses() -> None:
    """Verify chat commands (|c|, |chatmsg|, bot command output) are stripped from parsed battle protocol lines."""
    payload = _sample_replay_payload("chat-response")
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


def test_public_bo3_metadata_and_empty_protocol_commands_are_preserved() -> None:
    """Verify public Showdown best-of-3 HTML headers parse parent room IDs and game numbers correctly."""
    payload = _sample_replay_payload("gen9championsvgc2026regmbbo3-100")
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


def test_null_parent_is_an_orphan_instead_of_a_literal_series_id() -> None:
    """Verify replays with parent=None fall back to player pairing rather than creating a string 'None' series."""
    payload = _sample_replay_payload("orphan")
    payload["parent"] = None

    document = parse_replay_payload(payload)

    assert document.metadata.parent_room == ""
    assert group_replays((document,)).series[0].record.grouping_method.name == (
        "FALLBACK_SAME_PLAYERS"
    )


def test_link_extraction_is_same_format_and_model_agnostic() -> None:
    """Verify linked_replay_ids only extracts next-game hyperlinks matching the active battle format."""
    payload = _sample_replay_payload("gen9championsvgc2026regmbbo3-100")
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


def test_packed_open_team_sheet_imputation_is_deterministic() -> None:
    """Verify open team sheet EV/stat imputation is deterministic across repeated runs."""
    payload = _sample_replay_payload("packed")
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


def test_new_schema_records_round_trip() -> None:
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


def test_grouping_parent_and_fallback_are_deterministic() -> None:
    """Verify series grouping by parent room or player fallback resolves identical series records regardless of order."""
    first = parse_replay_payload(_sample_replay_payload("g1", parent="series-1"))
    second = parse_replay_payload(_sample_replay_payload("g2", parent="series-1", winner="Bob"))
    parent_result = group_replays((second, first), format_id=first.metadata.format_id)
    assert len(parent_result.series) == 1
    assert parent_result.series[0].record.game_replay_ids == ("g1", "g2")
    assert parent_result.series[0].record.score == (1, 1)
    fallback_first = parse_replay_payload(_sample_replay_payload("fallback-game-1", parent=""))
    fallback_second = parse_replay_payload(_sample_replay_payload("fallback-game-2", parent=""))
    fallback = group_replays((fallback_second, fallback_first))
    assert len(fallback.series) == 1
    assert fallback.series[0].record.grouping_method.name == "FALLBACK_SAME_PLAYERS"


def test_grouping_preserves_authoritative_numbers_and_stable_series_id() -> None:
    """Verify series grouping maintains stable series hashes as additional games in the match are discovered."""
    first = parse_replay_payload(_sample_replay_payload("g1", game_number=1))
    second = parse_replay_payload(_sample_replay_payload("g2", game_number=2))

    incomplete = group_replays((first,)).series[0]
    complete = group_replays((second, first)).series[0]

    assert incomplete.record.series_id == complete.record.series_id
    assert [membership.game_number for membership in complete.memberships] == [1, 2]
    assert complete.record.is_complete


def test_grouping_quarantines_missing_and_duplicate_game_numbers() -> None:
    """Verify series grouping marks series with non-contiguous or duplicate game numbers as incomplete with diagnostics."""
    second = parse_replay_payload(_sample_replay_payload("g2", game_number=2))
    third = parse_replay_payload(_sample_replay_payload("g3", game_number=3))
    missing = group_replays((third, second)).series[0]

    duplicate_a = parse_replay_payload(_sample_replay_payload("dup-a", game_number=1))
    duplicate_b = parse_replay_payload(_sample_replay_payload("dup-b", game_number=1))
    duplicate = group_replays((duplicate_a, duplicate_b)).series[0]

    assert [membership.game_number for membership in missing.memberships] == [2, 3]
    assert not missing.record.is_complete
    assert "non_contiguous_game_numbers" in {diagnostic.code for diagnostic in missing.diagnostics}
    assert [membership.game_number for membership in duplicate.memberships] == [1, 1]
    assert not duplicate.record.is_complete
    assert "duplicate_game_number" in {diagnostic.code for diagnostic in duplicate.diagnostics}


def test_grouping_quarantines_games_after_a_series_clinch() -> None:
    """Verify matches with extra games played after a 2-0 clinch are quarantined with game_after_series_clinch diagnostic."""
    games = tuple(
        parse_replay_payload(_sample_replay_payload(f"g{number}", game_number=number))
        for number in (1, 2, 3)
    )

    group = group_replays(games).series[0]

    assert group.record.score == (2, 0)
    assert not group.record.is_complete
    assert "game_after_series_clinch" in {diagnostic.code for diagnostic in group.diagnostics}
    assert validated_bo3_series(games) == ()


def test_grouping_quarantines_missing_outcomes_and_team_conflicts() -> None:
    """Verify series grouping flags incomplete outcomes and team roster changes across games in a BO3."""
    unresolved_payload = _sample_replay_payload(
        "unresolved", game_number=1, parent="unresolved-series"
    )
    unresolved_payload["log"] = "\n".join(
        line for line in str(unresolved_payload["log"]).splitlines() if not line.startswith("|win|")
    )
    unresolved = group_replays((parse_replay_payload(unresolved_payload),)).series[0]

    first = parse_replay_payload(
        _sample_replay_payload("team-1", game_number=1, parent="team-series")
    )
    changed_payload = _sample_replay_payload("team-2", game_number=2, parent="team-series")
    changed_payload["log"] = str(changed_payload["log"]).replace("Pikachu", "Raichu")
    conflicted = group_replays((first, parse_replay_payload(changed_payload))).series[0]

    assert not unresolved.record.is_complete
    assert "missing_outcome" in {diagnostic.code for diagnostic in unresolved.diagnostics}
    assert not conflicted.record.is_complete
    assert "team_identity_conflict" in {diagnostic.code for diagnostic in conflicted.diagnostics}
    assert all(
        "team_identity_conflict" in membership.diagnostics for membership in conflicted.memberships
    )


def test_side_roles_are_canonical_and_bo1_bo3_views_share_games() -> None:
    """Verify game_player_roles maps perspectives canonical to the match winner/loser across side swaps."""
    first = parse_replay_payload(_sample_replay_payload("g1", game_number=1))
    second = parse_replay_payload(
        _sample_replay_payload(
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
    inferred_first = parse_replay_payload(_sample_replay_payload("inferred-1"))
    inferred_second = parse_replay_payload(_sample_replay_payload("inferred-2"))
    assert validated_bo3_series((inferred_first, inferred_second)) == ()
    same_side_second = parse_replay_payload(_sample_replay_payload("g2", game_number=2))
    assert tuple(
        game.metadata.replay_id for game in validated_bo3_series((same_side_second, first))[0].games
    ) == ("g1", "g2")


def test_reconstruction_is_causal_symmetric_and_compilable() -> None:
    """Verify battle reconstruction is causal (pre_line < post_line) and produces candidate actions containing executed orders."""
    first = _sample_replay_payload("g1")
    second = _sample_replay_payload("g2", winner="Bob")
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


def test_fetcher_retries_and_writes_immutable_raw_cache(tmp_path) -> None:
    """Verify ReplayFetcher retries on HTTP 503 errors and writes raw gzip JSON replay logs to the disk cache."""
    payload = _sample_replay_payload("g1")
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


def test_fetcher_accepts_display_formats_and_follows_sibling_links(tmp_path) -> None:
    """Verify ReplayFetcher discovers search replay IDs and automatically fetches linked sibling games in a BO3."""
    format_id = "gen9championsvgc2026regmbbo3"
    first_id = f"{format_id}-100"
    second_id = f"{format_id}-101"
    first = _sample_replay_payload(first_id, game_number=1)
    first["format"] = "[Gen 9 Champions] VGC 2026 Reg M-B (Bo3)"
    first["formatid"] = format_id
    first["log"] = (
        f"|uhtml|bestof|<strong>Game 1</strong> of "
        f'<a href="/game-bestof3-{format_id}-99">a best-of-3</a>\n'
        f'|uhtml|next|<a href="/battle-{second_id}">Game 2 of 3</a>\n'
        f"{first['log']}"
    )
    second = _sample_replay_payload(second_id, game_number=2)
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


def test_fetcher_preserves_malformed_replay_json_for_compilation_audit(tmp_path) -> None:
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


def test_replay_fixture_compiles_to_runtime_bound_schema_v5_shard(tmp_path: Path) -> None:
    """Verify compile_payloads serializes into PyTorch schema v5 shards with valid action masks and CSR offsets."""
    result = compile_payloads((_sample_replay_payload("shard-fixture"),))
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
        torch.isfinite(tensor).all() for tensor in tensors.values() if tensor.is_floating_point()
    )
    assert tensors["categorical"].shape[0] == manifest.decisions
    assert tensors["action_mask"].shape == (4, 2, 49)
    assert tensors["candidate_offsets"].tolist() == [0, 12, 16, 28, 32]
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


def test_shard_bytes_are_deterministic_for_fixed_inputs(tmp_path: Path) -> None:
    """Verify write_tensor_shards produces byte-identical files and manifests across independent invocations with identical inputs."""
    result = compile_payloads((_sample_replay_payload("shard-fixture"),))
    first = write_tensor_shards(result, tmp_path / "first", created_at="2026-01-01T00:00:00Z")
    second = write_tensor_shards(result, tmp_path / "second", created_at="2026-01-01T00:00:00Z")
    first_path = first.manifest_path.parent / first.manifest.shards[0].filename
    second_path = second.manifest_path.parent / second.manifest.shards[0].filename
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first.manifest.to_dict() == second.manifest.to_dict()


def test_shard_manifest_rejects_runtime_contract_mismatch(tmp_path: Path) -> None:
    """Verify load_shard_manifest raises ValueError if the global contract SHA-256 does not match active runtime."""
    result = compile_payloads((_sample_replay_payload("shard-fixture"),))
    built = write_tensor_shards(result, tmp_path, created_at="2026-01-01T00:00:00Z")
    value = json.loads(built.manifest_path.read_text(encoding="utf-8"))
    value["global_contract_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="global contract"):
        load_shard_manifest(value, DEFAULT_RUNTIME_MANIFEST)


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


def test_dataset_rejects_tampered_golden_shard(tmp_path) -> None:
    """Verify LazyReplayDataset raises ValueError on tampered golden replay shards when verify_hashes=True."""
    built = _build_dataset(tmp_path, 1)
    shard_path = built.manifest_path.parent / built.manifest.shards[0].filename
    shard_path.write_bytes(shard_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        next(iter(LazyReplayDataset(built.manifest_path, verify_hashes=True)))


def test_dataset_rejects_missing_golden_shard(tmp_path) -> None:
    """Verify LazyReplayDataset detects missing physical shard files on disk and raises ValueError."""
    built = _build_dataset(tmp_path, 1)
    shard_path = built.manifest_path.parent / built.manifest.shards[0].filename
    shard_path.unlink()
    with pytest.raises(ValueError, match="Shard file is missing"):
        next(iter(LazyReplayDataset(built.manifest_path)))


def test_dataset_rejects_missing_and_duplicate_source_records(tmp_path) -> None:
    """Verify ShardManifest validation checks consistency between raw_replays and source_series records."""
    missing = _build_dataset_from_payloads(
        tmp_path / "missing", (golden_replay_payload("only", series_id="series"),)
    )
    missing_manifest = missing.manifest.to_dict()
    missing_manifest["raw_replays"] = {}
    altered = missing.manifest_path.parent / "missing.json"
    altered.write_text(json.dumps(missing_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="raw_replays|source_games|partition"):
        LazyReplayDataset(altered)

    duplicate_payloads = (
        golden_replay_payload("duplicate", series_id="duplicate-series"),
        golden_replay_payload("duplicate", series_id="duplicate-series"),
    )
    with pytest.raises((ValueError, KeyError), match="duplicate|already|unique|invalid"):
        _build_dataset_from_payloads(tmp_path / "duplicate", duplicate_payloads)


def test_compiler_retains_exact_partial_unknown_and_rejected_labels() -> None:
    """Verify compiler tracks counters for exact, partial, unknown, and rejected labels based on evidence and OTS validity."""
    exact = golden_replay_payload("exact", series_id="label-series")
    partial = golden_replay_payload("partial", series_id="label-series-2", first_move_target=None)
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
    rejected_result = compile_payloads((rejected,), format_id=rejected["formatid"])
    assert not rejected_result.games
    assert rejected_result.metrics.counters["rejected_games"] == 1


def test_compiler_closes_process_pool_after_worker_failure(monkeypatch) -> None:
    """Verify compile_payloads cleans up and shuts down worker pools when a worker crashes."""
    events: list[str] = []

    class FailingPool:
        def __init__(self, **kwargs):
            del kwargs

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

    monkeypatch.setattr(
        reconstruction_pipeline.concurrent.futures, "ProcessPoolExecutor", FailingPool
    )
    monkeypatch.setattr(reconstruction_pipeline.os, "cpu_count", lambda: 1)
    payload = golden_replay_payload("worker-failure", series_id="worker-failure-series")
    second_payload = golden_replay_payload("worker-failure-2", series_id="worker-failure-series-2")
    with pytest.raises(RuntimeError, match="injected worker failure"):
        compile_payloads((payload, second_payload), format_id=payload["formatid"])
    assert events == ["enter", "map", "exit"]


def test_replay_fetcher_filters_discovery_and_rejects_unsafe_cache_ids(tmp_path: Path) -> None:
    """Verify ReplayFetcher filters duplicate discovery IDs and prevents directory traversal attacks in cache paths."""
    config = ScrapeConfig(
        format_id="gen9stress",
        cache_dir=tmp_path,
        page_size=3,
        max_pages=3,
        retries=1,
        rate_limit_per_second=0,
    )

    def transport(url: str, timeout: float) -> HttpResponse:
        del timeout
        if "page=1" in url:
            body = json.dumps(
                [
                    {"id": "gen9stress-2", "format": "gen9stress"},
                    {"id": "other-1", "format": "other"},
                    {"id": "gen9stress-2", "format": "gen9stress"},
                ]
            ).encode()
            return HttpResponse(200, body)
        return HttpResponse(200, b"[]")

    fetcher = ReplayFetcher(config, transport=transport)
    assert fetcher.discover_ids() == ("gen9stress-2",)
    with pytest.raises(ReplayFetchError, match="unsafe path"):
        fetcher._write_immutable("../escape", b"{}")


@pytest.mark.parametrize("status, error", ((404, ReplayUnavailableError), (429, ReplayFetchError)))
def test_replay_fetcher_handles_http_error_matrix(tmp_path: Path, status: int, error) -> None:
    """Verify ReplayFetcher distinguishes 404 Not Found from retryable errors (429/500/503)."""
    config = ScrapeConfig(format_id="gen9stress", cache_dir=tmp_path, retries=1, backoff_seconds=0)

    def transport(url: str, timeout: float) -> HttpResponse:
        del url, timeout
        return HttpResponse(status, b"missing")

    fetcher = ReplayFetcher(config, transport=transport)
    if status == 404:
        assert fetcher.acquire(("gen9stress-missing",)) == ()
    else:
        with pytest.raises(error):
            fetcher.acquire(("gen9stress-missing",))


def test_replay_fetcher_recovers_corrupt_raw_cache_and_rejects_bad_index(tmp_path: Path) -> None:
    """Verify ReplayFetcher re-fetches when disk cache is corrupted and read_fetch_index validates JSONL schema."""
    config = ScrapeConfig(format_id="gen9stress", cache_dir=tmp_path, retries=1)

    def transport(url: str, timeout: float) -> HttpResponse:
        del timeout
        replay_id = url.rsplit("/", 1)[-1].removesuffix(".json")
        return HttpResponse(200, json.dumps({"log": [f"|turn|{replay_id}"]}).encode())

    fetcher = ReplayFetcher(config, transport=transport)
    raw = fetcher._raw_path("gen9stress-1")
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"not gzip")
    assert fetcher.acquire(("gen9stress-1",))
    fetcher.index_path.write_bytes(b"not-json\n")
    with pytest.raises((ValueError, ReplayFetchError)):
        read_fetch_index(fetcher.index_path)


def test_reconstructed_views_carry_open_team_sheet_nature() -> None:
    """Verify open team sheet natures are preserved for both projected sides."""
    payload = _sample_replay_payload("ots-nature")
    natures = {"Pikachu": "Jolly", "Eevee": "Adamant", "Bulbasaur": "Bold", "Charmander": "Timid"}

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
    assert opponent and None not in opponent
    assert opponent <= {*natures.values(), "Serious"}
    assert own and None not in own
    assert own <= {*natures.values(), "Serious"}


def _payload_with_ots_natures(replay_id: str) -> dict[str, object]:
    """A pipeline payload whose open team sheets declare natures, as real replays do."""
    natures = {"Pikachu": "Jolly", "Eevee": "Adamant", "Bulbasaur": "Bold", "Charmander": "Timid"}
    payload = _sample_replay_payload(replay_id)
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
            fields, _ = _perspective_tensors(
                game, perspective, builder=builder, stat_estimates=game.stat_estimates
            )
            for step in fields["numerical"]:
                rows.extend(tuple(token) for token in step)
    return rows


def test_replay_stat_imputation_is_consistent_with_runtime_dex() -> None:
    """Verify explicit replay stat estimates are stable when the runtime dex is passed explicitly."""
    dex = json.loads((DEFAULT_PATHS.data_root / "champions_dex.json").read_text(encoding="utf-8"))
    without = compile_payloads((_payload_with_ots_natures("metrics-off"),))
    with_dex = compile_payloads((_payload_with_ots_natures("metrics-on"),), dex=dex)

    assert _numerical_rows(without) == _numerical_rows(with_dex)

    assert without.metrics.counters["imputations"] == with_dex.metrics.counters["imputations"] > 0
    assert (
        without.metrics.counters["imputation_unknown"]
        == with_dex.metrics.counters["imputation_unknown"]
    )
    assert (
        without.metrics.counters["imputation_confidence_sum"]
        == with_dex.metrics.counters["imputation_confidence_sum"]
    )
    assert with_dex.metrics.counters["imputation_confidence_sum"] > 0.0


def test_replay_stats_are_applied_to_both_projected_sides() -> None:
    """Verify replay stat estimates cover both sides and projected natures remain explicit."""
    dex = json.loads((DEFAULT_PATHS.data_root / "champions_dex.json").read_text(encoding="utf-8"))
    result = compile_payloads((_payload_with_ots_natures("own-side"),), dex=dex)
    assert result.games

    # Estimates cover every stable roster member, and both projected sides retain
    # the corresponding open team sheet nature.
    for game in result.games:
        assert {estimate.member_id.side.side_index for estimate in game.stat_estimates} == {0, 1}
        for perspective in game.perspectives:
            snapshot = perspective.snapshots[0]
            assert all(mon.nature for mon in snapshot.view.team.values())


def test_candidate_cap_degrades_to_explicit_unknown_evidence() -> None:
    """Verify that when candidate action space exceeds max_candidates, the label degrades to LabelKind.UNKNOWN with empty candidates."""
    view = DecisionView(
        slots=(
            SlotDecision(move_targets=((-2, -1),)),
            SlotDecision(move_targets=((-2,),)),
        )
    )
    evidence = extract_action_evidence(
        EvidenceRequest(
            view=view,
            slots=(
                ObservedAction(alternatives=(7, 8), exact=False),
                ObservedAction(action=7),
            ),
            max_candidates=1,
        )
    )
    assert evidence.label_kind is LabelKind.UNKNOWN
    assert evidence.candidates == ()
    assert "candidate_cap_or_illegal" in evidence.tags


def test_scrape_config_validation(tmp_path: Path) -> None:
    """Verify ScrapeConfig raises ValueError for empty format_id, non-positive page_size, negative backoff, or zero timeout."""
    with pytest.raises(ValueError, match="format_id must be non-empty"):
        ScrapeConfig(format_id="", cache_dir=tmp_path)
    with pytest.raises(ValueError, match="page_size must be positive"):
        ScrapeConfig(format_id="test", cache_dir=tmp_path, page_size=0)
    with pytest.raises(ValueError, match="backoff and rate limit must be nonnegative"):
        ScrapeConfig(format_id="test", cache_dir=tmp_path, backoff_seconds=-1.0)
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        ScrapeConfig(format_id="test", cache_dir=tmp_path, timeout_seconds=0.0)


def test_scrape_soft_limit_completes_the_final_linked_series(tmp_path: Path) -> None:
    """Verify scrape limit_games acts as a soft limit that finishes downloading the remaining sibling games of the current series."""
    format_id = FORMAT.bo3_format
    seeds = [f"{format_id}-{number}" for number in (100, 200, 300)]
    siblings = [f"{format_id}-{number}" for number in (101, 201, 301)]
    bodies: dict[str, bytes] = {}
    for seed, sibling in zip(seeds, siblings, strict=True):
        first = _sample_replay_payload(seed, parent=f"series-{seed}")
        first["log"] = f'|uhtml|next|<a href="/battle-{sibling}">Game 2</a>\n{first["log"]}'
        bodies[seed] = json.dumps(first).encode()
        bodies[sibling] = json.dumps(
            _sample_replay_payload(sibling, parent=f"series-{seed}")
        ).encode()

    def transport(url: str, timeout: float) -> HttpResponse:
        del timeout
        if "search.invalid" in url:
            return HttpResponse(
                200,
                json.dumps([{"id": seed, "formatid": format_id} for seed in seeds]).encode(),
            )
        replay_id = url.rsplit("/", 1)[-1].removesuffix(".json")
        return HttpResponse(200, bodies[replay_id])

    config = ScrapeConfig(
        format_id=format_id,
        cache_dir=tmp_path,
        search_url="https://search.invalid",
        replay_url_template="https://replay.invalid/{replay_id}.json",
        page_size=50,
        limit_games=3,
        rate_limit_per_second=0,
    )
    entries = ReplayFetcher(config, transport=transport).acquire()

    assert {entry.replay_id for entry in entries} == {
        seeds[0],
        siblings[0],
        seeds[1],
        siblings[1],
    }


def test_raw_cache_keeps_malformed_bytes_for_later_quality_rejection(
    tmp_path: Path,
) -> None:
    """Verify malformed replay payloads are written to disk cache to prevent repeatedly querying broken replay endpoints."""
    replay_id = f"{FORMAT.bo3_format}-malformed"

    def transport(url: str, timeout: float) -> HttpResponse:
        del url, timeout
        return HttpResponse(200, b"not-json")

    config = ScrapeConfig(
        format_id=FORMAT.bo3_format,
        cache_dir=tmp_path,
        replay_url_template="https://replay.invalid/{replay_id}.json",
        rate_limit_per_second=0,
    )
    ReplayFetcher(config, transport=transport).acquire((replay_id,))

    assert (
        load_raw_replay(tmp_path / FORMAT.bo3_format / "raw" / f"{replay_id}.json.gz")
        == b"not-json"
    )


def test_fetcher_skips_404_without_writing_cache_entry(tmp_path: Path, caplog) -> None:
    """Verify ReplayFetcher logs a warning and returns empty results without creating cache artifacts on 404 Not Found."""
    replay_id = f"{FORMAT.bo3_format}-missing"

    def transport(url: str, timeout: float) -> HttpResponse:
        del timeout
        if "search.invalid" in url:
            return HttpResponse(200, json.dumps([{"id": replay_id}]).encode())
        return HttpResponse(404, b"not found")

    config = ScrapeConfig(
        format_id=FORMAT.bo3_format,
        cache_dir=tmp_path,
        search_url="https://search.invalid",
        replay_url_template="https://replay.invalid/{replay_id}.json",
        rate_limit_per_second=0,
    )
    with caplog.at_level("WARNING", logger="p0.replays.scrape"):
        assert ReplayFetcher(config, transport=transport).acquire() == ()
    assert not (tmp_path / FORMAT.bo3_format / "raw" / f"{replay_id}.json.gz").exists()
    assert not (tmp_path / FORMAT.bo3_format / "metadata" / f"{replay_id}.json").exists()
    assert "skipping unavailable replay" in caplog.text


def test_cache_build_is_dataset_bound_and_preserves_bo3_series(
    tmp_path: Path,
) -> None:
    """Verify compile_to_shards produces identical deterministic dataset hashes and separates accepted vs rejected games."""
    good_id = f"{FORMAT.bo3_format}-good"
    bad_id = f"{FORMAT.bo3_format}-bad"
    good = _sample_replay_payload(good_id, parent="source-series")
    bad = _sample_replay_payload(bad_id, parent="source-series")
    bad["log"] = "\n".join(
        line for line in str(bad["log"]).splitlines() if "|showteam|" not in line
    )
    bodies = {good_id: json.dumps(good).encode(), bad_id: json.dumps(bad).encode()}

    def transport(url: str, timeout: float) -> HttpResponse:
        del timeout
        replay_id = url.rsplit("/", 1)[-1].removesuffix(".json")
        return HttpResponse(200, bodies[replay_id])

    cache = tmp_path / "replays"
    config = ScrapeConfig(
        format_id=FORMAT.bo3_format,
        cache_dir=cache,
        replay_url_template="https://replay.invalid/{replay_id}.json",
        rate_limit_per_second=0,
    )
    ReplayFetcher(config, transport=transport).acquire((good_id, bad_id))

    docs = [
        parse_replay_payload(load_raw_replay(p))
        for p in (cache / FORMAT.bo3_format / "raw").glob("*.json.gz")
    ]
    first = compile_to_shards(docs, tmp_path / "shards", format_id=FORMAT.bo3_format)
    second = compile_to_shards(docs, tmp_path / "shards", format_id=FORMAT.bo3_format)

    assert first.manifest_path == second.manifest_path
    assert first.manifest.dataset_hash == second.manifest.dataset_hash
    assert first.manifest.source_games == 2
    assert first.manifest.accepted_games == 1
    assert first.manifest.rejected_games == 1
    chunks = list(LazyReplayDataset(first.manifest_path))
    assert len(chunks) == 2


def test_split_assignment_populates_all_requested_splits_when_possible() -> None:
    """Verify assign_series_splits allocates items to train, validation, and test partitions."""
    manifest = assign_series_splits(
        ("a", "b", "c", "d", "e"),
        global_contract_sha256="a" * 64,
        dataset_hash="b" * 64,
    )

    assert set(manifest.assignments.values()) == {"train", "validation", "test"}


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


def test_evidence_shapes() -> None:
    """Verify ActionEvidence enforces candidate counts and ranges according to EXACT/PARTIAL/UNKNOWN taxonomy."""
    assert _evidence(LabelKind.EXACT).exact_action == (7, 1)
    with pytest.raises(ValueError, match="only defined for EXACT"):
        _evidence(LabelKind.PARTIAL).exact_action
    with pytest.raises(ValueError, match="exactly one candidate"):
        ActionEvidence(LabelKind.EXACT, (), 1.0, MaskProvenance.CONSERVATIVE_RECONSTRUCTED)
    with pytest.raises(ValueError, match="two or more"):
        ActionEvidence(
            LabelKind.PARTIAL,
            ((7, 1),),
            0.5,
            MaskProvenance.CONSERVATIVE_RECONSTRUCTED,
        )
    with pytest.raises(ValueError, match="no candidates"):
        ActionEvidence(
            LabelKind.UNKNOWN,
            ((7, 1),),
            0.0,
            MaskProvenance.CONSERVATIVE_RECONSTRUCTED,
        )
    with pytest.raises(ValueError, match="outside"):
        ActionEvidence(
            LabelKind.EXACT,
            ((49, 0),),
            1.0,
            MaskProvenance.CONSERVATIVE_RECONSTRUCTED,
        )
    with pytest.raises(ValueError, match="Duplicate"):
        ActionEvidence(
            LabelKind.PARTIAL,
            ((7, 1), (7, 1)),
            0.5,
            MaskProvenance.CONSERVATIVE_RECONSTRUCTED,
        )


def test_ir_round_trips() -> None:
    """Verify SeriesRecord and FetchIndexEntry serialize and deserialize cleanly."""
    series = _series_record()
    assert SeriesRecord.from_dict(series.to_dict()) == series
    fetch = FetchIndexEntry(
        replay_id="r1",
        format_id="gen9championsvgc2026regmbbo3",
        source_url="https://replay.pokemonshowdown.com/r1",
        fetched_at="2026-07-17T00:00:00Z",
        http_status=200,
        content_sha256="d" * 64,
        byte_size=100,
    )
    assert FetchIndexEntry.from_dict(fetch.to_dict()) == fetch


def test_ir_rejects_bad_serializations() -> None:
    """Verify IR deserialization raises ValueError on schema version mismatch or missing/unknown fields."""
    payload = _series_record().to_dict()
    del payload["score"]
    payload["bogus"] = 1
    with pytest.raises(ValueError, match=r"missing=\['score'\], unknown=\['bogus'\]"):
        SeriesRecord.from_dict(payload)


def test_ir_validates_construction() -> None:
    """Verify SeriesRecord validates structural invariants on deserialization."""
    with pytest.raises(ValueError, match="two wins"):
        SeriesRecord.from_dict({**_series_record().to_dict(), "score": [1, 0]})


def test_observation_specs_are_derived() -> None:
    """Verify observation_field_specs and SHARD_TENSOR_SPECS match StructuredObservation field definitions."""
    specs = observation_field_specs()
    assert [spec[0] for spec in specs] == [spec[0] for spec in StructuredObservation._FIELD_SPECS]
    for (name, shape, dtype), (_, base_shape, base_dtype) in zip(
        specs, StructuredObservation._FIELD_SPECS, strict=True
    ):
        assert shape == (-1, *base_shape) and dtype is base_dtype, name
    assert [spec[0] for spec in SHARD_TENSOR_SPECS] == [
        "action_mask",
        "mask_provenance",
        "label_kind",
        "label_confidence",
        "loss_mask",
        "decision_type",
        "exact_action",
        "candidate_values",
        "candidate_offsets",
        "game_offsets",
        "series_offsets",
        "outcome",
    ]


def test_shard_manifest_contract() -> None:
    """Verify ShardManifest contract checks global SHA-256 validity and roundtrips through dictionary representation."""
    manifest = _shard_manifest_fixture_unit()
    assert ShardManifest.from_dict(manifest.to_dict()) == manifest
    assert manifest.decisions == 10 and manifest.games == 2 and manifest.series == 1
    assert load_shard_manifest(manifest.to_dict()) == manifest
    with pytest.raises(ValueError, match="incompatible"):
        load_shard_manifest({**manifest.to_dict(), "global_contract_sha256": "0" * 64})
    with pytest.raises(ValueError, match="unknown"):
        load_shard_manifest({**manifest.to_dict(), "runtime_manifest_sha256": "0" * 64})


def test_shard_manifest_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    """Verify ShardManifest verifies global runtime contract and detects tampered or mismatched source series."""
    vocab = tmp_path / "vocab.json"
    dex = tmp_path / "champions_dex.json"
    vocab.write_text(
        json.dumps({"species": {"pikachu": 1}, "moves": {"tackle": 1}}), encoding="utf-8"
    )
    dex.write_text('{"pikachu":{"base_stats":{"hp":35}}}', encoding="utf-8")
    runtime = current_manifest(vocab_path=vocab, dex_path=dex)

    entry = ShardIndexEntry("shard-000.pt", "c" * 64, 10, 2, 1, 100)
    manifest = ShardManifest(
        global_contract_sha256=runtime.global_sha256,
        shards=(entry,),
        diagnostics={"oov_ids": 0},
        created_at="2026-07-17T00:00:00Z",
        dataset_hash="d" * 64,
        source_format_id="gen9championsvgc2026regmbbo3",
        build_config={"seed": 3},
        raw_replays={"game-1": "f" * 64, "game-2": "e" * 64},
        source_series={"series-1": ("game-1", "game-2")},
        source_games=2,
        accepted_games=2,
        rejected_games=0,
        artifact_hashes={"shard-000.pt": "c" * 64},
    )
    assert ShardManifest.from_dict(manifest.to_dict()) == manifest
    manifest_path = tmp_path / "runtime_manifest.json"
    manifest_path.write_text(json.dumps(runtime.to_dict()), encoding="utf-8")
    with pytest.raises(ValueError, match="default global manifest"):
        load_shard_manifest(
            {**manifest.to_dict(), "global_contract_sha256": "b" * 64}, manifest_path
        )
    with pytest.raises(ValueError, match="source_series"):
        ShardManifest.from_dict({**manifest.to_dict(), "source_series": {"series-1": ("game-1",)}})
