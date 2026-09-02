"""Tests for team corpus manifests and source construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from p0.cli.corpus import main as corpus_main
from p0.evaluation.harness import DEFAULT_TEST_TEAM
from p0.format_config import FORMAT, current_manifest
from p0.model.tokenizer import PokemonTokenizer
from p0.teams.corpus import (
    CORPUS_MANIFEST_SCHEMA,
    CorpusEntry,
    CorpusSourceSpec,
    CorpusSplit,
    TeamCorpusManifest,
    corpus_content_hash,
    load_corpus_manifest,
)
from p0.teams.corpus_build import (
    build_corpus,
)
from p0.teams.corpus_source import CorpusTeamSource
from p0.teams.factory import build_team_source
from p0.teams.source import FileTeamSource
from p0.teams.spread_usage import (
    SPREAD_USAGE_SCHEMA,
)
from p0.teams.validation import validate_many
from tests.team_fixtures import team_variant


def vocabulary() -> dict[str, dict[str, int]]:
    return {
        "species": {
            "pikachu": 1,
            "charizard": 2,
            "whimsicott": 3,
            "garchomp": 4,
            "kingambit": 5,
            "glimmora": 6,
            "raichu": 7,
        },
        "items": {
            "lightball": 1,
            "charizarditey": 2,
            "focussash": 3,
            "sitrusberry": 4,
            "blackglasses": 5,
            "shucaberry": 6,
            "lifeorb": 7,
        },
        "abilities": {
            "static": 1,
            "blaze": 2,
            "prankster": 3,
            "roughskin": 4,
            "defiant": 5,
            "toxicdebris": 6,
        },
        "moves": {
            "fakeout": 1,
            "protect": 2,
            "thunderbolt": 3,
            "electroweb": 4,
            "heatwave": 5,
            "solarbeam": 6,
            "weatherball": 7,
            "moonblast": 8,
            "tailwind": 9,
            "encore": 10,
            "earthquake": 11,
            "dragonclaw": 12,
            "rockslide": 13,
            "kowtowcleave": 14,
            "suckerpunch": 15,
            "lowkick": 16,
            "powergem": 17,
            "sludgebomb": 18,
            "earthpower": 19,
        },
    }


def _make_entry(
    index: int,
    canonical_index: int | None = None,
    split: CorpusSplit = CorpusSplit.TRAIN,
    usage_count: int = 10,
) -> CorpusEntry:
    if canonical_index is None:
        canonical_index = index
    canonical = f"canonical_{canonical_index:04d}"
    canonical_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    packed = f"Nickname|Species{index}|item|ability|move1,move2|nature"
    packed_sha256 = hashlib.sha256(packed.encode("utf-8")).hexdigest()
    return CorpusEntry(
        canonical_hash=canonical_hash,
        packed=packed,
        packed_sha256=packed_sha256,
        split=split,
        usage_count=usage_count,
        spread_provenance="imputed",
    )


def _write_manifest(
    tmp_path: Path, entries: tuple[CorpusEntry, ...]
) -> tuple[Path, TeamCorpusManifest]:
    manifest = TeamCorpusManifest(
        artifact_schema=CORPUS_MANIFEST_SCHEMA,
        global_contract_sha256=current_manifest().global_sha256,
        format_id=FORMAT.battle_format,
        corpus_hash=corpus_content_hash(entries),
        entries=entries,
        created_at="2026-07-19T12:00:00Z",
        sampling_metadata={"pool_size": len(entries)},
    )
    path = tmp_path / "corpus_manifest.json"
    path.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return path, manifest


def _chaos(species: str, spreads: dict[str, float]) -> dict[str, Any]:
    """Build a minimal chaos export carrying one species' spread distribution."""
    return {"data": {species: {"Spreads": spreads}}}


def _dex(*species: dict[str, Any]) -> dict[str, Any]:
    """Build a minimal dex carrying only what forme aliasing reads."""
    return {"species": list(species), "moves": []}


_BASE_STATS = {"hp": 78, "atk": 65, "def": 68, "spa": 112, "spd": 154, "spe": 75}


def _payload(spreads: dict[str, Any], aliases: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "schema": SPREAD_USAGE_SCHEMA,
        "format_id": FORMAT.battle_format,
        "weight_scale": 1_000_000,
        "aliases": aliases or {},
        "spreads": spreads,
    }


_ONE_BUCKET = {"real": {"timid": [[2, 0, 0, 32, 0, 32, 1000]]}}


def _corpus_entry(packed: str = "packed-team") -> CorpusEntry:
    return CorpusEntry(
        canonical_hash=hashlib.sha256(packed.encode()).hexdigest(),
        packed=packed,
        packed_sha256=hashlib.sha256(packed.encode()).hexdigest(),
        split=CorpusSplit.TRAIN,
        usage_count=3,
    )


def _corpus_manifest(entries: tuple[CorpusEntry, ...]) -> TeamCorpusManifest:
    active_contract = current_manifest().global_sha256
    return TeamCorpusManifest(
        global_contract_sha256=active_contract,
        format_id="gen9championsvgc2026regmb",
        corpus_hash=corpus_content_hash(entries),
        entries=entries,
        created_at="2026-07-17T00:00:00Z",
        sampling_metadata={"sampling": "uniform_canonical"},
    )


class TestCorpusManifests:
    def test_corpus_manifest_contract(self) -> None:
        """Verify TeamCorpusManifest serializes losslessly and enforces canonical hash contracts."""
        entries = (_corpus_entry("team-a"), _corpus_entry("team-b"))
        assert entries[0].spread_provenance == "imputed"
        manifest = _corpus_manifest(entries)
        assert TeamCorpusManifest.from_dict(manifest.to_dict()) == manifest
        assert load_corpus_manifest(manifest.to_dict()) == manifest
        assert corpus_content_hash(entries) == corpus_content_hash(entries[::-1])
        with pytest.raises(ValueError, match="does not match the packed team"):
            CorpusEntry(
                canonical_hash="a" * 64,
                packed="team",
                packed_sha256="b" * 64,
                split=CorpusSplit.TRAIN,
                usage_count=1,
            )
        with pytest.raises(ValueError, match="does not match the entries"):
            TeamCorpusManifest.from_dict({**manifest.to_dict(), "corpus_hash": "0" * 64})
        with pytest.raises(ValueError, match="Duplicate corpus entry"):
            _corpus_manifest((entries[0], entries[0]))
        with pytest.raises(ValueError, match="unknown"):
            CorpusEntry.from_dict({**entries[0].to_dict(), "archetype_tags": []})

    def test_corpus_source_spec_validates(self) -> None:
        """Verify CorpusSourceSpec validates split parameter."""
        spec = CorpusSourceSpec(
            corpus_path="teams/corpus_manifest.json",
            corpus_hash="a" * 64,
            format_id="gen9championsvgc2026regmb",
            split=CorpusSplit.TRAIN,
        )
        assert spec.split is CorpusSplit.TRAIN
        with pytest.raises(ValueError, match="split"):
            CorpusSourceSpec(
                corpus_path="x",
                corpus_hash="a" * 64,
                format_id="f",
                split=CorpusSplit.UNSPECIFIED,
            )

    def test_build_team_source_resolves_corpus_manifest(self, tmp_path: Path) -> None:
        """Verify the team-source factory instantiates CorpusTeamSource for a manifest path."""
        tokenizer = PokemonTokenizer(vocabulary())
        contract_hash = current_manifest().global_sha256
        v1 = team_variant("Pikachu")
        manifest, _ = build_corpus(
            (v1,),
            tokenizer=tokenizer,
            validator=validate_many,
            global_contract_sha256=contract_hash,
            format_id=FORMAT.battle_format,
        )
        manifest_path = tmp_path / "corpus_manifest.json"
        manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

        source = build_team_source(manifest_path)
        assert isinstance(source, CorpusTeamSource)
        desc = source.describe()
        assert desc["kind"] == "corpus"
        assert desc["corpus_hash"] == manifest.corpus_hash
        assert desc["split"] == "TRAIN"
        assert desc["sampling"] == "uniform_canonical"

    def test_build_team_source_accepts_regular_manifest_for_bo3(self, tmp_path: Path) -> None:
        """Verify a regular-format corpus can feed a Bo3 model."""
        path, manifest = _write_manifest(tmp_path, (_make_entry(1),))

        source = build_team_source(path, expected_format_id=FORMAT.bo3_format)
        assert isinstance(source, CorpusTeamSource)
        assert source.describe()["format_id"] == manifest.format_id

    def test_build_team_source_rejects_incompatible_manifest_format(self, tmp_path: Path) -> None:
        """Verify unsupported corpus and model format pairs fail loudly."""
        path, manifest = _write_manifest(tmp_path, (_make_entry(1),))
        incompatible = replace(manifest, format_id="unsupported-format")
        path.write_text(json.dumps(incompatible.to_dict()), encoding="utf-8")

        with pytest.raises(ValueError, match="Corpus format mismatch"):
            build_team_source(path, expected_format_id=FORMAT.bo3_format)

    def test_build_team_source_falls_back_to_file_source(self, tmp_path: Path) -> None:
        """Verify the team-source factory falls back for a raw team directory."""
        pool_dir = tmp_path / "pool"
        pool_dir.mkdir()
        team_text = "\n\n".join(
            f"Pikachu{i} @ Light Ball\nAbility: Static\nJolly Nature\n- Fake Out\n- Protect\n- Thunderbolt\n- Electroweb"
            for i in range(1, 7)
        )
        (pool_dir / "team.txt").write_text(team_text, encoding="utf-8")
        source = build_team_source(pool_dir)
        assert isinstance(source, FileTeamSource)

    def test_build_team_source_rejects_invalid_manifest(self, tmp_path: Path) -> None:
        """Verify a present manifest is authoritative and malformed content raises."""
        pool_dir = tmp_path / "pool"
        pool_dir.mkdir()
        (pool_dir / "corpus_manifest.json").write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid corpus manifest"):
            build_team_source(pool_dir)

    def test_corpus_cli_build_and_audit(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify corpus CLI subcommands ('build' and 'audit') execute and output valid JSON audit results."""
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        team_text_1 = DEFAULT_TEST_TEAM
        team_text_2 = DEFAULT_TEST_TEAM.replace("Pikachu @ Light Ball", "Raichu @ Light Ball", 1)
        (input_dir / "v1.txt").write_text(team_text_1, encoding="utf-8")
        (input_dir / "v2.txt").write_text(team_text_2, encoding="utf-8")

        all_dir = tmp_path / "pools" / "all"

        corpus_main(
            [
                "build",
                "--input",
                str(input_dir),
                "--output-dir",
                str(all_dir),
                "--format-id",
                FORMAT.battle_format,
            ]
        )

        assert (all_dir / "corpus_manifest.json").is_file()

        captured = capsys.readouterr()
        audit_data = json.loads(captured.out.split("\n")[-2]) if captured.out.strip() else {}
        assert audit_data["admitted_count"] == 2
        assert audit_data["rejected_count"] == 0

        corpus_main(["audit", "--path", str(all_dir)])
        audit_captured = capsys.readouterr()
        re_audit_data = (
            json.loads(audit_captured.out.split("\n")[-2]) if audit_captured.out.strip() else {}
        )
        assert re_audit_data["admitted_count"] == 2

    def test_build_team_source_resolves_directory_manifest(self, tmp_path: Path) -> None:
        """Verify the team-source factory resolves a manifest inside a pool directory."""
        tokenizer = PokemonTokenizer(vocabulary())
        contract_hash = current_manifest().global_sha256
        v1 = team_variant("Pikachu")
        manifest, _ = build_corpus(
            (v1,),
            tokenizer=tokenizer,
            validator=validate_many,
            global_contract_sha256=contract_hash,
            format_id=FORMAT.battle_format,
        )
        pool_dir = tmp_path / "pool_all"
        pool_dir.mkdir(parents=True, exist_ok=True)
        (pool_dir / "corpus_manifest.json").write_text(
            json.dumps(manifest.to_dict()), encoding="utf-8"
        )
        (pool_dir / "invalid-team.txt").write_text("not a team", encoding="utf-8")

        source = build_team_source(pool_dir)
        assert isinstance(source, CorpusTeamSource)

    def test_corpus_manifest_hash_is_order_independent_but_packed_content_bound(self) -> None:
        """Verify corpus content hash calculation is invariant to entry permutation but sensitive to packed team changes."""
        entries = tuple(
            CorpusEntry(
                canonical_hash=hashlib.sha256(f"canonical-{letter}".encode()).hexdigest(),
                packed=f"team-{letter}",
                packed_sha256=hashlib.sha256(f"team-{letter}".encode()).hexdigest(),
                split=CorpusSplit.TRAIN,
                usage_count=index + 1,
            )
            for index, letter in enumerate(("a", "b", "c"))
        )
        manifest = TeamCorpusManifest(
            global_contract_sha256="d" * 64,
            format_id="gen9championsvgc2026regmb",
            corpus_hash=corpus_content_hash(entries),
            entries=entries,
            created_at="2026-07-17T00:00:00Z",
            sampling_metadata={"seed": 3},
        )
        assert TeamCorpusManifest.from_dict(manifest.to_dict()) == manifest
        assert corpus_content_hash(entries) == corpus_content_hash(entries[::-1])
        with pytest.raises(ValueError, match="does not match"):
            TeamCorpusManifest.from_dict({**manifest.to_dict(), "corpus_hash": "e" * 64})
