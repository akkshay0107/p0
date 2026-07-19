"""Tests for corpus CLI commands and training/runtime composition roots."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from p0.cli.corpus import main as corpus_main
from p0.format_config import FORMAT, current_manifest
from p0.model.tokenizer import PokemonTokenizer
from p0.teams.corpus_build import CorpusBuilder
from p0.teams.corpus_source import CorpusTeamSource
from p0.teams.source import FileTeamSource
from p0.teams.stat_points import StatPoints
from p0.teams.team import CanonicalTeam, TeamMember, TeamMetadata, TeamVariant
from p0.teams.validation import AdmissionResult
from p0.training.composition import _team_source
from p0.training.config import CorpusConfig, TeamSourceConfig


def _mock_vocab() -> dict[str, dict[str, int]]:
    return {
        "species": {
            "pikachu": 1,
            "charizard": 2,
        },
        "items": {
            "lightball": 1,
            "charizarditey": 2,
        },
        "abilities": {
            "static": 1,
            "blaze": 2,
        },
        "moves": {
            "fakeout": 1,
            "protect": 2,
            "thunderbolt": 3,
            "electroweb": 4,
            "heatwave": 5,
            "solarbeam": 6,
            "weatherball": 7,
        },
    }


def _mock_variant(species: str = "Pikachu") -> TeamVariant:
    if species == "Pikachu":
        members = tuple(
            TeamMember(
                species="Pikachu",
                item="Light Ball",
                ability="Static",
                moves=("Fake Out", "Protect", "Thunderbolt", "Electroweb"),
                nature="Jolly",
            )
            for _ in range(6)
        )
    else:
        members = tuple(
            TeamMember(
                species="Charizard",
                item="Charizardite Y",
                ability="Blaze",
                moves=("Heat Wave", "Solar Beam", "Protect", "Weather Ball"),
                nature="Modest",
            )
            for _ in range(6)
        )
    return TeamVariant(
        team=CanonicalTeam(members),
        spreads=tuple(StatPoints(hp=2, spa=32, spe=32) for _ in members),
        metadata=TeamMetadata(
            source_series=("test-series",),
            source_replays=("game-1",),
            first_seen="2026-01-01T00:00:00Z",
            last_seen="2026-01-02T00:00:00Z",
            usage_count=5 if species == "Pikachu" else 3,
            archetype_tags=("offense",),
        ),
    )


def _mock_validator(
    variants: Sequence[TeamVariant], **kwargs: object
) -> tuple[AdmissionResult, ...]:
    return tuple(
        AdmissionResult(
            team_hash=variant.team.team_hash,
            valid=True,
            packed_team="]".join(
                f"{m.species}|{m.species}|{m.item}|{m.ability}|{','.join(m.moves)}|{m.nature}"
                for m in variant.team.members
            ),
            problems=(),
        )
        for variant in variants
    )


def test_team_source_composition_resolves_corpus(tmp_path: Path) -> None:
    tokenizer = PokemonTokenizer(_mock_vocab())
    contract_hash = current_manifest().runtime_contract_sha256
    builder = CorpusBuilder(
        tokenizer=tokenizer,
        validator=_mock_validator,
        runtime_contract_sha256=contract_hash,
        format_id=FORMAT.battle_format,
    )
    v1 = _mock_variant("Pikachu")
    manifest, _ = builder.build((v1,))
    manifest_path = tmp_path / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

    source = _team_source(
        TeamSourceConfig(path=manifest_path),
        corpus_config=CorpusConfig(
            agent_split="train",
            sampling_policy="uniform_canonical",
            allow_mirror=False,
        ),
        seed=42,
        is_agent=True,
    )
    assert isinstance(source, CorpusTeamSource)
    desc = source.describe()
    assert desc["kind"] == "corpus"
    assert desc["corpus_hash"] == manifest.corpus_hash
    assert desc["split"] == "TRAIN"
    assert desc["sampling_policy"] == "UNIFORM_CANONICAL"
    assert desc["allow_mirror"] is False


def test_team_source_composition_falls_back_to_file_source(tmp_path: Path) -> None:
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    team_text = "\n\n".join(
        f"Pikachu{i} @ Light Ball\nAbility: Static\nJolly Nature\n- Fake Out\n- Protect\n- Thunderbolt\n- Electroweb"
        for i in range(1, 7)
    )
    (pool_dir / "team.txt").write_text(team_text, encoding="utf-8")
    source = _team_source(TeamSourceConfig(path=pool_dir))
    assert isinstance(source, FileTeamSource)


def test_corpus_cli_build_and_audit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Patch current_manifest and PokemonTokenizer to use our mock deterministic setup
    monkeypatch.setattr("p0.cli.corpus.current_manifest", lambda: current_manifest())
    monkeypatch.setattr(
        "p0.cli.corpus.PokemonTokenizer.from_file", lambda: PokemonTokenizer(_mock_vocab())
    )
    monkeypatch.setattr("p0.cli.corpus.validate_many", _mock_validator)

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    v1 = _mock_variant("Pikachu")
    v2 = _mock_variant("Charizard")
    (input_dir / "v1.json").write_text(json.dumps(v1.to_dict()), encoding="utf-8")
    (input_dir / "v2.json").write_text(json.dumps(v2.to_dict()), encoding="utf-8")

    output_manifest = tmp_path / "output" / "corpus_manifest.json"
    pool_dir = tmp_path / "pools"

    corpus_main(
        [
            "build",
            "--input",
            str(input_dir),
            "--output",
            str(output_manifest),
            "--pool-dir",
            str(pool_dir),
            "--format-id",
            FORMAT.battle_format,
        ]
    )

    assert output_manifest.is_file()
    assert (pool_dir / "all" / "corpus_manifest.json").is_file()

    captured = capsys.readouterr()
    audit_data = json.loads(captured.out)
    assert audit_data["admitted_count"] == 2
    assert audit_data["rejected_count"] == 0

    corpus_main(["audit", "--manifest", str(output_manifest)])
    audit_captured = capsys.readouterr()
    re_audit_data = json.loads(audit_captured.out)
    assert re_audit_data["admitted_count"] == 2


def test_team_source_composition_resolves_directory_manifest(tmp_path: Path) -> None:
    tokenizer = PokemonTokenizer(_mock_vocab())
    contract_hash = current_manifest().runtime_contract_sha256
    builder = CorpusBuilder(
        tokenizer=tokenizer,
        validator=_mock_validator,
        runtime_contract_sha256=contract_hash,
        format_id=FORMAT.battle_format,
    )
    v1 = _mock_variant("Pikachu")
    manifest, _ = builder.build((v1,))
    pool_dir = tmp_path / "pool_all"
    pool_dir.mkdir(parents=True, exist_ok=True)
    (pool_dir / "corpus_manifest.json").write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

    source = _team_source(
        TeamSourceConfig(path=pool_dir),
        corpus_config=CorpusConfig(agent_split="train"),
        seed=10,
        is_agent=True,
    )
    assert isinstance(source, CorpusTeamSource)
