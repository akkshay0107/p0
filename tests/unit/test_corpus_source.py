"""Unit tests for corpus team pools, corpus source sampling, and CLI operations."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from p0.cli.corpus import main as corpus_main
from p0.evaluation.harness import DEFAULT_TEST_TEAM
from p0.format_config import FORMAT, current_manifest
from p0.teams.corpus import (
    CORPUS_MANIFEST_SCHEMA,
    CorpusEntry,
    CorpusSourceSpec,
    CorpusSplit,
    TeamCorpusManifest,
    corpus_content_hash,
)
from p0.teams.corpus_source import CorpusTeamPool, CorpusTeamSource
from p0.teams.factory import build_team_source
from p0.teams.source import FileTeamSource


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


class TestCorpusTeamPool:
    def test_pool_split_filtering(self, tmp_path: Path) -> None:
        entries = (
            _make_entry(1, split=CorpusSplit.TRAIN),
            _make_entry(2, split=CorpusSplit.VALIDATION),
            _make_entry(3, split=CorpusSplit.TEST),
        )
        path, manifest = _write_manifest(tmp_path, entries)
        spec = CorpusSourceSpec(
            corpus_path=str(path),
            corpus_hash=manifest.corpus_hash,
            format_id=manifest.format_id,
            split=CorpusSplit.TRAIN,
        )
        pool = CorpusTeamPool.from_spec(spec)
        assert len(pool.entries(CorpusSplit.TRAIN)) == 1
        assert len(pool.entries(CorpusSplit.VALIDATION)) == 1
        assert len(pool.entries(CorpusSplit.TEST)) == 1

    def test_pool_spec_mismatch_raises(self, tmp_path: Path) -> None:
        path, manifest = _write_manifest(tmp_path, (_make_entry(1),))
        spec = CorpusSourceSpec(
            corpus_path=str(path),
            corpus_hash="0" * 64,
            format_id=manifest.format_id,
            split=CorpusSplit.TRAIN,
        )
        with pytest.raises(ValueError, match="Corpus hash does not match"):
            CorpusTeamPool.from_spec(spec)


class TestCorpusTeamSource:
    def test_sampling_uniform_canonical(self, tmp_path: Path) -> None:
        entries = (
            _make_entry(1, canonical_index=1),
            _make_entry(2, canonical_index=1),  # variant of canonical 1
            _make_entry(3, canonical_index=2),  # canonical 2
        )
        path, manifest = _write_manifest(tmp_path, entries)
        spec = CorpusSourceSpec(
            corpus_path=str(path),
            corpus_hash=manifest.corpus_hash,
            format_id=manifest.format_id,
            split=CorpusSplit.TRAIN,
        )
        source = CorpusTeamSource(spec)
        rng = random.Random(42)
        sample = source.sample(rng)
        assert sample.packed in {e.packed for e in entries}
        assert sample.team_hash in {e.packed_sha256 for e in entries}

    def test_describe(self, tmp_path: Path) -> None:
        path, manifest = _write_manifest(tmp_path, (_make_entry(1),))
        spec = CorpusSourceSpec(
            corpus_path=str(path),
            corpus_hash=manifest.corpus_hash,
            format_id=manifest.format_id,
            split=CorpusSplit.TRAIN,
        )
        source = CorpusTeamSource(spec)
        desc = source.describe()
        assert desc["kind"] == "corpus"
        assert desc["split"] == "TRAIN"
        assert desc["pool_size"] == 1


class TestBuildTeamSource:
    def test_resolves_corpus_manifest(self, tmp_path: Path) -> None:
        path, manifest = _write_manifest(tmp_path, (_make_entry(1),))
        manifest_path = tmp_path / "corpus_manifest.json"
        manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

        source = build_team_source(manifest_path)
        assert isinstance(source, CorpusTeamSource)
        assert source.describe()["kind"] == "corpus"

    def test_accepts_regular_manifest_for_bo3(self, tmp_path: Path) -> None:
        path, manifest = _write_manifest(tmp_path, (_make_entry(1),))
        source = build_team_source(path, expected_format_id=FORMAT.bo3_format)
        assert isinstance(source, CorpusTeamSource)
        assert source.describe()["format_id"] == manifest.format_id

    def test_rejects_incompatible_manifest_format(self, tmp_path: Path) -> None:
        path, manifest = _write_manifest(tmp_path, (_make_entry(1),))
        incompatible = replace(manifest, format_id="unsupported-format")
        path.write_text(json.dumps(incompatible.to_dict()), encoding="utf-8")

        with pytest.raises(ValueError, match="Corpus format mismatch"):
            build_team_source(path, expected_format_id=FORMAT.bo3_format)

    def test_falls_back_to_file_source(self, tmp_path: Path) -> None:
        pool_dir = tmp_path / "pool"
        pool_dir.mkdir()
        team_text = "\n\n".join(
            f"Pikachu{i} @ Light Ball\nAbility: Static\nJolly Nature\n- Fake Out\n- Protect\n- Thunderbolt\n- Electroweb"
            for i in range(1, 7)
        )
        (pool_dir / "team.txt").write_text(team_text, encoding="utf-8")
        source = build_team_source(pool_dir)
        assert isinstance(source, FileTeamSource)

    def test_rejects_invalid_manifest(self, tmp_path: Path) -> None:
        pool_dir = tmp_path / "pool"
        pool_dir.mkdir()
        (pool_dir / "corpus_manifest.json").write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid corpus manifest"):
            build_team_source(pool_dir)


class TestCorpusCLI:
    def test_cli_build_and_audit(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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
