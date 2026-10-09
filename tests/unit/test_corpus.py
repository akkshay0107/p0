"""Tests for the team corpus: manifest contents, sampling, and loading."""

import random
from collections import Counter
from pathlib import Path

import orjson
import pytest

from p0.format_config import FORMAT
from p0.teams.corpus import (
    TeamCorpus,
    corpus_from_team_files,
    load_team_corpus,
    pack_showdown_team,
)
from tests.team_fixtures import DEFAULT_TEST_TEAM


class TestTeamCorpus:
    def test_manifest_roundtrip(self) -> None:
        corpus = TeamCorpus(format_id="gen9championsvgc2026regmc", teams=(("a", "b"), ("c",)))

        assert corpus.to_dict() == {
            "format_id": "gen9championsvgc2026regmc",
            "teams": [["a", "b"], ["c"]],
        }
        assert TeamCorpus.from_dict(corpus.to_dict()) == corpus

    @pytest.mark.parametrize("teams", [(), ((),), (("a", ""),)])
    def test_rejects_empty_teams(self, teams: tuple[tuple[str, ...], ...]) -> None:
        with pytest.raises(ValueError, match="at least one team"):
            TeamCorpus(format_id="gen9championsvgc2026regmc", teams=teams)

    def test_rejects_empty_format(self) -> None:
        with pytest.raises(ValueError, match="format_id"):
            TeamCorpus(format_id="", teams=(("a",),))

    def test_rejects_unknown_fields(self) -> None:
        value = {**TeamCorpus(format_id="f", teams=(("a",),)).to_dict(), "obsolete": "value"}

        with pytest.raises(ValueError, match="unknown"):
            TeamCorpus.from_dict(value)


class TestTeamCorpusSampling:
    def test_samples_uniformly_by_canonical_team(self) -> None:
        """A team with 90 variants is drawn as often as a team with 10 variants."""
        many = tuple(f"many-{index}" for index in range(90))
        few = tuple(f"few-{index}" for index in range(10))
        corpus = TeamCorpus(format_id=FORMAT.battle_format, teams=(many, few))
        rng = random.Random(42)

        counts = Counter(corpus.sample(rng).split("-")[0] for _ in range(10000))

        assert counts["many"] + counts["few"] == 10000
        # Per-variant sampling would put about 9000 draws on the first team.
        assert 4500 < counts["many"] < 5500


class TestPackShowdownTeam:
    def test_packs_six_members(self) -> None:
        packed = pack_showdown_team(DEFAULT_TEST_TEAM)

        members = packed.split("]")
        assert len(members) == 6
        assert members[0].startswith("Pikachu|")

    def test_rejects_partial_team(self) -> None:
        one_member = DEFAULT_TEST_TEAM.strip().split("\n\n")[0]

        with pytest.raises(ValueError, match="Malformed Showdown team"):
            pack_showdown_team(one_member)


class TestCorpusFromTeamFiles:
    def test_each_file_is_one_canonical_team(self, tmp_path: Path) -> None:
        first = tmp_path / "first.txt"
        second = tmp_path / "second.txt"
        first.write_text(DEFAULT_TEST_TEAM, encoding="utf-8")
        second.write_text(DEFAULT_TEST_TEAM.replace("Pikachu", "Raichu", 1), encoding="utf-8")

        corpus = corpus_from_team_files((first, second), FORMAT.bo3_format)

        assert corpus.format_id == FORMAT.bo3_format
        assert len(corpus.teams) == 2
        assert corpus.teams[0][0].startswith("Pikachu|")
        assert corpus.teams[1][0].startswith("Raichu|")

    def test_rejects_malformed_and_missing_files(self, tmp_path: Path) -> None:
        malformed = tmp_path / "malformed.txt"
        malformed.write_text("not a team", encoding="utf-8")

        with pytest.raises(ValueError, match="Malformed team file"):
            corpus_from_team_files((malformed,), FORMAT.bo3_format)
        with pytest.raises(ValueError, match="Malformed team file"):
            corpus_from_team_files((tmp_path / "missing.txt",), FORMAT.bo3_format)

    def test_rejects_no_files(self) -> None:
        with pytest.raises(ValueError, match="at least one team"):
            corpus_from_team_files((), FORMAT.bo3_format)


class TestLoadTeamCorpus:
    def _write(self, directory: Path, value: object) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "corpus_manifest.json").write_bytes(orjson.dumps(value))

    def test_loads_manifest_from_pool_directory(self, tmp_path: Path) -> None:
        corpus = TeamCorpus(format_id=FORMAT.battle_format, teams=(("a", "b"), ("c",)))
        self._write(tmp_path, corpus.to_dict())

        assert load_team_corpus(tmp_path, FORMAT.battle_format) == corpus

    def test_accepts_regular_corpus_for_bo3(self, tmp_path: Path) -> None:
        corpus = TeamCorpus(format_id=FORMAT.battle_format, teams=(("a",),))
        self._write(tmp_path, corpus.to_dict())

        assert load_team_corpus(tmp_path, FORMAT.bo3_format) == corpus

    def test_rejects_incompatible_format(self, tmp_path: Path) -> None:
        self._write(tmp_path, TeamCorpus(format_id="gen9ou", teams=(("a",),)).to_dict())

        with pytest.raises(ValueError, match="Corpus format mismatch"):
            load_team_corpus(tmp_path, FORMAT.bo3_format)

    def test_directory_of_team_files_without_manifest_names_the_build_command(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "team.txt").write_text(DEFAULT_TEST_TEAM, encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="p0-corpus build"):
            load_team_corpus(tmp_path, FORMAT.bo3_format)

    def test_rejects_invalid_manifest(self, tmp_path: Path) -> None:
        self._write(tmp_path, [])

        with pytest.raises(ValueError, match="Invalid corpus manifest"):
            load_team_corpus(tmp_path, FORMAT.bo3_format)

    def test_manifest_with_older_fields_names_the_build_command(self, tmp_path: Path) -> None:
        self._write(
            tmp_path,
            {"artifact_schema": "p0.team_corpus.v2", "format_id": "f", "entries": []},
        )

        with pytest.raises(ValueError, match="rebuild it with `p0-corpus build"):
            load_team_corpus(tmp_path, FORMAT.bo3_format)
