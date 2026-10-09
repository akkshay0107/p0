"""Unit tests for corpus building and audit reports."""

from __future__ import annotations

from pathlib import Path

import orjson

from p0.format_config import FORMAT
from p0.teams.corpus import TeamCorpus
from p0.teams.corpus_build import audit_corpus, write_corpus_manifest


class TestWriteCorpusManifest:
    def test_writes_to_output_dir(self, tmp_path: Path) -> None:
        packed = "Pikachu|Static|Light Ball|Timid|Thunderbolt,Protect"
        corpus = TeamCorpus(format_id=FORMAT.battle_format, teams=((packed,),))
        out_dir = tmp_path / "pool"

        path = write_corpus_manifest(corpus, out_dir)

        assert path == out_dir / "corpus_manifest.json"
        assert orjson.loads(path.read_bytes()) == {
            "format_id": FORMAT.battle_format,
            "teams": [[packed]],
        }


class TestAuditCorpus:
    def test_counts_every_variant_and_collects_coverage(self) -> None:
        corpus = TeamCorpus(
            format_id=FORMAT.battle_format,
            teams=(
                ("Pikachu||Light Ball|Static|Thunderbolt,Protect", "Sparky|Raichu||Static|Protect"),
                ("Garchomp||Sitrus Berry|Rough Skin|Earthquake",),
            ),
        )

        audit = audit_corpus(corpus, total_candidates=4, rejections_by_reason={"oov_item: X": 1})

        assert audit == {
            "total_candidates": 4,
            "admitted_count": 3,
            "rejected_count": 1,
            "rejections_by_reason": {"oov_item: X": 1},
            "species_coverage": ("garchomp", "pikachu", "raichu"),
            "move_coverage": ("earthquake", "protect", "thunderbolt"),
            "item_coverage": ("lightball", "sitrusberry"),
        }
