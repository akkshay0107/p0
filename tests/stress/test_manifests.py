from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from p0.format_config import canonical_json_sha256, current_manifest, load_runtime_manifest
from p0.replays.shards import ShardIndexEntry, ShardManifest, load_shard_manifest
from p0.teams.corpus import CorpusEntry, CorpusSplit, TeamCorpusManifest, corpus_content_hash


def _runtime_files(tmp_path: Path) -> tuple[Path, Path]:
    vocab = tmp_path / "vocab.json"
    dex = tmp_path / "dex.json"
    vocab.write_text(
        json.dumps({"species": {"pikachu": 1}, "moves": {"tackle": 1}}), encoding="utf-8"
    )
    dex.write_text('{"pikachu":{"base_stats":{"hp":35}}}', encoding="utf-8")
    return vocab, dex


def _shard_manifest(contract: str) -> ShardManifest:
    entry = ShardIndexEntry("shard-000.pt", "c" * 64, 10, 2, 1, 100)
    return ShardManifest(
        runtime_contract_sha256=contract,
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


@pytest.mark.stress
def test_runtime_manifest_digest_is_semantic_and_round_trips(tmp_path: Path) -> None:
    vocab, dex = _runtime_files(tmp_path)
    manifest = current_manifest(vocab_path=vocab, dex_path=dex)
    reordered = json.loads(json.dumps(manifest.to_dict()))
    reordered["runtime_contract"] = {
        key: reordered["runtime_contract"][key]
        for key in reversed(tuple(reordered["runtime_contract"]))
    }
    assert canonical_json_sha256(reordered["runtime_contract"]) == manifest.runtime_contract_sha256
    path = tmp_path / "runtime_manifest.json"
    path.write_text(json.dumps(reordered), encoding="utf-8")
    assert load_runtime_manifest(path) == manifest


@pytest.mark.stress
def test_shard_manifest_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    vocab, dex = _runtime_files(tmp_path)
    runtime = current_manifest(vocab_path=vocab, dex_path=dex)
    contract = runtime.runtime_contract_sha256
    manifest = _shard_manifest(contract)
    assert ShardManifest.from_dict(manifest.to_dict()) == manifest
    manifest_path = tmp_path / "runtime_manifest.json"
    manifest_path.write_text(json.dumps(runtime.to_dict()), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible"):
        load_shard_manifest(
            {**manifest.to_dict(), "runtime_contract_sha256": "b" * 64}, manifest_path
        )
    with pytest.raises(ValueError, match="source_series"):
        ShardManifest.from_dict({**manifest.to_dict(), "source_series": {"series-1": ("game-1",)}})


@pytest.mark.stress
def test_corpus_manifest_hash_is_order_independent_but_packed_content_bound() -> None:
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
        runtime_contract_sha256="d" * 64,
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
