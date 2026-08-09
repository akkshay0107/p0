"""Corpus-backed TeamSource implementation with uniform team sampling.

This module implements CorpusTeamSource, which loads a validated TeamCorpusManifest
and provides allocation-free, pure-Python sampling of ValidatedTeam instances from
one configured corpus split.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path

import orjson

from p0.teams.corpus import (
    CorpusEntry,
    CorpusSourceSpec,
    CorpusSplit,
    TeamCorpusManifest,
    load_corpus_manifest,
)
from p0.teams.source import JsonScalar, ValidatedTeam


class CorpusTeamPool:
    """One validated corpus manifest with immutable split and sampling views."""

    def __init__(self, spec: CorpusSourceSpec, manifest: TeamCorpusManifest) -> None:
        self._corpus_path = spec.corpus_path
        self._corpus_hash = manifest.corpus_hash
        self._format_id = manifest.format_id
        entries_by_split: dict[CorpusSplit, tuple[CorpusEntry, ...]] = {}
        canonical_pools_by_split: dict[CorpusSplit, tuple[tuple[CorpusEntry, ...], ...]] = {}
        team_hashes_by_split: dict[CorpusSplit, tuple[str, ...]] = {}
        for split in (CorpusSplit.TRAIN, CorpusSplit.VALIDATION, CorpusSplit.TEST):
            entries = tuple(entry for entry in manifest.entries if entry.split is split)
            by_canonical: dict[str, list[CorpusEntry]] = {}
            for entry in entries:
                by_canonical.setdefault(entry.canonical_hash, []).append(entry)
            entries_by_split[split] = entries
            canonical_pools_by_split[split] = tuple(
                tuple(by_canonical[key]) for key in sorted(by_canonical)
            )
            team_hashes_by_split[split] = tuple(sorted(entry.packed_sha256 for entry in entries))
        self._entries_by_split = entries_by_split
        self._canonical_pools_by_split = canonical_pools_by_split
        self._team_hashes_by_split = team_hashes_by_split

    @classmethod
    def from_spec(cls, spec: CorpusSourceSpec) -> CorpusTeamPool:
        """Load and validate one manifest for one or more split-specific sources."""
        path = Path(spec.corpus_path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus manifest file not found: {path}")

        try:
            raw_data = orjson.loads(path.read_bytes())
        except (OSError, UnicodeError, orjson.JSONDecodeError) as exc:
            raise ValueError(f"Malformed corpus manifest file: {path}") from exc

        manifest = load_corpus_manifest(raw_data)
        pool = cls(spec, manifest)
        pool.validate_spec(spec)
        return pool

    def validate_spec(self, spec: CorpusSourceSpec) -> None:
        if spec.corpus_path != self._corpus_path:
            raise ValueError("Corpus source specs must share one manifest path")
        if spec.corpus_hash != self._corpus_hash:
            raise ValueError(
                f"Corpus hash does not match: declared={spec.corpus_hash}, actual={self._corpus_hash}"
            )
        if spec.format_id != self._format_id:
            raise ValueError(
                f"Format ID mismatch: declared={spec.format_id}, actual={self._format_id}"
            )

    def entries(self, split: CorpusSplit) -> tuple[CorpusEntry, ...]:
        return self._entries_by_split[split]

    def canonical_pools(self, split: CorpusSplit) -> tuple[tuple[CorpusEntry, ...], ...]:
        return self._canonical_pools_by_split[split]

    def team_hashes(self, split: CorpusSplit) -> tuple[str, ...]:
        return self._team_hashes_by_split[split]


class CorpusTeamSource:
    """A load-time verified, corpus-backed team sampling source."""

    def __init__(self, spec: CorpusSourceSpec, *, pool: CorpusTeamPool | None = None) -> None:
        if pool is None:
            pool = CorpusTeamPool.from_spec(spec)
        else:
            pool.validate_spec(spec)
        entries = pool.entries(spec.split)
        if not entries:
            raise ValueError(f"No corpus entries match split={spec.split.name}")

        self._spec = spec
        self._entries = entries
        self._canonical_pools = pool.canonical_pools(spec.split)
        self._team_hashes = pool.team_hashes(spec.split)

    def _sample_entry(
        self,
        rng: random.Random,
    ) -> CorpusEntry:
        return rng.choice(rng.choice(self._canonical_pools))

    def sample(self, rng: random.Random) -> ValidatedTeam:
        """Return a single validated team sampled uniformly by canonical team."""
        entry = self._sample_entry(rng)
        return ValidatedTeam(packed=entry.packed, team_hash=entry.packed_sha256)

    def describe(self) -> Mapping[str, JsonScalar | tuple[str, ...]]:
        """Describe the active corpus pool and sampling configuration."""
        return {
            "kind": "corpus",
            "corpus_path": self._spec.corpus_path,
            "corpus_hash": self._spec.corpus_hash,
            "format_id": self._spec.format_id,
            "split": self._spec.split.name,
            "sampling": "uniform_canonical",
            "pool_size": len(self._entries),
            "team_hashes": self._team_hashes,
        }
