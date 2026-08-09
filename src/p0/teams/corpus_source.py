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
    load_corpus_manifest,
)
from p0.teams.source import JsonScalar, ValidatedTeam


class CorpusTeamSource:
    """A load-time verified, corpus-backed team sampling source."""

    def __init__(self, spec: CorpusSourceSpec) -> None:
        path = Path(spec.corpus_path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus manifest file not found: {path}")

        try:
            raw_data = orjson.loads(path.read_bytes())
        except (OSError, UnicodeError, orjson.JSONDecodeError) as exc:
            raise ValueError(f"Malformed corpus manifest file: {path}") from exc

        manifest = load_corpus_manifest(raw_data)

        if manifest.corpus_hash != spec.corpus_hash:
            raise ValueError(
                f"Corpus hash does not match: declared={spec.corpus_hash}, actual={manifest.corpus_hash}"
            )
        if manifest.format_id != spec.format_id:
            raise ValueError(
                f"Format ID mismatch: declared={spec.format_id}, actual={manifest.format_id}"
            )

        filtered = [entry for entry in manifest.entries if entry.split == spec.split]

        if not filtered:
            raise ValueError(f"No corpus entries match split={spec.split.name}")

        self._spec = spec
        self._entries = tuple(filtered)
        self._by_canonical: dict[str, list[CorpusEntry]] | None = None
        self._canonical_keys: tuple[str, ...] | None = None

    def _get_canonical_index(self) -> tuple[dict[str, list[CorpusEntry]], tuple[str, ...]]:
        if self._by_canonical is None or self._canonical_keys is None:
            by_canonical: dict[str, list[CorpusEntry]] = {}
            for entry in self._entries:
                by_canonical.setdefault(entry.canonical_hash, []).append(entry)
            self._by_canonical = by_canonical
            self._canonical_keys = tuple(sorted(by_canonical.keys()))
        return self._by_canonical, self._canonical_keys

    def _sample_entry(
        self,
        rng: random.Random,
    ) -> CorpusEntry:
        by_canonical, canonical_keys = self._get_canonical_index()
        chosen_canonical = rng.choice(canonical_keys)
        return rng.choice(by_canonical[chosen_canonical])

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
            "team_hashes": tuple(sorted({entry.packed_sha256 for entry in self._entries})),
        }
