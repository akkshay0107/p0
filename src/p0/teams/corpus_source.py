"""Corpus-backed TeamSource implementation and offline sampling policies.

This module implements CorpusTeamSource, which loads a validated TeamCorpusManifest
and provides allocation-free, pure-Python sampling of ValidatedTeam instances
according to configured split bounds, curriculum stages, and diverse sampling
policies.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path

import orjson

from p0.teams.corpus import (
    CorpusEntry,
    CorpusSourceSpec,
    SamplingPolicy,
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
        if spec.curriculum_stage:
            filtered = [
                entry for entry in filtered if spec.curriculum_stage in entry.archetype_tags
            ]

        if not filtered:
            raise ValueError(
                f"No corpus entries match split={spec.split.name} and "
                f"curriculum_stage={spec.curriculum_stage!r}"
            )

        self._spec = spec
        self._entries = tuple(filtered)
        self._usage_weights: list[int] | None = None
        self._by_canonical: dict[str, list[CorpusEntry]] | None = None
        self._canonical_keys: tuple[str, ...] | None = None
        self._by_archetype: dict[str, list[CorpusEntry]] | None = None
        self._archetype_keys: tuple[str, ...] | None = None
        self._rare_weights: list[int] | None = None
        self._prepare_sampling(spec.sampling_policy)

    def _get_usage_weights(self) -> list[int]:
        if self._usage_weights is None:
            self._usage_weights = [entry.usage_count for entry in self._entries]
        return self._usage_weights

    def _get_canonical_index(self) -> tuple[dict[str, list[CorpusEntry]], tuple[str, ...]]:
        if self._by_canonical is None or self._canonical_keys is None:
            by_canonical: dict[str, list[CorpusEntry]] = {}
            for entry in self._entries:
                by_canonical.setdefault(entry.canonical_hash, []).append(entry)
            self._by_canonical = by_canonical
            self._canonical_keys = tuple(sorted(by_canonical.keys()))
        return self._by_canonical, self._canonical_keys

    def _get_archetype_index(self) -> tuple[dict[str, list[CorpusEntry]], tuple[str, ...]]:
        if self._by_archetype is None or self._archetype_keys is None:
            by_archetype: dict[str, list[CorpusEntry]] = {}
            for entry in self._entries:
                tags = entry.archetype_tags if entry.archetype_tags else ("_untagged_",)
                for tag in tags:
                    by_archetype.setdefault(tag, []).append(entry)

            # A wholly untagged pool would collapse this policy into uniform sampling
            # over every entry, which is indistinguishable from the default policy at
            # runtime. Refuse it rather than silently sample the wrong distribution.
            if set(by_archetype) == {"_untagged_"}:
                raise ValueError(
                    "UNIFORM_ARCHETYPE sampling requires archetype tags, but no corpus "
                    "entry carries one; archetype tagging is currently unwired"
                )

            self._by_archetype = by_archetype
            self._archetype_keys = tuple(sorted(by_archetype.keys()))
        return self._by_archetype, self._archetype_keys

    def _get_rare_weights(self) -> list[int]:
        if self._rare_weights is None:
            total_usage = sum(self._get_usage_weights())
            self._rare_weights = [
                max(1, total_usage // entry.usage_count) for entry in self._entries
            ]
        return self._rare_weights

    def _prepare_sampling(self, policy: SamplingPolicy) -> None:
        if policy == SamplingPolicy.USAGE_WEIGHTED:
            self._get_usage_weights()
        elif policy == SamplingPolicy.UNIFORM_CANONICAL:
            self._get_canonical_index()
        elif policy == SamplingPolicy.UNIFORM_ARCHETYPE:
            self._get_archetype_index()
        elif policy == SamplingPolicy.RARE_COVERAGE:
            self._get_rare_weights()

    def _sample_entry(
        self,
        rng: random.Random,
    ) -> CorpusEntry:
        entries = self._entries
        policy = self._spec.sampling_policy

        if policy == SamplingPolicy.USAGE_WEIGHTED:
            return rng.choices(entries, weights=self._get_usage_weights(), k=1)[0]

        if policy == SamplingPolicy.UNIFORM_CANONICAL:
            by_canonical, canonical_keys = self._get_canonical_index()
            chosen_canonical = rng.choice(canonical_keys)
            return rng.choice(by_canonical[chosen_canonical])

        if policy == SamplingPolicy.UNIFORM_ARCHETYPE:
            by_archetype, archetype_keys = self._get_archetype_index()
            chosen_arch = rng.choice(archetype_keys)
            return rng.choice(by_archetype[chosen_arch])

        if policy == SamplingPolicy.RARE_COVERAGE:
            return rng.choices(entries, weights=self._get_rare_weights(), k=1)[0]

        return rng.choice(entries)

    def sample(self, rng: random.Random) -> ValidatedTeam:
        """Return a single validated team sampled according to policy."""
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
            "sampling_policy": self._spec.sampling_policy.name,
            "curriculum_stage": self._spec.curriculum_stage,
            "pool_size": len(self._entries),
            "team_hashes": tuple(sorted({entry.packed_sha256 for entry in self._entries})),
        }
