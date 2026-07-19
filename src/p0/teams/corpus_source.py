"""Corpus-backed TeamSource implementation and offline sampling policies.

This module implements CorpusTeamSource, which loads a validated TeamCorpusManifest
and provides allocation-free, pure-Python sampling of ValidatedTeam instances
according to configured split bounds, curriculum stages, mirroring constraints,
and diverse sampling policies.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from pathlib import Path

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
            raw_data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
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
        self._prepare_sampling(spec.sampling_policy)

    def _prepare_sampling(self, policy: SamplingPolicy) -> None:
        self._usage_weights = [entry.usage_count for entry in self._entries]

        by_canonical: dict[str, list[CorpusEntry]] = {}
        for entry in self._entries:
            by_canonical.setdefault(entry.canonical_hash, []).append(entry)
        self._by_canonical = by_canonical
        self._canonical_keys = tuple(sorted(by_canonical.keys()))

        by_archetype: dict[str, list[CorpusEntry]] = {}
        for entry in self._entries:
            tags = entry.archetype_tags if entry.archetype_tags else ("_untagged_",)
            for tag in tags:
                by_archetype.setdefault(tag, []).append(entry)
        self._by_archetype = by_archetype
        self._archetype_keys = tuple(sorted(by_archetype.keys()))

        total_usage = sum(self._usage_weights)
        self._rare_weights = [max(1, total_usage // entry.usage_count) for entry in self._entries]

    def _sample_entry(
        self,
        rng: random.Random,
        exclude_canonical_hash: str | None = None,
    ) -> CorpusEntry:
        entries = self._entries
        policy = self._spec.sampling_policy

        if exclude_canonical_hash is not None:
            entries = tuple(
                entry for entry in self._entries if entry.canonical_hash != exclude_canonical_hash
            )
            if not entries:
                raise ValueError("No eligible corpus entries remain after exclusion")

            if policy == SamplingPolicy.USAGE_WEIGHTED:
                weights = [entry.usage_count for entry in entries]
                return rng.choices(entries, weights=weights, k=1)[0]
            if policy == SamplingPolicy.UNIFORM_CANONICAL:
                keys = tuple(sorted({entry.canonical_hash for entry in entries}))
                chosen_canonical = rng.choice(keys)
                candidates = [
                    entry for entry in entries if entry.canonical_hash == chosen_canonical
                ]
                return rng.choice(candidates)
            if policy == SamplingPolicy.UNIFORM_ARCHETYPE:
                by_arch: dict[str, list[CorpusEntry]] = {}
                for entry in entries:
                    tags = entry.archetype_tags if entry.archetype_tags else ("_untagged_",)
                    for tag in tags:
                        by_arch.setdefault(tag, []).append(entry)
                chosen_arch = rng.choice(tuple(sorted(by_arch.keys())))
                return rng.choice(by_arch[chosen_arch])
            if policy == SamplingPolicy.RARE_COVERAGE:
                tot = sum(entry.usage_count for entry in entries)
                rare_w = [max(1, tot // entry.usage_count) for entry in entries]
                return rng.choices(entries, weights=rare_w, k=1)[0]
            return rng.choice(entries)

        if policy == SamplingPolicy.USAGE_WEIGHTED:
            return rng.choices(entries, weights=self._usage_weights, k=1)[0]
        if policy == SamplingPolicy.UNIFORM_CANONICAL:
            chosen_canonical = rng.choice(self._canonical_keys)
            return rng.choice(self._by_canonical[chosen_canonical])
        if policy == SamplingPolicy.UNIFORM_ARCHETYPE:
            chosen_arch = rng.choice(self._archetype_keys)
            return rng.choice(self._by_archetype[chosen_arch])
        if policy == SamplingPolicy.RARE_COVERAGE:
            return rng.choices(entries, weights=self._rare_weights, k=1)[0]
        return rng.choice(entries)

    def sample(self, rng: random.Random) -> ValidatedTeam:
        """Return a single validated team sampled according to policy."""
        entry = self._sample_entry(rng)
        return ValidatedTeam(packed=entry.packed, team_hash=entry.packed_sha256)

    def sample_pair(self, rng: random.Random) -> tuple[ValidatedTeam, ValidatedTeam]:
        """Return a pair of validated teams respecting the mirroring constraint."""
        first_entry = self._sample_entry(rng)
        first_team = ValidatedTeam(packed=first_entry.packed, team_hash=first_entry.packed_sha256)

        if not self._spec.allow_mirror:
            if len(self._canonical_keys) < 2:
                raise ValueError(
                    "Cannot sample non-mirror pair from a single-canonical-team corpus"
                )
            second_entry = self._sample_entry(
                rng, exclude_canonical_hash=first_entry.canonical_hash
            )
            second_team = ValidatedTeam(
                packed=second_entry.packed, team_hash=second_entry.packed_sha256
            )
            return first_team, second_team

        second_entry = self._sample_entry(rng)
        second_team = ValidatedTeam(
            packed=second_entry.packed, team_hash=second_entry.packed_sha256
        )
        return first_team, second_team

    def describe(self) -> Mapping[str, JsonScalar | tuple[str, ...]]:
        """Describe the active corpus pool and sampling configuration."""
        return {
            "kind": "corpus",
            "corpus_path": self._spec.corpus_path,
            "corpus_hash": self._spec.corpus_hash,
            "format_id": self._spec.format_id,
            "split": self._spec.split.name,
            "sampling_policy": self._spec.sampling_policy.name,
            "allow_mirror": self._spec.allow_mirror,
            "curriculum_stage": self._spec.curriculum_stage,
            "pool_size": len(self._entries),
            "team_hashes": tuple(sorted({entry.packed_sha256 for entry in self._entries})),
        }
