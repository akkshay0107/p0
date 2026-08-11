"""Empirical Stat Point spread priors derived from Showdown usage statistics.

Champions hides Stat Point spreads, so an opponent's real stats are never observable.
Smogon's chaos usage exports publish spreads in Champions Stat Point units already
(Nature:hp/atk/def/spa/spd/spe), keyed jointly with the nature, which open team
sheets reveal. This module turns those exports into a compact prior keyed on
(species, nature) and provides the runtime lookup used to impute stats.

The build blends the Bo3 and Bo1 exports per bucket. Raw chaos weights are not
comparable across exports - the Bo3 file carries more total spread weight than the
Bo1 file despite covering roughly a ninth of the battles - so each source is
normalized to a probability distribution before mixing.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import orjson

from p0.paths import DEFAULT_PATHS
from p0.teams.stat_points import (
    STAT_POINT_LIMIT,
    STAT_POINT_TOTAL_LIMIT,
    StatPoints,
    fallback_points,
)
from p0.teams.team import normalize_id

SPREAD_USAGE_SCHEMA = 2

# Recorded on each estimate so downstream metrics can separate a usage-backed spread
# from a move-category shape, which are not comparably reliable.
IMPUTED_FROM_USAGE = "usage"
IMPUTED_FROM_FALLBACK = "fallback"

DEFAULT_SPREAD_TABLE_PATH = DEFAULT_PATHS.data_root / "spread_usage.json"

# Bo3 is the format replays are compiled against, so it leads the blend; Bo1 covers
# roughly 98 extra species and contributes the remainder.
BO3_BLEND_WEIGHT = 0.8
MAX_SPREADS_PER_BUCKET = 15

# Natures below this share of a species' spread mass are dropped. At 0.5% this
# discards roughly 63% of buckets while retaining 99.3% of usage coverage, and the
# discarded buckets are the ones whose rankings are dominated by sampling noise.
MIN_NATURE_SHARE = 0.005

# Weights are stored as scaled integers so the artifact hashes identically across
# platforms, which float formatting cannot guarantee.
WEIGHT_SCALE = 1_000_000


class SpreadPrior(NamedTuple):
    """One candidate spread and its share of its (species, nature) bucket."""

    points: StatPoints
    weight: float


class SpreadEstimate(NamedTuple):
    """A resolved spread, how much usage backs it, and where it came from."""

    points: StatPoints
    confidence: float
    origin: str


@dataclass(frozen=True, slots=True)
class SpreadTable:
    """Usage-derived spread priors, keyed by normalized species and lowercase nature."""

    buckets: Mapping[tuple[str, str], tuple[SpreadPrior, ...]]
    format_id: str

    def lookup(self, species: str, nature: str) -> tuple[SpreadPrior, ...]:
        """Return the bucket for a species and nature, empty when absent."""
        return self.buckets.get((normalize_id(species), nature.lower()), ())

    def best(self, species: str, nature: str) -> StatPoints | None:
        """Return the most-used spread for a bucket, or None when it is absent.

        Buckets are stored weight-descending, so the argmax is the first entry.
        """
        bucket = self.lookup(species, nature)
        return bucket[0].points if bucket else None

    def resolve(
        self, species: str, nature: str, move_categories: tuple[str, ...]
    ) -> SpreadEstimate | None:
        """Estimate a spread from the usage prior, falling back to move categories.

        Returns None when neither source can produce one, which callers record as an
        explicit UNKNOWN rather than substituting a blind guess.
        """
        bucket = self.lookup(species, nature)
        if bucket:
            return SpreadEstimate(bucket[0].points, bucket[0].weight, IMPUTED_FROM_USAGE)

        return self._fallback_estimate(move_categories)

    def sample(
        self, species: str, nature: str, move_categories: tuple[str, ...], rng: random.Random
    ) -> SpreadEstimate | None:
        """Draw a spread in proportion to usage, for generating varied teams.

        Mirrors resolve() except that a populated bucket is sampled by weight rather
        than reduced to its argmax.
        """
        bucket = self.lookup(species, nature)
        if not bucket:
            return self._fallback_estimate(move_categories)

        prior = rng.choices(bucket, weights=[entry.weight for entry in bucket], k=1)[0]
        return SpreadEstimate(prior.points, prior.weight, IMPUTED_FROM_USAGE)

    @staticmethod
    def _fallback_estimate(move_categories: tuple[str, ...]) -> SpreadEstimate | None:
        """Shape a spread from move categories when no usage bucket exists."""
        points = fallback_points(move_categories)
        if points is None:
            return None

        # A category fallback is a shape, not an observed frequency, so it carries no
        # usage share to report as confidence.
        return SpreadEstimate(points, 0.0, IMPUTED_FROM_FALLBACK)


def cosmetic_forme_aliases(dex: Mapping[str, Any]) -> dict[str, str]:
    """Map purely cosmetic formes onto the base species whose priors they can share.

    Usage exports report cosmetic formes under the base name, so without this a
    Florges-Blue sheet would miss the Florges bucket entirely. A forme qualifies only
    when it carries no base stats of its own or carries stats identical to the base:
    Floette's forme list mixes cosmetic colours with Floette-Eternal and Floette-Mega,
    which are separate Pokemon whose spreads must never be folded together.
    """
    entries = [entry for entry in dex.get("species", ()) if isinstance(entry, Mapping)]
    by_id: dict[str, Mapping[str, Any]] = {
        normalize_id(str(entry.get("id", entry.get("name", "")))): entry for entry in entries
    }

    def canonical_id(entry: Mapping[str, Any]) -> str:
        """Resolve one entry to the species whose priors it may share.

        Anchored on baseSpecies rather than on whichever entry happens to list the
        forme: Alcremie's variants each list all the others, so listing order would
        otherwise decide the target and could alias two formes onto each other.
        """
        own_id = normalize_id(str(entry.get("id", entry.get("name", ""))))
        base_id = normalize_id(str(entry.get("baseSpecies", "")))
        if not base_id or base_id == own_id:
            return own_id

        base_entry = by_id.get(base_id)
        if base_entry is None:
            return own_id

        own_stats = entry.get("baseStats")
        base_stats = base_entry.get("baseStats")
        if not isinstance(own_stats, Mapping) or not isinstance(base_stats, Mapping):
            return own_id
        if dict(own_stats) != dict(base_stats):
            return own_id

        return base_id

    aliases: dict[str, str] = {}
    for entry in entries:
        own_id = normalize_id(str(entry.get("id", entry.get("name", ""))))
        target = canonical_id(entry)
        if target != own_id:
            aliases[own_id] = target

        # Cosmetic formes carry no dex record of their own, so they exist only in the
        # listing entry's forme names and inherit its stats by construction.
        for name in (*entry.get("formeOrder", ()), *entry.get("cosmeticFormes", ())):
            if not isinstance(name, str) or not name:
                continue
            forme_id = normalize_id(name)
            if forme_id != target and forme_id not in by_id:
                aliases[forme_id] = target

    return aliases


def parse_spread_key(key: str) -> tuple[str, StatPoints] | None:
    """Parse a chaos Nature:hp/atk/def/spa/spd/spe key, rejecting illegal spreads."""
    nature, separator, allocation = key.partition(":")
    if not separator:
        return None

    parts = allocation.split("/")
    if len(parts) != 6:
        return None

    try:
        values = [int(part) for part in parts]
    except ValueError:
        return None

    # Usage exports occasionally carry spreads from other formats or malformed rows;
    # anything outside the Champions budget is discarded rather than repaired.
    if any(not 0 <= value <= STAT_POINT_LIMIT for value in values):
        return None
    if sum(values) > STAT_POINT_TOTAL_LIMIT:
        return None

    return nature.lower(), StatPoints(*values)


class SourceBuckets(NamedTuple):
    """One chaos export reduced to per-bucket distributions and their nature shares."""

    distributions: dict[tuple[str, str], dict[StatPoints, float]]
    nature_share: dict[tuple[str, str], float]


def _normalized_buckets(document: Mapping[str, Any]) -> SourceBuckets:
    """Group one chaos export into (species, nature) buckets that each sum to 1.

    Also records how much of each species' total spread mass its nature accounts
    for, which is what the rare-nature prune is applied against.
    """
    data = document.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("Chaos export is missing its 'data' object")

    raw: dict[tuple[str, str], dict[StatPoints, float]] = {}
    species_totals: dict[str, float] = {}
    for species, entry in data.items():
        if not isinstance(entry, Mapping):
            continue
        spreads = entry.get("Spreads")
        if not isinstance(spreads, Mapping):
            continue

        species_id = normalize_id(str(species))
        for key, weight in spreads.items():
            parsed = parse_spread_key(str(key))
            if parsed is None or weight <= 0:
                continue
            nature, points = parsed
            bucket = raw.setdefault((species_id, nature), {})
            bucket[points] = bucket.get(points, 0.0) + float(weight)
            species_totals[species_id] = species_totals.get(species_id, 0.0) + float(weight)

    nature_share: dict[tuple[str, str], float] = {}
    for key, bucket in raw.items():
        total = sum(bucket.values())
        nature_share[key] = total / species_totals[key[0]]
        for points in bucket:
            bucket[points] /= total

    return SourceBuckets(raw, nature_share)


def build_spread_table(
    bo1: Mapping[str, Any],
    bo3: Mapping[str, Any],
    *,
    format_id: str,
    dex: Mapping[str, Any],
    max_spreads: int = MAX_SPREADS_PER_BUCKET,
    min_nature_share: float = MIN_NATURE_SHARE,
) -> dict[str, Any]:
    """Blend two chaos exports into the serializable spread-prior artifact.

    Each (species, nature) bucket present in both exports is mixed
    BO3_BLEND_WEIGHT toward Bo3; a bucket present in only one export is taken from
    that export alone. Buckets are then truncated to the most-used max_spreads
    entries and renormalized so the stored weights describe the truncated bucket.

    Arguments:
        bo1: Parsed chaos export for the singles-ladder format.
        bo3: Parsed chaos export for the Bo3 format.
        format_id: Battle format the artifact is declared against.
        dex: Champions dex, used to alias cosmetic formes onto their base species.
        max_spreads: Maximum spreads retained per bucket.
        min_nature_share: Drop a bucket when its nature accounts for less than this
            share of its species' spread mass in every export that carries it.

    Returns:
        A JSON-serializable mapping ready for atomic_json_save.
    """
    if max_spreads <= 0:
        raise ValueError("max_spreads must be positive")
    if not 0.0 <= min_nature_share < 1.0:
        raise ValueError("min_nature_share must be in [0, 1)")

    bo1_buckets = _normalized_buckets(bo1)
    bo3_buckets = _normalized_buckets(bo3)

    serialized: dict[str, dict[str, list[list[int]]]] = {}
    for key in sorted(set(bo1_buckets.distributions) | set(bo3_buckets.distributions)):
        species_id, nature = key

        # A nature this rare is backed by too few observations for its ranked
        # spreads to mean anything; the caller's fallback is better than stored
        # noise. Kept when either export considers it common enough.
        if (
            max(bo1_buckets.nature_share.get(key, 0.0), bo3_buckets.nature_share.get(key, 0.0))
            < min_nature_share
        ):
            continue

        from_bo1 = bo1_buckets.distributions.get(key)
        from_bo3 = bo3_buckets.distributions.get(key)

        if from_bo1 is not None and from_bo3 is not None:
            merged: dict[StatPoints, float] = {}
            for points, share in from_bo3.items():
                merged[points] = merged.get(points, 0.0) + BO3_BLEND_WEIGHT * share
            for points, share in from_bo1.items():
                merged[points] = merged.get(points, 0.0) + (1.0 - BO3_BLEND_WEIGHT) * share
        else:
            # Exactly one source has this bucket; it is already normalized.
            merged = dict(from_bo3 if from_bo3 is not None else from_bo1 or {})

        # Sort by descending weight with the spread as a deterministic tiebreak, so
        # the artifact is byte-identical across runs and entry zero is the argmax.
        ordered = sorted(merged.items(), key=lambda item: (-item[1], item[0].as_tuple()))
        retained = ordered[:max_spreads]
        retained_total = sum(weight for _, weight in retained)

        rows = [
            [*points.as_tuple(), max(1, round(WEIGHT_SCALE * weight / retained_total))]
            for points, weight in retained
        ]
        serialized.setdefault(species_id, {})[nature] = rows

    # Only aliases that would actually resolve are stored: the target must carry
    # buckets, and a forme the exports report separately keeps its own.
    aliases = {
        forme: target
        for forme, target in sorted(cosmetic_forme_aliases(dex).items())
        if target in serialized and forme not in serialized
    }

    return {
        "schema": SPREAD_USAGE_SCHEMA,
        "format_id": format_id,
        "bo3_blend_weight": BO3_BLEND_WEIGHT,
        "max_spreads_per_bucket": max_spreads,
        "min_nature_share": min_nature_share,
        "weight_scale": WEIGHT_SCALE,
        "aliases": aliases,
        "spreads": serialized,
    }


def load_spread_table(payload: Mapping[str, Any]) -> SpreadTable:
    """Rebuild the runtime lookup from a serialized spread-prior artifact."""
    schema = payload.get("schema")
    if schema != SPREAD_USAGE_SCHEMA:
        raise ValueError(
            f"Unsupported spread table schema: expected={SPREAD_USAGE_SCHEMA}, actual={schema}"
        )

    spreads = payload.get("spreads")
    if not isinstance(spreads, Mapping):
        raise ValueError("Spread table is missing its 'spreads' object")

    buckets: dict[tuple[str, str], tuple[SpreadPrior, ...]] = {}
    for species_id, by_nature in spreads.items():
        if not isinstance(by_nature, Mapping):
            raise ValueError(f"Spread table entry is not a nature mapping: {species_id}")
        for nature, rows in by_nature.items():
            buckets[(str(species_id), str(nature))] = _load_bucket(rows, species_id, nature)

    # Cosmetic formes point at the base species' tuples rather than copying them, so
    # aliasing costs one dict entry per forme and nothing in the serialized artifact.
    aliases = payload.get("aliases", {})
    if not isinstance(aliases, Mapping):
        raise ValueError("Spread table 'aliases' must be a mapping")

    # Aliases are expanded against the real buckets only, so a target that is itself
    # an alias would resolve to nothing and a missing target would do nothing at all.
    # Both are silent, so reject them here instead of serving empty lookups.
    for forme, target in aliases.items():
        if str(target) not in spreads:
            raise ValueError(f"Spread alias points at an unknown species: {forme} -> {target}")
        if str(target) in aliases:
            raise ValueError(f"Spread alias resolves through another alias: {forme} -> {target}")

    natures_by_species: dict[str, list[tuple[str, tuple[SpreadPrior, ...]]]] = {}
    for (species_id, nature), bucket in buckets.items():
        natures_by_species.setdefault(species_id, []).append((nature, bucket))

    for forme, target in aliases.items():
        for nature, bucket in natures_by_species.get(str(target), ()):
            buckets.setdefault((str(forme), nature), bucket)

    return SpreadTable(buckets=buckets, format_id=str(payload.get("format_id", "")))


def _load_bucket(rows: Any, species_id: str, nature: str) -> tuple[SpreadPrior, ...]:
    """Deserialize and renormalize one bucket's rows."""
    if not isinstance(rows, Sequence) or not rows:
        raise ValueError(f"Spread bucket is empty or malformed: {species_id}/{nature}")

    parsed: list[tuple[StatPoints, int]] = []
    for row in rows:
        if not isinstance(row, Sequence) or len(row) != 7:
            raise ValueError(f"Spread row must hold six stats and a weight: {species_id}/{nature}")
        points = StatPoints(*(int(value) for value in row[:6]))
        weight = int(row[6])
        if weight <= 0:
            raise ValueError(f"Spread weight must be positive: {species_id}/{nature}")
        parsed.append((points, weight))

    # best() and resolve() take entry zero as the argmax, so a bucket that is not
    # weight-descending would silently hand back a spread that is not the most used.
    # The writer sorts, but nothing else guarantees the artifact on disk still is.
    weights = [weight for _, weight in parsed]
    if weights != sorted(weights, reverse=True):
        raise ValueError(f"Spread bucket is not weight-descending: {species_id}/{nature}")

    total = sum(weights)
    return tuple(SpreadPrior(points, weight / total) for points, weight in parsed)


@lru_cache(maxsize=2)
def load_spread_table_file(path: Path = DEFAULT_SPREAD_TABLE_PATH) -> SpreadTable:
    """Load and cache a spread table from disk.

    Cached because every observation build consults the table, and re-parsing a
    multi-megabyte artifact per battle would dominate the build cost.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Spread table not found: {path}. Build it with 'p0-build-spreads'."
        )
    try:
        payload = orjson.loads(path.read_bytes())
    except (OSError, orjson.JSONDecodeError) as exc:
        raise ValueError(f"Malformed spread table: {path}") from exc

    return load_spread_table(payload)
