"""Empirical Stat Point spread priors derived from Showdown usage statistics.

Champions hides Stat Point spreads, so an opponent's real stats are never observable.
Smogon's chaos usage exports publish spreads in Champions Stat Point units already
(``Nature:hp/atk/def/spa/spd/spe``), keyed jointly with the nature, which open team
sheets reveal. This module turns those exports into a compact prior keyed on
(species, nature) and provides the runtime lookup used to impute stats.

The build blends the Bo3 and Bo1 exports per bucket. Raw chaos weights are not
comparable across exports - the Bo3 file carries more total spread weight than the
Bo1 file despite covering roughly a ninth of the battles - so each source is
normalized to a probability distribution before mixing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

from p0.teams.stat_points import (
    STAT_POINT_LIMIT,
    STAT_POINT_TOTAL_LIMIT,
    StatPoints,
)
from p0.teams.team import normalize_id

SPREAD_USAGE_SCHEMA = 1

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


@dataclass(frozen=True, slots=True)
class SpreadTable:
    """Usage-derived spread priors, keyed by normalized species and lowercase nature."""

    buckets: Mapping[tuple[str, str], tuple[SpreadPrior, ...]]
    format_id: str
    schema: int = SPREAD_USAGE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SPREAD_USAGE_SCHEMA:
            raise ValueError(
                f"Unsupported spread table schema: expected={SPREAD_USAGE_SCHEMA}, "
                f"actual={self.schema}"
            )

    def lookup(self, species: str, nature: str) -> tuple[SpreadPrior, ...]:
        """Return the bucket for a species and nature, empty when absent."""
        return self.buckets.get((normalize_id(species), nature.lower()), ())

    def best(self, species: str, nature: str) -> StatPoints | None:
        """Return the most-used spread for a bucket, or None when it is absent.

        Buckets are stored weight-descending, so the argmax is the first entry.
        """
        bucket = self.lookup(species, nature)
        return bucket[0].points if bucket else None


def parse_spread_key(key: str) -> tuple[str, StatPoints] | None:
    """Parse a chaos ``Nature:hp/atk/def/spa/spd/spe`` key, rejecting illegal spreads."""
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
    max_spreads: int = MAX_SPREADS_PER_BUCKET,
    min_nature_share: float = MIN_NATURE_SHARE,
) -> dict[str, Any]:
    """Blend two chaos exports into the serializable spread-prior artifact.

    Each (species, nature) bucket present in both exports is mixed
    BO3_BLEND_WEIGHT toward Bo3; a bucket present in only one export is taken from
    that export alone. Buckets are then truncated to the most-used ``max_spreads``
    entries and renormalized so the stored weights describe the truncated bucket.

    Arguments:
        bo1: Parsed chaos export for the singles-ladder format.
        bo3: Parsed chaos export for the Bo3 format.
        format_id: Battle format the artifact is declared against.
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

    return {
        "schema": SPREAD_USAGE_SCHEMA,
        "format_id": format_id,
        "bo3_blend_weight": BO3_BLEND_WEIGHT,
        "max_spreads_per_bucket": max_spreads,
        "min_nature_share": min_nature_share,
        "weight_scale": WEIGHT_SCALE,
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

    total = sum(weight for _, weight in parsed)
    return tuple(SpreadPrior(points, weight / total) for points, weight in parsed)
