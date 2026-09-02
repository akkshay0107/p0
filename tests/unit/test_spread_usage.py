"""Tests for spread parsing, validation, and fallback."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import orjson
import pytest

from p0.cli.build_spreads import DEFAULT_CUTOFF, DEFAULT_MONTH, USAGE_URL
from p0.format_config import FORMAT, current_manifest
from p0.teams.corpus import (
    CORPUS_MANIFEST_SCHEMA,
    CorpusEntry,
    CorpusSplit,
    TeamCorpusManifest,
    corpus_content_hash,
)
from p0.teams.spread_usage import (
    BO3_BLEND_WEIGHT,
    DEFAULT_SPREAD_TABLE_PATH,
    IMPUTED_FROM_FALLBACK,
    IMPUTED_FROM_USAGE,
    SPREAD_USAGE_SCHEMA,
    build_spread_table,
    cosmetic_forme_aliases,
    load_spread_table,
    load_spread_table_file,
    parse_spread_key,
)
from p0.teams.stat_points import BaseStats, StatPoints, calculate_stats, fallback_points


def vocabulary() -> dict[str, dict[str, int]]:
    return {
        "species": {
            "pikachu": 1,
            "charizard": 2,
            "whimsicott": 3,
            "garchomp": 4,
            "kingambit": 5,
            "glimmora": 6,
            "raichu": 7,
        },
        "items": {
            "lightball": 1,
            "charizarditey": 2,
            "focussash": 3,
            "sitrusberry": 4,
            "blackglasses": 5,
            "shucaberry": 6,
            "lifeorb": 7,
        },
        "abilities": {
            "static": 1,
            "blaze": 2,
            "prankster": 3,
            "roughskin": 4,
            "defiant": 5,
            "toxicdebris": 6,
        },
        "moves": {
            "fakeout": 1,
            "protect": 2,
            "thunderbolt": 3,
            "electroweb": 4,
            "heatwave": 5,
            "solarbeam": 6,
            "weatherball": 7,
            "moonblast": 8,
            "tailwind": 9,
            "encore": 10,
            "earthquake": 11,
            "dragonclaw": 12,
            "rockslide": 13,
            "kowtowcleave": 14,
            "suckerpunch": 15,
            "lowkick": 16,
            "powergem": 17,
            "sludgebomb": 18,
            "earthpower": 19,
        },
    }


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


def _chaos(species: str, spreads: dict[str, float]) -> dict[str, Any]:
    """Build a minimal chaos export carrying one species' spread distribution."""
    return {"data": {species: {"Spreads": spreads}}}


def _dex(*species: dict[str, Any]) -> dict[str, Any]:
    """Build a minimal dex carrying only what forme aliasing reads."""
    return {"species": list(species), "moves": []}


_BASE_STATS = {"hp": 78, "atk": 65, "def": 68, "spa": 112, "spd": 154, "spe": 75}


def _payload(spreads: dict[str, Any], aliases: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "schema": SPREAD_USAGE_SCHEMA,
        "format_id": FORMAT.battle_format,
        "weight_scale": 1_000_000,
        "aliases": aliases or {},
        "spreads": spreads,
    }


_ONE_BUCKET = {"real": {"timid": [[2, 0, 0, 32, 0, 32, 1000]]}}


def _corpus_entry(packed: str = "packed-team") -> CorpusEntry:
    return CorpusEntry(
        canonical_hash=hashlib.sha256(packed.encode()).hexdigest(),
        packed=packed,
        packed_sha256=hashlib.sha256(packed.encode()).hexdigest(),
        split=CorpusSplit.TRAIN,
        usage_count=3,
    )


def _corpus_manifest(entries: tuple[CorpusEntry, ...]) -> TeamCorpusManifest:
    active_contract = current_manifest().global_sha256
    return TeamCorpusManifest(
        global_contract_sha256=active_contract,
        format_id="gen9championsvgc2026regmb",
        corpus_hash=corpus_content_hash(entries),
        entries=entries,
        created_at="2026-07-17T00:00:00Z",
        sampling_metadata={"sampling": "uniform_canonical"},
    )


class TestSpreadParsing:
    def test_parse_spread_key_rejects_illegal_and_malformed(self) -> None:
        """Verify parse_spread_key enforces stat cap of 32, total budget of 66, and 6 stat components."""
        parsed = parse_spread_key("Impish:32/0/21/0/11/2")
        assert parsed is not None
        nature, points = parsed
        assert nature == "impish"
        assert points == StatPoints(hp=32, defense=21, spd=11, spe=2)

        assert parse_spread_key("Impish:33/0/0/0/0/0") is None, "per-stat cap of 32"
        assert parse_spread_key("Impish:32/32/32/0/0/0") is None, "66-point budget"
        assert parse_spread_key("Impish:32/0/0/0/0") is None, "needs six stats"
        assert parse_spread_key("Impish:a/0/0/0/0/0") is None
        assert parse_spread_key("32/0/0/0/0/0") is None, "needs a nature"

    def test_spread_table_blends_shared_buckets_toward_bo3(self) -> None:
        """Verify spread table blends Bo3 and Bo1 distributions according to BO3_BLEND_WEIGHT (0.8/0.2)."""
        only_bo3 = "Timid:2/0/0/32/0/32"
        only_bo1 = "Timid:32/0/0/32/0/2"
        payload = build_spread_table(
            _chaos("Whimsicott", {only_bo1: 10.0}),
            _chaos("Whimsicott", {only_bo3: 1.0}),
            format_id=FORMAT.battle_format,
            dex=_dex(),
        )
        bucket = load_spread_table(payload).lookup("Whimsicott", "timid")

        # Each source normalizes to 1.0 first, so the raw 10:1 count gap is irrelevant
        # and the blend lands on the configured 0.8/0.2 split.
        assert [prior.points for prior in bucket] == [
            StatPoints(hp=2, spa=32, spe=32),
            StatPoints(hp=32, spa=32, spe=2),
        ]
        assert bucket[0].weight == pytest.approx(BO3_BLEND_WEIGHT, abs=1e-4)
        assert bucket[1].weight == pytest.approx(1.0 - BO3_BLEND_WEIGHT, abs=1e-4)

    def test_spread_table_uses_single_source_when_bucket_is_absent(self) -> None:
        """Verify spread table adopts 100% weight when a spread appears in only one format export."""
        payload = build_spread_table(
            _chaos("Sylveon", {"Calm:32/0/0/0/32/2": 5.0}),
            _chaos("Incineroar", {"Impish:32/0/32/0/2/0": 5.0}),
            format_id=FORMAT.battle_format,
            dex=_dex(),
        )
        table = load_spread_table(payload)

        # A bucket only one export carries is taken whole rather than down-weighted.
        assert table.best("Sylveon", "calm") == StatPoints(hp=32, spd=32, spe=2)
        assert table.best("Incineroar", "impish") == StatPoints(hp=32, defense=32, spd=2)
        assert table.lookup("Sylveon", "calm")[0].weight == pytest.approx(1.0)

    def test_spread_table_prunes_rare_natures(self) -> None:
        """Verify build_spread_table prunes natures below min_nature_share threshold (e.g. 5%)."""
        common = "Timid:2/0/0/32/0/32"
        rare = "Hardy:32/0/32/0/2/0"
        export = _chaos("Gholdengo", {common: 99.0, rare: 1.0})
        table = load_spread_table(
            build_spread_table(
                export, export, format_id=FORMAT.battle_format, dex=_dex(), min_nature_share=0.05
            )
        )

        assert table.best("Gholdengo", "timid") is not None
        assert table.best("Gholdengo", "hardy") is None, "1% share is below the 5% floor"

        kept = load_spread_table(
            build_spread_table(
                export, export, format_id=FORMAT.battle_format, dex=_dex(), min_nature_share=0.0
            )
        )
        assert kept.best("Gholdengo", "hardy") is not None

    def test_spread_table_truncates_and_renormalizes(self) -> None:
        """Verify spread tables truncate entries to max_spreads per bucket and re-normalize probabilities to 1.0."""
        # Each allocation sums to 64, so all twenty are legal and distinct.
        spreads = {f"Jolly:{n}/{32 - n}/0/0/0/32": float(20 - n) for n in range(20)}
        export = _chaos("Garchomp", spreads)
        bucket = load_spread_table(
            build_spread_table(
                export, export, format_id=FORMAT.battle_format, dex=_dex(), max_spreads=5
            )
        ).lookup("Garchomp", "jolly")

        assert len(bucket) == 5
        assert sum(prior.weight for prior in bucket) == pytest.approx(1.0)
        # Stored weight-descending, so the argmax is entry zero.
        assert [prior.weight for prior in bucket] == sorted(
            (prior.weight for prior in bucket), reverse=True
        )

    def test_spread_table_rejects_unsupported_schema(self) -> None:
        """Verify load_spread_table rejects invalid or unsupported schema versions."""
        payload = build_spread_table(
            _chaos("Garchomp", {"Jolly:2/32/0/0/0/32": 1.0}),
            _chaos("Garchomp", {"Jolly:2/32/0/0/0/32": 1.0}),
            format_id=FORMAT.battle_format,
            dex=_dex(),
        )
        payload["schema"] = SPREAD_USAGE_SCHEMA + 1
        with pytest.raises(ValueError, match="schema"):
            load_spread_table(payload)

    def test_spread_table_lookup_normalizes_species_and_nature(self) -> None:
        """Verify spread table lookups normalize species names and natures case-insensitively."""
        payload = build_spread_table(
            _chaos("Charizard-Mega-Y", {"Timid:2/0/0/32/0/32": 1.0}),
            _chaos("Charizard-Mega-Y", {"Timid:2/0/0/32/0/32": 1.0}),
            format_id=FORMAT.battle_format,
            dex=_dex(),
        )
        table = load_spread_table(payload)

        expected = StatPoints(hp=2, spa=32, spe=32)
        assert table.best("Charizard-Mega-Y", "Timid") == expected
        assert table.best("charizardmegay", "timid") == expected
        assert table.best("Missingno", "timid") is None
        assert table.lookup("Missingno", "timid") == ()

    def test_cosmetic_formes_alias_onto_their_base_species(self) -> None:
        """Verify cosmetic forme variations (e.g. Florges colours) automatically alias to base species."""
        dex = _dex(
            {
                "id": "florges",
                "name": "Florges",
                "baseSpecies": "Florges",
                "baseStats": _BASE_STATS,
                "formeOrder": ["Florges", "Florges-Blue", "Florges-White"],
            }
        )
        aliases = cosmetic_forme_aliases(dex)

        # Colour variants carry no dex record of their own, so they inherit the base.
        assert aliases["florgesblue"] == "florges"
        assert aliases["florgeswhite"] == "florges"
        assert "florges" not in aliases

    def test_formes_with_distinct_base_stats_are_never_aliased(self) -> None:
        """Verify forme variations with distinct base stats (e.g. Floette-Eternal) are excluded from cosmetic aliasing."""
        eternal_stats = {"hp": 74, "atk": 65, "def": 67, "spa": 125, "spd": 128, "spe": 92}
        dex = _dex(
            {
                "id": "floette",
                "name": "Floette",
                "baseSpecies": "Floette",
                "baseStats": {"hp": 54, "atk": 45, "def": 47, "spa": 75, "spd": 98, "spe": 52},
                "formeOrder": ["Floette", "Floette-Blue", "Floette-Eternal"],
                "otherFormes": ["Floette-Eternal"],
            },
            {
                "id": "floetteeternal",
                "name": "Floette-Eternal",
                "baseSpecies": "Floette",
                "baseStats": eternal_stats,
            },
        )
        aliases = cosmetic_forme_aliases(dex)

        assert aliases["floetteblue"] == "floette", "cosmetic colour still folds"
        assert "floetteeternal" not in aliases, "different base stats must stay separate"

    def test_aliases_resolve_in_one_hop_without_cycles(self) -> None:
        """Verify forme aliases resolve cleanly in a single lookup step without alias chaining or cycles."""
        shared = dict(_BASE_STATS)
        dex = _dex(
            *(
                {
                    "id": f"alcremie{flavour}",
                    "name": f"Alcremie-{flavour}",
                    "baseSpecies": "Alcremie",
                    "baseStats": shared,
                    "formeOrder": ["Alcremie-ruby", "Alcremie-matcha"],
                }
                for flavour in ("ruby", "matcha")
            ),
            {"id": "alcremie", "name": "Alcremie", "baseSpecies": "Alcremie", "baseStats": shared},
        )
        aliases = cosmetic_forme_aliases(dex)

        assert aliases["alcremieruby"] == "alcremie"
        assert aliases["alcremiematcha"] == "alcremie"
        assert not [key for key, value in aliases.items() if value in aliases], "no alias chains"
        assert not [key for key, value in aliases.items() if key == value], "no self references"

    def test_aliased_forme_shares_the_base_species_bucket(self) -> None:
        """Verify aliased cosmetic formes return the identical memory object as their base species."""
        dex = _dex(
            {
                "id": "florges",
                "name": "Florges",
                "baseSpecies": "Florges",
                "baseStats": _BASE_STATS,
                "formeOrder": ["Florges", "Florges-Blue"],
            }
        )
        export = _chaos("Florges", {"Modest:32/0/20/12/2/0": 1.0})
        table = load_spread_table(
            build_spread_table(export, export, format_id=FORMAT.battle_format, dex=dex)
        )

        expected = StatPoints(hp=32, defense=20, spa=12, spd=2)
        assert table.best("Florges", "modest") == expected
        assert table.best("Florges-Blue", "modest") == expected

        # The alias shares the base tuple rather than duplicating rows in the artifact.
        assert table.lookup("Florges-Blue", "modest") is table.lookup("Florges", "modest")


class TestSpreadValidation:
    def test_load_rejects_bucket_that_is_not_weight_descending(self) -> None:
        """Verify load_spread_table rejects spread buckets that are not strictly sorted in descending weight order."""
        unsorted_rows = {"real": {"timid": [[32, 0, 0, 32, 0, 2, 100], [2, 0, 0, 32, 0, 32, 900]]}}
        with pytest.raises(ValueError, match="weight-descending"):
            load_spread_table(_payload(unsorted_rows))

    def test_load_rejects_unresolvable_aliases(self) -> None:
        """Verify load_spread_table rejects missing targets or alias chains."""
        with pytest.raises(ValueError, match="unknown species"):
            load_spread_table(_payload(_ONE_BUCKET, {"ghost": "missing"}))
        with pytest.raises(ValueError, match="unknown species|through another alias"):
            load_spread_table(_payload(_ONE_BUCKET, {"first": "second", "second": "real"}))

    def test_load_accepts_a_single_hop_alias(self) -> None:
        """Verify load_spread_table accepts single-hop valid cosmetic aliases."""
        table = load_spread_table(_payload(_ONE_BUCKET, {"cosmetic": "real"}))
        assert table.best("cosmetic", "timid") == table.best("real", "timid")

    def test_usage_export_url_matches_the_published_layout(self) -> None:
        """Verify Smogon stats URL formatter generates valid external chaos data paths."""
        url = USAGE_URL.format(month="2026-07", format_id="gen9championsvgc2026regmb", cutoff=1760)
        assert url == (
            "https://www.smogon.com/stats/2026-07/chaos/gen9championsvgc2026regmb-1760.json.gz"
        )

    def test_shipped_spread_table_records_the_exports_it_came_from(self) -> None:
        """Verify committed spread_usage.json records source month, format filenames, and SHA-256 digests."""
        payload = orjson.loads(DEFAULT_SPREAD_TABLE_PATH.read_bytes())
        source = payload["source"]

        assert source["month"] == DEFAULT_MONTH
        assert set(source["exports"]) == {
            f"{FORMAT.battle_format}-{DEFAULT_CUTOFF}.json",
            f"{FORMAT.bo3_format}-{DEFAULT_CUTOFF}.json",
        }
        assert all(len(digest) == 64 for digest in source["exports"].values())

    def test_first_point_and_nature_truncation_match_showdown(self) -> None:
        """Verify calculate_stats stat formulas match Showdown Gen 9 stat calculations at level 50."""
        base = BaseStats(100, 100, 100, 100, 100, 100)
        zero = calculate_stats(base, StatPoints(), "adamant")
        one = calculate_stats(base, StatPoints(atk=1), "adamant")
        assert zero == (175, 132, 120, 108, 120, 120)
        assert one[1] == zero[1] + 1

    @pytest.mark.parametrize(
        "points, error",
        [
            (dict(hp=33), r"\[0, 32\]"),
            (dict(hp=32, atk=32, defense=3), "at most 66"),
        ],
    )
    def test_stat_point_validation(self, points, error) -> None:
        """Verify StatPoints enforces 0..32 per-stat bounds and 66 total point budget."""
        with pytest.raises(ValueError, match=error):
            StatPoints(**points)

    def test_fallback_follows_move_category_priority(self) -> None:
        """Verify fallback_points allocates 66 points using category priority: physical > special > status."""
        physical = StatPoints(hp=32, atk=32, spe=2)
        special = StatPoints(hp=32, spa=32, spe=2)
        status = StatPoints(hp=32, defense=17, spd=17)

        assert fallback_points(("physical", "physical", "status", "status")) == physical
        assert fallback_points(("special", "special", "status", "status")) == special
        assert fallback_points(("status", "status", "special", "physical")) == status

        # Physical wins a tie with special regardless of which stat the species favours.
        assert fallback_points(("physical", "physical", "special", "special")) == physical

        # Every shape spends the full budget and stays legal.
        for categories in (("physical",) * 2, ("special",) * 2, ("status",) * 2):
            points = fallback_points(categories)
            assert points is not None
            assert sum(points.as_tuple()) == 66

    def test_fallback_is_unknown_when_no_category_reaches_two(self) -> None:
        """Verify fallback_points returns None when known move categories are insufficient to establish archetype."""
        assert fallback_points(("physical", "special", "status")) is None
        assert fallback_points(()) is None
        assert fallback_points(("physical",)) is None

    def test_usage_prior_beats_fallback_and_reports_confidence(self) -> None:
        """Verify resolve() prioritizes empirical usage table over generic category fallback."""
        table = load_spread_table_file()
        categories = ("physical", "status", "physical", "status")

        covered = table.resolve("Incineroar", "Impish", categories)
        assert covered is not None
        assert covered.origin == IMPUTED_FROM_USAGE
        assert 0.0 < covered.confidence <= 1.0
        assert sum(covered.points.as_tuple()) <= 66

        # The prior is consulted before the fallback, so a covered species must not
        # collapse onto the generic physical shape.
        assert covered.points != StatPoints(hp=32, atk=32, spe=2)

        uncovered = table.resolve("Missingno", "Impish", categories)
        assert uncovered is not None
        assert uncovered.origin == IMPUTED_FROM_FALLBACK
        assert uncovered.confidence == 0.0
        assert uncovered.points == StatPoints(hp=32, atk=32, spe=2)

    def test_usage_prior_resolution_is_deterministic(self) -> None:
        """Verify resolve() produces identical StatPoints across multiple invocations."""
        table = load_spread_table_file()
        categories = ("special", "status", "special", "status")
        first = table.resolve("Charizard-Mega-Y", "Timid", categories)
        second = table.resolve("Charizard-Mega-Y", "Timid", categories)
        assert first == second
