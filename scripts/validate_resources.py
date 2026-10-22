"""Validate generated resources after initialization."""

import json
from pathlib import Path
from typing import Any

from p0.paths import DEFAULT_PATHS
from p0.replays.reconstruction.classification import CLASSIFICATION_REGISTRY

IMPOSSIBLE_TAGS_PATH = Path(__file__).with_name("impossible_protocol_tags.json")
_OWNER_KINDS = ("moves", "items", "abilities")
_REACHABLE = {"reachable-potential", "reachable-resolved"}


def _load(filename: str) -> dict[str, Any]:
    return json.loads((DEFAULT_PATHS.data_root / filename).read_text(encoding="utf-8"))


def _entry_tags(entry: dict[str, Any]) -> list[str]:
    tags = [entry["tag"]] if entry.get("tag") is not None else list(entry["resolved_tags"])
    if entry.get("split_tag"):
        tags.append(entry["split_tag"])
    return tags


def validate_sources(dex: dict[str, Any], inventory: dict[str, Any], contract: dict[str, Any]):
    """Check that the inventory and the contract were generated from the dex's simulator."""
    source = dex["source"]
    formats = [source["battleFormat"], source["bo3Format"]]
    for name, value in (("raw emission inventory", inventory), ("protocol contract", contract)):
        if value["showdown_commit"] != source["commit"]:
            raise ValueError(f"{name} has an unexpected Showdown revision")
        if value["formats"] != formats:
            raise ValueError(f"{name} formats disagree with dex metadata")


def validate_raw_emission_inventory(dex: dict[str, Any], inventory: dict[str, Any]) -> None:
    """Check that the source scan is complete and every emission site is classified."""
    showdown_root = DEFAULT_PATHS.showdown_root
    expected_files = {
        "config/formats.ts",
        "server/room-battle.ts",
        *(f"data/{name}.ts" for name in ("moves", "abilities", "items", "conditions", "rulesets")),
        *(
            source.relative_to(showdown_root).as_posix()
            for directory in ("sim", f"data/mods/{dex['source']['mod']}")
            for source in (showdown_root / directory).rglob("*.ts")
        ),
    }
    files = inventory["files"]
    if {item["path"] for item in files} != expected_files:
        raise ValueError("raw emission inventory omits or adds Showdown source files")
    if len(files) != len(expected_files):
        raise ValueError("raw emission inventory contains duplicate source files")

    counts: dict[str, int] = {}
    for entry in inventory["entries"]:
        if entry["path"] not in expected_files:
            raise ValueError("raw emission inventory entry has no scanned source file")
        counts[entry["reachability"]] = counts.get(entry["reachability"], 0) + 1
    if counts != inventory["reachability_counts"]:
        raise ValueError("raw emission inventory reachability counts disagree")
    if (
        counts.get("unresolved")
        or inventory["review_status"] != "raw_inventory_reachable_sites_classified"
    ):
        raise ValueError("raw emission inventory has unresolved source sites")
    covered = {
        (entry["path"], entry["call"], tag, entry["enclosing"], entry["data_owner"])
        for entry in inventory["entries"]
        if entry["reachability"] in _REACHABLE
        for tag in ([entry["tag"]] if entry["tag"] is not None else entry["resolved_tags"])
    }
    required = {
        ("sim/battle.ts", "add", "teampreview", "makeRequest", None),
        ("data/rulesets.ts", "add", "rule", "onBegin", "speciesclause"),
    }
    if not required <= covered:
        raise ValueError(
            f"raw emission inventory is missing source witnesses: {sorted(required - covered)}"
        )


def validate_protocol_contract(inventory: dict[str, Any], contract: dict[str, Any]) -> None:
    """Check that the contract matches the inventory and the engine's shape table."""
    missing = set(CLASSIFICATION_REGISTRY) - {entry["tag"] for entry in contract["entries"]}
    if missing:
        raise ValueError(f"protocol contract is missing classifiers: {sorted(missing)}")
    if contract["review_status"] != "source_inventory_reachable_sites_resolved":
        raise ValueError("protocol contract contains unresolved source emission sites")
    witnesses = contract["raw_witnesses"]
    witness_ids = [witness["id"] for witness in witnesses]
    if len(set(witness_ids)) != len(witness_ids):
        raise ValueError("raw source witness IDs must be unique")
    expected_ids = {
        f"{entry['path']}:{entry['line']}:{entry['call']}"
        for entry in inventory["entries"]
        if entry["reachability"] in _REACHABLE
    }
    if set(witness_ids) != expected_ids:
        raise ValueError("protocol contract source witnesses disagree with raw inventory")
    for witness in witnesses:
        tags = ([witness["tag"]] if witness["tag"] is not None else []) + witness["resolved_tags"]
        # The stream uses an empty tag for out-of-band text and separators.
        if tags == [""]:
            if witness["disposition"] != "no_state_change":
                raise ValueError(f"blank source witness must be state neutral: {witness}")
            continue
        if not tags or any(tag not in CLASSIFICATION_REGISTRY for tag in tags):
            raise ValueError(f"raw source witness has no classifier mapping: {witness}")
        dispositions = {CLASSIFICATION_REGISTRY[tag].classification.value for tag in tags}
        if dispositions != {witness["disposition"]}:
            raise ValueError(f"raw source witness disposition disagrees with classifier: {witness}")


def validate_tag_support(dex: dict[str, Any], inventory: dict[str, Any]) -> None:
    """Check that every emitted tag is handled or impossible, and no impossible tag is legal."""
    legality = dex["legality"]
    impossible = set(json.loads(IMPOSSIBLE_TAGS_PATH.read_text(encoding="utf-8")))

    unhandled: set[str] = set()
    legal_owners: set[tuple[str, str]] = set()
    for entry in inventory["entries"]:
        tags = _entry_tags(entry)
        if entry["reachability"] != "excluded":
            unhandled.update(tag for tag in tags if tag and tag not in CLASSIFICATION_REGISTRY)
        kind = Path(entry["path"]).stem
        if kind in _OWNER_KINDS and entry["data_owner"] in legality[kind]:
            legal_owners.update((tag, entry["data_owner"]) for tag in tags if tag in impossible)
    if unhandled:
        raise ValueError(
            f"Showdown can emit tags that are neither handled nor impossible: {sorted(unhandled)}"
        )
    if legal_owners:
        raise ValueError(f"impossible tags have a legal owner: {sorted(legal_owners)}")


def main() -> None:
    coverage = _load("champions_coverage.json")
    if coverage["missingLegalContent"] or coverage["unmappedLegalEffects"]:
        raise ValueError("Dex and vocabulary coverage is incomplete")
    dex = _load("champions_dex.json")
    inventory = _load("showdown_raw_emission_inventory.json")
    contract = _load("replay_protocol_contract.json")
    validate_sources(dex, inventory, contract)
    validate_raw_emission_inventory(dex, inventory)
    validate_protocol_contract(inventory, contract)
    validate_tag_support(dex, inventory)


if __name__ == "__main__":
    main()
