"""Generate the replay protocol contract from the selected source inventory."""

from __future__ import annotations

import json
from pathlib import Path

from p0.persistence import atomic_json_save
from p0.replays.identity import normalize_showdown_id
from p0.replays.reconstruction.classification import CLASSIFICATION_REGISTRY

PROJECT_ROOT = Path(__file__).parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


def main() -> None:
    raw_inventory_path = DATA_ROOT / "showdown_raw_emission_inventory.json"
    raw_inventory = json.loads(raw_inventory_path.read_text(encoding="utf-8"))
    raw_entries = raw_inventory["entries"]
    source_by_tag = {}
    raw_witnesses = []
    for raw in raw_entries:
        tag = raw.get("tag")
        if tag and raw.get("reachability") != "excluded":
            source_by_tag.setdefault(tag, raw)
        if raw.get("reachability") in {"reachable-potential", "reachable-resolved", "unresolved"}:
            tags = [tag] if tag is not None else raw.get("resolved_tags", [])
            dispositions = {
                CLASSIFICATION_REGISTRY[item].classification.value
                for item in tags
                if item in CLASSIFICATION_REGISTRY
            }
            if tags == [""]:
                disposition = "no_state_change"
            elif len(dispositions) == 1 and all(item in CLASSIFICATION_REGISTRY for item in tags):
                disposition = next(iter(dispositions))
            else:
                disposition = "unresolved"
            raw_witnesses.append(
                {
                    "id": f"{raw['path']}:{raw['line']}:{raw['call']}",
                    "path": raw["path"],
                    "line": raw["line"],
                    "call": raw["call"],
                    "tag": raw.get("tag"),
                    "resolved_tags": raw.get("resolved_tags", []),
                    "arguments": raw.get("arguments", []),
                    "reachability": raw["reachability"],
                    "disposition": disposition,
                    "normalized_effects": sorted(
                        {
                            normalize_showdown_id(argument.strip("'\"").split(":", 1)[-1])
                            for argument in raw.get("arguments", [])
                            if argument.startswith(
                                ("'move:", '"move:', "'ability:", '"ability:', "'item:", '"item:')
                            )
                        }
                    ),
                }
            )
    entries = []
    for tag, rule in sorted(CLASSIFICATION_REGISTRY.items()):
        source = source_by_tag.get(tag)
        entries.append(
            {
                "tag": tag,
                "shape": {
                    "minimum_arguments": rule.minimum_arguments,
                    "maximum_arguments": rule.maximum_arguments,
                    "required_nonempty": list(rule.required),
                },
                "disposition": rule.classification.value,
                "source": {
                    "path": source["path"] if source else "pokemon-showdown/sim/SIM-PROTOCOL.md",
                    "line": source["line"] if source else None,
                    "review": (
                        "reachable AST emission witness"
                        if source
                        else "protocol shape documented in SIM-PROTOCOL; no direct AST emission witness"
                    ),
                },
            }
        )
    value = {
        "showdown_commit": raw_inventory["showdown_commit"],
        "formats": raw_inventory["formats"],
        "review_status": (
            "source_inventory_reachable_sites_resolved"
            if raw_inventory.get("review_status") == "raw_inventory_reachable_sites_classified"
            and all(row["disposition"] != "unresolved" for row in raw_witnesses)
            else "source_inventory_unresolved_sites"
        ),
        "raw_witnesses": raw_witnesses,
        "entries": entries,
    }
    path = DATA_ROOT / "replay_protocol_contract.json"
    atomic_json_save(path, value)


if __name__ == "__main__":
    main()
