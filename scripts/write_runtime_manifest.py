"""Validate completed resources and write the runtime completion manifest last."""

import json

from p0.contracts import build_global_contract
from p0.paths import DEFAULT_PATHS
from p0.persistence import atomic_json_save
from p0.replays.reconstruction.contract import (
    validate_protocol_contract,
    validate_raw_emission_inventory,
)


def main() -> None:
    data_root = DEFAULT_PATHS.data_root
    coverage = json.loads((data_root / "champions_coverage.json").read_text(encoding="utf-8"))
    if coverage["missingLegalContent"] or coverage["unmappedLegalEffects"]:
        raise ValueError("Dex and vocabulary coverage is incomplete")
    validate_raw_emission_inventory()
    validate_protocol_contract()
    atomic_json_save(
        data_root / "runtime_manifest.json", build_global_contract(data_root).to_dict()
    )


if __name__ == "__main__":
    main()
