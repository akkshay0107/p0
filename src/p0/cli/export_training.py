"""Export one checkpoint, its recorded inputs, and a portable resume configuration."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Any

from p0.format_config import load_active_global_contract, sha256_file
from p0.paths import DEFAULT_PATHS
from p0.persistence import atomic_output
from p0.replays.dataset import LazyReplayDataset
from p0.teams.factory import build_team_source
from p0.training.checkpoint import CheckpointStore


def export_checkpoint(checkpoint: Path, output: Path) -> None:
    """
    Publish a checked archive without reading a mutable application config.

    Arguments:
        checkpoint: Training checkpoint with recorded input locations.
        output: Archive to replace after all checks pass.

    Returns:
        None.
    """
    loaded = CheckpointStore().read(checkpoint)
    metadata = loaded.artifact["provenance"]
    run = metadata.get("run")
    if not isinstance(run, dict) or "inputs" not in run:
        raise ValueError(
            "This checkpoint has no recorded inputs; resume and save it before exporting"
        )
    if loaded.artifact["artifact_type"] != "training":
        raise ValueError("Export requires a training checkpoint")
    files: dict[str, Path] = {"run/checkpoint.pt": checkpoint}
    for name in ("runtime_manifest.json", "vocab.json", "champions_dex.json", "spread_usage.json"):
        files[f"data/{name}"] = DEFAULT_PATHS.data_root / name
    expected = {name: sha256_file(path) for name, path in files.items()}
    expected["run/checkpoint.pt"] = loaded.sha256
    contract = load_active_global_contract()
    if loaded.artifact["global_contract_sha256"] != contract.global_sha256:
        raise ValueError("Export requires the checkpoint's original runtime resources")
    settings, inputs = run["settings"], run["inputs"]
    config: dict[str, Any] = {}
    if metadata["trainer_kind"] == "bc":
        manifest_path, split_path = Path(inputs["shard_manifest"]), Path(inputs["split_manifest"])
        files["run/inputs/manifest.json"] = manifest_path
        files["run/inputs/splits.json"] = split_path
        expected["run/inputs/manifest.json"] = sha256_file(manifest_path)
        expected["run/inputs/splits.json"] = metadata["split_manifest_sha256"]
        dataset = LazyReplayDataset(manifest_path, split_manifest=split_path, verify_hashes=True)
        if (
            dataset.manifest.dataset_hash != metadata["dataset_hash"]
            or sha256_file(split_path) != metadata["split_manifest_sha256"]
        ):
            raise ValueError("BC input files no longer match the checkpoint")
        for entry in dataset.manifest.shards:
            name = f"run/inputs/{entry.filename}"
            files[name] = manifest_path.parent / entry.filename
            expected[name] = entry.sha256
        bc = {
            name: value
            for name, value in settings.items()
            if name not in {"overfit", "gamma", "value_coef"}
        }
        config = {
            "training": {"gamma": settings["gamma"], "value_coef": settings["value_coef"]},
            "bc": {
                **bc,
                "epochs": metadata["epoch_budget"],
                "resume_checkpoint": "run/checkpoint.pt",
                "shard_manifest": "run/inputs/manifest.json",
                "split_manifest": "run/inputs/splits.json",
                "output_dir": "artifacts/resumed/bc",
            },
        }
    elif metadata["trainer_kind"] == "ppo":
        for index, name in enumerate(("agent_teams", "opponent_teams")):
            path = Path(inputs[name])
            if path.is_dir():
                manifest = path / "corpus_manifest.json"
                paths = (
                    [manifest]
                    if manifest.is_file()
                    else sorted(
                        item
                        for item in path.iterdir()
                        if item.is_file() and not item.name.startswith(".")
                    )
                )
            else:
                paths = [path]
            for item in paths:
                member = f"run/inputs/{name}/{item.name}"
                files[member] = item
                expected[member] = sha256_file(item)
            source = build_team_source(path)
            identity = {
                key: value for key, value in source.describe().items() if key != "corpus_path"
            }
            if identity != settings["teams"][index]:
                raise ValueError(f"Team input no longer matches the checkpoint: {path}")
        config = {
            "training": settings["training"],
            "teams": {"all": "run/inputs/opponent_teams", "reduced": "run/inputs/agent_teams"},
            "paths": {
                "teams_root": ".",
                "resume_checkpoint": "run/checkpoint.pt",
                "checkpoint_path": "artifacts/resumed/ppo_checkpoint.pt",
                "runs_dir": "artifacts/resumed/runs",
            },
        }
    else:
        raise ValueError("Unsupported trainer in checkpoint")
    if output.resolve() in {path.resolve() for path in files.values()}:
        raise ValueError("Export destination must not replace an input file")
    hashes: dict[str, str] = {}
    with atomic_output(output) as temporary:
        with tarfile.open(temporary, "w:gz") as archive:
            for name, path in files.items():
                # Keep a single open file across hashing and copying; training replaces files atomically.
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    if digest != expected[name]:
                        raise ValueError(f"Input changed during export: {name}; retry the export")
                    stream.seek(0)
                    info = tarfile.TarInfo(name)
                    info.size = os.fstat(stream.fileno()).st_size
                    archive.addfile(info, stream)
                    hashes[name] = digest
            encoded = (json.dumps(config, indent=2) + "\n").encode()
            info = tarfile.TarInfo("run/config.yaml")
            info.size = len(encoded)
            archive.addfile(info, io.BytesIO(encoded))
            hashes[info.name] = hashlib.sha256(encoded).hexdigest()
            encoded = json.dumps(hashes, indent=2).encode()
            info = tarfile.TarInfo("run/checksums.json")
            info.size = len(encoded)
            archive.addfile(info, io.BytesIO(encoded))
        # Also detect inputs edited in place while they were copied.
        with tarfile.open(temporary, "r:gz") as archive:
            for name, digest in hashes.items():
                stream: Any = archive.extractfile(name)
                if stream is None or hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError(f"File changed during export: {name}")


def main(argv: list[str] | None = None) -> int:
    """Export CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("training_export.tar.gz"))
    args = parser.parse_args(argv)
    try:
        export_checkpoint(args.checkpoint, args.output)
    except (OSError, ValueError, tarfile.TarError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    print(f"Exported {args.output}. Extract into a compatible p0 checkout.")
    print("Resume BC with p0-bc train --config run/config.yaml.")
    print("Resume PPO with p0-train --config run/config.yaml --agent-team-source reduced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
