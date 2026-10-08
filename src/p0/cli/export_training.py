"""Export one checkpoint, its recorded inputs, and a portable resume configuration."""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

from p0.paths import DEFAULT_PATHS
from p0.persistence import atomic_output
from p0.replays.dataset import LazyReplayDataset
from p0.teams.source import build_team_source
from p0.training.checkpoint import CheckpointStore


def export_checkpoint(checkpoint: Path, output: Path) -> None:
    """
    Publish an archive with recorded inputs without reading a mutable application config.

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
    for name in ("vocab.json", "champions_dex.json", "spread_usage.json"):
        files[f"data/{name}"] = DEFAULT_PATHS.data_root / name
    settings, inputs = run["settings"], run["inputs"]
    config: dict[str, Any] = {}
    if metadata["trainer_kind"] == "bc":
        manifest_path, split_path = Path(inputs["shard_manifest"]), Path(inputs["split_manifest"])
        files["run/inputs/manifest.json"] = manifest_path
        files["run/inputs/splits.json"] = split_path
        dataset = LazyReplayDataset(manifest_path, split_manifest=split_path)
        if (
            dataset.manifest.dataset_id != metadata["dataset_id"]
            or dataset.split_manifest is None
            or dataset.split_manifest.split_id != metadata["split_id"]
        ):
            raise ValueError("BC input files no longer match the checkpoint")
        for entry in dataset.manifest.shards:
            name = f"run/inputs/{entry.filename}"
            files[name] = manifest_path.parent / entry.filename
        bc = {
            name: value for name, value in settings.items() if name not in {"gamma", "value_coef"}
        }
        config = {
            "training": {"gamma": settings["gamma"], "value_coef": settings["value_coef"]},
            "bc": {
                **bc,
                "epochs": metadata["epoch_budget"],
                "resume_checkpoint": "run/checkpoint.pt",
                "shard_manifest": "run/inputs/manifest.json",
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
    with atomic_output(output) as temporary:
        with tarfile.open(temporary, "w:gz") as archive:
            for name, path in files.items():
                archive.add(path, arcname=name, recursive=False)
            encoded = (json.dumps(config, indent=2) + "\n").encode()
            _add_bytes(archive, "run/config.yaml", encoded)


def _add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


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
