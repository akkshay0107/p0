"""Export training artifacts and their runtime contracts."""

import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path

from omegaconf import OmegaConf

from p0.paths import DEFAULT_PATHS


def format_size(size_bytes: float) -> str:
    """Format a byte count into a human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def gather_directory_files(
    directory: Path, project_root: Path, targets: list[tuple[Path, str, int]]
):
    """Walk a directory and append file metadata to the target list."""
    if not directory.exists():
        return
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            relative = p.relative_to(project_root)
            if relative == Path("artifacts/config.yaml"):
                continue
            targets.append((p, str(relative), p.stat().st_size))


def collect_export_files(project_root: Path, artifacts: Path) -> list[tuple[Path, str, int]]:
    """Collect runtime artifacts plus the contracts needed to interpret them."""
    targets: list[tuple[Path, str, int]] = []
    gather_directory_files(artifacts, project_root, targets)
    for relative in (
        Path("data/runtime_manifest.json"),
        Path("data/vocab.json"),
        Path("data/champions_dex.json"),
    ):
        source = project_root / relative
        if source.exists():
            targets.append((source, str(relative), source.stat().st_size))
    return targets


def redacted_config_snapshot(config_path: Path) -> bytes | None:
    """Serialize a schema-shaped configuration snapshot with credentials removed."""
    if not config_path.is_file():
        return None
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    secret_names = {
        "password",
        "token",
        "secret",
        "api_key",
        "apikey",
        "access_token",
        "credentials",
    }

    def redact(value: object, key: str = "") -> object:
        if key.casefold() in secret_names:
            return "<redacted>"
        if isinstance(value, dict):
            return {str(name): redact(item, str(name)) for name, item in value.items()}
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    snapshot = redact(raw)
    return (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main() -> int:
    """Export CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    project_root = DEFAULT_PATHS.repository_root
    output_path = project_root / "ppo_training_export.tar.gz"

    artifacts = project_root / "artifacts"
    if not artifacts.exists():
        print("No artifacts directory found.", file=sys.stderr)
        return 1

    targets = collect_export_files(project_root, artifacts)
    config_snapshot = redacted_config_snapshot(project_root / "config.yaml")

    files_to_archive = [
        (p, arc, size) for p, arc, size in targets if p.resolve() != output_path.resolve()
    ]

    if not files_to_archive:
        print("No training files found to export.", file=sys.stderr)
        return 1

    total_bytes = sum(size for _, _, size in files_to_archive) + (
        len(config_snapshot) if config_snapshot is not None else 0
    )
    print(
        f"Discovered {len(files_to_archive)} files to export (Total size: {format_size(total_bytes)})."
    )

    print(f"Creating archive: {output_path}")
    t0 = time.time()

    try:
        with tarfile.open(output_path, "w:gz") as tar:
            for filepath, arcname, size in files_to_archive:
                if size > 10 * 1024 * 1024:
                    print(f"Adding: {arcname} ({format_size(size)})")
                tar.add(filepath, arcname=arcname)
            if config_snapshot is not None:
                info = tarfile.TarInfo("artifacts/config.redacted.json")
                info.size = len(config_snapshot)
                info.mtime = time.time()
                tar.addfile(info, io.BytesIO(config_snapshot))
    except (OSError, tarfile.TarError, TypeError, ValueError) as e:
        print(f"Error creating archive: {e}", file=sys.stderr)
        return 1

    duration = time.time() - t0
    archive_size = output_path.stat().st_size

    print(f"Export completed in {duration:.1f}s.")
    print(f"Archive file: {output_path} ({format_size(archive_size)})")

    # export instructions
    print()
    print("To restore on the target machine, copy the archive to the 'p0/' directory and run:")
    print(f"    tar -xzf {output_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
