"""Export training artifacts and their runtime contracts."""

import argparse
import shutil
import sys
import tarfile
import time
from pathlib import Path

from p0.paths import DEFAULT_PATHS


def format_size(size_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def gather_directory_files(
    directory: Path, project_root: Path, targets: list[tuple[Path, str, int]]
):
    if not directory.exists():
        return
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            targets.append((p, str(p.relative_to(project_root)), p.stat().st_size))


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    project_root = DEFAULT_PATHS.repository_root
    output_path = project_root / "ppo_training_export.tar.gz"

    artifacts = project_root / "artifacts"
    if not artifacts.exists():
        print("No artifacts directory found.", file=sys.stderr)
        return 1

    config = project_root / "config.yaml"
    if config.exists():
        shutil.copy(config, artifacts / "config.yaml")

    targets = collect_export_files(project_root, artifacts)

    files_to_archive = [
        (p, arc, size) for p, arc, size in targets if p.resolve() != output_path.resolve()
    ]

    if not files_to_archive:
        print("No training files found to export.", file=sys.stderr)
        return 1

    total_bytes = sum(size for _, _, size in files_to_archive)
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
    except Exception as e:
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
