#!/usr/bin/env python3
import argparse
import sys
import tarfile
import time
from pathlib import Path


def find_project_root() -> Path:
    # iterate over all ancestors till project root found
    # fallback if no pyproject file is current dirs parent
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path(__file__).resolve().parent


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


def gather_files(project_root: Path, exclude_logs: bool) -> list[tuple[Path, str, int]]:
    targets = []

    ppoconfig = project_root / ".ppoconfig"
    if ppoconfig.exists():
        targets.append((ppoconfig, ".ppoconfig", ppoconfig.stat().st_size))

    artifacts = project_root / "artifacts"
    checkpoint = artifacts / "checkpoints" / "ppo_checkpoint.pt"
    if checkpoint.exists():
        targets.append(
            (checkpoint, "artifacts/checkpoints/ppo_checkpoint.pt", checkpoint.stat().st_size)
        )

    gather_directory_files(artifacts / "checkpoints" / "pool", project_root, targets)
    gather_directory_files(artifacts / "backups", project_root, targets)

    if not exclude_logs:
        training_log = artifacts / "training.log"
        if training_log.exists():
            targets.append((training_log, "artifacts/training.log", training_log.stat().st_size))

        gather_directory_files(artifacts / "runs", project_root, targets)
        gather_directory_files(artifacts / "eval", project_root, targets)

    return targets


def main():
    parser = argparse.ArgumentParser(description="Export training checkpoints and state.")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="ppo_training_export.tar.gz",
        help="Path to save the export archive.",
    )
    parser.add_argument(
        "--exclude-logs", action="store_true", help="Exclude TensorBoard runs and training.log."
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose file lists.")
    args = parser.parse_args()

    project_root = find_project_root()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # avoid archiving the output file into itself
    # if output path is within the project root.
    files_to_archive = [
        (p, arc, size)
        for p, arc, size in gather_files(project_root, args.exclude_logs)
        if p.resolve() != output_path
    ]

    if not files_to_archive:
        print("No training files found to export.", file=sys.stderr)
        sys.exit(1)

    total_bytes = sum(size for _, _, size in files_to_archive)
    print(
        f"Discovered {len(files_to_archive)} files to export (Total size: {format_size(total_bytes)})."
    )

    print(f"Creating archive: {output_path}")
    t0 = time.time()
    try:
        with tarfile.open(output_path, "w:gz") as tar:
            for filepath, arcname, size in files_to_archive:
                if args.verbose or size > 10 * 1024 * 1024:
                    print(f"Adding: {arcname} ({format_size(size)})")
                tar.add(filepath, arcname=arcname)
    except Exception as e:
        print(f"Error creating archive: {e}", file=sys.stderr)
        sys.exit(1)

    duration = time.time() - t0
    archive_size = output_path.stat().st_size
    print(f"Export completed in {duration:.1f}s.")
    print(f"Archive file: {output_path} ({format_size(archive_size)})")

    # export instructions
    print()
    print("To restore on the target machine, copy the archive to the 'p0/' directory and run:")
    print(f"    tar -xzf {output_path.name}")


if __name__ == "__main__":
    main()
