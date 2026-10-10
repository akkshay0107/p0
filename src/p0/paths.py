"""Single ownership point for repository and application paths."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    repository_root: Path
    data_root: Path
    teams_root: Path
    artifacts_root: Path
    showdown_root: Path

    @classmethod
    def from_root(cls, repository_root: str | Path) -> ProjectPaths:
        """Construct ProjectPaths anchored at repository_root."""
        root = Path(repository_root).expanduser().resolve()

        return cls(
            repository_root=root,
            data_root=root / "data",
            teams_root=root / "teams",
            artifacts_root=root / "artifacts",
            showdown_root=root / "pokemon-showdown",
        )


def _default_paths() -> ProjectPaths:
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").is_file():
        return ProjectPaths.from_root(source_root)

    paths = ProjectPaths.from_root(Path.cwd())
    candidates = (
        Path(__file__).resolve().parents[1] / "share" / "p0",
        Path(sys.prefix) / "share" / "p0",
    )

    for data_root in candidates:
        if (data_root / "vocab.json").is_file():
            return replace(paths, data_root=data_root)

    return paths


DEFAULT_PATHS = _default_paths()
