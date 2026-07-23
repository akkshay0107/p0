"""Single ownership point for repository and application paths."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    repository_root: Path
    data_root: Path
    teams_root: Path
    artifacts_root: Path
    showdown_root: Path
    gauntlet_dir: Path
    checkpoint_path: Path
    runs_dir: Path
    replays_dir: Path
    backups_dir: Path
    log_path: Path
    resume_checkpoint: Path | None = None
    initial_policy_checkpoint: Path | None = None

    def __post_init__(self) -> None:
        if self.resume_checkpoint is not None and self.initial_policy_checkpoint is not None:
            raise ValueError(
                "paths.resume_checkpoint and paths.initial_policy_checkpoint are mutually exclusive"
            )

    @classmethod
    def from_root(cls, repository_root: str | Path) -> ProjectPaths:
        root = Path(repository_root).expanduser().resolve()
        artifacts = root / "artifacts"
        return cls(
            repository_root=root,
            data_root=root / "data",
            teams_root=root / "teams",
            artifacts_root=artifacts,
            showdown_root=root / "pokemon-showdown",
            gauntlet_dir=artifacts / "checkpoints" / "gauntlet",
            checkpoint_path=artifacts / "checkpoints" / "ppo_checkpoint.pt",
            runs_dir=artifacts / "runs",
            replays_dir=artifacts / "replays",
            backups_dir=artifacts / "backups",
            log_path=artifacts / "training.log",
            resume_checkpoint=None,
            initial_policy_checkpoint=None,
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
            return ProjectPaths(
                repository_root=paths.repository_root,
                data_root=data_root,
                teams_root=paths.teams_root,
                artifacts_root=paths.artifacts_root,
                showdown_root=paths.showdown_root,
                gauntlet_dir=paths.gauntlet_dir,
                checkpoint_path=paths.checkpoint_path,
                runs_dir=paths.runs_dir,
                replays_dir=paths.replays_dir,
                backups_dir=paths.backups_dir,
                log_path=paths.log_path,
                resume_checkpoint=paths.resume_checkpoint,
                initial_policy_checkpoint=paths.initial_policy_checkpoint,
            )
    return paths


DEFAULT_PATHS = _default_paths()
