"""Tests for repository and application path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from p0.paths import DEFAULT_PATHS, ProjectPaths


class TestProjectPaths:
    def test_from_root_anchors_expected_directories(self, tmp_path: Path) -> None:
        """Verify ProjectPaths.from_root sets up all default subdirectories under the provided root."""
        paths = ProjectPaths.from_root(tmp_path)

        assert paths.repository_root == tmp_path.resolve()
        assert paths.data_root == tmp_path.resolve() / "data"
        assert paths.teams_root == tmp_path.resolve() / "teams"
        assert paths.artifacts_root == tmp_path.resolve() / "artifacts"
        assert paths.showdown_root == tmp_path.resolve() / "pokemon-showdown"
        assert (
            paths.checkpoint_path
            == tmp_path.resolve() / "artifacts" / "checkpoints" / "ppo_checkpoint.pt"
        )
        assert paths.runs_dir == tmp_path.resolve() / "artifacts" / "runs"
        assert paths.replays_dir == tmp_path.resolve() / "artifacts" / "replays"
        assert paths.log_path == tmp_path.resolve() / "artifacts" / "training.log"
        assert paths.resume_checkpoint is None
        assert paths.initial_policy_checkpoint is None

    def test_mutually_exclusive_checkpoints_raise_value_error(self, tmp_path: Path) -> None:
        """Verify that setting both resume_checkpoint and initial_policy_checkpoint raises ValueError."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            ProjectPaths(
                repository_root=tmp_path,
                data_root=tmp_path / "data",
                teams_root=tmp_path / "teams",
                artifacts_root=tmp_path / "artifacts",
                showdown_root=tmp_path / "pokemon-showdown",
                checkpoint_path=tmp_path / "artifacts" / "ppo.pt",
                runs_dir=tmp_path / "artifacts" / "runs",
                replays_dir=tmp_path / "artifacts" / "replays",
                log_path=tmp_path / "artifacts" / "training.log",
                resume_checkpoint=tmp_path / "resume.pt",
                initial_policy_checkpoint=tmp_path / "initial.pt",
            )

    def test_default_paths_anchor_repository(self) -> None:
        """Verify DEFAULT_PATHS points to an existing project root containing data and manifest."""
        assert DEFAULT_PATHS.repository_root.is_dir()
        assert DEFAULT_PATHS.data_root.is_dir()
        assert (DEFAULT_PATHS.data_root / "runtime_manifest.json").is_file()
