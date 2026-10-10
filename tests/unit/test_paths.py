"""Tests for repository and application path resolution."""

from __future__ import annotations

from pathlib import Path

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

    def test_default_paths_anchor_the_source_checkout(self) -> None:
        """Verify source-mode DEFAULT_PATHS resolves to the checkout containing this test suite."""
        checkout = Path(__file__).resolve().parents[2]

        assert (checkout / "pyproject.toml").is_file()
        assert DEFAULT_PATHS.repository_root == checkout
        assert DEFAULT_PATHS.data_root == checkout / "data"
        assert (DEFAULT_PATHS.data_root / "champions_dex.json").is_file()
