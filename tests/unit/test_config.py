"""Tests for application configuration loading and validation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from p0.format_config import FORMAT
from p0.training.config import (
    BCConfig,
    BotConfig,
    GlobalConfig,
    TrainingConfig,
    load_config,
)

ROOT = Path(__file__).resolve().parents[2]


def write_config(tmp_path: Path, contents: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(contents, encoding="utf-8")
    return path


class TestConfig:
    def test_load_config_requires_file(self, tmp_path: Path) -> None:
        """Verify load_config raises FileNotFoundError when the file does not exist."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_config(tmp_path / "missing.yaml")

    def test_load_config_applies_partial_yaml_to_source_defaults(self, tmp_path: Path) -> None:
        """Verify partial YAML overrides update specified sections while retaining defaults."""
        config = load_config(
            write_config(tmp_path, "training:\n  n_envs: 8\n  magnet_alpha: 0.05\n")
        )

        assert isinstance(config, GlobalConfig)
        assert config.training.n_envs == 8
        assert config.training.magnet_alpha == 0.05
        assert config.training.num_episodes == TrainingConfig().num_episodes
        assert config.training.magnet_refresh_interval == TrainingConfig().magnet_refresh_interval

    @pytest.mark.parametrize(
        ("label", "contents", "message"),
        [
            (
                "unknown training field",
                "training:\n  unknown_value: 1\n",
                "unknown TrainingConfig field",
            ),
            (
                "magnet refresh exceeds episodes",
                "training:\n  num_episodes: 10\n  magnet_refresh_interval: 20\n",
                "magnet_refresh_interval",
            ),
            (
                "mismatched bot format",
                "bot:\n  battle_format: gen9anythinggoes\n",
                "battle_format",
            ),
            (
                "removed bo3 switch",
                "bo3: 1\n",
                "unknown root configuration section",
            ),
        ],
    )
    def test_load_config_rejects_invalid_contracts_with_specific_errors(
        self, tmp_path: Path, label: str, contents: str, message: str
    ) -> None:
        """Verify load_config detects and rejects schema violations with informative errors."""
        with pytest.raises(ValueError, match=message):
            load_config(write_config(tmp_path, contents))

    def test_bot_config_accepts_configured_bo3_format(self) -> None:
        """Verify live bot configuration exposes the checked-in Bo3 format."""
        config = BotConfig()
        assert config.battle_format == FORMAT.bo3_format

    def test_bot_config_rejects_bo1_format(self) -> None:
        """Verify bot configuration rejects non-Bo3 battle formats."""
        with pytest.raises(ValueError, match="must be the Bo3 format"):
            BotConfig(battle_format=FORMAT.battle_format)

    def test_bot_config_rejects_live_concurrency_above_one(self) -> None:
        """Live Bo3 series tracking is intentionally single-battle."""
        with pytest.raises(ValueError, match="fixed at 1"):
            BotConfig(max_concurrent_battles=2)

    def test_config_is_immutable(self, tmp_path: Path) -> None:
        """Verify GlobalConfig dataclasses are frozen against accidental mutation."""
        config = load_config(write_config(tmp_path, "{}\n"))

        with pytest.raises(FrozenInstanceError):
            setattr(config, "training", TrainingConfig())
        with pytest.raises(FrozenInstanceError):
            setattr(config.training, "n_envs", 1)

    def test_paths_and_team_pool_paths_resolve_once_from_project_root(self, tmp_path: Path) -> None:
        """Verify relative paths in YAML resolve deterministically against repository root."""
        config = load_config(
            write_config(
                tmp_path,
                """
    paths:
      data_root: relative-data
    teams:
      all: team-pool
      reduced: reduced-pool
    """,
            )
        )

        assert config.paths.repository_root.is_absolute()
        assert config.paths.data_root == (ROOT / "relative-data").resolve()
        assert config.teams.all == (ROOT / "teams" / "team-pool").resolve()
        assert config.teams.reduced == (ROOT / "teams" / "reduced-pool").resolve()

    def test_config_sections(self, tmp_path: Path) -> None:
        """Verify config sections load correctly and reject mismatched objectives."""
        config = load_config("config.example.yaml")
        assert config.bc.batch_decisions == 256
        assert config.bc.gamma == config.training.gamma
        assert config.bc.value_coef == config.training.value_coef
        assert config.teams.all == (ROOT / "teams" / "all").resolve()
        assert config.teams.reduced == (ROOT / "teams" / "reduced").resolve()
        assert config.evaluation.episodes_per_matchup == 20

        bad = tmp_path / "config.yaml"
        bad.write_text("bc:\n  bogus: 1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unknown BCConfig field"):
            load_config(bad)

        duplicate_objective = tmp_path / "duplicate-objective.yaml"
        duplicate_objective.write_text(
            "training:\n  gamma: 0.95\nbc:\n  gamma: 0.9\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="bc.gamma is derived from training"):
            load_config(duplicate_objective)

        with pytest.raises(ValueError, match="bc.gamma must match training.gamma"):
            GlobalConfig(training=TrainingConfig(gamma=0.95), bc=BCConfig(gamma=0.9))
