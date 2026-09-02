"""Tests for model and bot configuration loading."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from p0.format_config import (
    FORMAT,
)
from p0.model.config import ModelConfig
from p0.training.config import (
    BotConfig,
    GlobalConfig,
    TrainingConfig,
    load_config,
)


def write_config(tmp_path: Path, contents: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(contents, encoding="utf-8")
    return path


def _resources(
    tmp_path: Path, *, extra_species: bool = False, base_power: int = 90
) -> tuple[Path, Path]:
    vocab = tmp_path / "vocab.json"
    species = {"pikachu": 1}
    if extra_species:
        species["raichu"] = 2
    vocab.write_text(json.dumps({"species": species}), encoding="utf-8")
    dex = tmp_path / "champions_dex.json"
    dex.write_text(json.dumps({"moves": [{"id": "test", "basePower": base_power}]}))
    return vocab, dex


ROOT = Path(__file__).resolve().parents[2]


def _runtime_files(tmp_path: Path) -> tuple[Path, Path]:
    vocab = tmp_path / "vocab.json"
    dex = tmp_path / "champions_dex.json"
    vocab.write_text(
        json.dumps({"species": {"pikachu": 1}, "moves": {"tackle": 1}}), encoding="utf-8"
    )
    dex.write_text('{"pikachu":{"base_stats":{"hp":35}}}', encoding="utf-8")
    return vocab, dex


class TestConfig:
    def test_load_config_requires_file(self, tmp_path: Path) -> None:
        """Verify load_config raises FileNotFoundError when the specified config file does not exist."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_config(tmp_path / "missing.yaml")

    def test_load_config_applies_partial_yaml_to_source_defaults(self, tmp_path: Path) -> None:
        """Verify that partial YAML overrides selectively update target configuration sections while retaining default values."""
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
        """Verify load_config detects and rejects schema violations with informative error messages."""
        with pytest.raises(ValueError, match=message):
            load_config(write_config(tmp_path, contents))

    def test_bot_config_accepts_configured_bo3_format(self) -> None:
        """Verify live bot configuration exposes the checked-in Bo3 format."""
        config = BotConfig()

        assert config.battle_format == FORMAT.bo3_format

    def test_bot_config_rejects_bo1_format(self) -> None:
        """Verify the live bot configuration rejects formats unsupported by RLPlayer."""
        with pytest.raises(ValueError, match="must be the Bo3 format"):
            BotConfig(battle_format=FORMAT.battle_format)

    def test_bot_config_rejects_live_concurrency_above_one(self) -> None:
        """Live Bo3 series tracking is intentionally single-battle for now."""
        with pytest.raises(ValueError, match="fixed at 1"):
            BotConfig(max_concurrent_battles=2)

    def test_config_is_immutable(self, tmp_path: Path) -> None:
        """Verify GlobalConfig dataclasses are frozen to prevent accidental in-place mutation during execution."""
        config = load_config(write_config(tmp_path, "{}\n"))

        with pytest.raises(FrozenInstanceError):
            setattr(config, "training", TrainingConfig())
        with pytest.raises(FrozenInstanceError):
            setattr(config.training, "n_envs", 1)

    def test_paths_and_team_pool_paths_resolve_once_from_project_root(self, tmp_path: Path) -> None:
        """Verify relative paths configured in YAML resolve deterministically against the project root directory."""
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
        assert config.paths.data_root == (Path(__file__).parents[2] / "relative-data").resolve()
        assert config.teams.all == (Path(__file__).parents[2] / "teams" / "team-pool").resolve()
        assert (
            config.teams.reduced == (Path(__file__).parents[2] / "teams" / "reduced-pool").resolve()
        )

    def test_model_config_is_checkpoint_local_and_validated(self) -> None:
        """Verify ModelConfig enforces divisibility requirements (d_model % nhead == 0)."""
        config = ModelConfig.baseline()
        assert config.d_model == 384
        assert config.reducer_layers == 8
        assert config.dim_feedforward == 1536
        with pytest.raises(ValueError, match="divisible"):
            ModelConfig(63, 8, 1, 256)
