"""Tests for application configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from p0.training.config import (
    BCConfig,
    GlobalConfig,
    PPOConfig,
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

    def test_empty_file_keeps_every_default(self, tmp_path: Path) -> None:
        config = load_config(write_config(tmp_path, "{}\n"))

        assert isinstance(config, GlobalConfig)
        assert config.ppo.num_episodes == 2000
        assert config.ppo.gamma == 0.99
        assert config.ppo.output_dir == ROOT / "artifacts" / "runs" / "ppo"
        assert config.bc.batch_decisions == 256
        assert config.bc.dataset_dir == ROOT / "artifacts" / "datasets"
        assert config.bot.username == "Bot"

    def test_partial_yaml_changes_only_the_named_settings(self, tmp_path: Path) -> None:
        config = load_config(
            write_config(
                tmp_path,
                """
                ppo:
                  n_envs: 4
                  magnet_alpha: 0.05
                bc:
                  epochs: 3
                bot:
                  username: Tester
                """,
            )
        )

        assert config.ppo.n_envs == 4
        assert config.ppo.magnet_alpha == 0.05
        assert config.ppo.rollout_steps == 320
        assert config.bc.epochs == 3
        assert config.bc.learning_rate == 3e-4
        assert config.bot.username == "Tester"

    def test_shared_settings_reach_both_trainers(self, tmp_path: Path) -> None:
        """gamma, seed, and enable_optim are written once and read by PPO and BC."""
        config = load_config(write_config(tmp_path, "gamma: 0.95\nseed: 7\nenable_optim: false\n"))

        assert (config.ppo.gamma, config.bc.gamma) == (0.95, 0.95)
        assert (config.ppo.seed, config.bc.seed) == (7, 7)
        assert (config.ppo.enable_optim, config.bc.enable_optim) == (False, False)

    @pytest.mark.parametrize(
        ("label", "contents", "message"),
        [
            ("unknown ppo setting", "ppo:\n  unknown_value: 1\n", "unknown ppo setting"),
            ("unknown bc setting", "bc:\n  bogus: 1\n", "unknown bc setting"),
            ("unknown bot setting", "bot:\n  battle_format: x\n", "unknown bot setting"),
            ("unknown root setting", "bo3: 1\n", "unknown root setting"),
            ("old training section", "training:\n  n_envs: 2\n", "unknown root setting"),
            ("old paths section", "paths:\n  runs_dir: runs\n", "unknown root setting"),
            ("old evaluation section", "evaluation:\n  seed: 1\n", "unknown root setting"),
            ("ppo value fixed in code", "ppo:\n  clip_range: 0.3\n", "unknown ppo setting"),
            ("bc value fixed in code", "bc:\n  weight_decay: 0.1\n", "unknown bc setting"),
            ("gamma is a root setting", "bc:\n  gamma: 0.9\n", "unknown bc setting"),
            ("seed is a root setting", "ppo:\n  seed: 3\n", "unknown ppo setting"),
            ("bot option that is a p0-play flag", "bot:\n  top_p: 0.5\n", "unknown bot setting"),
            ("section is not a mapping", "ppo: 3\n", "ppo must be a mapping"),
            ("false section is not a mapping", "ppo: false\n", "ppo must be a mapping"),
            ("run too short for warmup", "ppo:\n  num_episodes: 5\n", "too short"),
            (
                "resume and initial policy together",
                "ppo:\n  resume_checkpoint: a.pt\n  initial_policy_checkpoint: b.pt\n",
                "mutually exclusive",
            ),
            (
                "one server url without the other",
                "bot:\n  websocket_url: ws://localhost:8000\n",
                "must be configured together",
            ),
        ],
    )
    def test_load_config_rejects_invalid_files_with_specific_errors(
        self, tmp_path: Path, label: str, contents: str, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            load_config(write_config(tmp_path, contents))

    def test_config_is_immutable(self, tmp_path: Path) -> None:
        """Verify configuration values cannot be changed after construction."""
        config = load_config(write_config(tmp_path, "{}\n"))

        with pytest.raises(AttributeError):
            setattr(config, "ppo", PPOConfig())
        with pytest.raises(AttributeError):
            setattr(config.ppo, "n_envs", 1)

    def test_relative_paths_start_at_the_repository_root(self, tmp_path: Path) -> None:
        config = load_config(
            write_config(
                tmp_path,
                """
                ppo:
                  output_dir: relative-run
                  resume_checkpoint: /absolute/ppo.pt
                bc:
                  dataset_dir: relative-data
                  output_dir: relative-bc
                """,
            )
        )

        assert config.ppo.output_dir == ROOT / "relative-run"
        assert config.ppo.resume_checkpoint == Path("/absolute/ppo.pt")
        assert config.ppo.initial_policy_checkpoint is None
        assert config.bc.dataset_dir == ROOT / "relative-data"
        assert config.bc.output_dir == ROOT / "relative-bc"

    def test_ppo_checkpoint_is_written_inside_the_output_directory(self, tmp_path: Path) -> None:
        assert PPOConfig(output_dir=tmp_path).checkpoint_path == tmp_path / "ppo_checkpoint.pt"

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("learning_rate", float("nan")),
            ("learning_rate", float("inf")),
            ("max_grad_norm", float("nan")),
            ("max_grad_norm", float("inf")),
            ("value_coef", float("nan")),
            ("value_coef", float("inf")),
            ("weight_decay", float("nan")),
            ("weight_decay", float("inf")),
        ],
    )
    def test_bc_config_rejects_nonfinite_optimizer_values(
        self,
        field: str,
        value: float,
    ) -> None:
        values: dict[str, Any] = {field: value}
        with pytest.raises(ValueError, match=field):
            BCConfig(**values)
