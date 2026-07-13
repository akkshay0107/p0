from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from p0.model.config import ModelConfig
from p0.train.config import GlobalConfig, TrainingConfig, load_config


def write_config(tmp_path, contents: str):
    path = tmp_path / "config.yaml"
    path.write_text(contents, encoding="utf-8")
    return path


def test_load_config_requires_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config(tmp_path / "missing.yaml")


def test_load_config_applies_partial_yaml_to_source_defaults(tmp_path):
    config = load_config(write_config(tmp_path, "training:\n  n_envs: 8\n  n_self_envs: 2\n"))

    assert isinstance(config, GlobalConfig)
    assert config.training.n_envs == 8
    assert config.training.n_self_envs == 2
    assert config.training.num_episodes == TrainingConfig().num_episodes
    assert config.pool.pool_size == 50


def test_load_config_rejects_unknown_fields(tmp_path):
    path = write_config(tmp_path, "training:\n  unknown_value: 1\n")

    with pytest.raises(ValueError, match="unknown TrainingConfig field"):
        load_config(path)


def test_load_config_rejects_invalid_values(tmp_path):
    path = write_config(tmp_path, "training:\n  n_envs: 2\n  n_self_envs: 3\n")

    with pytest.raises(ValueError, match="n_self_envs"):
        load_config(path)


def test_config_is_immutable(tmp_path):
    config = load_config(write_config(tmp_path, "{}\n"))

    with pytest.raises(FrozenInstanceError):
        setattr(config, "training", TrainingConfig())
    with pytest.raises(FrozenInstanceError):
        setattr(config.training, "n_envs", 1)


def test_paths_and_team_source_paths_resolve_once_from_project_root(tmp_path):
    config = load_config(
        write_config(
            tmp_path,
            """
paths:
  data_root: relative-data
environment:
  agent_team_source:
    path: team-pool
""",
        )
    )

    assert config.paths.repository_root.is_absolute()
    assert config.paths.data_root == (Path(__file__).parents[1] / "relative-data").resolve()
    assert config.environment.agent_team_source.path == (
        Path(__file__).parents[1] / "team-pool"
    ).resolve()


def test_team_source_config_rejects_unknown_kind(tmp_path):
    path = write_config(
        tmp_path,
        "environment:\n  agent_team_source:\n    kind: directory_magic\n",
    )
    with pytest.raises(ValueError, match="file_pool"):
        load_config(path)


def test_bot_format_must_match_application_format(tmp_path):
    path = write_config(tmp_path, "bot:\n  battle_format: gen9anythinggoes\n")
    with pytest.raises(ValueError, match="battle_format"):
        load_config(path)


def test_model_config_is_checkpoint_local_and_validated():
    config = ModelConfig.baseline()
    assert config.d_model == 512
    assert config.history_tokens == 8
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(63, 8, 1, 8, 256)
