from dataclasses import FrozenInstanceError

import pytest

from src.train.config import GlobalConfig, TrainingConfig, load_config


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
        config.training = TrainingConfig()
    with pytest.raises(FrozenInstanceError):
        config.training.n_envs = 1
