import re
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from p0.model.config import ModelConfig
from p0.training.config import GlobalConfig, TrainingConfig, load_config


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


def test_load_config_rejects_invalid_contracts_with_specific_errors(tmp_path):
    cases = (
        (
            "unknown training field",
            "training:\n  unknown_value: 1\n",
            "unknown TrainingConfig field",
        ),
        ("invalid environment split", "training:\n  n_envs: 2\n  n_self_envs: 3\n", "n_self_envs"),
        (
            "removed team-source kind",
            "environment:\n  agent_team_source:\n    kind: directory_magic\n",
            "unknown TeamSourceConfig field",
        ),
        (
            "mismatched bot format",
            "bot:\n  battle_format: gen9anythinggoes\n",
            "battle_format",
        ),
    )
    for label, contents, message in cases:
        try:
            load_config(write_config(tmp_path, contents))
        except ValueError as exc:
            assert re.search(message, str(exc)), f"{label}: unexpected error: {exc}"
        else:
            pytest.fail(f"{label}: expected ValueError")


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
    assert (
        config.environment.agent_team_source.path
        == (Path(__file__).parents[1] / "teams" / "team-pool").resolve()
    )


def test_model_config_is_checkpoint_local_and_validated():
    config = ModelConfig.baseline()
    assert config.d_model == 512
    assert config.history_tokens == 8
    assert config.prelude_layers == 1
    assert config.coda_layers == 1
    assert config.core_repeats == 1
    assert not config.core_weights_tied
    assert config.pass_embedding_enabled
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(63, 8, 1, 8, 256)


def test_model_config_depth_round_trip_and_supported_variants():
    config = ModelConfig(
        d_model=64,
        nhead=4,
        prelude_layers=1,
        history_tokens=8,
        dim_feedforward=128,
        coda_layers=1,
        core_repeats=3,
        core_weights_tied=True,
        pass_embedding_enabled=False,
    )

    assert ModelConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"core_repeats": 0}, "core_repeats"),
        ({"prelude_layers": 2}, "prelude_layers"),
        ({"coda_layers": 2}, "coda_layers"),
        ({"core_weights_tied": True}, "core_weights_tied"),
    ),
)
def test_model_config_rejects_unsupported_depth_combinations(kwargs, message):
    config_payload = ModelConfig(32, 4, 1, 8, 128).to_dict()
    config_payload.update(kwargs)

    with pytest.raises(ValueError, match=message):
        ModelConfig.from_dict(config_payload)


@pytest.mark.parametrize(
    "field",
    (
        "d_model",
        "nhead",
        "prelude_layers",
        "history_tokens",
        "dim_feedforward",
        "coda_layers",
        "core_repeats",
        "series_tokens",
    ),
)
def test_model_config_rejects_boolean_integer_fields(field):
    payload = ModelConfig.baseline().to_dict()
    payload[field] = True

    with pytest.raises(ValueError, match="positive integer"):
        ModelConfig.from_dict(payload)


@pytest.mark.parametrize(
    "field",
    ("core_weights_tied", "pass_embedding_enabled", "series_context_enabled"),
)
def test_model_config_rejects_non_boolean_flags(field):
    payload = ModelConfig.baseline().to_dict()
    payload[field] = 1

    with pytest.raises(ValueError, match="boolean"):
        ModelConfig.from_dict(payload)


def test_model_config_rejects_malformed_missing_and_unknown_fields():
    payload = ModelConfig.baseline().to_dict()
    del payload["prelude_layers"]
    payload["unexpected"] = 1

    with pytest.raises(
        ValueError,
        match=r"missing=\['prelude_layers'\], unknown=\['unexpected'\]",
    ):
        ModelConfig.from_dict(payload)
