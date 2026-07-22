import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from p0.battle.actions import ACT_SIZE
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import NUMERICAL_WIDTH, SEQUENCE_LENGTH, StructuredObservation
from p0.training.checkpoint import DEFAULT_POLICY_STORE

_SPEC = importlib.util.spec_from_file_location(
    "benchmark_reducer_depth",
    Path(__file__).parents[1] / "bench" / "benchmark_reducer_depth.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
BenchmarkConfig = _MODULE.BenchmarkConfig
run_benchmark = _MODULE.run_benchmark


def _benchmark_config(**overrides):
    values = {
        "device": "cpu",
        "dtype": "float32",
        "seed": 7,
        "warmup": 1,
        "iterations": 1,
        "repeats": 2,
        "batch_size": 1,
        "time_steps": 1,
        "d_model": 8,
        "nhead": 2,
        "dim_feedforward": 32,
        "deep_reducer_layers": 2,
    }
    values.update(overrides)
    return BenchmarkConfig(**values)


def _small_policy(reducer_layers: int = 1):
    return build_policy(
        ModelConfig(8, 2, reducer_layers, 32),
        default_runtime_resources(),
    )


def test_benchmark_config_requires_explicit_and_compatible_inputs():
    with pytest.raises(ValueError, match="positive integer"):
        _benchmark_config(iterations=0)
    with pytest.raises(ValueError, match="supplied together"):
        _benchmark_config(checkpoint=Path("checkpoint.pt"))
    with pytest.raises(ValueError, match="at least two"):
        _benchmark_config(deep_reducer_layers=1)


def test_benchmark_uses_project_default_device_when_not_overridden():
    assert _MODULE._resolve_device(None) == _MODULE.default_device()


def test_benchmark_model_defaults_match_baseline_config():
    baseline = ModelConfig.baseline()

    assert _MODULE.DEFAULT_MODEL_CONFIG == baseline
    assert _MODULE.DEFAULT_MODEL_CONFIG.d_model == 512
    assert _MODULE.DEFAULT_MODEL_CONFIG.nhead == 8
    assert _MODULE.DEFAULT_MODEL_CONFIG.dim_feedforward == 2048
    assert _MODULE.DEFAULT_MODEL_CONFIG.reducer_layers == 5


def test_benchmark_cli_defaults_use_baseline_and_bench_timing(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark_reducer_depth"])

    args = _MODULE.parse_args()
    config = _MODULE._config_from_args(args)

    assert config.d_model == ModelConfig.baseline().d_model
    assert config.nhead == ModelConfig.baseline().nhead
    assert config.dim_feedforward == ModelConfig.baseline().dim_feedforward
    assert config.deep_reducer_layers == 3
    assert config.warmup == 10
    assert config.iterations == 10
    assert config.repeats == 5


def test_benchmark_covers_depths_and_labels_gated_metrics():
    result = run_benchmark(_benchmark_config())

    assert result["benchmark_schema"] == "p0.reducer_depth_benchmark.v1"
    assert len(result["variants"]) == 2
    assert {variant["mode"] for variant in result["variants"]} == {"baseline", "deep"}
    for variant in result["variants"]:
        assert variant["parameter_count"] > 0
        assert len(variant["samples_seconds_per_batch"]) == 2
        assert variant["median_seconds_per_batch"] > 0
        assert variant["iqr_seconds_per_batch"] >= 0
        assert variant["tokens_per_second"] > 0
        assert variant["peak_training_memory_bytes"] > 0
        assert variant["validation_bc_nll"]["status"] == "unavailable"
        assert variant["self_play_strength"]["status"] == "unavailable"


def test_benchmark_restores_global_rng_state_and_reports_architecture():
    torch.manual_seed(123)
    before = torch.random.get_rng_state()
    result = run_benchmark(_benchmark_config())
    after = torch.random.get_rng_state()

    assert torch.equal(after, before)
    baseline = next(variant for variant in result["variants"] if variant["mode"] == "baseline")
    assert baseline["architecture"]["reducer_layers"] == 1


def test_benchmark_rejects_incompatible_checkpoint(tmp_path):
    checkpoint = tmp_path / "incompatible.pt"
    DEFAULT_POLICY_STORE.save_policy(checkpoint, _small_policy(reducer_layers=4))

    with pytest.raises(ValueError, match="incompatible with the requested benchmark"):
        run_benchmark(
            _benchmark_config(
                checkpoint=checkpoint,
                validation_artifact=tmp_path / "validation.pt",
            )
        )


def test_benchmark_reports_validation_nll_for_compatible_artifacts(tmp_path):
    checkpoint = tmp_path / "baseline.pt"
    validation = tmp_path / "validation.pt"
    policy = _small_policy()
    observation = StructuredObservation.empty_batch(1)
    action_mask = torch.ones((1, 2, ACT_SIZE), dtype=torch.bool)
    encoded = policy.encode(observation, action_mask)
    memory = policy.empty_memory(1)
    torch.save(
        {
            "tokens": encoded.tokens,
            "aux": encoded.aux,
            "numerical": encoded.numerical,
            "action_mask": action_mask,
            "actions": torch.zeros((1, 2), dtype=torch.long),
            "series_tokens": memory[0],
            "series_mask": memory[1],
            "history_tokens": memory[2],
            "history_mask": memory[3],
            "history_age_ids": memory[4],
        },
        validation,
    )
    DEFAULT_POLICY_STORE.save_policy(checkpoint, policy)

    result = run_benchmark(_benchmark_config(checkpoint=checkpoint, validation_artifact=validation))

    assert all(
        variant["validation_bc_nll"]["status"] == "available"
        for variant in result["variants"]
    )
    assert all(variant["validation_bc_nll"]["value"] >= 0 for variant in result["variants"])
    assert encoded.numerical.shape == (1, SEQUENCE_LENGTH, NUMERICAL_WIDTH)
