"""Benchmark fixed memory-reducer depth variants."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.profiler
from torch import Tensor

from p0.battle.actions import ACT_SIZE
from p0.model.architecture_contract import CURRENT_REDUCER_TOKEN_COUNT
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.policy import EncodedObs, PolicyNet
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.training.checkpoint import CheckpointStore
from p0.training.utils import default_device

BENCHMARK_SCHEMA = "p0.reducer_depth_benchmark.v1"
MODE_SETTINGS = (("baseline", 1), ("deep", None))
DTYPE_NAMES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}
DEFAULT_MODEL_CONFIG = ModelConfig.baseline()


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Inputs and timing controls for a reducer-depth benchmark."""

    dtype: str
    seed: int
    warmup: int
    iterations: int
    repeats: int
    batch_size: int
    time_steps: int
    d_model: int
    nhead: int
    dim_feedforward: int
    deep_reducer_layers: int
    device: str | None = None
    checkpoint: Path | None = None
    validation_artifact: Path | None = None

    def __post_init__(self) -> None:
        if type(self.dtype) is not str or self.dtype not in DTYPE_NAMES:
            raise ValueError(f"unsupported benchmark dtype {self.dtype!r}")
        if type(self.seed) is not int:
            raise ValueError("benchmark seed must be an integer")
        for name, value in (
            ("warmup", self.warmup),
            ("iterations", self.iterations),
            ("repeats", self.repeats),
            ("batch_size", self.batch_size),
            ("time_steps", self.time_steps),
            ("d_model", self.d_model),
            ("nhead", self.nhead),
            ("dim_feedforward", self.dim_feedforward),
            ("deep_reducer_layers", self.deep_reducer_layers),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"benchmark {name} must be a positive integer")
        if self.deep_reducer_layers < 2:
            raise ValueError("benchmark deep_reducer_layers must be at least two")
        if self.d_model % self.nhead:
            raise ValueError("benchmark d_model must be divisible by nhead")
        if (self.checkpoint is None) != (self.validation_artifact is None):
            raise ValueError(
                "checkpoint and validation_artifact must be supplied together for BC validation"
            )


@dataclass(frozen=True, slots=True)
class Metric:
    """A metric with an explicit availability gate."""

    status: str
    value: float | None = None
    reason: str | None = None

    @classmethod
    def unavailable(cls, reason: str) -> Metric:
        return cls(status="unavailable", reason=reason)

    @classmethod
    def available(cls, value: float) -> Metric:
        return cls(status="available", value=value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VariantResult:
    mode: str
    architecture: dict[str, int | bool]
    parameter_count: int
    samples_seconds_per_batch: tuple[float, ...]
    median_seconds_per_batch: float
    iqr_seconds_per_batch: float
    tokens_per_second: float
    peak_training_memory_bytes: int
    validation_bc_nll: Metric
    self_play_strength: Metric

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["validation_bc_nll"] = self.validation_bc_nll.to_dict()
        result["self_play_strength"] = self.self_play_strength.to_dict()
        return result


def _resolve_device(name: str | None) -> torch.device:
    if name is None:
        return default_device()
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"requested benchmark device {name!r} is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError(f"requested benchmark device {name!r} is unavailable")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary(samples: list[float]) -> tuple[float, float]:
    median = statistics.median(samples)
    if len(samples) < 2:
        return median, 0.0
    quartiles = statistics.quantiles(samples, n=4, method="inclusive")
    return median, quartiles[2] - quartiles[0]


def _make_config(benchmark: BenchmarkConfig, reducer_layers: int) -> ModelConfig:
    return ModelConfig(
        d_model=benchmark.d_model,
        nhead=benchmark.nhead,
        reducer_layers=reducer_layers,
        dim_feedforward=benchmark.dim_feedforward,
    )


def _repeat_memory(memory: tuple[Tensor, ...], repeats: int) -> tuple[Tensor, ...]:
    return tuple(value.repeat_interleave(repeats, dim=0) for value in memory)


def _build_inputs(
    policy: PolicyNet,
    benchmark: BenchmarkConfig,
    device: torch.device,
) -> tuple[EncodedObs, Tensor, tuple[Tensor, ...]]:
    observations = StructuredObservation.empty_batch(benchmark.batch_size).to(device)
    action_mask = torch.ones(
        (benchmark.batch_size, 2, ACT_SIZE), dtype=torch.bool, device=device
    )
    encoded = policy.encode(observations, action_mask)
    if benchmark.time_steps == 1:
        return encoded, action_mask, policy.empty_memory(benchmark.batch_size)
    return (
        EncodedObs(
            tokens=encoded.tokens.repeat_interleave(benchmark.time_steps, dim=0),
            aux=encoded.aux.repeat_interleave(benchmark.time_steps, dim=0),
            numerical=encoded.numerical.repeat_interleave(benchmark.time_steps, dim=0),
        ),
        action_mask.repeat_interleave(benchmark.time_steps, dim=0),
        _repeat_memory(policy.empty_memory(benchmark.batch_size), benchmark.time_steps),
    )


def _sample(
    policy: PolicyNet,
    encoded: EncodedObs,
    memory: tuple[Tensor, ...],
    device: torch.device,
    iterations: int,
) -> float:
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iterations):
            policy.actor.reducer(encoded.tokens, *memory)
    synchronize(device)
    return (time.perf_counter() - start) / iterations


def _peak_memory(
    policy: PolicyNet,
    encoded: EncodedObs,
    memory: tuple[Tensor, ...],
    device: torch.device,
) -> int:
    policy.zero_grad(set_to_none=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=False,
    ) as profile:
        reduced = policy.actor.reducer(encoded.tokens, *memory)
        reduced.cls.square().mean().backward()
    current = 0
    peak = 0
    for event in profile.events() or ():
        current += event.cpu_memory_usage
        peak = max(peak, current)
    policy.zero_grad(set_to_none=True)
    if device.type == "cuda":
        peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
    return max(1, peak)


def _load_validation_artifact(path: Path) -> dict[str, Tensor]:
    value = torch.load(path, weights_only=True, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError(f"Validation artifact {path} must be a mapping")
    expected = {
        "tokens",
        "aux",
        "numerical",
        "action_mask",
        "actions",
        "series_tokens",
        "series_mask",
        "history_tokens",
        "history_mask",
        "history_age_ids",
    }
    if set(value) != expected or not all(isinstance(value[name], Tensor) for name in expected):
        raise ValueError(f"Invalid validation artifact fields in {path}")
    return {name: value[name] for name in expected}


def _validation_metric(
    policy: PolicyNet,
    artifact: Mapping[str, Tensor] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> Metric:
    if artifact is None:
        return Metric.unavailable("compatible BC validation inputs were not supplied")
    encoded = EncodedObs(
        tokens=artifact["tokens"].to(device=device, dtype=dtype),
        aux=artifact["aux"].to(device=device, dtype=dtype),
        numerical=artifact["numerical"].to(device=device, dtype=dtype),
    )
    action_mask = artifact["action_mask"].to(device=device, dtype=torch.bool)
    actions = artifact["actions"].to(device=device, dtype=torch.long)
    memory = tuple(
        artifact[name].to(
            device=device,
            dtype=(
                torch.long
                if name == "history_age_ids"
                else torch.bool
                if name in {"series_mask", "history_mask"}
                else dtype
            ),
        )
        for name in (
            "series_tokens",
            "series_mask",
            "history_tokens",
            "history_mask",
            "history_age_ids",
        )
    )
    with torch.inference_mode():
        output = policy.evaluate(encoded, action_mask, actions, *memory)
    return Metric.available(float((-output.log_probs).mean().item()))


def _run_variant(
    benchmark: BenchmarkConfig,
    *,
    mode: str,
    reducer_layers: int,
    device: torch.device,
    dtype: torch.dtype,
    validation_artifact: Mapping[str, Tensor] | None,
    checkpoint_policy: PolicyNet | None,
) -> VariantResult:
    config = _make_config(benchmark, reducer_layers)
    policy = (
        checkpoint_policy
        if checkpoint_policy is not None and checkpoint_policy.config == config
        else build_policy(config, default_runtime_resources())
    ).to(device=device, dtype=dtype).eval()
    encoded, _, memory = _build_inputs(policy, benchmark, device)
    with torch.inference_mode():
        for _ in range(benchmark.warmup):
            policy.actor.reducer(encoded.tokens, *memory)
    samples = [
        _sample(policy, encoded, memory, device, benchmark.iterations)
        for _ in range(benchmark.repeats)
    ]
    median, iqr = _summary(samples)
    token_count = encoded.tokens.size(0) * CURRENT_REDUCER_TOKEN_COUNT
    return VariantResult(
        mode=mode,
        architecture=config.to_dict(),
        parameter_count=sum(parameter.numel() for parameter in policy.parameters()),
        samples_seconds_per_batch=tuple(samples),
        median_seconds_per_batch=median,
        iqr_seconds_per_batch=iqr,
        tokens_per_second=token_count / median,
        peak_training_memory_bytes=_peak_memory(policy, encoded, memory, device),
        validation_bc_nll=_validation_metric(policy, validation_artifact, device, dtype),
        self_play_strength=Metric.unavailable("self-play smoke configuration was not supplied"),
    )


def run_benchmark(benchmark: BenchmarkConfig) -> dict[str, Any]:
    """Run baseline and deeper fixed reducer variants."""
    device = _resolve_device(benchmark.device)
    dtype = DTYPE_NAMES[benchmark.dtype]
    cpu_rng_state = torch.random.get_rng_state()
    try:
        configs = tuple(
            _make_config(benchmark, benchmark.deep_reducer_layers if layers is None else layers)
            for _, layers in MODE_SETTINGS
        )
        checkpoint_policy = None
        if benchmark.checkpoint is not None:
            checkpoint_policy = CheckpointStore().load_policy(benchmark.checkpoint, device)
            if checkpoint_policy.config not in configs:
                raise ValueError(
                    "checkpoint architecture is incompatible with the requested benchmark variants"
                )
        validation_artifact = (
            _load_validation_artifact(benchmark.validation_artifact)
            if benchmark.validation_artifact is not None
            else None
        )
        variants = []
        for mode, layers in MODE_SETTINGS:
            torch.manual_seed(benchmark.seed)
            variants.append(
                _run_variant(
                    benchmark,
                    mode=mode,
                    reducer_layers=benchmark.deep_reducer_layers if layers is None else layers,
                    device=device,
                    dtype=dtype,
                    validation_artifact=validation_artifact,
                    checkpoint_policy=checkpoint_policy,
                ).to_dict()
            )
        return {
            "benchmark_schema": BENCHMARK_SCHEMA,
            "inputs": {
                **asdict(benchmark),
                "device": str(device),
                "checkpoint": None if benchmark.checkpoint is None else str(benchmark.checkpoint),
                "validation_artifact": (
                    None
                    if benchmark.validation_artifact is None
                    else str(benchmark.validation_artifact)
                ),
            },
            "variants": variants,
        }
    finally:
        torch.random.set_rng_state(cpu_rng_state)


def _config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
        batch_size=args.batch_size,
        time_steps=args.time_steps,
        d_model=args.d_model,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        deep_reducer_layers=args.deep_reducer_layers,
        checkpoint=args.checkpoint,
        validation_artifact=args.validation_artifact,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=sorted(DTYPE_NAMES), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--time-steps", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=DEFAULT_MODEL_CONFIG.d_model)
    parser.add_argument("--nhead", type=int, default=DEFAULT_MODEL_CONFIG.nhead)
    parser.add_argument("--dim-feedforward", type=int, default=DEFAULT_MODEL_CONFIG.dim_feedforward)
    parser.add_argument("--deep-reducer-layers", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--validation-artifact", type=Path)
    args = parser.parse_args()
    for name in (
        "warmup",
        "iterations",
        "repeats",
        "batch_size",
        "time_steps",
        "d_model",
        "nhead",
        "dim_feedforward",
        "deep_reducer_layers",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    try:
        _config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _print_metric(name: str, metric: dict[str, Any]) -> None:
    print(f"{name}_status={metric['status']}")
    if metric["status"] == "available":
        print(f"{name}_value={metric['value']}")
    else:
        print(f"{name}_reason={metric['reason']}")


def benchmark(args: argparse.Namespace) -> None:
    result = run_benchmark(_config_from_args(args))
    inputs = result["inputs"]
    print(
        f"device={inputs['device']} dtype={inputs['dtype']} batch={inputs['batch_size']} "
        f"time_steps={inputs['time_steps']} d_model={inputs['d_model']} "
        f"nhead={inputs['nhead']} dim_feedforward={inputs['dim_feedforward']} "
        f"deep_reducer_layers={inputs['deep_reducer_layers']} warmup={inputs['warmup']} "
        f"iterations={inputs['iterations']} repeats={inputs['repeats']}"
    )
    for variant in result["variants"]:
        print(f"variant={variant['mode']}")
        print(f"architecture={json.dumps(variant['architecture'], sort_keys=True)}")
        print(f"parameter_count={variant['parameter_count']}")
        print(
            "samples_seconds_per_batch="
            + ",".join(f"{sample:.8f}" for sample in variant["samples_seconds_per_batch"])
        )
        print(f"median_seconds_per_batch={variant['median_seconds_per_batch']:.8f}")
        print(f"iqr_seconds_per_batch={variant['iqr_seconds_per_batch']:.8f}")
        print(f"median_tokens_per_second={variant['tokens_per_second']:.2f}")
        print(f"peak_training_memory_bytes={variant['peak_training_memory_bytes']}")
        _print_metric("validation_bc_nll", variant["validation_bc_nll"])
        _print_metric("self_play_strength", variant["self_play_strength"])


if __name__ == "__main__":
    benchmark(parse_args())
