"""Benchmark the unified reducer depth variants."""

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
import torch.nn as nn
import torch.profiler
from torch import Tensor

from p0.battle.actions import ACT_SIZE
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.policy import EncodedObs, PolicyNet
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import NUMERICAL_WIDTH, SEQUENCE_LENGTH
from p0.training.checkpoint import CheckpointStore
from p0.training.utils import default_device

BENCHMARK_SCHEMA = "p0.reducer_depth_benchmark.v1"
PASS_EMBEDDING_SETTINGS = (False, True)
MODE_SETTINGS = (
    ("baseline_untied", 1, False),
    ("deeper_untied", None, False),
    ("tied_core", None, True),
)
DTYPE_NAMES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}
DEFAULT_MODEL_CONFIG = ModelConfig.baseline()


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Inputs and timing controls for a reducer benchmark."""

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
    history_tokens: int
    deep_core_repeats: int
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
            ("history_tokens", self.history_tokens),
            ("deep_core_repeats", self.deep_core_repeats),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"benchmark {name} must be a positive integer")
        if self.deep_core_repeats < 2:
            raise ValueError("benchmark deep_core_repeats must be at least two")
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
    """One reducer mode and pass-embedding setting."""

    mode: str
    architecture: dict[str, int | bool]
    parameter_count: int
    samples_seconds_per_batch: tuple[float, ...]
    median_seconds_per_batch: float
    iqr_seconds_per_batch: float
    tokens_per_second: float
    peak_bptt_memory_bytes: int
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


def _sample(
    reducer: nn.Module,
    tokens: Tensor,
    state: Tensor,
    device: torch.device,
    iterations: int,
) -> float:
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iterations):
            _run_reducer_sequence(reducer, tokens, state)
    synchronize(device)
    return (time.perf_counter() - start) / iterations


def _summary(samples: list[float]) -> tuple[float, float]:
    median = statistics.median(samples)
    if len(samples) < 2:
        return median, 0.0
    quartiles = statistics.quantiles(samples, n=4, method="inclusive")
    return median, quartiles[2] - quartiles[0]


def _make_config(
    benchmark: BenchmarkConfig,
    *,
    core_repeats: int,
    core_weights_tied: bool,
    pass_embedding_enabled: bool,
) -> ModelConfig:
    return ModelConfig(
        d_model=benchmark.d_model,
        nhead=benchmark.nhead,
        prelude_layers=1,
        history_tokens=benchmark.history_tokens,
        dim_feedforward=benchmark.dim_feedforward,
        coda_layers=1,
        core_repeats=core_repeats,
        core_weights_tied=core_weights_tied,
        pass_embedding_enabled=pass_embedding_enabled,
    )


def _build_tokens(
    policy: PolicyNet,
    benchmark: BenchmarkConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    sequence_length = policy.actor.reducer.seq_len
    tokens = torch.randn(
        benchmark.batch_size,
        benchmark.time_steps,
        sequence_length,
        benchmark.d_model,
        device=device,
        dtype=dtype,
    )
    state = policy.initial_state(benchmark.batch_size).to(device=device, dtype=dtype)
    return tokens, state


def _run_reducer_sequence(
    reducer: nn.Module,
    tokens: Tensor,
    state: Tensor,
) -> tuple[Tensor, Tensor]:
    loss = tokens.new_zeros(())
    for time_index in range(tokens.size(1)):
        cls, state, _ = reducer(tokens[:, time_index], state)
        loss = loss + cls.square().mean()
    return state, loss


def _peak_cpu_bptt_memory(
    reducer: nn.Module,
    tokens: Tensor,
    state: Tensor,
) -> int:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=False,
        acc_events=True,
    ) as profile:
        _, loss = _run_reducer_sequence(reducer, tokens, state)
        loss.backward()

    current = 0
    peak = 0
    for event in profile.events() or ():
        current += event.cpu_memory_usage
        peak = max(peak, current)
    return max(0, peak)


def _peak_bptt_memory(
    reducer: nn.Module,
    tokens: Tensor,
    state: Tensor,
    device: torch.device,
) -> int:
    reducer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        _, loss = _run_reducer_sequence(reducer, tokens, state)
        loss.backward()
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        peak = _peak_cpu_bptt_memory(reducer, tokens, state)
    reducer.zero_grad(set_to_none=True)
    return peak


def _load_validation_artifact(path: Path) -> dict[str, Tensor]:
    try:
        value = torch.load(path, weights_only=True, map_location="cpu")
    except (OSError, RuntimeError, EOFError, ValueError, IndexError) as exc:
        raise ValueError(f"Unable to read validation artifact {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"Validation artifact {path} must be a mapping")
    expected = {"tokens", "aux", "numerical", "action_mask", "actions", "state"}
    unknown = sorted(
        (key for key in value if type(key) is not str or key not in expected),
        key=repr,
    )
    missing = sorted(expected - set(value))
    if unknown or missing:
        raise ValueError(
            f"Invalid validation artifact fields: missing={missing}, unknown={unknown}"
        )
    if not all(isinstance(value[name], Tensor) for name in expected):
        raise ValueError(f"Validation artifact {path} must contain tensors only")
    return {name: value[name] for name in expected}


def _validate_validation_artifact(
    artifact: Mapping[str, Tensor],
    policy: PolicyNet,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[EncodedObs, Tensor, Tensor, Tensor]:
    tokens = artifact["tokens"].to(device=device, dtype=dtype)
    aux = artifact["aux"].to(device=device, dtype=dtype)
    numerical = artifact["numerical"].to(device=device, dtype=dtype)
    action_mask = artifact["action_mask"].to(device=device, dtype=torch.bool)
    actions = artifact["actions"].to(device=device, dtype=torch.long)
    state = artifact["state"].to(device=device, dtype=dtype)
    batch_size = tokens.size(0)
    if tokens.shape != (
        batch_size,
        policy.actor.reducer.seq_len,
        policy.d_model,
    ):
        raise ValueError("validation tokens are incompatible with the benchmark architecture")
    if aux.dim() != 4 or aux.size(0) != batch_size or aux.size(1) != 2:
        raise ValueError("validation aux has an incompatible shape")
    if numerical.shape != (batch_size, SEQUENCE_LENGTH, NUMERICAL_WIDTH):
        raise ValueError("validation numerical features have an incompatible shape")
    if action_mask.shape != (batch_size, 2, ACT_SIZE):
        raise ValueError("validation action_mask has an incompatible shape")
    if actions.shape != (batch_size, 2):
        raise ValueError("validation actions have an incompatible shape")
    if state.shape != (batch_size, policy.config.history_tokens, policy.d_model):
        raise ValueError("validation recurrent state has an incompatible shape")
    return EncodedObs(tokens=tokens, aux=aux, numerical=numerical), state, action_mask, actions


def _checkpoint_policy(
    benchmark: BenchmarkConfig,
    device: torch.device,
    dtype: torch.dtype,
    configs: tuple[ModelConfig, ...],
) -> PolicyNet | None:
    if benchmark.checkpoint is None:
        return None
    policy = CheckpointStore().load_policy(benchmark.checkpoint, device)
    if policy.config not in configs:
        raise ValueError(
            "checkpoint architecture is incompatible with the requested benchmark variants"
        )
    return policy.to(device=device, dtype=dtype)


def _validation_metric(
    policy: PolicyNet,
    artifact: Mapping[str, Tensor] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> Metric:
    if artifact is None:
        return Metric.unavailable("compatible BC validation inputs were not supplied")
    enc, state, action_mask, actions = _validate_validation_artifact(
        artifact, policy, device, dtype
    )
    with torch.inference_mode():
        output = policy.evaluate(enc, action_mask, actions, state)
    return Metric.available(float((-output.log_probs).mean().item()))


def _snapshot_state(policy: PolicyNet) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in policy.state_dict().items()}


def _assert_state_unchanged(policy: PolicyNet, snapshot: Mapping[str, Tensor]) -> None:
    current = policy.state_dict()
    if set(current) != set(snapshot) or any(
        not torch.equal(current[name], snapshot[name]) for name in snapshot
    ):
        raise RuntimeError("benchmark mutated model state")


def _run_variant(
    benchmark: BenchmarkConfig,
    *,
    mode: str,
    core_repeats: int,
    core_weights_tied: bool,
    pass_embedding_enabled: bool,
    device: torch.device,
    dtype: torch.dtype,
    validation_artifact: Mapping[str, Tensor] | None,
    checkpoint_policy: PolicyNet | None,
) -> VariantResult:
    config = _make_config(
        benchmark,
        core_repeats=core_repeats,
        core_weights_tied=core_weights_tied,
        pass_embedding_enabled=pass_embedding_enabled,
    )
    policy = (
        checkpoint_policy
        if checkpoint_policy is not None and checkpoint_policy.config == config
        else build_policy(config, default_runtime_resources())
    )
    policy = policy.to(device=device, dtype=dtype).eval()
    tokens, initial_state = _build_tokens(policy, benchmark, device, dtype)
    snapshot = _snapshot_state(policy)
    reducer = policy.actor.reducer

    with torch.inference_mode():
        for _ in range(benchmark.warmup):
            _run_reducer_sequence(reducer, tokens, initial_state)
    synchronize(device)
    samples = [
        _sample(
            reducer,
            tokens,
            initial_state,
            device,
            benchmark.iterations,
        )
        for _ in range(benchmark.repeats)
    ]
    median, iqr = _summary(samples)
    token_count = benchmark.batch_size * benchmark.time_steps * reducer.seq_len
    peak_memory = _peak_bptt_memory(reducer, tokens, initial_state, device)
    _assert_state_unchanged(policy, snapshot)

    return VariantResult(
        mode=f"{mode}_{'pass' if pass_embedding_enabled else 'no_pass'}",
        architecture=config.to_dict(),
        parameter_count=sum(parameter.numel() for parameter in policy.parameters()),
        samples_seconds_per_batch=tuple(samples),
        median_seconds_per_batch=median,
        iqr_seconds_per_batch=iqr,
        tokens_per_second=token_count / median,
        peak_bptt_memory_bytes=peak_memory,
        validation_bc_nll=_validation_metric(
            policy, validation_artifact, device, dtype
        ),
        self_play_strength=Metric.unavailable(
            "self-play smoke configuration was not supplied"
        ),
    )


def run_benchmark(benchmark: BenchmarkConfig) -> dict[str, Any]:
    """Run all configured variants and return a serializable result."""
    device = _resolve_device(benchmark.device)
    dtype = DTYPE_NAMES[benchmark.dtype]
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    )
    try:
        configs = tuple(
            _make_config(
                benchmark,
                core_repeats=benchmark.deep_core_repeats if repeats is None else repeats,
                core_weights_tied=tied,
                pass_embedding_enabled=pass_enabled,
            )
            for _, repeats, tied in MODE_SETTINGS
            for pass_enabled in PASS_EMBEDDING_SETTINGS
        )
        checkpoint_policy = _checkpoint_policy(benchmark, device, dtype, configs)
        validation_artifact = (
            _load_validation_artifact(benchmark.validation_artifact)
            if benchmark.validation_artifact is not None
            else None
        )
        results: list[VariantResult] = []
        for mode, repeats, tied in MODE_SETTINGS:
            for pass_enabled in PASS_EMBEDDING_SETTINGS:
                torch.manual_seed(benchmark.seed)
                results.append(
                    _run_variant(
                        benchmark,
                        mode=mode,
                        core_repeats=benchmark.deep_core_repeats if repeats is None else repeats,
                        core_weights_tied=tied,
                        pass_embedding_enabled=pass_enabled,
                        device=device,
                        dtype=dtype,
                        validation_artifact=validation_artifact,
                        checkpoint_policy=checkpoint_policy,
                    )
                )
        return {
            "benchmark_schema": BENCHMARK_SCHEMA,
            "inputs": {
                **asdict(benchmark),
                "device": str(device),
                "checkpoint": (
                    None if benchmark.checkpoint is None else str(benchmark.checkpoint)
                ),
                "validation_artifact": (
                    None
                    if benchmark.validation_artifact is None
                    else str(benchmark.validation_artifact)
                ),
            },
            "variants": [result.to_dict() for result in results],
        }
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, device)


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
        history_tokens=args.history_tokens,
        deep_core_repeats=args.deep_core_repeats,
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
    parser.add_argument(
        "--dim-feedforward",
        type=int,
        default=DEFAULT_MODEL_CONFIG.dim_feedforward,
    )
    parser.add_argument(
        "--history-tokens",
        type=int,
        default=DEFAULT_MODEL_CONFIG.history_tokens,
    )
    parser.add_argument("--deep-core-repeats", type=int, default=3)
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
        "history_tokens",
        "deep_core_repeats",
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
        f"device={inputs['device']} dtype={inputs['dtype']} "
        f"batch={inputs['batch_size']} time_steps={inputs['time_steps']} "
        f"d_model={inputs['d_model']} nhead={inputs['nhead']} "
        f"dim_feedforward={inputs['dim_feedforward']} "
        f"history_tokens={inputs['history_tokens']} "
        f"deep_core_repeats={inputs['deep_core_repeats']} "
        f"warmup={inputs['warmup']} iterations={inputs['iterations']} "
        f"repeats={inputs['repeats']}"
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
        print(f"peak_bptt_memory_bytes={variant['peak_bptt_memory_bytes']}")
        _print_metric("validation_bc_nll", variant["validation_bc_nll"])
        _print_metric("self_play_strength", variant["self_play_strength"])



if __name__ == "__main__":
    benchmark(parse_args())
