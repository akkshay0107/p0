"""Benchmark comparing uncompiled baseline vs. Strategy A compiled policy execution speed."""

from __future__ import annotations

import argparse
import copy
import statistics
import time
from collections.abc import Callable

import torch

from p0.format_config import FORMAT
from p0.model.config import ModelConfig
from p0.model.factory import build_policy, compile_policy
from p0.model.policy import PolicyNet
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.training.utils import default_device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(operation: Callable[[], object], device: torch.device, iterations: int) -> float:
    _synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            operation()
    _synchronize(device)
    return (time.perf_counter() - start) / iterations


def _summary(samples: list[float]) -> tuple[float, float]:
    if len(samples) == 1:
        return samples[0], 0.0
    quartiles = statistics.quantiles(samples, n=4, method="inclusive")
    return statistics.median(samples), quartiles[2] - quartiles[0]


def run_benchmark(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = default_device()
    resources = default_runtime_resources()

    config = ModelConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        reducer_layers=args.reducer_layers,
        dim_feedforward=args.dim_feedforward,
    )

    print("=" * 80)
    print(" Strategy A torch.compile Performance Benchmark")
    print(f" Device: {device}")
    print(
        f" Config: d_model={config.d_model}, nhead={config.nhead}, reducer_layers={config.reducer_layers}"
    )
    print("=" * 80)

    # 1. Instantiate baseline policy
    policy_eager = build_policy(config, resources).to(device).eval()

    # 2. Deepcopy for compiled policy so parameters are identical
    policy_compiled = copy.deepcopy(policy_eager)
    if device.type == "cuda" and hasattr(torch, "compile"):
        print("\nCompiling sub-modules (Strategy A: .encoder, .actor, .critic, .series)...")
        compile_policy(policy_compiled, enable=True, dynamic=True)
    else:
        print(
            "\nNote: CUDA not detected or torch.compile unavailable. Running in eager mode comparison."
        )

    # Prepare workloads
    batch_sizes = [1, 8, 128] if args.batch_size is None else [args.batch_size]

    for batch_size in batch_sizes:
        print(f"\n--- Workload Benchmark (Batch Size B = {batch_size}) ---")
        obs = StructuredObservation.empty_batch(batch_size).to(device)
        action_mask = torch.ones(
            (batch_size, 2, FORMAT.action_size), dtype=torch.bool, device=device
        )
        series, series_mask, history, history_mask, history_age_ids = policy_eager.empty_memory(
            batch_size
        )

        def run_act(p: PolicyNet):
            return p.act_obs(
                obs, action_mask, series, series_mask, history, history_mask, history_age_ids
            )

        # Warmup Eager
        _synchronize(device)
        for _ in range(args.warmup):
            run_act(policy_eager)
        _synchronize(device)

        eager_samples = [
            _measure(lambda: run_act(policy_eager), device, args.iterations)
            for _ in range(args.repeats)
        ]
        eager_med, eager_iqr = _summary(eager_samples)

        # Warmup Compiled (includes Triton kernel compilation pass on first invocation)
        _synchronize(device)
        compile_warmup_start = time.perf_counter()
        for _ in range(args.warmup):
            run_act(policy_compiled)
        _synchronize(device)
        compile_warmup_time = time.perf_counter() - compile_warmup_start

        compiled_samples = [
            _measure(lambda: run_act(policy_compiled), device, args.iterations)
            for _ in range(args.repeats)
        ]
        compiled_med, compiled_iqr = _summary(compiled_samples)

        speedup = eager_med / compiled_med if compiled_med > 0 else 1.0
        eager_fps = (batch_size / eager_med) if eager_med > 0 else 0
        compiled_fps = (batch_size / compiled_med) if compiled_med > 0 else 0

        print(
            f" Eager Mode    : {eager_med * 1000:8.3f} ms/step | {eager_fps:10.1f} dec/sec | IQR: {eager_iqr * 1000:.3f} ms"
        )
        print(
            f" Compiled Mode : {compiled_med * 1000:8.3f} ms/step | {compiled_fps:10.1f} dec/sec | IQR: {compiled_iqr * 1000:.3f} ms"
        )
        print(f" Speedup       : {speedup:8.2f}x faster")
        if device.type == "cuda":
            print(f" Warmup/Compilation Time: {compile_warmup_time:.2f} s")

    print("\n" + "=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Specific batch size to test (default: test 1, 8, 128)",
    )
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--reducer-layers", type=int, default=5)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
