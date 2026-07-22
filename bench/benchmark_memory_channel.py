"""Benchmark the fixed event, reducer, and policy inference stages."""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from functools import partial

import torch

from p0.format_config import FORMAT
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.policy import EncodedObs, PolicyNet
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.training.utils import default_device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(operation: Callable[[], object], device: torch.device, iterations: int) -> float:
    _synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iterations):
            operation()
    _synchronize(device)
    return (time.perf_counter() - start) / iterations


def _summary(samples: list[float]) -> tuple[float, float]:
    if len(samples) == 1:
        return samples[0], 0.0
    quartiles = statistics.quantiles(samples, n=4, method="inclusive")
    return statistics.median(samples), quartiles[2] - quartiles[0]


def _event_pooler(
    policy: PolicyNet, observations: StructuredObservation, device: torch.device
) -> object:
    return policy.encoder._encode_events(observations, device)


def _fixed_reducer(
    policy: PolicyNet,
    encoded: EncodedObs,
    memory: tuple[torch.Tensor, ...],
) -> object:
    return policy.actor.reducer(encoded.tokens, *memory)


def _policy_inference(
    policy: PolicyNet,
    encoded: EncodedObs,
    action_mask: torch.Tensor,
    memory: tuple[torch.Tensor, ...],
) -> object:
    return policy.act(encoded, action_mask, *memory)


def benchmark(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = default_device()
    policy = build_policy(
        ModelConfig(
            d_model=args.d_model,
            nhead=args.nhead,
            reducer_layers=args.reducer_layers,
            dim_feedforward=args.dim_feedforward,
        ),
        default_runtime_resources(),
    ).to(device)
    observations = StructuredObservation.empty_batch(args.batch_size).to(device)
    action_mask = torch.ones(
        (args.batch_size, 2, FORMAT.action_size), dtype=torch.bool, device=device
    )
    encoded = policy.encode(observations, action_mask)
    memory = policy.empty_memory(args.batch_size)

    operations: dict[str, Callable[[], object]] = {
        "event_pooler": partial(_event_pooler, policy, observations, device),
        "fixed_reducer": partial(_fixed_reducer, policy, encoded, memory),
        "policy_inference": partial(_policy_inference, policy, encoded, action_mask, memory),
    }
    with torch.inference_mode():
        for _ in range(args.warmup):
            for operation in operations.values():
                operation()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(
        f"device={device} batch={args.batch_size} d_model={args.d_model} "
        f"nhead={args.nhead} reducer_layers={args.reducer_layers} "
        f"iterations={args.iterations} repeats={args.repeats}"
    )
    for name, operation in operations.items():
        samples = [_measure(operation, device, args.iterations) for _ in range(args.repeats)]
        median, iqr = _summary(samples)
        print(f"{name}_median_seconds={median:.8f}")
        print(f"{name}_iqr_seconds={iqr:.8f}")

    if device.type == "cuda":
        print(f"peak_cuda_memory_bytes={torch.cuda.max_memory_allocated(device)}")
    print(
        f"cpu_history_buffer_bytes={args.batch_size * 48 * args.d_model * torch.empty((), dtype=torch.float32).element_size()}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--reducer-layers", type=int, default=5)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for name in vars(args):
        if name != "seed" and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


if __name__ == "__main__":
    benchmark(parse_args())
