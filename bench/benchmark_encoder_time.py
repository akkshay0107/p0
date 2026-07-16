"""Deterministic reducer-encoder microbenchmark used by the refactor gates."""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn as nn

from p0.model.structured_observation import EVENT_COUNT, SEQUENCE_LENGTH
from p0.model.swiglu_encoder import SwiGLUTransformerEncoder
from p0.training.utils import default_device

ACTION_MASK_TOKEN_COUNT = 1
HISTORY_TOKEN_COUNT = 8
REDUCER_INPUT_LENGTH = SEQUENCE_LENGTH + ACTION_MASK_TOKEN_COUNT + EVENT_COUNT
REDUCER_ATTENTION_LENGTH = REDUCER_INPUT_LENGTH + HISTORY_TOKEN_COUNT


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sample(
    encoder: nn.Module,
    src: torch.Tensor,
    device: torch.device,
    iterations: int,
) -> float:
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iterations):
            encoder(src)
    synchronize(device)
    return (time.perf_counter() - start) / iterations


def _summary(samples: list[float]) -> tuple[float, float]:
    quartiles = statistics.quantiles(samples, n=4, method="inclusive")
    return statistics.median(samples), quartiles[2] - quartiles[0]


def benchmark(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = default_device()
    encoder = SwiGLUTransformerEncoder(
        d_model=args.d_model,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        num_layers=args.layers,
    ).to(device)
    src = torch.randn(args.batch_size, REDUCER_ATTENTION_LENGTH, args.d_model, device=device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            encoder(src)
    synchronize(device)

    samples = [_sample(encoder, src, device, args.iterations) for _ in range(args.repeats)]
    median, iqr = _summary(samples)
    print(
        f"device={device} batch={args.batch_size} tokens={REDUCER_ATTENTION_LENGTH} "
        f"d_model={args.d_model} layers={args.layers} iterations={args.iterations} "
        f"repeats={args.repeats}"
    )
    print(f"samples_seconds_per_batch={','.join(f'{sample:.8f}' for sample in samples)}")
    print(f"median_seconds_per_batch={median:.8f}")
    print(f"iqr_seconds_per_batch={iqr:.8f}")
    print(f"median_tokens_per_second={args.batch_size * REDUCER_ATTENTION_LENGTH / median:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for name in (
        "batch_size",
        "d_model",
        "nhead",
        "dim_feedforward",
        "layers",
        "warmup",
        "iterations",
        "repeats",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


if __name__ == "__main__":
    benchmark(parse_args())
