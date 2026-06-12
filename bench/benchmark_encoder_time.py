import time
from typing import cast

import torch
import torch.nn as nn

from src.model.structured_observation import EVENT_COUNT, SEQUENCE_LENGTH
from src.model.swiglu_encoder import SwiGLUTransformerEncoder
from src.train.utils import default_device

ACTION_MASK_TOKEN_COUNT = 1
HISTORY_TOKEN_COUNT = 8
REDUCER_INPUT_LENGTH = SEQUENCE_LENGTH + ACTION_MASK_TOKEN_COUNT + EVENT_COUNT
REDUCER_ATTENTION_LENGTH = REDUCER_INPUT_LENGTH + HISTORY_TOKEN_COUNT
BATCH_SIZE = 32
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def average_forward_time(
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


def benchmark():
    d_model = 512
    nhead = 8
    dim_feedforward = 2048
    num_layers = 3
    seq_len = REDUCER_ATTENTION_LENGTH

    print(
        "Benchmarking reducer attention length "
        f"{seq_len} ({SEQUENCE_LENGTH} observation + {ACTION_MASK_TOKEN_COUNT} action mask "
        f"+ {EVENT_COUNT} event + {HISTORY_TOKEN_COUNT} history tokens)"
    )

    # Standard Transformer
    enc_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=0.0,
        batch_first=True,
        norm_first=True,
        activation="gelu",
    )
    standard_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    # SwiGLU Transformer
    swiglu_encoder = SwiGLUTransformerEncoder(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_layers=num_layers,
    )

    # Compiled SwiGLU transformer
    compiled_swiglu_encoder = cast(
        SwiGLUTransformerEncoder,
        torch.compile(
            SwiGLUTransformerEncoder(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                num_layers=num_layers,
            ),
        ),
    )

    device = default_device()
    standard_encoder.to(device)
    swiglu_encoder.to(device)
    compiled_swiglu_encoder.to(device)

    src = torch.randn(BATCH_SIZE, seq_len, d_model).to(device)

    with torch.inference_mode():
        for _ in range(WARMUP_ITERATIONS):
            standard_encoder(src)
            swiglu_encoder(src)
            compiled_swiglu_encoder(src)
    synchronize(device)

    standard_time = average_forward_time(
        standard_encoder,
        src,
        device,
        BENCHMARK_ITERATIONS,
    )
    print(f"Standard Encoder Time: {standard_time:.6f} s/batch")

    swiglu_time = average_forward_time(
        swiglu_encoder,
        src,
        device,
        BENCHMARK_ITERATIONS,
    )
    swiglu_ratio = standard_time / swiglu_time
    print(f"\nSwiGLU Encoder Time: {swiglu_time:.6f} s/batch")
    print(f"Relative performance: {swiglu_ratio:.3f}x")

    compiled_swiglu_time = average_forward_time(
        compiled_swiglu_encoder,
        src,
        device,
        BENCHMARK_ITERATIONS,
    )
    compiled_ratio = standard_time / compiled_swiglu_time
    print(f"\nCompiled SwiGLU Encoder Time: {compiled_swiglu_time:.6f} s/batch")
    print(f"Relative performance: {compiled_ratio:.3f}x")


if __name__ == "__main__":
    benchmark()
