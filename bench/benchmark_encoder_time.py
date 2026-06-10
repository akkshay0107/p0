import time
from typing import cast

import torch
import torch.nn as nn

from src.model.swiglu_encoder import SwiGLUTransformerEncoder


def benchmark():
    d_model = 512
    nhead = 8
    dim_feedforward = 2048
    num_layers = 3
    batch_size = 32
    seq_len = 50

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    standard_encoder.to(device)
    swiglu_encoder.to(device)
    compiled_swiglu_encoder.to(device)

    src = torch.randn(batch_size, seq_len, d_model).to(device)

    # warmup loops
    for _ in range(10):
        _ = standard_encoder(src)
        _ = swiglu_encoder(src)
        _ = compiled_swiglu_encoder(src)

    start = time.perf_counter()
    for _ in range(100):
        _ = standard_encoder(src)
    end = time.perf_counter()
    standard_time = (end - start) / 100
    print(f"Standard Encoder Time: {standard_time:.6f} s/batch")

    start = time.perf_counter()
    for _ in range(100):
        _ = swiglu_encoder(src)
    end = time.perf_counter()
    swiglu_time = (end - start) / 100
    swiglu_ratio = standard_time / swiglu_time
    print(f"\nSwiGLU Encoder Time: {swiglu_time:.6f} s/batch")
    print(f"Relative performance: {swiglu_ratio:.3f}x")

    start = time.perf_counter()
    for _ in range(100):
        _ = compiled_swiglu_encoder(src)
    end = time.perf_counter()
    compiled_swiglu_time = (end - start) / 100
    compiled_ratio = standard_time / compiled_swiglu_time
    print(f"\nCompiled SwiGLU Encoder Time: {compiled_swiglu_time:.6f} s/batch")
    print(f"Relative performance: {compiled_ratio:.3f}x")


if __name__ == "__main__":
    benchmark()
