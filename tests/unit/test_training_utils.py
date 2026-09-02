"""Tests for training precision selection."""

from __future__ import annotations

import torch

from p0.training.utils import select_optimization_precision


class TestTrainingUtils:
    def test_optimization_precision_waterfall(self) -> None:
        """Verify BF16, FP16, and FP32 are selected in preference order."""
        assert select_optimization_precision(True, torch.device("cpu")) == (
            torch.float32,
            False,
            False,
        )
        assert select_optimization_precision(True, torch.device("cuda"), bf16_supported=True) == (
            torch.bfloat16,
            True,
            False,
        )
        assert select_optimization_precision(True, torch.device("cuda"), bf16_supported=False) == (
            torch.float16,
            True,
            True,
        )
        assert select_optimization_precision(False, torch.device("cuda")) == (
            torch.float32,
            False,
            False,
        )
