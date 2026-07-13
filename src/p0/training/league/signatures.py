"""Pure signature comparison helpers."""

from __future__ import annotations

import torch

SCORE_EPS = 0.05
DIV_MIN_SPREAD = 0.01


def normalize_scores(
    values: dict[str, float], min_spread: float = DIV_MIN_SPREAD, eps: float = SCORE_EPS
) -> dict[str, float]:
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < min_spread:
        return {key: 1.0 for key in values}
    return {key: eps + (1 - eps) * (value - lo) / (hi - lo) for key, value in values.items()}


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    midpoint = 0.5 * (p + q)
    safe_midpoint = midpoint.clamp_min(1e-12)
    kl_p = torch.sum(torch.where(p > 0, p * torch.log2(p / safe_midpoint), 0.0), dim=-1)
    kl_q = torch.sum(torch.where(q > 0, q * torch.log2(q / safe_midpoint), 0.0), dim=-1)
    return (0.5 * kl_p + 0.5 * kl_q).mean().item()
