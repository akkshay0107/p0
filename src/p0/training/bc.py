"""Behaviour-cloning objectives over exact and ragged replay labels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from p0.replays.schema import LabelKind


@dataclass(frozen=True, slots=True)
class BCObjective:
    """Loss and detached reporting values for one candidate-scored batch."""

    loss: Tensor
    exact_nll: Tensor
    partial_nll: Tensor
    marginal_log_probs: Tensor
    exact_count: int
    partial_count: int
    labeled_count: int


def _validate_objective_inputs(
    candidate_log_probs: Tensor,
    candidate_offsets: Tensor,
    label_kind: Tensor,
    loss_mask: Tensor,
) -> Tensor:
    if candidate_log_probs.dim() != 1:
        raise ValueError("candidate_log_probs must be one-dimensional")
    if candidate_offsets.dim() != 1 or candidate_offsets.dtype != torch.long:
        raise ValueError("candidate_offsets must be a one-dimensional torch.long tensor")
    if label_kind.dim() != 1 or loss_mask.dim() != 1:
        raise ValueError("label_kind and loss_mask must be one-dimensional")
    if label_kind.numel() + 1 != candidate_offsets.numel():
        raise ValueError("candidate_offsets must have one boundary per decision")
    if label_kind.numel() != loss_mask.numel():
        raise ValueError("label_kind and loss_mask must have matching lengths")
    if candidate_offsets.device != candidate_log_probs.device:
        candidate_offsets = candidate_offsets.to(candidate_log_probs.device)
    if (
        label_kind.device != candidate_log_probs.device
        or loss_mask.device != candidate_log_probs.device
    ):
        raise ValueError("objective tensors must share a device")
    if (
        candidate_offsets[0].item() != 0
        or candidate_offsets[-1].item() != candidate_log_probs.numel()
    ):
        raise ValueError("candidate_offsets must start at zero and end at candidate count")
    if torch.any(candidate_offsets[1:] < candidate_offsets[:-1]):
        raise ValueError("candidate_offsets must be nondecreasing")
    if torch.any((loss_mask < 0) | (loss_mask > 1)):
        raise ValueError("loss_mask values must be in [0, 1]")
    return candidate_offsets


def _ragged_logsumexp(candidate_log_probs: Tensor, offsets: Tensor) -> Tensor:
    decision_count = offsets.numel() - 1
    counts = offsets[1:] - offsets[:-1]
    row_ids = torch.repeat_interleave(
        torch.arange(decision_count, device=candidate_log_probs.device), counts
    )
    finite_scores = torch.where(
        torch.isfinite(candidate_log_probs),
        candidate_log_probs,
        torch.full_like(candidate_log_probs, float("-inf")),
    )
    row_max = torch.full(
        (decision_count,),
        float("-inf"),
        dtype=candidate_log_probs.dtype,
        device=candidate_log_probs.device,
    )
    if finite_scores.numel():
        row_max.scatter_reduce_(0, row_ids, finite_scores, reduce="amax", include_self=True)
    safe_max = row_max[row_ids] if row_ids.numel() else row_max.new_empty((0,))
    safe_delta = torch.where(
        torch.isfinite(safe_max),
        finite_scores - safe_max,
        torch.zeros_like(finite_scores),
    )
    row_sum = torch.zeros(
        (decision_count,), dtype=candidate_log_probs.dtype, device=candidate_log_probs.device
    )
    if safe_delta.numel():
        row_sum.scatter_add_(0, row_ids, torch.exp(safe_delta))
    result = row_max + torch.log(row_sum)
    return torch.where(
        (counts > 0) & torch.isfinite(row_max),
        result,
        torch.full_like(result, float("-inf")),
    )


def compute_bc_objective(
    candidate_log_probs: Tensor,
    candidate_offsets: Tensor,
    label_kind: Tensor,
    loss_mask: Tensor,
) -> BCObjective:
    """Compute exact and candidate-marginalized NLL without dropping unknown steps."""
    offsets = _validate_objective_inputs(
        candidate_log_probs, candidate_offsets, label_kind, loss_mask
    )
    exact = label_kind == int(LabelKind.EXACT)
    partial = label_kind == int(LabelKind.PARTIAL)
    unknown = label_kind == int(LabelKind.UNKNOWN)
    if torch.any(~(exact | partial | unknown)):
        raise ValueError("label_kind contains an unsupported label")
    counts = offsets[1:] - offsets[:-1]
    if torch.any(exact & (counts != 1)):
        raise ValueError("EXACT labels must have exactly one candidate")
    if torch.any(partial & (counts < 2)):
        raise ValueError("PARTIAL labels must have at least two candidates")
    if torch.any(unknown & (counts != 0)):
        raise ValueError("UNKNOWN labels must not have candidates")
    if torch.any(unknown & (loss_mask != 0)):
        raise ValueError("UNKNOWN labels must have a zero loss mask")
    if torch.any((exact | partial) & (loss_mask == 0)):
        raise ValueError("Labeled decisions must have a nonzero loss mask")

    marginal_log_probs = _ragged_logsumexp(candidate_log_probs, offsets)
    exact_scores = candidate_log_probs[offsets[:-1][exact]]
    exact_score_rows = torch.zeros_like(marginal_log_probs)
    exact_score_rows[exact] = exact_scores
    selected_log_probs = torch.where(exact, exact_score_rows, marginal_log_probs)
    per_decision_loss = torch.where(
        loss_mask > 0,
        -selected_log_probs * loss_mask,
        torch.zeros_like(selected_log_probs),
    )
    labeled_count = int(loss_mask.sum().item())
    denominator = loss_mask.sum().clamp_min(1.0)
    exact_count = int(exact.sum().item())
    partial_count = int(partial.sum().item())
    exact_nll = (-exact_scores).sum() / max(exact_count, 1)
    partial_nll = (-marginal_log_probs[partial]).sum() / max(partial_count, 1)
    return BCObjective(
        loss=per_decision_loss.sum() / denominator,
        exact_nll=exact_nll,
        partial_nll=partial_nll,
        marginal_log_probs=marginal_log_probs,
        exact_count=exact_count,
        partial_count=partial_count,
        labeled_count=labeled_count,
    )


__all__ = ["BCObjective", "compute_bc_objective"]
