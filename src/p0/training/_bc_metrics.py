"""Behavior-cloning objectives and vectorized evaluation aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from p0.model.structured_observation import (
    NUM_IDX_SLOT_LEGALITY_UNKNOWN,
    TOKEN_IDX_ALLY_SIDE,
)
from p0.replays.schema import LabelKind


@dataclass(frozen=True, slots=True)
class BCObjective:
    """Loss and detached reporting values for one candidate-scored batch."""

    loss: Tensor
    exact_nll: Tensor
    partial_nll: Tensor
    marginal_log_probs: Tensor
    exact_count: Tensor
    partial_count: Tensor
    labeled_count: Tensor
    loss_weight: Tensor


@dataclass(frozen=True, slots=True)
class BCEvaluationMetrics:
    overall_nll: float
    exact_nll: float
    partial_nll: float
    exact_joint_accuracy: float
    value_loss: float
    illegal_probability_mass: float
    unknown_label_fraction: float
    non_finite_values: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "overall_nll": self.overall_nll,
            "exact_nll": self.exact_nll,
            "partial_nll": self.partial_nll,
            "exact_joint_accuracy": self.exact_joint_accuracy,
            "value_loss": self.value_loss,
            "illegal_probability_mass": self.illegal_probability_mass,
            "unknown_label_fraction": self.unknown_label_fraction,
            "non_finite_values": self.non_finite_values,
        }


def _validate_objective_inputs(
    candidate_count: int,
    candidate_offsets: Tensor,
    label_kind: Tensor,
    loss_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if candidate_offsets.dim() != 1 or candidate_offsets.dtype != torch.long:
        raise ValueError("candidate_offsets must be a one-dimensional torch.long tensor")
    if label_kind.dim() != 1 or loss_mask.dim() != 1:
        raise ValueError("label_kind and loss_mask must be one-dimensional")
    if label_kind.numel() + 1 != candidate_offsets.numel():
        raise ValueError("candidate_offsets must have one boundary per decision")
    if label_kind.numel() != loss_mask.numel():
        raise ValueError("label_kind and loss_mask must have matching lengths")
    if (
        candidate_offsets.device != label_kind.device
        or candidate_offsets.device != loss_mask.device
    ):
        raise ValueError("candidate label-contract tensors must share a device")

    if candidate_offsets[0].item() != 0 or candidate_offsets[-1].item() != candidate_count:
        raise ValueError("candidate_offsets must start at zero and end at candidate count")
    if torch.any(candidate_offsets[1:] < candidate_offsets[:-1]):
        raise ValueError("candidate_offsets must be nondecreasing")
    if torch.any((loss_mask < 0) | (loss_mask > 1)):
        raise ValueError("loss_mask values must be in [0, 1]")

    exact = label_kind == int(LabelKind.EXACT)
    partial = label_kind == int(LabelKind.PARTIAL)
    unknown = label_kind == int(LabelKind.UNKNOWN)
    labeled = exact | partial

    if torch.any(~(labeled | unknown)):
        raise ValueError("label_kind contains an unsupported label")

    counts = candidate_offsets[1:] - candidate_offsets[:-1]
    if torch.any(exact & (counts != 1)):
        raise ValueError("EXACT labels must have exactly one candidate")
    if torch.any(partial & (counts < 2)):
        raise ValueError("PARTIAL labels must have at least two candidates")
    if torch.any(unknown & (counts != 0)):
        raise ValueError("UNKNOWN labels must not have candidates")
    if torch.any(unknown & (loss_mask != 0)):
        raise ValueError("UNKNOWN labels must have a zero loss mask")
    if torch.any(labeled & (loss_mask == 0)):
        raise ValueError("Labeled decisions must have a nonzero loss mask")

    return exact, partial, unknown, labeled


def _ragged_logsumexp(candidate_log_probs: Tensor, offsets: Tensor) -> Tensor:
    counts = offsets[1:] - offsets[:-1]
    row_max = torch.segment_reduce(candidate_log_probs, reduce="max", lengths=counts, unsafe=True)
    row_max = torch.where(counts > 0, row_max, float("-inf"))
    row_ids = torch.repeat_interleave(
        torch.arange(counts.numel(), device=candidate_log_probs.device),
        counts,
    )
    gathered_max = row_max[row_ids]
    shifted = torch.where(
        torch.isfinite(gathered_max),
        candidate_log_probs - gathered_max,
        torch.zeros_like(candidate_log_probs),
    )
    row_sum = torch.segment_reduce(torch.exp(shifted), reduce="sum", lengths=counts, unsafe=True)
    return torch.where(
        torch.isfinite(row_max),
        row_max + torch.log(row_sum),
        torch.full_like(row_max, float("-inf")),
    )


def compute_bc_objective(
    candidate_log_probs: Tensor,
    candidate_offsets: Tensor,
    label_kind: Tensor,
    loss_mask: Tensor,
) -> BCObjective:
    """Compute exact and candidate-marginalized NLL without dropping unknown steps."""
    if candidate_log_probs.dim() != 1:
        raise ValueError("candidate_log_probs must be one-dimensional")
    if (
        candidate_offsets.device != candidate_log_probs.device
        or label_kind.device != candidate_log_probs.device
        or loss_mask.device != candidate_log_probs.device
    ):
        raise ValueError("objective tensors must share a device")

    exact, partial, _, labeled = _validate_objective_inputs(
        candidate_log_probs.numel(),
        candidate_offsets,
        label_kind,
        loss_mask,
    )
    return _compute_bc_objective_unchecked(
        candidate_log_probs,
        candidate_offsets,
        loss_mask,
        exact,
        partial,
        labeled,
        exact_count=exact.sum(),
        partial_count=partial.sum(),
        labeled_count=labeled.sum(),
        loss_weight=loss_mask.sum(),
    )


def _compute_bc_objective_unchecked(
    candidate_log_probs: Tensor,
    candidate_offsets: Tensor,
    loss_mask: Tensor,
    exact: Tensor,
    partial: Tensor,
    labeled: Tensor,
    *,
    exact_count: Tensor,
    partial_count: Tensor,
    labeled_count: Tensor,
    loss_weight: Tensor,
) -> BCObjective:
    """Compute the objective after the batch contract was validated at its boundary."""
    marginal_log_probs = _ragged_logsumexp(candidate_log_probs, candidate_offsets)

    # Selection avoids the undefined zero-times-negative-infinity result for UNKNOWN rows.
    per_decision_loss = torch.where(
        loss_mask > 0,
        -marginal_log_probs * loss_mask,
        torch.zeros_like(marginal_log_probs),
    )
    exact_nll = (-marginal_log_probs[exact]).sum() / exact_count.clamp_min(1)
    partial_nll = (-marginal_log_probs[partial]).sum() / partial_count.clamp_min(1)

    return BCObjective(
        loss=per_decision_loss.sum() / loss_weight.clamp_min(1.0),
        exact_nll=exact_nll,
        partial_nll=partial_nll,
        marginal_log_probs=marginal_log_probs,
        exact_count=exact_count,
        partial_count=partial_count,
        labeled_count=labeled_count,
        loss_weight=loss_weight,
    )


def slot_legality_unknown(numerical: Tensor) -> Tensor:
    """Per-decision flag for whether either active slot's legality is unproven."""
    gates = numerical[
        :,
        TOKEN_IDX_ALLY_SIDE,
        NUM_IDX_SLOT_LEGALITY_UNKNOWN : NUM_IDX_SLOT_LEGALITY_UNKNOWN + 2,
    ]
    return gates.gt(0).any(dim=-1)


@dataclass(slots=True)
class _BCEvaluationAccumulator:
    counts: Tensor
    nll_sums: Tensor
    value_loss_sum: Tensor
    value_count: Tensor
    illegal_mass_sum: Tensor
    illegal_mass_count: Tensor

    @classmethod
    def create(cls, device: torch.device) -> _BCEvaluationAccumulator:
        return cls(
            counts=torch.zeros(7, dtype=torch.long, device=device),
            nll_sums=torch.zeros(3, dtype=torch.float64, device=device),
            value_loss_sum=torch.zeros((), dtype=torch.float64, device=device),
            value_count=torch.zeros((), dtype=torch.long, device=device),
            illegal_mass_sum=torch.zeros((), dtype=torch.float64, device=device),
            illegal_mass_count=torch.zeros((), dtype=torch.long, device=device),
        )

    def add_value(self, predictions: Tensor, targets: Tensor, mask: Tensor) -> None:
        """Accumulate discounted terminal-outcome value diagnostics."""
        if mask.any():
            error = predictions[mask] - targets[mask]
            self.value_loss_sum += error.square().to(torch.float64).sum()
            self.value_count += mask.sum()

    def add_legality(self, logits: Tensor, action_mask: Tensor, numerical: Tensor) -> None:
        """Accumulate probability mass on request-illegal first-slot actions."""
        unknown = slot_legality_unknown(numerical)
        proven = ~unknown

        # Only proven rows have an authoritative notion of illegal to measure against,
        # so they are weighted in rather than indexed out, which would force a sync.
        probabilities = torch.softmax(logits.float(), dim=-1)
        illegal_mass = probabilities.mul(~action_mask[:, 0].bool()).sum(dim=-1)

        self.illegal_mass_sum += illegal_mass.mul(proven).sum().to(torch.float64)
        self.illegal_mass_count += proven.sum()

    def add(
        self,
        *,
        exact_actions: Tensor,
        masks: tuple[Tensor, Tensor, Tensor, Tensor],
        marginal_nll: Tensor,
        safe_nll: Tensor,
        predicted: Tensor,
        best_scores: Tensor,
    ) -> None:
        exact, partial, unknown, labeled = masks
        exact_correct = torch.all(predicted[exact] == exact_actions[exact], dim=1).sum()
        self.counts += torch.stack(
            (
                exact.sum() + partial.sum() + unknown.sum(),
                labeled.sum(),
                unknown.sum(),
                exact.sum(),
                partial.sum(),
                exact_correct,
                (~torch.isfinite(best_scores)).sum()
                + ((~torch.isfinite(marginal_nll)) & labeled).sum(),
            )
        )
        safe_nll_64 = safe_nll.to(dtype=torch.float64)
        self.nll_sums += torch.stack(
            (
                safe_nll_64[labeled].sum(),
                safe_nll_64[exact].sum(),
                safe_nll_64[partial].sum(),
            )
        )

    def finalize(self) -> BCEvaluationMetrics:
        counts = self.counts.cpu().tolist()
        nll_sums = self.nll_sums.cpu().tolist()
        value_count = int(self.value_count.item())
        illegal_mass_count = int(self.illegal_mass_count.item())
        decisions, labeled, unknown, exact, _ = (int(value) for value in counts[:5])
        return BCEvaluationMetrics(
            overall_nll=float(nll_sums[0]) / max(labeled, 1),
            exact_nll=float(nll_sums[1]) / max(exact, 1),
            partial_nll=float(nll_sums[2]) / max(int(counts[4]), 1),
            exact_joint_accuracy=int(counts[5]) / max(exact, 1),
            value_loss=float(self.value_loss_sum.item()) / max(value_count, 1),
            illegal_probability_mass=float(self.illegal_mass_sum.item())
            / max(illegal_mass_count, 1),
            unknown_label_fraction=unknown / max(decisions, 1),
            non_finite_values=int(counts[6]),
        )
