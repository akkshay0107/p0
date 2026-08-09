"""Behavior-cloning objectives and vectorized evaluation aggregation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from p0.model.structured_observation import (
    NUM_IDX_SLOT_LEGALITY_UNKNOWN,
    TOKEN_IDX_ALLY_SIDE,
)
from p0.replays.schema import DecisionType, LabelKind
from p0.training._bc_batch import BCDecisionBatch


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
    loss_weight: float


@dataclass(frozen=True, slots=True)
class BCEvaluationMetrics:
    overall_nll: float
    exact_nll: float
    partial_nll: float
    exact_joint_accuracy: float
    decisions: int
    labeled_decisions: int
    unknown_decisions: int
    exact_decisions: int
    partial_decisions: int
    illegal_predictions: int
    non_finite_values: int
    by_decision_type: Mapping[str, Mapping[str, float | int]]
    confidence_buckets: Mapping[str, Mapping[str, float | int]]
    candidate_set_sizes: Mapping[str, int]
    value_loss: float = 0.0
    value_decisions: int = 0
    # Mean probability mass the unmasked policy puts on actions the authoritative
    # request calls illegal, over decisions whose legality is proven. Output masking
    # hides this at inference, so it is the only view of learned legality behaviour.
    illegal_probability_mass: float = 0.0
    unknown_legality_decisions: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "overall_nll": self.overall_nll,
            "exact_nll": self.exact_nll,
            "partial_nll": self.partial_nll,
            "exact_joint_accuracy": self.exact_joint_accuracy,
            "decisions": self.decisions,
            "labeled_decisions": self.labeled_decisions,
            "unknown_decisions": self.unknown_decisions,
            "exact_decisions": self.exact_decisions,
            "partial_decisions": self.partial_decisions,
            "illegal_predictions": self.illegal_predictions,
            "non_finite_values": self.non_finite_values,
            "by_decision_type": dict(self.by_decision_type),
            "confidence_buckets": dict(self.confidence_buckets),
            "candidate_set_sizes": dict(self.candidate_set_sizes),
            "value_loss": self.value_loss,
            "value_decisions": self.value_decisions,
            "illegal_probability_mass": self.illegal_probability_mass,
            "unknown_legality_decisions": self.unknown_legality_decisions,
        }


_CONFIDENCE_BUCKET_NAMES = ("[0,.25)", "[.25,.5)", "[.5,.75)", "[.75,1]")
_DECISION_TYPE_BIN_COUNT = max(int(decision_type) for decision_type in DecisionType) + 1


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
    _validate_label_rows(candidate_offsets, loss_mask, exact, partial, unknown, labeled)
    return exact, partial, unknown, labeled


def _validate_label_rows(
    candidate_offsets: Tensor,
    loss_mask: Tensor,
    exact: Tensor,
    partial: Tensor,
    unknown: Tensor,
    labeled: Tensor,
) -> None:
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
        exact_count=int(exact.sum().item()),
        partial_count=int(partial.sum().item()),
        labeled_count=int(labeled.sum().item()),
        loss_weight=float(loss_mask.sum().item()),
    )


def _compute_bc_objective_unchecked(
    candidate_log_probs: Tensor,
    candidate_offsets: Tensor,
    loss_mask: Tensor,
    exact: Tensor,
    partial: Tensor,
    labeled: Tensor,
    *,
    exact_count: int,
    partial_count: int,
    labeled_count: int,
    loss_weight: float,
) -> BCObjective:
    """Compute the objective after the batch contract was validated at its boundary."""
    marginal_log_probs = _ragged_logsumexp(candidate_log_probs, candidate_offsets)

    # Selection avoids the undefined zero-times-negative-infinity result for UNKNOWN rows.
    per_decision_loss = torch.where(
        loss_mask > 0,
        -marginal_log_probs * loss_mask,
        torch.zeros_like(marginal_log_probs),
    )
    exact_nll = (-marginal_log_probs[exact]).sum() / max(exact_count, 1)
    partial_nll = (-marginal_log_probs[partial]).sum() / max(partial_count, 1)

    return BCObjective(
        loss=per_decision_loss.sum() / max(loss_weight, 1.0),
        exact_nll=exact_nll,
        partial_nll=partial_nll,
        marginal_log_probs=marginal_log_probs,
        exact_count=exact_count,
        partial_count=partial_count,
        labeled_count=labeled_count,
        loss_weight=loss_weight,
    )


def _add_fixed_bin_totals(
    counts: Tensor,
    nll_sums: Tensor,
    bin_ids: Tensor,
    labeled: Tensor,
    safe_nll: Tensor,
) -> None:
    width = counts.size(1)
    counts[0].add_(torch.bincount(bin_ids, minlength=width))
    counts[1].add_(torch.bincount(bin_ids[labeled], minlength=width))
    labeled_nll = torch.where(
        labeled,
        safe_nll.to(dtype=torch.float64),
        torch.zeros((), dtype=torch.float64, device=safe_nll.device),
    )
    nll_sums.add_(torch.bincount(bin_ids, weights=labeled_nll, minlength=width))


def _finalize_evaluation_bins(
    counts: Tensor,
    nll_sums: Tensor,
    *,
    names: tuple[str, ...] | None = None,
) -> dict[str, Mapping[str, float | int]]:
    decision_values = counts[0].cpu().tolist()
    labeled_values = counts[1].cpu().tolist()
    nll_values = nll_sums.cpu().tolist()
    result: dict[str, Mapping[str, float | int]] = {}
    for index, decision_count in enumerate(decision_values):
        if not decision_count:
            continue
        key = names[index] if names is not None else str(index)
        labeled_count = int(labeled_values[index])
        result[key] = {
            "decisions": int(decision_count),
            "labeled": labeled_count,
            "nll": float(nll_values[index]) / max(labeled_count, 1),
        }
    return result


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
    decision_type_counts: Tensor
    decision_type_nll_sums: Tensor
    confidence_counts: Tensor
    confidence_nll_sums: Tensor
    confidence_boundaries: Tensor
    candidate_counts: Tensor
    value_loss_sum: Tensor
    value_count: Tensor
    illegal_mass_sum: Tensor
    illegal_mass_count: Tensor
    unknown_legality_count: Tensor

    @classmethod
    def create(cls, device: torch.device) -> _BCEvaluationAccumulator:
        return cls(
            counts=torch.zeros(8, dtype=torch.long, device=device),
            nll_sums=torch.zeros(3, dtype=torch.float64, device=device),
            decision_type_counts=torch.zeros(
                (2, _DECISION_TYPE_BIN_COUNT),
                dtype=torch.long,
                device=device,
            ),
            decision_type_nll_sums=torch.zeros(
                _DECISION_TYPE_BIN_COUNT,
                dtype=torch.float64,
                device=device,
            ),
            confidence_counts=torch.zeros((2, 4), dtype=torch.long, device=device),
            confidence_nll_sums=torch.zeros(4, dtype=torch.float64, device=device),
            confidence_boundaries=torch.tensor((0.25, 0.5, 0.75), device=device),
            candidate_counts=torch.zeros(0, dtype=torch.long, device=device),
            value_loss_sum=torch.zeros((), dtype=torch.float64, device=device),
            value_count=torch.zeros((), dtype=torch.long, device=device),
            illegal_mass_sum=torch.zeros((), dtype=torch.float64, device=device),
            illegal_mass_count=torch.zeros((), dtype=torch.long, device=device),
            unknown_legality_count=torch.zeros((), dtype=torch.long, device=device),
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

        # Only proven rows have an authoritative notion of "illegal" to measure against,
        # so they are weighted in rather than indexed out, which would force a sync.
        probabilities = torch.softmax(logits.float(), dim=-1)
        illegal_mass = probabilities.mul(~action_mask[:, 0].bool()).sum(dim=-1)

        self.unknown_legality_count += unknown.sum()
        self.illegal_mass_sum += illegal_mass.mul(proven).sum().to(torch.float64)
        self.illegal_mass_count += proven.sum()

    def add(
        self,
        batch: BCDecisionBatch,
        *,
        candidate_offsets: Tensor,
        masks: tuple[Tensor, Tensor, Tensor, Tensor],
        marginal_nll: Tensor,
        safe_nll: Tensor,
        predicted: Tensor,
        best_scores: Tensor,
    ) -> None:
        exact, partial, unknown, labeled = masks
        exact_actions = batch.exact_action.to(predicted.device)
        exact_correct = torch.all(predicted[exact] == exact_actions[exact], dim=1).sum()
        self.counts += torch.stack(
            (
                exact.sum() + partial.sum() + unknown.sum(),
                labeled.sum(),
                unknown.sum(),
                exact.sum(),
                partial.sum(),
                exact_correct,
                (~torch.isfinite(best_scores)).sum(),
                ((~torch.isfinite(marginal_nll)) & labeled).sum(),
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

        decision_types = batch.decision_type.to(exact.device)
        _add_fixed_bin_totals(
            self.decision_type_counts,
            self.decision_type_nll_sums,
            decision_types,
            labeled,
            safe_nll,
        )
        confidence = batch.label_confidence.to(exact.device)
        confidence_bins = torch.bucketize(confidence, self.confidence_boundaries)
        _add_fixed_bin_totals(
            self.confidence_counts,
            self.confidence_nll_sums,
            confidence_bins,
            labeled,
            safe_nll,
        )

        candidate_sizes = candidate_offsets[1:] - candidate_offsets[:-1]
        batch_candidate_counts = torch.bincount(candidate_sizes)
        if self.candidate_counts.numel() < batch_candidate_counts.numel():
            padding = self.candidate_counts.new_zeros(
                batch_candidate_counts.numel() - self.candidate_counts.numel()
            )
            self.candidate_counts = torch.cat((self.candidate_counts, padding))
        self.candidate_counts[: batch_candidate_counts.numel()].add_(batch_candidate_counts)

    def finalize(self) -> BCEvaluationMetrics:
        counts = self.counts.cpu().tolist()
        nll_sums = self.nll_sums.cpu().tolist()
        candidate_counts = self.candidate_counts.cpu().tolist()
        value_count = int(self.value_count.item())
        illegal_mass_count = int(self.illegal_mass_count.item())
        decisions, labeled, unknown, exact, partial = (int(value) for value in counts[:5])
        return BCEvaluationMetrics(
            overall_nll=float(nll_sums[0]) / max(labeled, 1),
            exact_nll=float(nll_sums[1]) / max(exact, 1),
            partial_nll=float(nll_sums[2]) / max(partial, 1),
            exact_joint_accuracy=int(counts[5]) / max(exact, 1),
            decisions=decisions,
            labeled_decisions=labeled,
            unknown_decisions=unknown,
            exact_decisions=exact,
            partial_decisions=partial,
            illegal_predictions=int(counts[6]),
            non_finite_values=int(counts[7]),
            by_decision_type=_finalize_evaluation_bins(
                self.decision_type_counts,
                self.decision_type_nll_sums,
            ),
            confidence_buckets=_finalize_evaluation_bins(
                self.confidence_counts,
                self.confidence_nll_sums,
                names=_CONFIDENCE_BUCKET_NAMES,
            ),
            candidate_set_sizes={
                str(size): int(count) for size, count in enumerate(candidate_counts) if count
            },
            value_loss=float(self.value_loss_sum.item()) / max(value_count, 1),
            value_decisions=value_count,
            illegal_probability_mass=float(self.illegal_mass_sum.item())
            / max(illegal_mass_count, 1),
            unknown_legality_decisions=int(self.unknown_legality_count.item()),
        )
