"""Behavior-cloning objectives and vectorized evaluation aggregation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from p0.battle.actions import TEAM_SIZE
from p0.model.structured_observation import (
    NUM_IDX_SLOT_LEGALITY_UNKNOWN,
    TOKEN_IDX_ALLY_SIDE,
)
from p0.replays.schema import LabelKind

LOG_PROBABILITY_TOLERANCE = 1e-6


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
    decisions: int
    labeled_count: int
    exact_count: int
    partial_count: int
    value_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


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
    if label_kind.dtype != torch.long:
        raise ValueError("label_kind must use torch.long labels")
    if not loss_mask.is_floating_point():
        raise ValueError("loss_mask must use a floating-point dtype")
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
    if torch.any(~torch.isfinite(loss_mask)) or torch.any((loss_mask < 0) | (loss_mask > 1)):
        raise ValueError("loss_mask values must be finite and in [0, 1]")

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


def _policy_loss_sum(
    marginal_log_probs: Tensor,
    loss_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the weighted policy numerator and its positive-weight denominator."""
    loss_weight = loss_mask.sum()
    # Selection avoids the undefined zero-times-negative-infinity result for UNKNOWN rows.
    policy_sum = torch.where(
        loss_mask > 0,
        -marginal_log_probs * loss_mask,
        torch.zeros_like(marginal_log_probs),
    ).sum()
    return policy_sum, loss_weight


def compute_bc_objective(
    candidate_log_probs: Tensor,
    candidate_offsets: Tensor,
    label_kind: Tensor,
    loss_mask: Tensor,
) -> BCObjective:
    """Compute exact and candidate-marginalized NLL without dropping unknown steps."""
    if candidate_log_probs.dim() != 1:
        raise ValueError("candidate_log_probs must be one-dimensional")
    if not candidate_log_probs.is_floating_point():
        raise ValueError("candidate_log_probs must use a floating-point dtype")
    if (
        candidate_offsets.device != candidate_log_probs.device
        or label_kind.device != candidate_log_probs.device
        or loss_mask.device != candidate_log_probs.device
    ):
        raise ValueError("objective tensors must share a device")

    if torch.any(torch.isnan(candidate_log_probs)) or torch.any(
        candidate_log_probs > LOG_PROBABILITY_TOLERANCE
    ):
        raise ValueError("candidate_log_probs must contain valid log probabilities")

    exact, partial, _, labeled = _validate_objective_inputs(
        candidate_log_probs.numel(),
        candidate_offsets,
        label_kind,
        loss_mask,
    )
    marginal_log_probs = _ragged_logsumexp(candidate_log_probs, candidate_offsets)
    labeled_marginals = marginal_log_probs[labeled]
    if torch.any(~torch.isfinite(labeled_marginals)):
        raise ValueError("Every labeled decision must have positive finite candidate mass")
    if torch.any(labeled_marginals > LOG_PROBABILITY_TOLERANCE):
        raise ValueError("Candidate probability mass must not exceed one")
    weighted_loss_sum, loss_weight = _policy_loss_sum(marginal_log_probs, loss_mask)
    exact_count = exact.sum()
    partial_count = partial.sum()
    exact_nll = (-marginal_log_probs[exact]).sum() / exact_count.clamp_min(1)
    partial_nll = (-marginal_log_probs[partial]).sum() / partial_count.clamp_min(1)
    denominator = torch.where(
        loss_weight > 0,
        loss_weight,
        torch.ones_like(loss_weight),
    )
    loss = weighted_loss_sum / denominator

    return BCObjective(
        loss=loss,
        exact_nll=exact_nll,
        partial_nll=partial_nll,
        marginal_log_probs=marginal_log_probs,
        exact_count=exact_count,
        partial_count=partial_count,
        labeled_count=labeled.sum(),
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
        selected_predictions = predictions[mask].float()
        selected_targets = targets[mask].float()
        finite = torch.isfinite(selected_predictions) & torch.isfinite(selected_targets)
        self.counts[6] += (~finite).sum()
        error = torch.where(
            finite,
            selected_predictions - selected_targets,
            torch.zeros_like(selected_predictions),
        )
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
        finite = torch.isfinite(illegal_mass)
        self.counts[6] += (proven & ~finite).sum()
        safe_illegal_mass = torch.where(finite, illegal_mass, torch.zeros_like(illegal_mass))

        self.illegal_mass_sum += safe_illegal_mass.mul(proven).sum().to(torch.float64)
        self.illegal_mass_count += proven.sum()

    def add(
        self,
        *,
        exact_actions: Tensor,
        masks: tuple[Tensor, Tensor, Tensor, Tensor],
        marginal_nll: Tensor,
        predicted: Tensor,
        best_scores: Tensor,
        team_preview: Tensor,
    ) -> None:
        exact, partial, unknown, labeled = masks
        ordered_correct = torch.all(predicted == exact_actions, dim=1)
        target_first, target_second = exact_actions.unbind(dim=-1)
        swapped_target = torch.stack(
            (
                (target_first % TEAM_SIZE) * TEAM_SIZE + target_first // TEAM_SIZE,
                (target_second % TEAM_SIZE) * TEAM_SIZE + target_second // TEAM_SIZE,
            ),
            dim=-1,
        )
        preview_correct = (
            (predicted[:, 0] == exact_actions[:, 0]) | (predicted[:, 0] == swapped_target[:, 0])
        ) & ((predicted[:, 1] == exact_actions[:, 1]) | (predicted[:, 1] == swapped_target[:, 1]))
        exact_correct = torch.where(team_preview, preview_correct, ordered_correct)[exact].sum()
        exact_count = exact.sum()
        partial_count = partial.sum()
        unknown_count = unknown.sum()
        labeled_count = exact_count + partial_count
        self.counts += torch.stack(
            (
                labeled_count + unknown_count,
                labeled_count,
                unknown_count,
                exact_count,
                partial_count,
                exact_correct,
                (~torch.isfinite(best_scores)).sum()
                + ((~torch.isfinite(marginal_nll)) & labeled).sum(),
            )
        )
        safe_nll = torch.where(
            torch.isfinite(marginal_nll),
            marginal_nll,
            torch.zeros_like(marginal_nll),
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
        decisions, labeled, unknown, exact, partial, correct, nonfinite = map(
            int, self.counts.cpu().tolist()
        )
        overall_sum, exact_sum, partial_sum = self.nll_sums.cpu().tolist()
        value_count = int(self.value_count.item())
        illegal_mass_count = int(self.illegal_mass_count.item())
        return BCEvaluationMetrics(
            overall_nll=overall_sum / max(labeled, 1),
            exact_nll=exact_sum / max(exact, 1),
            partial_nll=partial_sum / max(partial, 1),
            exact_joint_accuracy=correct / max(exact, 1),
            value_loss=float(self.value_loss_sum.item()) / max(value_count, 1),
            illegal_probability_mass=float(self.illegal_mass_sum.item())
            / max(illegal_mass_count, 1),
            unknown_label_fraction=unknown / max(decisions, 1),
            non_finite_values=nonfinite,
            decisions=decisions,
            labeled_count=labeled,
            exact_count=exact,
            partial_count=partial,
            value_count=value_count,
        )
