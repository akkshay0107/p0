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


def _swap_preview_pair[Action: (Tensor, int)](action: Action) -> Action:
    """Return the team-preview action that names the same two Pokemon in the other order."""
    return (action % TEAM_SIZE) * TEAM_SIZE + action // TEAM_SIZE


def label_masks(label_kind: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return the exact, partial, unknown, and labeled decision masks."""
    exact = label_kind == int(LabelKind.EXACT)
    partial = label_kind == int(LabelKind.PARTIAL)
    unknown = label_kind == int(LabelKind.UNKNOWN)
    return exact, partial, unknown, exact | partial


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
        proven = ~slot_legality_unknown(numerical)

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
        swapped_target = _swap_preview_pair(exact_actions)
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
