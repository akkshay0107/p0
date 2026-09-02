"""Tests for behavioral-cloning objectives."""

from __future__ import annotations

import math

import pytest
import torch

from p0.format_config import FORMAT
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import (
    StructuredObservation,
)
from p0.replays.dataset import ReplayGameChunk
from p0.replays.schema import LabelKind
from p0.training.bc import (
    BCTrainer,
    compute_bc_objective,
)
from p0.training.config import BCConfig


def _chunk(
    label_kind: list[int],
    candidate_values: list[tuple[int, int]],
    offsets: list[int],
    *,
    game_number: int = 1,
    is_series_end: bool = False,
) -> ReplayGameChunk:
    """Helper creating a synthetic ReplayGameChunk for BC trainer unit tests."""
    length = len(label_kind)
    observations = StructuredObservation.empty_batch(length)
    action_mask = torch.zeros((length, 2, FORMAT.action_size), dtype=torch.bool)
    action_mask[:, 0, 7] = True
    action_mask[:, 0, 9] = True
    action_mask[:, 1, 8] = True
    action_mask[:, 1, 10] = True
    return ReplayGameChunk(
        series_id="series-1",
        game_number=game_number,
        player=0,
        canonical_player=0,
        observations=observations,
        action_mask=action_mask,
        mask_provenance=torch.ones(length, dtype=torch.long),
        label_kind=torch.tensor(label_kind, dtype=torch.long),
        label_confidence=torch.ones(length),
        loss_mask=torch.tensor([float(kind != int(LabelKind.UNKNOWN)) for kind in label_kind]),
        decision_type=torch.ones(length, dtype=torch.long),
        exact_action=torch.tensor([(7, 8)] * length, dtype=torch.long),
        candidate_values=torch.tensor(candidate_values, dtype=torch.long).reshape(-1, 2),
        candidate_offsets=torch.tensor(offsets, dtype=torch.long),
        outcome=torch.ones(length),
        is_series_end=is_series_end,
    )


def _trainer(chunk: ReplayGameChunk, *, minibatch_size: int = 2) -> BCTrainer:
    """Helper building a minimal BCTrainer instance over a single game chunk."""
    policy = build_policy(
        ModelConfig(64, 4, 1, 128),
        default_runtime_resources(),
    )
    return BCTrainer(
        policy,
        (chunk,),
        BCConfig(
            batch_decisions=minibatch_size,
            learning_rate=1e-3,
            epochs=1,
            num_workers=0,
            enable_optim=False,
        ),
        device="cpu",
    )


class TestBCObjectives:
    def test_exact_and_partial_losses_match_probability_definitions(self) -> None:
        """
        Verify behavior cloning objective calculations for EXACT, PARTIAL, and UNKNOWN label kinds.

        Mathematical Formulation:
        - EXACT label: Standard Negative Log Likelihood: NLL = -log(P(action))
        - PARTIAL label: Marginal Negative Log Likelihood over candidate set: NLL = -log(sum_{c in C} P(c))
        - UNKNOWN label: loss_mask == 0, excluded from loss (loss == 0.0)
        """
        log_probs = torch.tensor([math.log(0.25), math.log(0.5), math.log(0.25)])
        offsets = torch.tensor([0, 1, 3, 3], dtype=torch.long)
        labels = torch.tensor(
            [int(LabelKind.EXACT), int(LabelKind.PARTIAL), int(LabelKind.UNKNOWN)]
        )
        loss_mask = torch.tensor([1.0, 1.0, 0.0])

        result = compute_bc_objective(log_probs, offsets, labels, loss_mask)

        expected_exact = -math.log(0.25)
        # Partial candidate set contains probs 0.5 and 0.25 -> sum = 0.75
        expected_partial = -math.log(0.75)
        assert result.exact_count == 1 and result.partial_count == 1
        assert result.labeled_count == 2
        assert result.loss_weight == 2.0
        assert result.exact_nll.item() == pytest.approx(expected_exact)
        assert result.partial_nll.item() == pytest.approx(expected_partial)
        assert result.loss.item() == pytest.approx((expected_exact + expected_partial) / 2)
        assert result.marginal_log_probs[2].isneginf()

    def test_fractional_loss_weights_do_not_change_labeled_counts(self) -> None:
        """Verify fractional sample weights scale total loss without distorting discrete labeled count metrics."""
        result = compute_bc_objective(
            torch.log(torch.tensor([0.25, 0.75])),
            torch.tensor([0, 1, 2], dtype=torch.long),
            torch.tensor([int(LabelKind.EXACT), int(LabelKind.EXACT)]),
            torch.tensor([0.25, 0.75]),
        )

        assert result.labeled_count == 2
        assert result.loss_weight == 1.0
        assert result.loss.item() == pytest.approx(-0.25 * math.log(0.25) - 0.75 * math.log(0.75))

    def test_unknown_steps_have_zero_loss_and_preserve_boundaries(self) -> None:
        """Verify UNKNOWN labels produce zero loss and empty gradients without breaking backprop graph."""
        log_probs = torch.empty(0, requires_grad=True)
        offsets = torch.tensor([0, 0, 0], dtype=torch.long)
        labels = torch.tensor([int(LabelKind.UNKNOWN), int(LabelKind.UNKNOWN)])
        loss_mask = torch.zeros(2)

        result = compute_bc_objective(log_probs, offsets, labels, loss_mask)
        assert result.loss.item() == 0.0
        result.loss.backward()
        assert log_probs.grad is not None and log_probs.grad.numel() == 0

    def test_partial_loss_is_candidate_order_invariant(self) -> None:
        """Verify marginal log-sum-exp over candidate actions is invariant to internal candidate permutation."""
        first = compute_bc_objective(
            torch.log(torch.tensor([0.2, 0.3, 0.5])),
            torch.tensor([0, 3], dtype=torch.long),
            torch.tensor([int(LabelKind.PARTIAL)]),
            torch.ones(1),
        )
        second = compute_bc_objective(
            torch.log(torch.tensor([0.5, 0.2, 0.3])),
            torch.tensor([0, 3], dtype=torch.long),
            torch.tensor([int(LabelKind.PARTIAL)]),
            torch.ones(1),
        )
        torch.testing.assert_close(first.loss, second.loss)

    def test_candidate_objective_preserves_gradients(self) -> None:
        """Verify backward gradient flow through marginal candidate loss calculations."""
        probabilities = torch.tensor([0.2, 0.3, 0.5], requires_grad=True)
        log_probs = probabilities.log()
        result = compute_bc_objective(
            log_probs,
            torch.tensor([0, 3], dtype=torch.long),
            torch.tensor([int(LabelKind.PARTIAL)]),
            torch.ones(1),
        )
        result.loss.backward()
        assert probabilities.grad is not None
        assert torch.isfinite(probabilities.grad).all()

    @pytest.mark.parametrize(
        ("labels", "offsets", "mask", "message"),
        [
            ([int(LabelKind.EXACT)], [0, 2], [1.0], "EXACT"),
            ([int(LabelKind.PARTIAL)], [0, 1], [1.0], "PARTIAL"),
            ([int(LabelKind.UNKNOWN)], [0, 1], [0.0], "UNKNOWN"),
            ([99], [0, 0], [0.0], "unsupported"),
        ],
    )
    def test_invalid_label_and_candidate_shapes_are_rejected(
        self, labels, offsets, mask, message
    ) -> None:
        """Verify compute_bc_objective detects candidate count and label type mismatches."""
        with pytest.raises(ValueError, match=message):
            compute_bc_objective(
                torch.full((offsets[-1],), math.log(0.5)),
                torch.tensor(offsets, dtype=torch.long),
                torch.tensor(labels),
                torch.tensor(mask),
            )
