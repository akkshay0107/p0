import math

import pytest
import torch

from p0.replays.schema import LabelKind
from p0.training.bc import compute_bc_objective


def test_exact_and_partial_losses_match_probability_definitions() -> None:
    log_probs = torch.tensor([math.log(0.25), math.log(0.5), math.log(0.25)])
    offsets = torch.tensor([0, 1, 3, 3], dtype=torch.long)
    labels = torch.tensor([int(LabelKind.EXACT), int(LabelKind.PARTIAL), int(LabelKind.UNKNOWN)])
    loss_mask = torch.tensor([1.0, 1.0, 0.0])

    result = compute_bc_objective(log_probs, offsets, labels, loss_mask)

    expected_exact = -math.log(0.25)
    expected_partial = -math.log(0.75)
    assert result.exact_count == 1 and result.partial_count == 1
    assert result.labeled_count == 2
    assert result.exact_nll.item() == pytest.approx(expected_exact)
    assert result.partial_nll.item() == pytest.approx(expected_partial)
    assert result.loss.item() == pytest.approx((expected_exact + expected_partial) / 2)
    assert result.marginal_log_probs[2].isneginf()


def test_unknown_steps_have_zero_loss_and_preserve_boundaries() -> None:
    log_probs = torch.empty(0, requires_grad=True)
    offsets = torch.tensor([0, 0, 0], dtype=torch.long)
    labels = torch.tensor([int(LabelKind.UNKNOWN), int(LabelKind.UNKNOWN)])
    loss_mask = torch.zeros(2)

    result = compute_bc_objective(log_probs, offsets, labels, loss_mask)
    assert result.loss.item() == 0.0
    result.loss.backward()
    assert log_probs.grad is not None and log_probs.grad.numel() == 0


def test_partial_loss_is_candidate_order_invariant() -> None:
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


def test_candidate_objective_preserves_gradients() -> None:
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
    "labels, offsets, mask, message",
    [
        ([int(LabelKind.EXACT)], [0, 2], [1.0], "EXACT"),
        ([int(LabelKind.PARTIAL)], [0, 1], [1.0], "PARTIAL"),
        ([int(LabelKind.UNKNOWN)], [0, 1], [0.0], "UNKNOWN"),
        ([99], [0, 0], [0.0], "unsupported"),
    ],
)
def test_invalid_label_and_candidate_shapes_are_rejected(labels, offsets, mask, message) -> None:
    with pytest.raises(ValueError, match=message):
        compute_bc_objective(
            torch.full((offsets[-1],), math.log(0.5)),
            torch.tensor(offsets, dtype=torch.long),
            torch.tensor(labels),
            torch.tensor(mask),
        )
