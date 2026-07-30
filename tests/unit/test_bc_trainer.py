from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from p0.format_config import FORMAT
from p0.model.architecture_contract import HISTORY_WINDOW
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.replays.dataset import ReplayGameChunk
from p0.replays.schema import LabelKind
from p0.training.bc import BCGameWindow, BCTrainer, collate_bc_batches
from p0.training.config import BCConfig


def _chunk(
    label_kind: list[int],
    candidate_values: list[tuple[int, int]],
    offsets: list[int],
    *,
    game_number: int = 1,
    is_series_end: bool = False,
):
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
            amp=False,
        ),
        device="cpu",
    )


def test_bc_trainer_updates_policy_in_game_local_chunks() -> None:
    chunk = _chunk(
        [int(LabelKind.EXACT), int(LabelKind.EXACT)],
        [(7, 8), (7, 8)],
        [0, 1, 2],
    )
    trainer = _trainer(chunk, minibatch_size=1)
    before = {
        name: parameter.detach().clone() for name, parameter in trainer.policy.named_parameters()
    }

    metrics = trainer.train()

    assert metrics["decisions"] == 2
    assert metrics["labeled_decisions"] == 2
    assert metrics["updates"] == 2
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in trainer.policy.named_parameters()
    )


def test_unknown_decision_is_excluded_without_breaking_game_context() -> None:
    chunk = _chunk(
        [int(LabelKind.UNKNOWN), int(LabelKind.EXACT)],
        [(7, 8)],
        [0, 0, 1],
    )
    metrics = _trainer(chunk).train()
    assert metrics["decisions"] == 2
    assert metrics["labeled_decisions"] == 1
    assert metrics["exact_decisions"] == 1


def test_bc_target_windows_keep_only_past_48_local_tokens() -> None:
    chunk = _chunk([int(LabelKind.EXACT)] * 4, [(7, 8)] * 4, [0, 1, 2, 3, 4])
    trainer = _trainer(chunk, minibatch_size=2)
    local_tokens = torch.randn(52, trainer.policy.d_model)

    whole = trainer._history_inputs(local_tokens)
    window = trainer._history_inputs(local_tokens, slice(48, 52))
    for whole_part, window_part in zip(whole, window, strict=True):
        torch.testing.assert_close(whole_part[48:], window_part)

    changed_ancient = local_tokens.clone()
    changed_ancient[0] += 1000.0
    original_last = trainer._history_inputs(local_tokens, slice(51, 52))
    changed_last = trainer._history_inputs(changed_ancient, slice(51, 52))
    for original_part, changed_part in zip(original_last, changed_last, strict=True):
        torch.testing.assert_close(original_part, changed_part)


def test_collated_context_is_compact_and_never_crosses_game_boundaries() -> None:
    first = _chunk(
        [int(LabelKind.EXACT), int(LabelKind.EXACT)],
        [(7, 8), (7, 8)],
        [0, 1, 2],
    )
    second = _chunk(
        [int(LabelKind.EXACT), int(LabelKind.EXACT)],
        [(7, 8), (7, 8)],
        [0, 1, 2],
    )
    second = replace(second, series_id="series-2")

    batch = next(collate_bc_batches((first, second), 4))

    assert all(isinstance(window, BCGameWindow) for window in batch.windows)
    assert batch.target_indices.tolist() == [0, 1, 2, 3]
    assert not torch.any(batch.history_mask[2])
    assert batch.history_indices[3, -1] == 2
    assert batch.history_mask[3, -1]
    for tensor in batch.observations.tensors():
        assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()


def test_model_inputs_encode_all_game_windows_once() -> None:
    first = _chunk(
        [int(LabelKind.EXACT), int(LabelKind.EXACT)],
        [(7, 8), (7, 8)],
        [0, 1, 2],
    )
    second = _chunk(
        [int(LabelKind.EXACT), int(LabelKind.EXACT)],
        [(7, 8), (7, 8)],
        [0, 1, 2],
    )
    second = replace(second, canonical_player=1)
    trainer = _trainer(first, minibatch_size=4)
    batch = next(collate_bc_batches((first, second), 4))

    with (
        torch.inference_mode(),
        patch.object(trainer.policy, "encode", wraps=trainer.policy.encode) as encode,
    ):
        model_inputs = trainer._model_inputs(batch)

    assert encode.call_count == 1
    assert model_inputs[0].tokens.size(0) == batch.decisions
    assert model_inputs[4].shape[:2] == (batch.decisions, HISTORY_WINDOW)


def test_continued_chunk_matches_per_window_reference_inputs() -> None:
    length = 100
    game = _chunk(
        [int(LabelKind.EXACT)] * length,
        [(7, 8)] * length,
        list(range(length + 1)),
    )
    trainer = _trainer(game, minibatch_size=64)
    trainer.policy.eval()
    batch = list(collate_bc_batches((game,), 64))[1]

    with torch.inference_mode():
        actual = trainer._model_inputs(batch)
        context_start = 64 - HISTORY_WINDOW
        encoded = trainer.policy.encode(
            game.observations[context_start:],
            game.action_mask[context_start:],
        )
        local_tokens = trainer.policy.local_history_tokens(encoded)
        expected_history = trainer._history_inputs(
            local_tokens,
            slice(HISTORY_WINDOW, game.length - context_start),
        )

    torch.testing.assert_close(actual[0].tokens, encoded.tokens[HISTORY_WINDOW:])
    torch.testing.assert_close(actual[0].aux, encoded.aux[HISTORY_WINDOW:])
    torch.testing.assert_close(actual[0].numerical, encoded.numerical[HISTORY_WINDOW:])
    for actual_part, expected_part in zip(actual[4:], expected_history, strict=True):
        torch.testing.assert_close(actual_part, expected_part)


def test_completed_game_does_not_emit_a_second_observation_payload() -> None:
    length = 100
    game = _chunk(
        [int(LabelKind.EXACT)] * length,
        [(7, 8)] * length,
        list(range(length + 1)),
    )

    first, second = collate_bc_batches((game,), 64)

    assert not first.windows[0].is_game_end
    assert second.windows[0].is_game_end
    assert not hasattr(second, "completed_games")
    assert second.observations.categorical.size(0) == HISTORY_WINDOW + length - 64


def test_raw_game_history_caches_each_target_token_once() -> None:
    length = 100
    game = _chunk(
        [int(LabelKind.EXACT)] * length,
        [(7, 8)] * length,
        list(range(length + 1)),
    )
    trainer = _trainer(game, minibatch_size=64)
    first, second = collate_bc_batches((game,), 64)

    with torch.inference_mode():
        trainer._model_inputs(first)
        trainer._model_inputs(second)

    cached = trainer._series_history.completed_before(game.series_key, 2)
    assert len(cached) == 1
    assert next(iter(cached.values())).shape == (length, trainer.policy.d_model)


def test_same_batch_next_game_receives_differentiable_series_context() -> None:
    first = _chunk(
        [int(LabelKind.EXACT)] * 2,
        [(7, 8)] * 2,
        [0, 1, 2],
        game_number=1,
    )
    second = _chunk(
        [int(LabelKind.EXACT)] * 2,
        [(7, 8)] * 2,
        [0, 1, 2],
        game_number=2,
        is_series_end=True,
    )
    trainer = _trainer(first, minibatch_size=4)
    batch = next(collate_bc_batches((first, second), 4))

    prepared = trainer._prepare_model_inputs(batch)
    series_mask = prepared.model_inputs[3]

    assert not series_mask[:2].any()
    assert series_mask[2:, :4].all()
    assert not series_mask[2:, 4:].any()
    assert prepared.model_inputs[2].requires_grad


def test_bc_loss_trains_series_resampler() -> None:
    first = _chunk(
        [int(LabelKind.EXACT)] * 2,
        [(7, 8)] * 2,
        [0, 1, 2],
        game_number=1,
    )
    second = _chunk(
        [int(LabelKind.EXACT)] * 2,
        [(7, 8)] * 2,
        [0, 1, 2],
        game_number=2,
        is_series_end=True,
    )
    trainer = _trainer(first, minibatch_size=4)
    batch = next(collate_bc_batches((first, second), 4))
    totals: dict[str, float | int] = {
        "loss": 0.0,
        "exact_nll": 0.0,
        "partial_nll": 0.0,
        "decisions": 0,
        "labeled_decisions": 0,
        "exact_decisions": 0,
        "partial_decisions": 0,
        "updates": 0,
        "games": 0,
    }

    trainer._backward_chunk(batch, totals)

    gradients = [
        parameter.grad
        for parameter in trainer.policy.series.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert any(torch.count_nonzero(gradient) for gradient in gradients)


def test_evaluate_encodes_each_collated_batch_once() -> None:
    first = _chunk(
        [int(LabelKind.EXACT)] * 3,
        [(7, 8)] * 3,
        [0, 1, 2, 3],
        game_number=1,
    )
    second = _chunk(
        [int(LabelKind.EXACT)] * 3,
        [(7, 8)] * 3,
        [0, 1, 2, 3],
        game_number=2,
        is_series_end=True,
    )
    trainer = _trainer(first, minibatch_size=2)

    with patch.object(trainer.policy, "encode", wraps=trainer.policy.encode) as encode:
        trainer.evaluate((first, second))

    assert encode.call_count == 3


def test_multi_epoch_training_rejects_one_shot_dataset() -> None:
    chunk = _chunk([int(LabelKind.EXACT)], [(7, 8)], [0, 1])
    trainer = _trainer(chunk)
    trainer.dataset = iter((chunk,))
    trainer.config = BCConfig(epochs=2, amp=False)

    with pytest.raises(ValueError, match="re-iterable"):
        trainer.train()
