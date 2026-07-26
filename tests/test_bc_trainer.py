import pytest
import torch

from p0.format_config import FORMAT
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.replays.dataset import ReplayGameChunk
from p0.replays.schema import LabelKind
from p0.training.bc import BCTrainer
from p0.training.config import BCConfig


def _chunk(label_kind: list[int], candidate_values: list[tuple[int, int]], offsets: list[int]):
    length = len(label_kind)
    observations = StructuredObservation.empty_batch(length)
    action_mask = torch.zeros((length, 2, FORMAT.action_size), dtype=torch.bool)
    action_mask[:, 0, 7] = True
    action_mask[:, 0, 9] = True
    action_mask[:, 1, 8] = True
    action_mask[:, 1, 10] = True
    return ReplayGameChunk(
        series_id="series-1",
        game_number=1,
        player=0,
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
        summary_inputs=(),
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

    assert metrics.decisions == 2
    assert metrics.labeled_decisions == 2
    assert metrics.updates == 2
    assert torch.isfinite(torch.tensor(metrics.loss))
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
    assert metrics.decisions == 2
    assert metrics.labeled_decisions == 1
    assert metrics.exact_decisions == 1


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


def test_multi_epoch_training_rejects_one_shot_dataset() -> None:
    chunk = _chunk([int(LabelKind.EXACT)], [(7, 8)], [0, 1])
    trainer = _trainer(chunk)
    trainer.dataset = iter((chunk,))
    trainer.config = BCConfig(epochs=2, amp=False)

    with pytest.raises(ValueError, match="re-iterable"):
        trainer.train()
