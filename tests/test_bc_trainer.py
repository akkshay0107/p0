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
        summary=None,
    )


def _trainer(chunk: ReplayGameChunk, *, chunk_length: int = 2) -> BCTrainer:
    policy = build_policy(
        ModelConfig(64, 4, 1, 8, 128),
        default_runtime_resources(),
    )
    return BCTrainer(
        policy,
        (chunk,),
        BCConfig(
            chunk_length=chunk_length,
            batch_decisions=2,
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
    trainer = _trainer(chunk, chunk_length=1)
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


def test_unknown_decision_advances_recurrence_without_policy_loss() -> None:
    chunk = _chunk(
        [int(LabelKind.UNKNOWN), int(LabelKind.EXACT)],
        [(7, 8)],
        [0, 0, 1],
    )
    metrics = _trainer(chunk).train()
    assert metrics.decisions == 2
    assert metrics.labeled_decisions == 1
    assert metrics.exact_decisions == 1
