import pytest
import torch

from p0.format_config import FORMAT
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.policy import EncodedObs
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import StructuredObservation


@pytest.fixture
def policy():
    result = build_policy(
        ModelConfig(64, 4, 1, 8, 128),
        default_runtime_resources(),
    )
    result.eval()
    return result


def _inputs(policy, batch_size: int = 2):
    observations = StructuredObservation.empty_batch(batch_size)
    action_mask = torch.ones((batch_size, 2, FORMAT.action_size), dtype=torch.bool)
    encoded = policy.encode(observations, action_mask)
    state = policy.initial_state(batch_size)
    return encoded, action_mask, state


def test_singleton_candidate_scores_match_standard_joint_scoring(policy) -> None:
    encoded, action_mask, state = _inputs(policy)
    candidates = torch.tensor([[7, 8], [9, 10]], dtype=torch.long)
    offsets = torch.tensor([0, 1, 2], dtype=torch.long)

    with torch.no_grad():
        candidate_scores = policy.actor.score_joint_candidates(
            encoded, action_mask, state, candidates, offsets
        )
        expected = torch.stack(
            [
                policy.actor.score(
                    EncodedObs(
                        encoded.tokens[index : index + 1],
                        encoded.aux[index : index + 1],
                        encoded.numerical[index : index + 1],
                    ),
                    action_mask[index : index + 1],
                    candidates[index : index + 1],
                    state[index : index + 1],
                )[1][0]
                for index in range(2)
            ]
        )
    torch.testing.assert_close(candidate_scores, expected)


def test_candidate_scoring_runs_reducer_once_per_observation_batch(policy) -> None:
    encoded, action_mask, state = _inputs(policy)
    candidates = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.long)
    offsets = torch.tensor([0, 2, 3], dtype=torch.long)
    calls = []

    def record_call(*_args):
        calls.append(True)

    handle = policy.actor.reducer.register_forward_hook(record_call)
    try:
        policy.actor.score_joint_candidates(encoded, action_mask, state, candidates, offsets)
    finally:
        handle.remove()
    assert calls == [True]


def test_candidate_order_does_not_change_scores(policy) -> None:
    encoded, action_mask, state = _inputs(policy, batch_size=1)
    candidates = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.long)
    offsets = torch.tensor([0, 3], dtype=torch.long)
    with torch.no_grad():
        first = policy.actor.score_joint_candidates(
            encoded, action_mask, state, candidates, offsets
        )
        permutation = torch.tensor([2, 0, 1])
        second = policy.actor.score_joint_candidates(
            encoded,
            action_mask,
            state,
            candidates[permutation],
            offsets,
        )
    torch.testing.assert_close(first, second[torch.argsort(permutation)])


def test_candidate_scoring_applies_second_action_mask(policy) -> None:
    encoded, _, state = _inputs(policy, batch_size=1)
    action_mask = torch.zeros((1, 2, FORMAT.action_size), dtype=torch.bool)
    action_mask[:, 0, 7] = True
    action_mask[:, 1, 8] = True
    candidates = torch.tensor([[7, 8], [7, 7]], dtype=torch.long)
    offsets = torch.tensor([0, 2], dtype=torch.long)
    scores = policy.actor.score_joint_candidates(encoded, action_mask, state, candidates, offsets)
    assert torch.isfinite(scores[0])
    assert torch.isneginf(scores[1])


def test_candidate_scoring_rejects_malformed_ragged_inputs(policy) -> None:
    encoded, action_mask, state = _inputs(policy, batch_size=1)
    candidates = torch.tensor([[7, 8]], dtype=torch.long)
    with pytest.raises(ValueError, match="one boundary"):
        policy.actor.score_joint_candidates(
            encoded, action_mask, state, candidates, torch.tensor([0, 1, 1])
        )
    with pytest.raises(ValueError, match="action ids"):
        policy.actor.score_joint_candidates(
            encoded,
            action_mask,
            state,
            candidates.to(torch.float32),
            torch.tensor([0, 1]),
        )
