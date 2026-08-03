from __future__ import annotations

import pytest
import torch

from p0.battle.actions import FORCED_ACTION, MOVE_START
from p0.format_config import FORMAT
from p0.model.architecture_contract import HISTORY_WINDOW
from p0.model.structured_observation import (
    CAT_KNOWNNESS_START,
    CAT_KNOWNNESS_WIDTH,
    StructuredObservation,
)
from tests.stress._helpers import (
    capture_showdown_decisions,
    stress_batch_sizes,
    stress_repetitions,
)

ACT_SIZE = FORMAT.action_size


def _minimal_model_action_mask(batch_size: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((batch_size, 2, ACT_SIZE), dtype=torch.bool, device=device)
    mask[:, :, 0] = True
    mask[:, :, MOVE_START] = True
    mask[:, :, FORCED_ACTION] = True
    return mask


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", stress_batch_sizes())
async def test_policy_handles_showdown_captured_batches(
    showdown_server,
    stress_policy,
    stress_device: torch.device,
    batch_size: int,
) -> None:
    decisions = await capture_showdown_decisions(
        showdown_server,
        game_count=1,
    )
    assert decisions
    selected = tuple(decisions[index % len(decisions)] for index in range(batch_size))
    observation = StructuredObservation.stack([decision.observation for decision in selected]).to(
        stress_device
    )
    action_mask = torch.zeros((batch_size, 2, ACT_SIZE), dtype=torch.bool, device=stress_device)
    for index, decision in enumerate(selected):
        for position, actions in enumerate(decision.legal_actions):
            action_mask[index, position, list(actions)] = True
    memory = stress_policy.empty_memory(batch_size)

    with torch.inference_mode():
        acted = stress_policy.act_obs(observation, action_mask, *memory)
        evaluated = stress_policy.evaluate_obs(
            observation,
            action_mask,
            acted.actions,
            *memory,
        )

    assert acted.actions.shape == (batch_size, 2)
    assert torch.all((acted.actions >= 0) & (acted.actions < ACT_SIZE))
    assert torch.all(action_mask.gather(2, acted.actions.unsqueeze(-1)).squeeze(-1))
    assert torch.isfinite(acted.log_probs).all()
    assert torch.isfinite(acted.value).all()
    assert torch.isfinite(evaluated.log_probs).all()
    assert torch.isfinite(evaluated.entropy).all()
    assert torch.isfinite(evaluated.value).all()
    for index, decision in enumerate(selected):
        actions = tuple(int(value) for value in acted.actions[index].tolist())
        assert actions in decision.legal_joint_actions

    assert torch.isfinite(evaluated.logits.masked_select(action_mask)).all()


@pytest.mark.stress
@pytest.mark.parametrize("batch_size", stress_batch_sizes())
def test_policy_handles_empty_and_maximum_memory_inputs(
    stress_policy,
    stress_device: torch.device,
    batch_size: int,
) -> None:
    """Exercise the documented zero-memory and full Bo3/history contracts."""
    observation = StructuredObservation.empty_batch(batch_size).to(stress_device)
    # The replay fixture contains unknown identifiers; this is the explicit
    # observation-side encoding for those values, independent of tokenizer output.
    observation.categorical[
        ..., CAT_KNOWNNESS_START : CAT_KNOWNNESS_START + CAT_KNOWNNESS_WIDTH
    ] = 4
    action_mask = _minimal_model_action_mask(batch_size, stress_device)

    empty_memory = stress_policy.empty_memory(batch_size)
    with torch.inference_mode():
        empty_acted = stress_policy.act_obs(observation, action_mask, *empty_memory)
    assert torch.isfinite(empty_acted.log_probs).all()
    assert torch.isfinite(empty_acted.value).all()
    assert torch.all(action_mask.gather(2, empty_acted.actions.unsqueeze(-1)).squeeze(-1))

    history = torch.linspace(
        -1.0,
        1.0,
        steps=batch_size * HISTORY_WINDOW * stress_policy.d_model,
        device=stress_device,
    ).reshape(batch_size, HISTORY_WINDOW, stress_policy.d_model)
    history_mask = torch.ones((batch_size, HISTORY_WINDOW), dtype=torch.bool, device=stress_device)
    history_age_ids = torch.arange(HISTORY_WINDOW, device=stress_device).expand(batch_size, -1)
    prior_games = [
        [
            torch.zeros((HISTORY_WINDOW, stress_policy.d_model), device=stress_device),
            torch.ones((HISTORY_WINDOW, stress_policy.d_model), device=stress_device),
        ]
        for _ in range(batch_size)
    ]
    series_tokens, series_mask = stress_policy.encode_series(prior_games)

    with torch.inference_mode():
        acted = stress_policy.act_obs(
            observation,
            action_mask,
            series_tokens,
            series_mask,
            history,
            history_mask,
            history_age_ids,
        )

    assert acted.actions.shape == (batch_size, 2)
    assert torch.isfinite(acted.log_probs).all()
    assert torch.isfinite(acted.value).all()
    assert torch.all(action_mask.gather(2, acted.actions.unsqueeze(-1)).squeeze(-1))
    assert series_mask.all()
    assert history_mask.all()

    # The same full-memory path must remain stable across repeated calls.
    for _ in range(stress_repetitions(default=2)):
        with torch.inference_mode():
            repeated = stress_policy.act_obs(
                observation,
                action_mask,
                series_tokens,
                series_mask,
                history,
                history_mask,
                history_age_ids,
            )
        assert torch.isfinite(repeated.log_probs).all()


@pytest.mark.stress
def test_policy_scores_ragged_candidates_with_empty_decision_rows(
    stress_policy,
    stress_device: torch.device,
) -> None:
    """Verify candidate offsets support variable counts without flattening assumptions."""
    batch_size = 3
    observation = StructuredObservation.empty_batch(batch_size).to(stress_device)
    action_mask = torch.ones((batch_size, 2, ACT_SIZE), dtype=torch.bool, device=stress_device)
    memory = stress_policy.empty_memory(batch_size)
    encoded = stress_policy.encode(observation, action_mask)
    candidate_values = torch.tensor(
        ((MOVE_START, MOVE_START), (FORCED_ACTION, 0), (1, 2), (MOVE_START, FORCED_ACTION)),
        dtype=torch.long,
        device=stress_device,
    )
    # Decision 1 intentionally owns no candidates: [0:2], [2:2], [2:4].
    candidate_offsets = torch.tensor((0, 2, 2, 4), dtype=torch.long)

    with torch.inference_mode():
        log_probs = stress_policy.actor.score_joint_candidates(
            encoded,
            action_mask,
            *memory,
            candidate_values,
            candidate_offsets,
        )

    assert log_probs.shape == (candidate_values.shape[0],)
    assert torch.isfinite(log_probs).all()
