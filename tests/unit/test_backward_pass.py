import pytest
import torch

from p0.format_config import FORMAT
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import (
    CAT_EFFECT_START,
    CAT_IDX_STATUS_COUNTER_KIND,
    CAT_KNOWNNESS_START,
    CAT_KNOWNNESS_WIDTH,
    CATEGORICAL_WIDTH,
    EFFECT_CATEGORICAL_WIDTH,
    EFFECT_NUMERICAL_WIDTH,
    EVENT_CATEGORICAL_WIDTH,
    EVENT_COUNT,
    EVENT_NUMERICAL_WIDTH,
    NUM_EFFECT_START,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    SideId,
    StructuredObservation,
    TokenType,
)
from p0.training.config import TrainingConfig
from p0.training.magnet import Magnet
from p0.training.ppo import _run_batched_ppo
from p0.training.trajectory import TrajectoryBatch

ACT_SIZE = FORMAT.action_size


@pytest.fixture
def dummy_obs():
    B = 2

    token_type_ids = torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long)
    token_type_ids[:, 0:12] = TokenType.POKEMON
    token_type_ids[:, 12:15] = TokenType.FIELD

    side_ids = torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long)
    side_ids[:, 0:6] = SideId.ALLY
    side_ids[:, 6:12] = SideId.OPPONENT
    side_ids[:, 12] = SideId.NONE
    side_ids[:, 13] = SideId.ALLY
    side_ids[:, 14] = SideId.OPPONENT

    slot_ids = torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long)
    for i in range(6):
        slot_ids[:, i] = i + 1
        slot_ids[:, 6 + i] = i + 1

    # Populate categorical with random IDs respecting vocab limits
    categorical = torch.zeros((B, SEQUENCE_LENGTH, CATEGORICAL_WIDTH), dtype=torch.long)

    # Pokemon tokens (0-11)
    # species (0): 1-34
    categorical[:, 0:12, 0] = torch.randint(1, 35, (B, 12))
    # ability (1): 1-23
    categorical[:, 0:12, 1] = torch.randint(1, 24, (B, 12))
    # item (2): 1-18
    categorical[:, 0:12, 2] = torch.randint(1, 19, (B, 12))
    # types (3,4): 1-18
    categorical[:, 0:12, 3:5] = torch.randint(1, 19, (B, 12, 2))
    # moves (5-8): 1-69
    categorical[:, 0:12, 5:9] = torch.randint(1, 70, (B, 12, 4))
    # move_types (9-12): 1-18
    categorical[:, 0:12, 9:13] = torch.randint(1, 19, (B, 12, 4))
    # move_categories (13-16): 1-3
    categorical[:, 0:12, 13:17] = torch.randint(1, 4, (B, 12, 4))
    # status (17): 1-6
    categorical[:, 0:12, 17] = torch.randint(1, 7, (B, 12))
    # status counter kind: 0-4
    categorical[:, 0:12, CAT_IDX_STATUS_COUNTER_KIND] = torch.randint(0, 5, (B, 12))
    categorical[:, 0:12, CAT_KNOWNNESS_START : CAT_KNOWNNESS_START + CAT_KNOWNNESS_WIDTH] = (
        torch.randint(1, 5, (B, 12, CAT_KNOWNNESS_WIDTH))
    )

    # Numerical features
    numerical = torch.randn((B, SEQUENCE_LENGTH, NUMERICAL_WIDTH))

    for token_idx in range(15):
        categorical[
            :, token_idx, CAT_EFFECT_START : CAT_EFFECT_START + EFFECT_CATEGORICAL_WIDTH
        ] = torch.tensor((1, 1, 1))
        numerical[:, token_idx, NUM_EFFECT_START : NUM_EFFECT_START + EFFECT_NUMERICAL_WIDTH] = 1.0

    # Populate valid orig_idxs to prevent random switch actions from crashing
    for i, idx in enumerate(range(0, 6)):
        numerical[:, idx, 26] = (i + 1) / 6.0

    numerical[:, 12, 2] = 1.0

    events_cat = torch.zeros((B, EVENT_COUNT, EVENT_CATEGORICAL_WIDTH), dtype=torch.long)
    events_cat[..., 0] = torch.randint(1, 19, (B, EVENT_COUNT))
    events_cat[..., 1] = torch.randint(1, 70, (B, EVENT_COUNT))
    events_cat[..., 2] = torch.randint(1, 19, (B, EVENT_COUNT))
    events_cat[..., 3] = torch.randint(1, 7, (B, EVENT_COUNT))
    events_cat[..., 4] = torch.randint(1, 25, (B, EVENT_COUNT))
    events_cat[..., 5] = torch.randint(1, 6, (B, EVENT_COUNT))
    events_cat[..., 6] = torch.randint(1, 19, (B, EVENT_COUNT))
    events_cat[..., 7] = torch.randint(0, 8, (B, EVENT_COUNT))
    events_cat[..., 8] = torch.randint(0, 3, (B, EVENT_COUNT))
    events_cat[..., 9] = torch.randint(0, 7, (B, EVENT_COUNT))

    events_num = torch.randn((B, EVENT_COUNT, EVENT_NUMERICAL_WIDTH))
    events_side_ids = torch.randint(0, 3, (B, EVENT_COUNT), dtype=torch.long)
    events_slot_ids = torch.randint(0, 7, (B, EVENT_COUNT), dtype=torch.long)

    return StructuredObservation(
        token_type_ids=token_type_ids,
        side_ids=side_ids,
        slot_ids=slot_ids,
        categorical=categorical,
        numerical=numerical,
        events_cat=events_cat,
        events_num=events_num,
        events_side_ids=events_side_ids,
        events_slot_ids=events_slot_ids,
        events_metadata=torch.zeros((B, 2)),
    )


def test_gradient_flow(dummy_obs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # smaller model for faster testing
    policy = build_policy(ModelConfig(64, 2, 1, 256), default_runtime_resources()).to(device)
    policy.train()

    obs = dummy_obs.to(device)

    # allow all actions for now
    action_mask = torch.ones((2, 2, ACT_SIZE), dtype=torch.uint8).to(device)

    out = policy.act_obs(obs, action_mask, *policy.empty_memory(2))
    # random loss fn involving both value and policy paths
    loss = out.value.mean() - out.log_probs.mean()

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()

    missing_grads = []
    zero_grads = []

    # not all used every turn
    conditional_params = [
        "series.",  # this single-step test supplies masked Game 1 memory slots
        "actor.pass_emb",
        "actor.switch_meta_emb",
        "actor.move_meta_emb",
        "actor.mega_meta_emb",
        "actor.target_ally_emb",
        "actor.target_opp_emb",
        "actor.target_self_multi_emb",
        "actor.move_proj",
        "actor.tp_meta_emb",
        "actor.q_switch_proj1",
        "actor.q_pass_proj1",
    ]

    for name, param in policy.named_parameters():
        if any(cond in name for cond in conditional_params):
            continue
        if param.requires_grad:
            if param.grad is None:
                missing_grads.append(name)
            elif torch.all(param.grad == 0):
                zero_grads.append(name)

    assert not missing_grads, f"Parameters missing gradients: {missing_grads}"

    if zero_grads:
        print(f"Warning: Parameters with zero gradients: {zero_grads}")

    # verify gradient coverage across logical components
    components = {
        "shared_encoder": "encoder",
        "actor_reducer": "actor.reducer",
        "actor_w_k": "actor.w_k",
        "actor_q_proj": "actor.q_",
        "critic_head": "critic.net",
    }

    failed_components = []
    for comp_name, prefix in components.items():
        comp_params = [p for n, p in policy.named_parameters() if n.startswith(prefix)]
        if not comp_params:
            failed_components.append(f"{comp_name} (no parameters found with prefix {prefix})")
            continue

        has_grad = any(p.grad is not None and torch.abs(p.grad).sum() > 0 for p in comp_params)
        if not has_grad:
            failed_components.append(comp_name)

    assert not failed_components, (
        f"The following components are not receiving gradients: {failed_components}"
    )

    # verify gradient coverage for embedding matrix
    embeddings = [
        "species_emb",
        "ability_emb",
        "item_emb",
        "move_emb",
        "type_emb",
        "category_emb",
        "status_emb",
        "effect_emb",
        "counter_kind_emb",
        "effect_namespace_emb",
        "knownness_emb",
        "token_type_emb",
        "side_emb",
        "slot_emb",
        "event_type_emb",
        "order_pos_emb",
    ]

    missing_emb_grads = []
    for emb in embeddings:
        found = False
        for n, p in policy.encoder.named_parameters():
            if emb in n:
                found = True
                if p.grad is None or torch.abs(p.grad).sum() == 0:
                    missing_emb_grads.append(emb)
        if not found:
            missing_emb_grads.append(f"{emb} (not found)")

    assert not missing_emb_grads, (
        f"The following embeddings are not receiving gradients: {missing_emb_grads}"
    )


def test_ppo_warmup(dummy_obs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = build_policy(ModelConfig(64, 2, 1, 256), default_runtime_resources()).to(device)
    policy.train()

    episode = TrajectoryBatch(
        observations=dummy_obs[0].unsqueeze(0),
        actions=torch.tensor([[1, 2]], dtype=torch.long),
        log_probs=torch.zeros(1),
        advantages=torch.ones(1),
        returns=torch.ones(1),
        values=torch.zeros(1),
        rewards=torch.zeros(1),
        dones=torch.ones(1),
        action_masks=torch.ones((1, 2, ACT_SIZE), dtype=torch.bool),
        length=1,
    )
    config = TrainingConfig(warmup_episodes=10)
    magnet = Magnet(policy)

    loss, _, steps = _run_batched_ppo(
        [episode], policy, magnet, config, device, episode=0, alpha=config.magnet_alpha
    )
    assert steps == 1

    policy.zero_grad(set_to_none=True)
    loss.backward()

    assert all(p.grad is None for p in policy.encoder.parameters())
    assert all(p.grad is None for p in policy.actor.parameters())
    assert any(
        p.grad is not None and not torch.all(p.grad == 0) for p in policy.critic.parameters()
    )


if __name__ == "__main__":
    pytest.main([__file__])
