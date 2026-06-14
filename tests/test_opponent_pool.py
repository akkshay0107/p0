from types import SimpleNamespace
from typing import Any, cast

import torch

from src.train.config import PPOConfig
from src.train.opponent_pool import INIT_WR, OpponentPool


def _fake_policy(value: float = 0.0) -> Any:
    return SimpleNamespace(
        state_dict=lambda: {
            "weight": torch.tensor([value], dtype=torch.float32),
            "index": torch.tensor([1], dtype=torch.long),
        }
    )


def test_sample_many_builds_roster_from_shadow_anchors_and_regulars(tmp_path):
    pool = OpponentPool(tmp_path, PPOConfig(n_pool_opponents=4))
    pool.shadow_id = "shadow"
    pool.anchor_ids = ["seed-a", "seed-b"]
    pool.regular_ids = ["ep20", "ep40"]
    pool.win_rates = {opponent_id: INIT_WR for opponent_id in pool.active_ids()}

    sampled = pool.sample_many(10)

    assert set(sampled) == {"shadow", "seed-a", "seed-b", "ep20", "ep40"}
    assert len(sampled) == len(set(sampled))


def test_add_regular_evicts_lowest_win_rate_regular_and_keeps_anchors(tmp_path):
    config = PPOConfig(pool_size=4)
    pool = OpponentPool(tmp_path, config)
    pool.shadow_id = "shadow"
    pool.anchor_ids = ["seed"]
    pool.regular_ids = ["weak", "strong"]
    pool.win_rates = {
        "shadow": 0.5,
        "seed": 0.5,
        "weak": 0.1,
        "strong": 0.9,
    }

    added = pool.add(cast(Any, _fake_policy(2.0)), "latest")

    assert added
    assert pool.anchor_ids == ["seed"]
    assert pool.regular_ids == ["strong", "latest"]
    assert pool.win_rates["latest"] == INIT_WR
    assert "weak" not in pool.win_rates
