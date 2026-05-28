import json
import random
from pathlib import Path
from typing import Self

import torch

from src.model.policy import PolicyNet
from src.train.config import PPOConfig


# manager for past checkpoints of the model
# past checkpoints used as opponents in a league manner
# similar to alphastar (no specific exploiter agents though)
class OpponentPool:
    def __init__(self, pool_dir: Path, config: PPOConfig):
        self.pool_dir = pool_dir
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.opponent_ids: list[str] = []  # ordered list of opponents (first = oldest)
        self.win_rates: dict[str, float] = {}  # ema win rate for each of the policies in the pool

    def __len__(self) -> int:
        return len(self.opponent_ids)

    def __repr__(self) -> str:
        return f"OpponentPool(size={len(self)}/{self.config.pool_size}, ids={self.opponent_ids})"

    def save_state(self) -> None:
        state = {
            "opponent_ids": self.opponent_ids,
            "win_rates": self.win_rates,
        }
        with open(self.pool_dir / "pool_state.json", "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        path = self.pool_dir / "pool_state.json"
        if path.exists():
            with open(path) as f:
                state = json.load(f)
            self.opponent_ids = state.get("opponent_ids", [])
            self.win_rates = state.get("win_rates", {})

        # sync with pool directory (in case json got deleted)
        for pt_file in sorted(self.pool_dir.glob("*.pt")):
            opponent_id = pt_file.stem
            if opponent_id not in self.opponent_ids:
                self.opponent_ids.append(opponent_id)
                if opponent_id not in self.win_rates:
                    self.win_rates[opponent_id] = 0.5

        # repair json file to match the models in the pool
        existing_ids = {pt.stem for pt in self.pool_dir.glob("*.pt")}
        self.opponent_ids = [oid for oid in self.opponent_ids if oid in existing_ids]
        self.win_rates = {oid: wr for oid, wr in self.win_rates.items() if oid in existing_ids}

    @classmethod
    def load_or_create(cls, pool_dir: Path, config: PPOConfig) -> Self:
        """Load an existing pool from disk, or create an empty one."""
        pool = cls(pool_dir, config)
        pool._load_state()
        return pool

    # TODO: add a gating rule that can decide whether the latest
    # model should enter the pool or not. Ideally should return
    # true if new policy added, else false
    def add(self, policy: PolicyNet, opponent_id: str) -> bool:
        raise NotImplementedError("")

    def load_policy(self, opponent_id: str) -> PolicyNet:
        checkpoint = torch.load(
            self.pool_dir / f"{opponent_id}.pt",
            weights_only=True,
        )
        net = PolicyNet(obs_dim=OBS_DIM, act_size=ACT_SIZE)
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()
        return net

    def update_win_rate(self, opponent_id: str, won: int, num_games: int) -> None:
        if opponent_id not in self.win_rates:
            return

        # won => number of games the latest policy won
        # equivalent to games lost by the opp id against latest policy
        observed_wr = 1.0 - won / num_games
        curr_wr = self.win_rates[opponent_id]

        alpha = self.config.pool_win_rate_smoothing
        self.win_rates[opponent_id] = (1 - alpha) * curr_wr + alpha * observed_wr

    def sample(self) -> tuple[PolicyNet, str]:
        """Returns a frozen policy from the pool sampled using win rates as weights."""
        if not self.opponent_ids:
            raise RuntimeError("OpponentPool is empty. Call pool.add() before pool.sample().")

        floor = self.config.pool_wr_floor
        weights = [max(floor, self.win_rates[oid]) for oid in self.opponent_ids]
        (opponent_id,) = random.choices(self.opponent_ids, weights=weights, k=1)
        return self.load_policy(opponent_id), opponent_id
