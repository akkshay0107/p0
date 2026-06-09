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
        # sort by creation time to maintain order
        existing_files = sorted(self.pool_dir.glob("*.pt"), key=lambda p: p.stat().st_ctime)
        existing_ids = {pt.stem for pt in existing_files}

        for pt_file in existing_files:
            opponent_id = pt_file.stem
            if opponent_id not in self.opponent_ids:
                self.opponent_ids.append(opponent_id)
                if opponent_id not in self.win_rates:
                    self.win_rates[opponent_id] = 0.5

        # remove entries that do not exist in pool dir anymore
        self.opponent_ids = [oid for oid in self.opponent_ids if oid in existing_ids]
        self.win_rates = {oid: wr for oid, wr in self.win_rates.items() if oid in existing_ids}

    @classmethod
    def load_or_create(cls, pool_dir: Path, config: PPOConfig) -> Self:
        """Load an existing pool from disk, or create an empty one."""
        pool = cls(pool_dir, config)
        pool._load_state()
        return pool

    def add(self, policy: PolicyNet, id: str, pool_wr: float) -> bool:
        if id in self.opponent_ids or pool_wr < 0.5:
            return False

        # evict lowest wr policy
        if len(self.opponent_ids) >= self.config.pool_size:
            evict_id = min(self.opponent_ids, key=lambda opp_id: self.win_rates[opp_id])
            if pool_wr <= self.win_rates[evict_id]:
                return False

            evict_path = self.pool_dir / f"{evict_id}.pt"
            if evict_path.exists():
                evict_path.unlink()

            self.opponent_ids.remove(evict_id)
            del self.win_rates[evict_id]

        save_path = self.pool_dir / f"{id}.pt"
        torch.save(
            {
                "model_state_dict": policy.state_dict(),
            },
            save_path,
        )

        self.opponent_ids.append(id)
        self.win_rates[id] = pool_wr
        return True

    def load_policy(self, opponent_id: str, device: str) -> PolicyNet:
        path = self.pool_dir / f"{opponent_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint for opponent '{opponent_id}' not found at {path}")

        checkpoint = torch.load(path, weights_only=True, map_location=device)
        net = PolicyNet()
        net.load_state_dict(checkpoint["model_state_dict"])
        return net.to(device).eval()

    def update_win_rate(self, opponent_id: str, agent_wins: int, num_games: int = 1) -> None:
        if opponent_id not in self.win_rates:
            return

        # agent_wins <=> games lost by the opp id against latest policy
        observed_wr = 1.0 - agent_wins / num_games
        curr_wr = self.win_rates[opponent_id]

        alpha = self.config.pool_win_rate_smoothing
        self.win_rates[opponent_id] = (1 - alpha) * curr_wr + alpha * observed_wr

    def sample(self, device: str) -> tuple[PolicyNet, str]:
        """Returns a frozen policy (loaded to device) from the pool sampled using win rates as weights."""
        opponent_id = self.sample_id()
        return self.load_policy(opponent_id, device), opponent_id

    def sample_id(self) -> str:
        """Returns an opponent ID sampled using win rates as weights."""
        if not self.opponent_ids:
            raise RuntimeError("OpponentPool is empty. Call pool.add() before pool.sample().")

        floor = self.config.pool_wr_floor
        weights = [max(floor, self.win_rates[oid]) for oid in self.opponent_ids]
        (opponent_id,) = random.choices(self.opponent_ids, weights=weights, k=1)
        return opponent_id

    def sample_many(self, count: int) -> list[str]:
        """Sample opponent IDs without replacement using win rates as weights."""
        if count <= 0:
            raise ValueError("Opponent count must be greater than zero.")
        if not self.opponent_ids:
            raise RuntimeError("OpponentPool is empty. Call pool.add() before pool.sample_many().")

        count = min(count, len(self.opponent_ids))
        weights = torch.tensor(
            [
                max(self.config.pool_wr_floor, self.win_rates[opponent_id])
                for opponent_id in self.opponent_ids
            ],
            dtype=torch.float32,
        )
        indices = torch.multinomial(weights, count, replacement=False)
        return [self.opponent_ids[index] for index in indices.tolist()]
