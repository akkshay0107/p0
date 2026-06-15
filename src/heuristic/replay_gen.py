import argparse
import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable

import numpy as np
import torch
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import (
    DefaultBattleOrder,
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
    SingleBattleOrder,
)

from src.env import MegaEnv
from src.heuristic.heuristic import FuzzyHeuristic
from src.lookups import ACT_SIZE
from src.model import observation_builder
from src.team_picker import RandomTeamFromPool
from src.train.config import PPOConfig


def _modify_mask(action_mask: torch.Tensor, action1):
    mask2 = action_mask[1].clone().bool()
    if 1 <= action1 and action1 <= 6:
        mask2[action1] = 0
    elif (26 < action1) and (action1 <= 46):
        mask2[27:47] = 0
    elif action1 == 0:
        mask2[0] = 0

    no_valid = mask2.sum(-1) == 0
    if no_valid:
        mask2[0] = 1

    return mask2


class ReplayRecordingPlayer(Player, ABC):
    def __init__(self, save_dir, shard_size=12, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_dir = save_dir
        self.shard_size = shard_size
        self.current_episodes = {}
        self.shard = []

    def _battle_finished_callback(self, battle: AbstractBattle):
        super()._battle_finished_callback(battle)
        tag = battle.battle_tag
        if tag in self.current_episodes:
            steps = self.current_episodes.pop(tag)
            if steps:
                self.shard.append(steps)
        if len(self.shard) >= self.shard_size:
            self.save_shard()

    def save_shard(self):
        # save all buffered episodes in the shard to a single file
        if not self.shard:
            return None

        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        uid = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:10]}"
        save_path = save_dir / f"{uid}.replay"
        tmp_path = save_dir / f"{uid}.replay.tmp"

        torch.save(self.shard, tmp_path)
        tmp_path.replace(save_path)
        print(f"saved shard with {len(self.shard)} episodes to {save_path}")

        self.shard = []
        return str(save_path)

    def get_observation(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        return observation_builder.from_battle(battle)

    async def choose_move(self, battle: AbstractBattle):
        assert isinstance(battle, DoubleBattle)
        if battle._wait:
            return DefaultBattleOrder()

        obs = self.get_observation(battle)
        action_mask_list = MegaEnv.get_action_mask(battle)
        action_mask = torch.tensor([action_mask_list[:ACT_SIZE], action_mask_list[ACT_SIZE:]])

        action_np = await self.get_action(battle, action_mask)
        tag = battle.battle_tag
        if tag not in self.current_episodes:
            self.current_episodes[tag] = []
        self.current_episodes[tag].append(
            {"obs": obs, "mask": action_mask, "action": torch.from_numpy(action_np)}
        )
        return MegaEnv.action_to_order(action_np, battle, strict=False)

    async def _handle_battle_request(
        self, battle: AbstractBattle, maybe_default_order: bool = False
    ):
        if battle.teampreview:
            assert isinstance(battle, DoubleBattle)
            obs = self.get_observation(battle)
            action_mask_list = MegaEnv.get_action_mask(battle)
            action_mask = torch.tensor([action_mask_list[:ACT_SIZE], action_mask_list[ACT_SIZE:]])
            action_np = await self.get_action(battle, action_mask)
            tag = battle.battle_tag
            if tag not in self.current_episodes:
                self.current_episodes[tag] = []
            self.current_episodes[tag].append(
                {"obs": obs, "mask": action_mask, "action": torch.from_numpy(action_np)}
            )
            order = MegaEnv.action_to_order(action_np, battle, strict=False)
            await self.ps_client.send_message(order.message, battle.battle_tag)
        else:
            await super()._handle_battle_request(battle, maybe_default_order)

    @abstractmethod
    async def get_action(self, battle: DoubleBattle, action_mask: torch.Tensor) -> np.ndarray:
        pass

    def teampreview(self, battle: AbstractBattle) -> str:
        # This is now handled in _handle_battle_request to allow recording
        return super().random_teampreview(battle)


class StrategyRecordingPlayer(ReplayRecordingPlayer):
    def __init__(self, strategy_player: Player, save_dir, shard_size=12, *args, **kwargs):
        super().__init__(save_dir, shard_size, *args, **kwargs)
        self.strategy_player = strategy_player

    async def get_action(self, battle: DoubleBattle, action_mask: torch.Tensor) -> np.ndarray:
        if battle.teampreview:
            res = self.strategy_player.teampreview(battle)
            if isinstance(res, Awaitable):
                res = await res
            order = SingleBattleOrder(res)
        else:
            order = self.strategy_player.choose_move(battle)
            if isinstance(order, Awaitable):
                order = await order

        action = MegaEnv.order_to_action(order, battle, fake=True)

        if not battle.teampreview:
            if action[0] < 0:
                action[0] = 0
            if not action_mask[0, action[0]]:
                valid_indices = torch.where(action_mask[0])[0]
                action[0] = valid_indices[0].item()

            mask2 = _modify_mask(action_mask, action[0])

            if action[1] < 0:
                action[1] = 0
            if not mask2[action[1]]:
                valid_indices = torch.where(mask2)[0]
                action[1] = valid_indices[0].item()
        else:
            action[0] = np.clip(action[0], 0, 35)
            action[1] = np.clip(action[1], 0, 35)

        return action


async def main():
    parser = argparse.ArgumentParser(description="Replay Generator")
    parser.add_argument(
        "-n", type=int, default=100, help="Number of battles per recording strategy"
    )
    args = parser.parse_args()

    teams_dir = "./teams"
    team_files = [
        path.read_text(encoding="utf-8")
        for path in Path(teams_dir).iterdir()
        if path.is_file() and not path.name.startswith(".")
    ]
    team = RandomTeamFromPool(team_files)
    fmt = "gen9championsvgc2026regma"

    def get_kwargs(name):
        return {
            "account_configuration": AccountConfiguration(name, None),
            "battle_format": fmt,
            "server_configuration": LocalhostServerConfiguration,
            "team": team,
            "accept_open_team_sheet": True,
            "max_concurrent_battles": 16,
        }

    # Initialize Strategies (start_listening=False as they are wrapped)
    fuzzy_strat = FuzzyHeuristic(start_listening=False, **get_kwargs("FuzzyStrat"))
    sh_strat = SimpleHeuristicsPlayer(start_listening=False, **get_kwargs("SHStrat"))
    mbp_strat = MaxBasePowerPlayer(start_listening=False, **get_kwargs("MBPStrat"))

    n = args.n
    shard_size = max(1, n // 8)

    replays_dir = PPOConfig().replays_dir
    rec_players = {
        "fuzzy": StrategyRecordingPlayer(
            strategy_player=fuzzy_strat,
            save_dir=str(replays_dir / "fuzzy_heuristic"),
            shard_size=shard_size,
            **get_kwargs("RecFuzzy"),
        ),
        "sh": StrategyRecordingPlayer(
            strategy_player=sh_strat,
            save_dir=str(replays_dir / "simple_heuristic"),
            shard_size=shard_size,
            **get_kwargs("RecSH"),
        ),
        "mbp": StrategyRecordingPlayer(
            strategy_player=mbp_strat,
            save_dir=str(replays_dir / "max_base_power"),
            shard_size=shard_size,
            **get_kwargs("RecMBP"),
        ),
    }

    # non-recording opponents for self-play and random matchups
    opponents = {
        "fuzzy": FuzzyHeuristic(**get_kwargs("OppFuzzy")),
        "sh": SimpleHeuristicsPlayer(**get_kwargs("OppSH")),
        "mbp": MaxBasePowerPlayer(**get_kwargs("OppMBP")),
        "random": RandomPlayer(**get_kwargs("OppRandom")),
    }

    fuzzy_bound = int(0.4 * n)
    sh_bound = int(0.75 * n)
    mbp_bound = int(0.90 * n)

    for name, player in rec_players.items():
        print(f"\n--- Generating replays for {name.upper()} ---")

        # Count battles per opponent type
        opp_counts = {}
        for i in range(n):
            if i <= fuzzy_bound:
                opp_type = "fuzzy"
            elif i <= sh_bound:
                opp_type = "sh"
            elif i <= mbp_bound:
                opp_type = "mbp"
            else:
                opp_type = "random"
            opp_counts[opp_type] = opp_counts.get(opp_type, 0) + 1

        for opp_type, count in opp_counts.items():
            if count <= 0:
                continue

            # use another recording player if types differ
            if opp_type in rec_players and opp_type != name:
                opp = rec_players[opp_type]
            else:
                opp = opponents[opp_type]

            print(f"Starting {count} battles: {player.username} vs {opp.username} ({opp_type})")
            await player.battle_against(opp, n_battles=count)

    # final cleanup
    for p in rec_players.values():
        p.save_shard()  # ensure all remaining data is saved
        await p.ps_client.stop_listening()
    for o in opponents.values():
        await o.ps_client.stop_listening()


if __name__ == "__main__":
    asyncio.run(main())
