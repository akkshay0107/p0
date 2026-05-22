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

import observation_builder
from env import Gen9VGCEnv
from heuristic import FuzzyHeuristic
from teams import RandomTeamFromPool


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
    def __init__(self, save_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_dir = save_dir
        self.current_episode = []
        self.shard = []

    def complete_episode(self):
        # move current episode steps to the shard buffer
        if self.current_episode:
            self.shard.append(list(self.current_episode))
            self.current_episode = []

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
        action_mask = observation_builder.get_action_mask(battle)

        action_np = await self.get_action(battle, action_mask)
        self.current_episode.append(
            {"obs": obs, "mask": action_mask, "action": torch.from_numpy(action_np)}
        )
        return Gen9VGCEnv.action_to_order(action_np, battle)

    async def _handle_battle_request(
        self, battle: AbstractBattle, maybe_default_order: bool = False
    ):
        if battle.teampreview:
            obs = self.get_observation(battle)
            action_mask = observation_builder.get_action_mask(battle)
            action_np = await self.get_action(battle, action_mask)
            self.current_episode.append(
                {"obs": obs, "mask": action_mask, "action": torch.from_numpy(action_np)}
            )
            order = Gen9VGCEnv.action_to_order(action_np, battle)
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
    def __init__(self, strategy_player: Player, save_dir, *args, **kwargs):
        super().__init__(save_dir, *args, **kwargs)
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

        action = Gen9VGCEnv.order_to_action(order, battle, fake=True, strict=False)

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
    fmt = "gen9vgc2025regh"

    def get_kwargs(name):
        return {
            "account_configuration": AccountConfiguration(name, None),
            "battle_format": fmt,
            "server_configuration": LocalhostServerConfiguration,
            "team": team,
            "accept_open_team_sheet": True,
        }

    # Initialize Strategies (start_listening=False as they are wrapped)
    fuzzy_strat = FuzzyHeuristic(k=3, start_listening=False, **get_kwargs("FuzzyStrat"))
    sh_strat = SimpleHeuristicsPlayer(start_listening=False, **get_kwargs("SHStrat"))
    mbp_strat = MaxBasePowerPlayer(start_listening=False, **get_kwargs("MBPStrat"))

    rec_players = {
        "fuzzy": StrategyRecordingPlayer(
            strategy_player=fuzzy_strat,
            save_dir="./replays/fuzzy_heuristic",
            max_concurrent_battles=1,
            **get_kwargs("RecFuzzy"),
        ),
        "sh": StrategyRecordingPlayer(
            strategy_player=sh_strat,
            save_dir="./replays/simple_heuristic",
            max_concurrent_battles=1,
            **get_kwargs("RecSH"),
        ),
        "mbp": StrategyRecordingPlayer(
            strategy_player=mbp_strat,
            save_dir="./replays/max_base_power",
            max_concurrent_battles=1,
            **get_kwargs("RecMBP"),
        ),
    }

    # non-recording opponents for self-play and random matchups
    opponents = {
        "fuzzy": FuzzyHeuristic(k=3, **get_kwargs("OppFuzzy")),
        "sh": SimpleHeuristicsPlayer(**get_kwargs("OppSH")),
        "mbp": MaxBasePowerPlayer(**get_kwargs("OppMBP")),
        "random": RandomPlayer(**get_kwargs("OppRandom")),
    }

    n = args.n
    fuzzy_bound = int(0.35 * n)
    sh_bound = int(0.70 * n)
    mbp_bound = int(0.90 * n)
    shard_size = max(1, args.n // 8)

    for name, player in rec_players.items():
        print(f"\n--- Generating replays for {name.upper()} ---")
        for i in range(n):
            if i <= fuzzy_bound:
                opp_type = "fuzzy"
            elif i <= sh_bound:
                opp_type = "sh"
            elif i <= mbp_bound:
                opp_type = "mbp"
            else:
                opp_type = "random"

            # use another recording player if types differ
            if opp_type in rec_players and opp_type != name:
                opp = rec_players[opp_type]
            else:
                opp = opponents[opp_type]

            print(f"Battle {i + 1}/{n}: {player.username} vs {opp.username} ({opp_type})")
            await player.battle_against(opp, n_battles=1)

            player.complete_episode()
            if isinstance(opp, ReplayRecordingPlayer):
                opp.complete_episode()

            # save for player / opp only if shard size is reached
            if len(player.shard) >= shard_size:
                player.save_shard()
            if isinstance(opp, ReplayRecordingPlayer) and len(opp.shard) >= shard_size:
                opp.save_shard()

        # save remaining episodes for this player
        player.save_shard()

    # final cleanup
    for p in rec_players.values():
        p.save_shard()  # ensure all remaining data is saved
        await p.ps_client.stop_listening()
    for o in opponents.values():
        await o.ps_client.stop_listening()


if __name__ == "__main__":
    asyncio.run(main())
