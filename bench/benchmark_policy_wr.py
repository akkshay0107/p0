import argparse
import asyncio
from pathlib import Path

import torch
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from src.heuristic.heuristic import FuzzyHeuristic
from src.model.policy import PolicyNet
from src.rl_player import RLPlayer
from src.team_picker import RandomTeamFromPool
from src.train.utils import load_checkpoint


async def main(n_battles: int, checkpoint_path: str):
    root_dir = Path(__file__).resolve().parent.parent
    teams_dir = root_dir / "teams"
    checkpoint_file = Path(checkpoint_path)

    if not teams_dir.exists():
        print(f"Teams directory not found: {teams_dir}")
        return

    team_files = [
        path.read_text(encoding="utf-8")
        for path in Path(teams_dir).iterdir()
        if path.is_file() and not path.name.startswith(".")
    ]
    if not team_files:
        print("No team files found. Please ensure there are team files in the teams directory.")
        return

    team = RandomTeamFromPool(team_files)
    fmt = "gen9championsvgc2026regma"

    # Load policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PolicyNet().to(device)

    if checkpoint_file.exists():
        print(f"Loading checkpoint from: {checkpoint_file}")
        load_checkpoint(checkpoint_file, policy)
    else:
        print(
            f"Warning: Checkpoint not found at {checkpoint_file}. Using randomly initialized policy."
        )

    policy.eval()

    def create_player(player_class, name, **kwargs):
        return player_class(
            account_configuration=AccountConfiguration(name, None),
            battle_format=fmt,
            server_configuration=LocalhostServerConfiguration,
            max_concurrent_battles=10,
            team=team,
            accept_open_team_sheet=True,
            **kwargs,
        )

    rl_player = RLPlayer(
        policy=policy,
        account_configuration=AccountConfiguration("RL_Policy", None),
        battle_format=fmt,
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=10,
        team=team,
        accept_open_team_sheet=True,
    )

    opponents = [
        ("Random", create_player(RandomPlayer, "Random")),
        ("MaxBP", create_player(MaxBasePowerPlayer, "MaxBP")),
        ("SimpleH", create_player(SimpleHeuristicsPlayer, "SimpleH")),
        ("FuzzyH", create_player(FuzzyHeuristic, "FuzzyH")),
    ]

    print(
        f"Starting evaluation of policy against {len(opponents)} bots ({n_battles} games each)..."
    )
    print("=" * 60)

    results = {}

    for name, opponent in opponents:
        print(f"Battling RL_Policy vs {name}...")
        await rl_player.battle_against(opponent, n_battles=n_battles)

        winrate = rl_player.win_rate
        results[name] = winrate

        print(f"  Result: {winrate:.2%} win rate")

        # Reset for next matches
        rl_player.reset_battles()
        opponent.reset_battles()

    print("\n" + "=" * 30)
    print(f"{'Opponent':<15} | {'Win Rate':>10}")
    print("-" * 30)
    for name, winrate in results.items():
        print(f"{name:<15} | {winrate:>10.2%}")
    print("=" * 30)

    # Cleanup
    await rl_player.ps_client.stop_listening()
    for _, opponent in opponents:
        await opponent.ps_client.stop_listening()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL policy against various bots.")
    parser.add_argument(
        "-n",
        "--n-battles",
        type=int,
        default=100,
        help="Number of battles against each opponent (default: 100)",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default="checkpoints/ppo_checkpoint.pt",
        help="Path to the policy checkpoint (default: checkpoints/ppo_checkpoint.pt)",
    )
    args = parser.parse_args()

    asyncio.run(main(args.n_battles, args.checkpoint))
