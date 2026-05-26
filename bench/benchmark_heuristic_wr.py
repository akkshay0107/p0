import argparse
import asyncio
from pathlib import Path

from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from src.heuristic.heuristic import FuzzyHeuristic
from src.team_picker import RandomTeamFromPool


async def main(n_battles: int):
    root_dir = Path(__file__).resolve().parent.parent
    teams_dir = root_dir / "teams"
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

    players = [
        create_player(RandomPlayer, "Random"),
        create_player(MaxBasePowerPlayer, "MaxBP"),
        create_player(SimpleHeuristicsPlayer, "SimpleH"),
        create_player(FuzzyHeuristic, "FuzzyH"),
    ]

    bot_names = ["Random", "MaxBP", "SimpleH", "FuzzyH"]
    n = len(players)
    winrate_matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    print(f"Starting {n * (n - 1) // 2 * n_battles} matches total...")

    for i in range(n):
        for j in range(i + 1, n):
            p1 = players[i]
            p2 = players[j]

            print(f"Battling {bot_names[i]} vs {bot_names[j]} ({n_battles} games)...")
            await p1.battle_against(p2, n_battles=n_battles)

            winrate = p1.win_rate
            winrate_matrix[i][j] = winrate
            winrate_matrix[j][i] = 1 - winrate

            print(f"  {bot_names[i]} winrate: {winrate:.2%}")

            # Reset for next matches if any
            p1.reset_battles()
            p2.reset_battles()

    print("\n" + "=" * 60)

    header = " " * 10
    for name in bot_names:
        header += f"{name:>10}"
    print(header)

    for i in range(n):
        row = f"{bot_names[i]:<10}"
        for j in range(n):
            if i == j:
                row += f"{'-':>10}"
            else:
                row += f"{winrate_matrix[i][j]:>10.2%}"
        print(row)
    print("=" * 60)

    for p in players:
        await p.ps_client.stop_listening()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Battle bots and calculate winrates.")
    parser.add_argument(
        "-b",
        type=int,
        default=100,
        help="Number of battles between any two sets of bots (default: 100)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.b))
