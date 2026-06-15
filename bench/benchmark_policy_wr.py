import asyncio
from pathlib import Path

from src.eval import HEURISTIC_OPPONENTS, evaluate_against_heuristics, load_team
from src.model.policy import PolicyNet
from src.train.config import load_config
from src.train.utils import default_device, load_checkpoint

N_BATTLES = 100
PORT = 8000


async def main():
    root_dir = Path(__file__).resolve().parent.parent
    config = load_config(root_dir / ".ppoconfig")

    if not config.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint_path}")

    device = default_device()
    policy = PolicyNet().to(device)
    print(f"Loading checkpoint from: {config.checkpoint_path}")
    load_checkpoint(config.checkpoint_path, policy)
    policy.eval()

    team = load_team(root_dir / "teams")

    print(
        f"Starting evaluation of policy against {len(HEURISTIC_OPPONENTS)} bots "
        f"({N_BATTLES} games each)..."
    )
    print("=" * 60)

    results = await evaluate_against_heuristics(policy, team=team, port=PORT, n_battles=N_BATTLES)

    print("\n" + "=" * 30)
    print(f"{'Opponent':<15} | {'Win Rate':>10}")
    print("-" * 30)
    for name in HEURISTIC_OPPONENTS:
        print(f"{name:<15} | {results[name]:>10.2%}")
    print("-" * 30)
    print(f"{'Mean':<15} | {results['mean']:>10.2%}")
    print("=" * 30)


if __name__ == "__main__":
    from src.showdown_server import spawned_showdown

    with spawned_showdown(port=PORT):
        asyncio.run(main())
