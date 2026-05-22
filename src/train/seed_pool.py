import shutil
import sys
from pathlib import Path

from behaviour_cloning import ReplayDataset, train_behavior_cloning
from ppo_utils import OpponentPool, PPOConfig


def _get_dataset(replays_base: Path, subdir: str) -> ReplayDataset | None:
    path = replays_base / subdir
    if not path.exists() or not list(path.rglob("*.replay")):
        print(f"No replays found in {path}. Skipping.")
        return None
    return ReplayDataset(str(path))


def main():
    root_dir = Path(__file__).resolve().parent.parent
    replays_base = root_dir / "replays"
    backup_dir = root_dir / "backups"
    backup_dir.mkdir(exist_ok=True)

    if not replays_base.exists():
        print(f"Replays directory {replays_base} does not exist. Run replay_gen.py first.")
        sys.exit(1)

    config = PPOConfig()
    pool = OpponentPool.load_or_create(config.pool_dir, config)
    added_seeds = []

    bc_kwargs = {
        "batch_size": 32,
        "num_epochs": 5,
        "learning_rate": 5e-4,
        "val_split_ratio": 0.1,
    }

    print("\n" + "=" * 60)
    print("Seeding pool with behaviour cloning policies")
    print("=" * 60)

    # 1. Max Base Power
    if "seed_max_base_power" not in pool.opponent_ids:
        print("--- Training seed_max_base_power ---")
        ds_mbp = _get_dataset(replays_base, "max_base_power")
        if ds_mbp:
            policy = train_behavior_cloning(ds_mbp, **bc_kwargs)
            if policy:
                pool.add(policy, "seed_max_base_power")
                added_seeds.append("seed_max_base_power")
                shutil.copy(config.pool_dir / "seed_max_base_power.pt", backup_dir / "seed_max_base_power.pt")
    else:
        print("seed_max_base_power already exists.")

    # 2. Simple Heuristic
    if "seed_simple_heuristic" not in pool.opponent_ids:
        print("--- Training seed_simple_heuristic ---")
        ds_sh = _get_dataset(replays_base, "simple_heuristic")
        if ds_sh:
            policy = train_behavior_cloning(ds_sh, **bc_kwargs)
            if policy:
                pool.add(policy, "seed_simple_heuristic")
                added_seeds.append("seed_simple_heuristic")
                shutil.copy(config.pool_dir / "seed_simple_heuristic.pt", backup_dir / "seed_simple_heuristic.pt")
    else:
        print("seed_simple_heuristic already exists.")

    # 3. Fuzzy Heuristic
    if "seed_fuzzy_heuristic" not in pool.opponent_ids:
        print("--- Training seed_fuzzy_heuristic ---")
        ds_fuzzy = _get_dataset(replays_base, "fuzzy_heuristic")
        if ds_fuzzy:
            policy = train_behavior_cloning(ds_fuzzy, **bc_kwargs)
            if policy:
                pool.add(policy, "seed_fuzzy_heuristic")
                added_seeds.append("seed_fuzzy_heuristic")
                shutil.copy(config.pool_dir / "seed_fuzzy_heuristic.pt", backup_dir / "seed_fuzzy_heuristic.pt")
    else:
        print("seed_fuzzy_heuristic already exists.")

    if added_seeds:
        pool.save_state()
        print(f"\nPool state saved. Added {len(added_seeds)} seeds: {', '.join(added_seeds)}")
    else:
        print("\nNo new seeds were added to the pool.")

    print(f"Current pool state: {pool}")


if __name__ == "__main__":
    main()
