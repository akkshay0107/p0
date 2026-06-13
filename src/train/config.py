from dataclasses import dataclass, fields
from pathlib import Path

# all runtime artifacts (checkpoints, pool, tensorboard runs, replays,
# backups, logs) live under a single directory to keep the project root tidy.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


# unfortunately the config is not static
# and is reused as a vessel to carry the changing
# hyperparams (lr, entropy_coef) for
# simplicity in the train loop
@dataclass(slots=True)
class PPOConfig:
    # default values provided are meant for the colab
    # T4 runtime with 15 GB VRAM and some unknown CPU
    # with 16 GB RAM
    num_episodes: int = 2000  # ~7.5M steps total
    n_envs: int = 8
    n_self_envs: int = 4
    n_pool_opponents: int = 4
    rollout_steps: int = 320
    batch_size: int = 128
    chunk_size: int = 32  # with BPTT, backward pass takes 14 GB RAM

    gamma: float = 0.99
    gae_lambda: float = 0.97
    clip_low: float = 0.2
    clip_high: float = 0.28  # DAPO style, entropy regularizer

    lr: float = 6e-5
    value_coef: float = 0.05
    entropy_coef: float = 0.03
    max_grad_norm: float = 1.0
    target_kl: float = 0.015  # for kl skipping
    ppo_epochs: int = 6  # number of ppo loops per episode
    # skew importance of team preview step
    teampreview_loss_mult: float = 1.5
    teampreview_entropy_mult: float = 2.0

    enable_optim: bool = True  # enable FP16 autocast + CUDA-graph compile of the rollout actor

    warmup_episodes: int = 20  # policy gradients frozen, allow value head to catchup to bc seeds
    ramp_up_phase: float = 0.1  # frac of epochs spent in linear lr increase
    ramp_down_phase: float = 0.2  # frac of epochs spent in decaying entropy coef

    artifacts_dir: Path = ARTIFACTS_DIR
    pool_dir: Path = ARTIFACTS_DIR / "checkpoints" / "pool"
    checkpoint_path: Path = ARTIFACTS_DIR / "checkpoints" / "ppo_checkpoint.pt"
    runs_dir: Path = ARTIFACTS_DIR / "runs"
    replays_dir: Path = ARTIFACTS_DIR / "replays"
    backups_dir: Path = ARTIFACTS_DIR / "backups"
    log_path: Path = ARTIFACTS_DIR / "training.log"
    pool_size: int = 50
    snapshot_interval: int = 50
    # promote the strongest snapshot to the permanent anchor pool once every
    # this many rotating snapshots are admitted
    pool_anchor_every: int = 5
    pool_win_rate_smoothing: float = 0.1
    pool_wr_floor: float = 0.1

    def __post_init__(self) -> None:
        if not 0 <= self.n_self_envs <= self.n_envs:
            raise ValueError("n_self_envs must be between 0 and n_envs.")
        if self.rollout_steps <= 0:
            raise ValueError("rollout_steps must be greater than zero.")
        if self.n_pool_opponents <= 0:
            raise ValueError("n_pool_opponents must be greater than zero.")


def load_config(config_path: str | Path = ".ppoconfig") -> PPOConfig:
    """
    Loads a PPOConfig from a flat key=value file.
    Merges specified values for hyperparameters with the default values
    for unspecified hyperparameters.

    Defaults to returning default PPOConfig on any error.
    """
    path = Path(config_path)
    if not path.exists():
        return PPOConfig()

    config_dict = {}
    valid_fields = {f.name: f.type for f in fields(PPOConfig)}

    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                if key in valid_fields:
                    target_type = valid_fields[key]
                    try:
                        if target_type is int:
                            config_dict[key] = int(float(value))
                        elif target_type is float:
                            config_dict[key] = float(value)
                        elif target_type is bool:
                            config_dict[key] = value.lower() in ("true", "1", "yes")
                        elif target_type is Path:
                            config_dict[key] = Path(value)
                        else:
                            config_dict[key] = value
                    except (ValueError, TypeError):
                        continue

        return PPOConfig(**config_dict)
    except Exception:
        return PPOConfig()
