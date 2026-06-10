from dataclasses import dataclass, fields
from pathlib import Path


# unfortunately the config is not static
# and is reused as a vessel to carry the changing
# hyperparams (lr, entropy_coef) for
# simplicity in the train loop
@dataclass(slots=True)
class PPOConfig:
    # default values provided are meant for the colab
    # T4 runtime with 15 GB VRAM and some unknown CPU
    # with 16 GB RAM
    num_episodes: int = 12500
    n_envs: int = 8
    n_self_envs: int = 4
    n_pool_opponents: int = 4
    rollout_steps: int = 320
    batch_size: int = 128
    chunk_size: int = 32  # with BPTT, backward pass takes 14 GB RAM

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_low: float = 0.2
    clip_high: float = 0.28  # DAPO style, entropy regularizer

    lr: float = 6e-5
    value_coef: float = 0.5
    entropy_coef: float = 0.04
    max_grad_norm: float = 1.0
    target_kl: float = 0.05  # for early kl stopping
    ppo_epochs: int = 2  # number of ppo loops per episode
    # skew importance of team preview step
    teampreview_loss_mult: float = 1.5
    teampreview_entropy_mult: float = 2.0

    warmup_episodes: int = 100  # policy gradients frozen, allow value head to catchup to bc seeds
    ramp_up_phase: float = 0.1  # frac of epochs spent in linear lr increase
    ramp_down_phase: float = 0.2  # frac of epochs spent in decaying entropy coef

    pool_dir: Path = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "pool"
    checkpoint_path: Path = (
        Path(__file__).resolve().parent.parent.parent / "checkpoints" / "ppo_checkpoint.pt"
    )
    pool_size: int = 40
    snapshot_interval: int = 50
    # admit a snapshot unconditionally every N episodes so pool diversity
    # keeps growing even when the win-rate gate would reject it
    pool_force_admit_every: int = 250
    pool_win_rate_smoothing: float = 0.1
    pool_wr_floor: float = 0.1

    def __post_init__(self) -> None:
        if not 0 <= self.n_self_envs <= self.n_envs:
            raise ValueError("n_self_envs must be between 0 and n_envs.")
        if self.rollout_steps <= 0:
            raise ValueError("rollout_steps must be greater than zero.")
        if self.n_pool_opponents <= 0:
            raise ValueError("n_pool_opponents must be greater than zero.")


def load_config(config_path: str = ".ppoconfig") -> PPOConfig:
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
