from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class PPOConfig:
    num_episodes: int = 12500
    n_envs = 16
    self_play_steps = 640
    pool_play_steps = 640

    lr: float = 3e-5
    batch_size: int = 32
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    target_kl: float = 0.05  # for early kl stopping
    ppo_epochs: int = 4  # number of ppo loops per episode
    # value head warmup episodes. policy receives no gradients for the first
    # warmup_episodes episodes. was having trouble with huge kl swings on the
    # shared features without this at the start.
    warmup_episodes: int = 100
    checkpoint_path: Path = (
        Path(__file__).resolve().parent.parent.parent / "checkpoints" / "ppo_checkpoint.pt"
    )
    # skew importance of team preview step
    teampreview_loss_mult: float = 1.5
    teampreview_entropy_mult: float = 2.0

    pool_dir: Path = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "pool"
    pool_size: int = 40
    snapshot_interval: int = 50
    pool_win_rate_smoothing: float = 0.1
    pool_wr_floor: float = 0.1


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
