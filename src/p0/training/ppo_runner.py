"""Application composition for PPO training."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import torch.optim as optim
from poke_env import AccountConfiguration
from torch.amp import GradScaler

from p0.format_config import FORMAT
from p0.model.config import ModelConfig
from p0.model.factory import build_policy, compile_policy
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.paths import DEFAULT_PATHS
from p0.runtime.composition import build_sim_env
from p0.runtime.env import SimEnv
from p0.runtime.showdown import start_showdown_servers
from p0.teams.corpus import load_team_corpus
from p0.training.checkpoint import (
    DEFAULT_CHECKPOINT_STORE,
    CheckpointStore,
)
from p0.training.config import PPOConfig
from p0.training.files import training_run
from p0.training.magnet import Magnet
from p0.training.rollout import RolloutCollector
from p0.training.trainer import PPOTrainer
from p0.training.utils import (
    PPOScheduler,
    adamw_param_groups,
    default_device,
    seed_everything,
    select_optimization_precision,
)
from p0.training.vector_env import ThreadVecEnv

LOGGER = logging.getLogger(__name__)
PPO_WEIGHT_DECAY = 1e-4
ADAM_EPSILON = 1e-6


def _close_environments(envs: list[SimEnv]) -> None:
    """Close each Pokemon Showdown simulation environment."""
    for env in envs:
        try:
            env.close()
        except Exception:
            LOGGER.exception("Failed to close a simulation environment cleanly")


def run_training(
    config: PPOConfig,
    *,
    policy_store: CheckpointStore = DEFAULT_CHECKPOINT_STORE,
    cancel_requested: Callable[[], bool] = lambda: False,
    reduced: bool = False,
    teams_root: Path = DEFAULT_PATHS.teams_root,
) -> None:
    """
    Set up self-play environments and run PPO training.

    Arguments:
        config: Validated PPO training configuration.
        policy_store: Checkpoint persistence implementation.
        cancel_requested: Callback polled for cooperative cancellation.
        reduced: Sample one seat's teams from the reduced corpus instead of all.
        teams_root: Directory holding the built "all" and "reduced" team pools.

    Returns:
        None.
    """
    opponent_source = load_team_corpus(teams_root / "all", FORMAT.bo3_format)
    agent_source = (
        load_team_corpus(teams_root / "reduced", FORMAT.bo3_format) if reduced else opponent_source
    )
    with training_run(
        policy_store,
        config.checkpoint_path,
        config.output_dir,
        trainer_kind="ppo",
        source_path=config.resume_checkpoint or config.initial_policy_checkpoint,
        resume=config.resume_checkpoint is not None,
    ) as files:
        seed_everything(config.seed)
        resources = default_runtime_resources()
        device = default_device()
        policy = (
            policy_store.load_policy(files.source, device, gamma=config.gamma)
            if files.source is not None
            else build_policy(ModelConfig.baseline(), resources).to(device)
        )
        optimizer = optim.AdamW(
            adamw_param_groups(policy, weight_decay=PPO_WEIGHT_DECAY),
            lr=config.lr,
            eps=ADAM_EPSILON,
        )
        precision = select_optimization_precision(config.enable_optim, device)
        scaler = GradScaler(device.type, enabled=precision.grad_scaler)
        magnet = Magnet(policy)
        scheduler = PPOScheduler(config)
        start = 0
        if config.resume_checkpoint is not None and files.source is not None:
            start = policy_store.load_training(
                files.source,
                policy,
                trainer_kind="ppo",
                optimizer=optimizer,
                scaler=scaler,
                magnet=magnet,
            )
        policy = compile_policy(policy, enable=config.enable_optim and device.type == "cuda")

        files.start(start)
        if start >= config.num_episodes:
            return
        with start_showdown_servers(config.n_envs, log_dir=files.metrics_dir / "logs") as servers:
            envs = []
            vector_env = None

            try:
                for index, server in enumerate(servers):
                    envs.append(
                        build_sim_env(
                            account_configuration1=AccountConfiguration(
                                f"TrainAgent_{index}", None
                            ),
                            account_configuration2=AccountConfiguration(f"BestAgent_{index}", None),
                            server_port=server.port,
                            agent_team_source=agent_source,
                            opponent_team_source=opponent_source,
                            observation_builder=ObservationBuilder(resources=resources),
                            agent_seed=config.seed + index * 2,
                            opponent_seed=config.seed + index * 2 + 1,
                        )
                    )
                vector_env = ThreadVecEnv(envs)
                collector = RolloutCollector(
                    vector_env,
                    policy,
                    config,
                )
                trainer = PPOTrainer(
                    policy=policy,
                    files=files,
                    collector=collector,
                    optimizer=optimizer,
                    scaler=scaler,
                    magnet=magnet,
                    scheduler=scheduler,
                    training_config=config,
                    cancel_requested=cancel_requested,
                )
                trainer.run(start)
            finally:
                if vector_env is not None:
                    vector_env.shutdown()
                else:
                    _close_environments(envs)
