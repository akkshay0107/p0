"""Application composition for PPO training."""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch.optim as optim
from poke_env import AccountConfiguration
from torch.amp import GradScaler

from p0.format_config import FORMAT
from p0.model.config import ModelConfig
from p0.model.factory import build_policy, compile_policy
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.runtime.composition import build_sim_env
from p0.runtime.env import SimEnv
from p0.runtime.showdown import start_showdown_servers
from p0.teams.corpus import load_team_corpus
from p0.training.checkpoint import (
    DEFAULT_CHECKPOINT_STORE,
    CheckpointStore,
    value_objective_metadata,
)
from p0.training.config import GlobalConfig
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
    config: GlobalConfig,
    *,
    policy_store: CheckpointStore = DEFAULT_CHECKPOINT_STORE,
    cancel_requested: Callable[[], bool] = lambda: False,
    reduced: bool = False,
) -> None:
    """
    Set up self-play environments and run PPO training.

    Arguments:
        config: Validated global runtime and training configuration.
        policy_store: Checkpoint persistence implementation.
        cancel_requested: Callback polled for cooperative cancellation.
        reduced: Sample one seat's teams from the reduced corpus instead of all.

    Returns:
        None.
    """
    training, paths = config.training, config.paths
    opponent_source = load_team_corpus(config.teams.all, FORMAT.bo3_format)
    agent_source = (
        load_team_corpus(config.teams.reduced, FORMAT.bo3_format) if reduced else opponent_source
    )
    with training_run(
        policy_store,
        paths.checkpoint_path,
        paths.runs_dir / "ppo_training",
        trainer_kind="ppo",
        source_path=paths.resume_checkpoint or paths.initial_policy_checkpoint,
        resume=paths.resume_checkpoint is not None,
    ) as files:
        seed_everything(training.seed)
        resources = default_runtime_resources()
        device = default_device()
        policy = (
            policy_store.load_policy(
                files.source,
                device,
                expected_objective=value_objective_metadata(training.gamma),
            )
            if files.source is not None
            else build_policy(ModelConfig.baseline(), resources).to(device)
        )
        optimizer = optim.AdamW(
            adamw_param_groups(policy, weight_decay=PPO_WEIGHT_DECAY),
            lr=training.lr,
            eps=ADAM_EPSILON,
        )
        precision = select_optimization_precision(training.enable_optim, device)
        scaler = GradScaler(device.type, enabled=precision.grad_scaler)
        magnet = Magnet(policy)
        scheduler = PPOScheduler(training)
        start = 0
        if paths.resume_checkpoint is not None and files.source is not None:
            start = policy_store.load_training(
                files.source,
                policy,
                trainer_kind="ppo",
                optimizer=optimizer,
                scaler=scaler,
                magnet=magnet,
            )
        policy = compile_policy(policy, enable=training.enable_optim and device.type == "cuda")

        files.start(start)
        if start >= training.num_episodes:
            return
        with start_showdown_servers(
            training.n_envs,
            showdown_root=paths.showdown_root,
            log_dir=files.metrics_dir / "logs",
        ) as servers:
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
                            agent_seed=training.seed + index * 2,
                            opponent_seed=training.seed + index * 2 + 1,
                        )
                    )
                vector_env = ThreadVecEnv(envs)
                collector = RolloutCollector(
                    vector_env,
                    policy,
                    training,
                )
                trainer = PPOTrainer(
                    policy=policy,
                    files=files,
                    collector=collector,
                    optimizer=optimizer,
                    scaler=scaler,
                    magnet=magnet,
                    scheduler=scheduler,
                    training_config=training,
                    cancel_requested=cancel_requested,
                )
                trainer.run(start)
            finally:
                if vector_env is not None:
                    vector_env.shutdown()
                else:
                    _close_environments(envs)
