"""Application composition for PPO training."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict

import torch.optim as optim
from poke_env import AccountConfiguration
from torch.amp import GradScaler

from p0.model.config import ModelConfig
from p0.model.factory import build_policy, compile_policy
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.runtime.composition import build_sim_env
from p0.runtime.env import SimEnv
from p0.runtime.showdown import start_showdown_servers
from p0.teams.source import build_team_source
from p0.training.checkpoint import DEFAULT_CHECKPOINT_STORE, CheckpointStore
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
    agent_team_source: str = "all",
) -> None:
    """
    Set up self-play environments and run PPO training.

    Arguments:
        config: Validated global runtime and training configuration.
        policy_store: Checkpoint persistence implementation.
        cancel_requested: Callback polled for cooperative cancellation.
        agent_team_source: Team pool name used for the self-play agent.

    Returns:
        None.
    """
    training, paths = config.training, config.paths
    if agent_team_source == "all":
        agent_team_path = config.teams.all
    elif agent_team_source == "reduced":
        agent_team_path = config.teams.reduced
    else:
        raise ValueError("agent_team_source must be 'all' or 'reduced'")

    agent_source = build_team_source(agent_team_path, expected_format_id=config.bot.battle_format)
    opponent_source = build_team_source(
        config.teams.all, expected_format_id=config.bot.battle_format
    )
    settings = {
        "training": asdict(training),
        "teams": [
            {key: value for key, value in source.describe().items() if key != "corpus_path"}
            for source in (agent_source, opponent_source)
        ],
    }
    with training_run(
        policy_store,
        paths.checkpoint_path,
        paths.runs_dir / "ppo_training",
        trainer_kind="ppo",
        settings=settings,
        source_path=paths.resume_checkpoint or paths.initial_policy_checkpoint,
        resume=paths.resume_checkpoint is not None,
    ) as files:
        files.metadata["inputs"] = {
            "agent_teams": str(agent_team_path.resolve()),
            "opponent_teams": str(config.teams.all.resolve()),
        }
        seed_everything(training.seed)
        resources = default_runtime_resources()
        device = default_device()
        policy = (
            policy_store.load_policy(
                files.source,
                device,
                expected_metadata={
                    "gamma": training.gamma,
                    "value_target_semantics": "discounted_terminal_outcome.v1",
                },
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
        resume_environment_state: object | None = None
        resume_collector_state: object | None = None
        start = (
            policy_store.load_training(
                files.source,
                policy,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                magnet=magnet,
                expected_trainer_kind="ppo",
                require_training_state=True,
            )
            if paths.resume_checkpoint is not None and files.source is not None
            else 0
        )
        if paths.resume_checkpoint is not None and files.source is not None:
            resume_metadata = policy_store.load_metadata(files.source)
            environment_state = resume_metadata.get("environment_state")
            collector_state = resume_metadata.get("collector_state")
            if isinstance(environment_state, (tuple, list)) and isinstance(
                collector_state, Mapping
            ):
                resume_environment_state = environment_state
                resume_collector_state = collector_state
            else:
                LOGGER.warning("PPO checkpoint has incomplete rollout state; starting fresh series")
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
                if resume_environment_state is not None and resume_collector_state is not None:
                    fresh_environment_state = vector_env.training_state()
                    try:
                        vector_env.restore_training_state(resume_environment_state)
                        collector.restore_training_state(resume_collector_state)
                    except (TypeError, ValueError) as exc:
                        vector_env.restore_training_state(fresh_environment_state)
                        collector = RolloutCollector(vector_env, policy, training)
                        LOGGER.warning(
                            "PPO checkpoint has malformed rollout state; starting fresh series: %s",
                            exc,
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
