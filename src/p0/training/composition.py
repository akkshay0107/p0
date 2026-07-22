"""Application composition for PPO training."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Mapping

import torch.optim as optim
from poke_env import AccountConfiguration
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from p0.format_config import FORMAT
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.runtime.composition import build_sim_env
from p0.runtime.env import SimEnv
from p0.runtime.showdown import start_showdown_servers
from p0.teams.source import FileTeamSource, TeamSource
from p0.training.checkpoint import DEFAULT_POLICY_STORE, PolicyStore
from p0.training.config import CorpusConfig, GlobalConfig, TeamSourceConfig
from p0.training.magnet import Magnet
from p0.training.ppo import PPOUpdater
from p0.training.rollout import RolloutCollector, SeriesContextProvider
from p0.training.trainer import PPOTrainer
from p0.training.utils import PPOScheduler, adamw_param_groups, default_device
from p0.training.vector_env import ThreadVecEnv


def _team_source(
    config: TeamSourceConfig,
    *,
    corpus_config: CorpusConfig | None = None,
    seed: int = 0,
    is_agent: bool = True,
) -> TeamSource:
    path = config.path
    if path.is_dir() and (path / "corpus_manifest.json").exists():
        path = path / "corpus_manifest.json"

    if path.suffix == ".json" or (path.is_file() and path.name.endswith(".json")):
        from p0.teams.corpus import CorpusSourceSpec, CorpusSplit, SamplingPolicy
        from p0.teams.corpus_source import CorpusTeamSource

        raw = json.loads(path.read_text(encoding="utf-8"))
        corpus_hash = str(raw.get("corpus_hash", ""))
        format_id = str(raw.get("format_id", FORMAT.battle_format))

        split = CorpusSplit.TRAIN
        policy = SamplingPolicy.USAGE_WEIGHTED
        allow_mirror = True
        if corpus_config is not None:
            split_name = corpus_config.agent_split.upper() if is_agent else "TRAIN"
            split = CorpusSplit[split_name]
            policy_name = corpus_config.sampling_policy.upper()
            policy = SamplingPolicy[policy_name]
            allow_mirror = corpus_config.allow_mirror

        spec = CorpusSourceSpec(
            corpus_path=str(path),
            corpus_hash=corpus_hash,
            format_id=format_id,
            split=split,
            seed=seed,
            sampling_policy=policy,
            allow_mirror=allow_mirror,
        )
        return CorpusTeamSource(spec)
    return FileTeamSource(config.path)


def _tensorboard_sink(writer: SummaryWriter):
    def emit(metrics: Mapping[str, float], step: int, phase: str) -> None:
        for name, value in metrics.items():
            writer.add_scalar(f"{phase}/{name}", value, step)

    return emit


def _close_environments(envs: list[SimEnv]) -> None:
    for env in envs:
        try:
            env.close()
            for player in (env.agent1, env.agent2):
                asyncio.run_coroutine_threadsafe(
                    player.ps_client.stop_listening(), player.ps_client.loop
                ).result(timeout=2.0)
        except Exception:
            logging.exception("Failed to close a simulation environment cleanly")


def _close_vector_env(vector_env: ThreadVecEnv) -> None:
    vector_env.shutdown()
    _close_environments(vector_env.envs)


def run_training(
    config: GlobalConfig,
    *,
    policy_store: PolicyStore = DEFAULT_POLICY_STORE,
    cancel_requested: Callable[[], bool] = lambda: False,
    series_context: SeriesContextProvider | None = None,
) -> None:
    training, paths = config.training, config.paths
    resources = default_runtime_resources()
    device = default_device()
    policy = (
        policy_store.load_policy(paths.checkpoint_path, device)
        if paths.checkpoint_path.exists()
        else build_policy(ModelConfig.baseline(), resources).to(device)
    )
    optimizer = optim.AdamW(adamw_param_groups(policy, weight_decay=1e-4), lr=training.lr, eps=1e-6)
    scaler = GradScaler(
        "cuda", enabled=training.enable_optim and device.type == "cuda", init_scale=512.0
    )
    magnet = Magnet(policy)
    start = policy_store.load_training_state(
        paths.checkpoint_path, policy, optimizer=optimizer, scaler=scaler, magnet=magnet
    )
    agent_source = _team_source(
        config.environment.agent_team_source,
        corpus_config=config.corpus,
        seed=0,
        is_agent=True,
    )
    opponent_source = _team_source(
        config.environment.opponent_team_source,
        corpus_config=config.corpus,
        seed=1,
        is_agent=False,
    )

    with start_showdown_servers(
        training.n_envs,
        showdown_root=paths.showdown_root,
    ) as servers:
        envs = []
        vector_env = None
        writer = None
        try:
            for index, server in enumerate(servers):
                envs.append(
                    build_sim_env(
                        account_configuration1=AccountConfiguration(f"TrainAgent_{index}", None),
                        account_configuration2=AccountConfiguration(f"BestAgent_{index}", None),
                        server_port=server.port,
                        agent_team_source=agent_source,
                        opponent_team_source=opponent_source,
                        observation_builder=ObservationBuilder(resources=resources),
                        agent_seed=index * 2,
                        opponent_seed=index * 2 + 1,
                    )
                )
            vector_env = ThreadVecEnv(envs)
            writer = SummaryWriter(log_dir=str(paths.runs_dir / "ppo_training"))
            collector = RolloutCollector(
                vector_env,
                policy,
                training,
                series_context=series_context,
            )
            updater = PPOUpdater(
                policy,
                optimizer,
                scaler,
                training,
                magnet,
                cancel_requested=cancel_requested,
            )
            trainer = PPOTrainer(
                policy=policy,
                policy_store=policy_store,
                checkpoint_path=paths.checkpoint_path,
                collector=collector,
                updater=updater,
                magnet=magnet,
                scheduler=PPOScheduler(training),
                training_config=training,
                metric_sink=_tensorboard_sink(writer),
                cancel_requested=cancel_requested,
            )
            trainer.run(start)
        finally:
            if writer is not None:
                writer.close()
            if vector_env is not None:
                _close_vector_env(vector_env)
            else:
                _close_environments(envs)
