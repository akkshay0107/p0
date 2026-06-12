import asyncio
import cProfile
import io
import pstats
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import SimpleHeuristicsPlayer

from src.env import MegaEnv
from src.lookups import ACT_SIZE
from src.model.policy import PolicyNet
from src.model.tokenizer import tokenizer
from src.rl_player import RLPlayer
from src.team_picker import RandomTeamFromPool
from src.train.config import load_config
from src.train.utils import default_device, load_checkpoint

N_BATTLES = 5
PROFILE_SORT = "tottime"
PROFILE_LIMIT = 50


@dataclass(slots=True)
class TimeTracker:
    tokenizer_time: float = 0.0
    observation_builder_time: float = 0.0
    action_mask_time: float = 0.0
    tensor_prep_time: float = 0.0
    encoder_time: float = 0.0
    policy_action_time: float = 0.0
    post_process_time: float = 0.0


TIMINGS = TimeTracker()


def get_time():
    return time.perf_counter()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# Instrument tokenizer methods to measure tokenizer stage
tokenizer_methods = [
    "id_for",
    "status_id",
    "volatile_ids",
    "species_id",
    "ability_id",
    "item_id",
    "type_id",
    "move_id",
    "move_type_id",
    "move_category_id",
]
for name in tokenizer_methods:
    original_func = getattr(tokenizer, name)

    def make_wrapped(orig):
        def wrapped(*args, **kwargs):
            start = get_time()
            try:
                return orig(*args, **kwargs)
            finally:
                TIMINGS.tokenizer_time += get_time() - start

        return wrapped

    setattr(tokenizer, name, make_wrapped(original_func))


class ProfiledRLPlayer(RLPlayer):
    """
    RLPlayer that profiles its own forward pass (_get_action method).
    This includes detailed stage-by-stage timing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiler = cProfile.Profile()
        self.call_count = 0

    def _get_action(self, battle):
        self.call_count += 1
        self.profiler.enable()
        try:
            # tokenizer stage
            start_obs = get_time()
            tok_start = TIMINGS.tokenizer_time
            obs = self.get_observation(battle)
            tok_end = TIMINGS.tokenizer_time
            obs_duration = get_time() - start_obs

            # non tokenizing observation building stage
            tok_duration_call = tok_end - tok_start
            TIMINGS.observation_builder_time += obs_duration - tok_duration_call

            # action mask fetch
            start_mask = get_time()
            action_mask_list = MegaEnv.get_action_mask(battle)
            action_mask = torch.tensor([action_mask_list[:ACT_SIZE], action_mask_list[ACT_SIZE:]])
            TIMINGS.action_mask_time += get_time() - start_mask

            # cpu to gpu moving stage (underrepresented if device is cpu)
            device = self.policy.device
            synchronize(device)
            start_prep = get_time()
            obs_t = obs.unsqueeze(0).to(device)
            action_mask_t = action_mask.unsqueeze(0).to(device)
            synchronize(device)
            TIMINGS.tensor_prep_time += get_time() - start_prep

            with torch.no_grad():
                if self.state is None:
                    self.state = self.policy.initial_state(1)

                synchronize(device)
                start_inference = get_time()
                enc = self.policy.encode(obs_t, action_mask_t)
                synchronize(device)
                TIMINGS.encoder_time += get_time() - start_inference

                start_inference = get_time()
                out = self.policy.act(
                    enc,
                    action_mask_t,
                    self.state,
                    top_p=self.top_p,
                )
                synchronize(device)
                TIMINGS.policy_action_time += get_time() - start_inference
                self.state = out.state

            start_post = get_time()
            result = out.actions[0].cpu().numpy()
            synchronize(device)
            TIMINGS.post_process_time += get_time() - start_post

            return result
        finally:
            self.profiler.disable()

    def print_profiling_results(self, sort_by="tottime", limit=50):
        # cProfile results
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats(sort_by)
        ps.print_stats(limit)

        print(f"\n{'=' * 20} cProfile Results for {self.username} {'=' * 20}")
        print(f"Total _get_action calls: {self.call_count}")
        print(s.getvalue())
        print(f"{'=' * 80}\n")

        # stagewise profiling
        tot = (
            TIMINGS.tokenizer_time
            + TIMINGS.observation_builder_time
            + TIMINGS.action_mask_time
            + TIMINGS.tensor_prep_time
            + TIMINGS.encoder_time
            + TIMINGS.policy_action_time
            + TIMINGS.post_process_time
        )
        tot = max(tot, 1e-9)

        print(f"\n{'=' * 20} Stage Breakdown for {self.username} {'=' * 20}")
        print(f"Total calls: {self.call_count}")
        print(f"Total measured time: {tot:.4f} s")
        print(f"Avg time per call:   {tot / max(1, self.call_count) * 1000:.3f} ms")
        print(
            f"\n{'Stage':<35} | {'Total Time (s)':<15} | {'Avg/Call (ms)':<15} | {'Percentage':<10}"
        )
        print("-" * 83)

        stages = [
            ("Tokenizer", TIMINGS.tokenizer_time),
            ("Observation Builder (non-tok)", TIMINGS.observation_builder_time),
            ("Action Masking", TIMINGS.action_mask_time),
            ("Tensor Prep / Device Transfer", TIMINGS.tensor_prep_time),
            ("Policy Encoding", TIMINGS.encoder_time),
            ("Policy Act / Top-P", TIMINGS.policy_action_time),
            ("Post-processing", TIMINGS.post_process_time),
        ]
        for name, val in stages:
            avg_ms = (val / max(1, self.call_count)) * 1000
            pct = (val / tot) * 100
            print(f"{name:<35} | {val:<15.4f} | {avg_ms:<15.3f} | {pct:<9.1f}%")
        print(f"{'=' * 80}\n")


async def main():
    root_dir = Path(__file__).resolve().parent.parent
    teams_dir = root_dir / "teams"
    config = load_config(root_dir / ".ppoconfig")

    if not teams_dir.exists():
        print(f"Teams directory not found: {teams_dir}")
        return

    team_files = [
        path.read_text(encoding="utf-8")
        for path in teams_dir.iterdir()
        if path.is_file() and not path.name.startswith(".")
    ]
    if not team_files:
        print("No team files found in teams directory.")
        return

    team = RandomTeamFromPool(team_files)
    fmt = "gen9championsvgc2026regma"

    device = default_device()
    print(f"Using device: {device}")
    if not config.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint_path}")
    policy = PolicyNet().to(device)
    load_checkpoint(config.checkpoint_path, policy)
    policy.eval()

    rl_player = ProfiledRLPlayer(
        policy=policy,
        account_configuration=AccountConfiguration("RL_Profiler", None),
        battle_format=fmt,
        server_configuration=LocalhostServerConfiguration,
        team=team,
        accept_open_team_sheet=True,
    )

    opponent = SimpleHeuristicsPlayer(
        account_configuration=AccountConfiguration("Opponent", None),
        battle_format=fmt,
        server_configuration=LocalhostServerConfiguration,
        team=team,
        accept_open_team_sheet=True,
    )

    print(f"Starting {N_BATTLES} battles to profile the forward pass...")
    try:
        await rl_player.battle_against(opponent, n_battles=N_BATTLES)
    except Exception as e:
        print(f"Error during battles: {e}")
        print("Make sure a local Pokémon Showdown server is running.")
        return

    rl_player.print_profiling_results(sort_by=PROFILE_SORT, limit=PROFILE_LIMIT)


if __name__ == "__main__":
    from src.showdown_server import spawned_showdown

    with spawned_showdown(port=8000):
        asyncio.run(main())
