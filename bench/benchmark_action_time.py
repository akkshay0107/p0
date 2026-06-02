import argparse
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
from torch.distributions import Categorical

from src.env import MegaEnv
from src.lookups import ACT_SIZE
from src.model.policy import PolicyNet
from src.model.tokenizer import tokenizer
from src.rl_player import RLPlayer
from src.team_picker import RandomTeamFromPool


@dataclass(slots=True)
class TimeTracker:
    tokenizer_time = 0.0
    observation_builder_time = 0.0
    action_mask_time = 0.0
    tensor_prep_time = 0.0
    policy_inference_time = 0.0
    top_p_time = 0.0
    post_process_time = 0.0


def get_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


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
                TimeTracker.tokenizer_time += get_time() - start

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

    def _get_action(self, battle, is_tp):
        self.call_count += 1
        self.profiler.enable()
        try:
            # tokenizer stage
            start_obs = get_time()
            tok_start = TimeTracker.tokenizer_time
            obs = self.get_observation(battle)
            tok_end = TimeTracker.tokenizer_time
            obs_duration = get_time() - start_obs

            # non tokenizing observation building stage
            tok_duration_call = tok_end - tok_start
            TimeTracker.observation_builder_time += obs_duration - tok_duration_call

            # action mask fetch
            start_mask = get_time()
            action_mask_list = MegaEnv.get_action_mask(battle)
            action_mask = torch.tensor([action_mask_list[:ACT_SIZE], action_mask_list[ACT_SIZE:]])
            TimeTracker.action_mask_time += get_time() - start_mask

            # cpu to gpu moving stage (underrepresented if device is cpu)
            start_prep = get_time()
            obs_t = obs.unsqueeze(0).to(self.policy.device)
            action_mask_t = action_mask.unsqueeze(0).to(self.policy.device)
            TimeTracker.tensor_prep_time += get_time() - start_prep

            # forward pass + sampling (overrepresented if device is cpu)
            with torch.no_grad():
                actions = self._top_p(
                    obs_t,
                    action_mask_t,
                    is_tp,
                )

            # gpu to cpu (underrepresented if device is cpu)
            start_post = get_time()
            result = actions[0].cpu().numpy()
            TimeTracker.post_process_time += get_time() - start_post

            return result
        finally:
            self.profiler.disable()

    def _top_p(self, obs, action_mask, is_tp: bool):
        if self.state is None:
            reducer = self.policy.actor.reducer
            batch_size = 1
            device = self.policy.device
            cls = reducer.cls_base.detach().expand(batch_size, -1, -1).squeeze(1).to(device)
            hg = reducer.hg_init.detach().expand(batch_size, -1, -1).to(device)
            self.state = (cls, hg)

        start_inference = get_time()
        with torch.no_grad():
            tokens = self.policy.encoder(obs)
            numerical = obs.numerical
            if numerical.dim() == 2:
                numerical = numerical.unsqueeze(0)
            padding_mask = self.policy._get_padding_mask(numerical)
            z, self.state = self.policy.actor.reducer(tokens, self.state, padding_mask)
            # Pokemon 1: P(a1 | z)
            logits1 = self.policy.actor.head1(z)
            if action_mask is not None:
                logits1 = logits1.masked_fill(action_mask[:, 0] == 0, float("-inf"))

            end_inference = get_time()
            TimeTracker.policy_inference_time += end_inference - start_inference

            start_topp = get_time()
            p1_logits_top_p = self._apply_top_p(logits1)
            cat1 = Categorical(logits=p1_logits_top_p)
            action1 = cat1.sample()  # (B,)
            TimeTracker.top_p_time += get_time() - start_topp

            start_inference = get_time()
            # Pokemon 2: P(a2 | z, a1)
            a1_emb = self.policy.actor.action_embedding(action1)
            logits2 = self.policy.actor.head2(torch.cat([z, a1_emb], dim=-1))

            # Combine and apply sequential masks
            logits = torch.stack([logits1, logits2], dim=1)
            if action_mask is not None:
                is_tp_t = torch.tensor([is_tp], device=self.policy.device, dtype=torch.bool)
                logits = self.policy.actor._apply_sequential_masks(
                    logits, action1, action_mask, is_tp_t
                )
            end_inference = get_time()
            TimeTracker.policy_inference_time += end_inference - start_inference

            start_topp = get_time()
            p2_logits_top_p = self._apply_top_p(logits[:, 1])
            cat2 = Categorical(logits=p2_logits_top_p)
            action2 = cat2.sample()  # (B,)
            TimeTracker.top_p_time += get_time() - start_topp

            return torch.stack([action1, action2], dim=-1)

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
            TimeTracker.tokenizer_time
            + TimeTracker.observation_builder_time
            + TimeTracker.action_mask_time
            + TimeTracker.tensor_prep_time
            + TimeTracker.policy_inference_time
            + TimeTracker.top_p_time
            + TimeTracker.post_process_time
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
            ("Tokenizer", TimeTracker.tokenizer_time),
            ("Observation Builder (non-tok)", TimeTracker.observation_builder_time),
            ("Action Masking", TimeTracker.action_mask_time),
            ("Tensor Prep / Device Transfer", TimeTracker.tensor_prep_time),
            ("Policy Inference", TimeTracker.policy_inference_time),
            ("Top-P Sampling", TimeTracker.top_p_time),
            ("Post-processing", TimeTracker.post_process_time),
        ]
        for name, val in stages:
            avg_ms = (val / max(1, self.call_count)) * 1000
            pct = (val / tot) * 100
            print(f"{name:<35} | {val:<15.4f} | {avg_ms:<15.3f} | {pct:<9.1f}%")
        print(f"{'=' * 80}\n")


async def main():
    parser = argparse.ArgumentParser(description="Profile the RL forward pass during battles.")
    parser.add_argument(
        "-n", "--n-battles", type=int, default=5, help="Number of battles to run for profiling."
    )
    parser.add_argument(
        "--sort", type=str, default="tottime", help="Sort criteria for pstats (tottime, cumtime)."
    )
    parser.add_argument(
        "--limit", type=int, default=50, help="Number of lines to show in the profile output."
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    teams_dir = root_dir / "teams"

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

    # random policy weights for benchmarking
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    policy = PolicyNet().to(device)
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

    print(f"Starting {args.n_battles} battles to profile the forward pass...")
    try:
        await rl_player.battle_against(opponent, n_battles=args.n_battles)
    except Exception as e:
        print(f"Error during battles: {e}")
        print("Make sure a local Pokémon Showdown server is running.")
        return

    rl_player.print_profiling_results(sort_by=args.sort, limit=args.limit)


if __name__ == "__main__":
    asyncio.run(main())
