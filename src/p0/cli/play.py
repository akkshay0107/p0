"""Live-play command-line composition root."""

from __future__ import annotations

import sys

from p0.rl_player import main as run_player


def main() -> int:
    if any(value in {"-h", "--help"} for value in sys.argv[1:]):
        print("usage: p0-play [showdown bot options]")
        print(
            "Run the p0 policy as a Pokémon Showdown bot; configuration defaults come from config.yaml."
        )
        return 0
    return run_player(sys.argv[1:])
