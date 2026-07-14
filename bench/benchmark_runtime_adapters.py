"""Deterministic microbenchmarks for checkpoint-two runtime adapters."""

from __future__ import annotations

import argparse
import logging
import statistics
import time

from poke_env.battle import DoubleBattle, Pokemon

from p0.battle.events import RawBattleEvent, parse_events
from p0.model.resources import default_runtime_resources
from p0.runtime.live_event_capture import capture_message, consume_raw_events
from p0.runtime.poke_env_action_adapter import action_to_single_order


def _battle_fixture() -> DoubleBattle:
    battle = DoubleBattle("adapter-bench", "benchmark-user", logging.getLogger(__name__), 9)
    battle._player_role = "p1"
    ally = Pokemon(gen=9, species="charizard")
    ally._active = True
    battle._team = {"p1: Charizard": ally}
    battle._active_pokemon = {"p1a": ally}
    return battle


def _measure(operation, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        operation()
    return (time.perf_counter() - start) / iterations


def _median(operation, iterations: int, repeats: int) -> tuple[float, float]:
    samples = [_measure(operation, iterations) for _ in range(repeats)]
    quartiles = statistics.quantiles(samples, n=4, method="inclusive")
    return statistics.median(samples), quartiles[2] - quartiles[0]


def benchmark(args: argparse.Namespace) -> None:
    battle = _battle_fixture()

    def order_conversion() -> None:
        action_to_single_order(0, battle, True, 0)

    def event_capture() -> None:
        capture_message(battle, ["", "-weather", "none"])
        consume_raw_events(battle)

    raw_events = [RawBattleEvent(("", "move", "p1a: Charizard", "Protect", "p1a: Charizard"))]
    resolver = default_runtime_resources().tokenizer

    def event_parse() -> None:
        parse_events(raw_events, resolver)

    order_median, order_iqr = _median(order_conversion, args.iterations, args.repeats)
    event_median, event_iqr = _median(event_capture, args.iterations, args.repeats)
    parse_median, parse_iqr = _median(event_parse, args.iterations, args.repeats)
    print(f"iterations={args.iterations} repeats={args.repeats}")
    print(f"single_order_conversion_median_seconds={order_median:.8f}")
    print(f"single_order_conversion_iqr_seconds={order_iqr:.8f}")
    print(f"live_event_capture_median_seconds={event_median:.8f}")
    print(f"live_event_capture_iqr_seconds={event_iqr:.8f}")
    print(f"protocol_event_parse_median_seconds={parse_median:.8f}")
    print(f"protocol_event_parse_iqr_seconds={parse_iqr:.8f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.iterations <= 0 or args.repeats < 5:
        parser.error("iterations must be positive and repeats must be at least five")
    return args


if __name__ == "__main__":
    benchmark(parse_args())
