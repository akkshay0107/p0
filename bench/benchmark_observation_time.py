"""Deterministic live-object observation-builder benchmark."""

from __future__ import annotations

import argparse
import logging
import statistics
import time

from poke_env.battle import DoubleBattle, Pokemon

from p0.battle.legality import action_mask
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.runtime.poke_env_battle_adapter import battle_view, decision_view


def _battle_fixture() -> DoubleBattle:
    battle = DoubleBattle("benchmark", "benchmark-user", logging.getLogger(__name__), 9)
    battle._player_role = "p1"
    allies = [Pokemon(gen=9, species=name) for name in ("charizard", "venusaur", "tyranitar")]
    opponents = [Pokemon(gen=9, species=name) for name in ("pikachu", "gengar", "dragonite")]
    for pokemon in allies + opponents:
        pokemon._current_hp = 100
        pokemon._max_hp = 100
    allies[0]._active = True
    opponents[0]._active = True
    battle._team = {f"p1: {pokemon.species}": pokemon for pokemon in allies}
    battle._opponent_team = {f"p2: {pokemon.species}": pokemon for pokemon in opponents}
    battle._active_pokemon = {"p1a": allies[0]}
    battle._opponent_active_pokemon = {"p2a": opponents[0]}
    battle._available_switches = [allies[1:], []]
    return battle


def _measure(operation, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        operation()
    return (time.perf_counter() - start) / iterations


def _summary(samples: list[float]) -> tuple[float, float]:
    quartiles = statistics.quantiles(samples, n=4, method="inclusive")
    return statistics.median(samples), quartiles[2] - quartiles[0]


def benchmark(args: argparse.Namespace) -> None:
    battle = _battle_fixture()
    target = StructuredObservation.empty_batch(1)[0]
    builder = ObservationBuilder(default_runtime_resources())
    view = battle_view(battle)
    for _ in range(args.warmup):
        builder.build_into(battle_view(battle), target)

    write_samples = [
        _measure(lambda: builder.build_into(battle_view(battle), target), args.iterations)
        for _ in range(args.repeats)
    ]
    allocation_samples = [
        _measure(lambda: builder.build(battle_view(battle)), args.iterations)
        for _ in range(args.repeats)
    ]
    view_write_samples = [
        _measure(lambda: builder.build_into(view, target), args.iterations)
        for _ in range(args.repeats)
    ]
    adapter_samples = [
        _measure(lambda: battle_view(battle), args.iterations) for _ in range(args.repeats)
    ]
    legality_samples = [
        _measure(
            lambda: action_mask(decision_view(battle)),
            args.iterations,
        )
        for _ in range(args.repeats)
    ]
    write_median, write_iqr = _summary(write_samples)
    allocation_median, allocation_iqr = _summary(allocation_samples)
    view_write_median, view_write_iqr = _summary(view_write_samples)
    adapter_median, adapter_iqr = _summary(adapter_samples)
    legality_median, legality_iqr = _summary(legality_samples)
    print(f"iterations={args.iterations} repeats={args.repeats} fixture_pokemon=6")
    print(f"write_samples_seconds={','.join(f'{sample:.8f}' for sample in write_samples)}")
    print(f"write_median_seconds={write_median:.8f}")
    print(f"write_iqr_seconds={write_iqr:.8f}")
    print(
        "allocate_and_write_samples_seconds="
        + ",".join(f"{sample:.8f}" for sample in allocation_samples)
    )
    print(f"allocate_and_write_median_seconds={allocation_median:.8f}")
    print(f"allocate_and_write_iqr_seconds={allocation_iqr:.8f}")
    print(f"view_write_median_seconds={view_write_median:.8f}")
    print(f"view_write_iqr_seconds={view_write_iqr:.8f}")
    print(f"cached_adapter_refresh_median_seconds={adapter_median:.8f}")
    print(f"cached_adapter_refresh_iqr_seconds={adapter_iqr:.8f}")
    print(f"decision_and_legality_median_seconds={legality_median:.8f}")
    print(f"decision_and_legality_iqr_seconds={legality_iqr:.8f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.warmup <= 0 or args.iterations <= 0 or args.repeats < 5:
        parser.error("warmup and iterations must be positive; repeats must be at least five")
    return args


if __name__ == "__main__":
    benchmark(parse_args())
