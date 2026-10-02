import asyncio
from typing import Any, cast

import pytest
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import RandomPlayer
from poke_env.player.battle_order import DoubleBattleOrder, ForfeitBattleOrder

from p0.battle.views import TransformedPokemonView
from p0.format_config import FORMAT
from p0.runtime import poke_env_patches
from p0.runtime.poke_env_battle_adapter import battle_view

DITTO_TEAM = """
Ditto @ Choice Scarf
Ability: Imposter
Level: 50
Jolly Nature
- Transform

Charizard @ Charizardite Y
Ability: Blaze
Level: 50
Modest Nature
- Heat Wave
- Solar Beam
- Protect
- Weather Ball

Whimsicott @ Focus Sash
Ability: Prankster
Level: 50
Timid Nature
- Moonblast
- Tailwind
- Encore
- Protect

Garchomp @ Sitrus Berry
Ability: Rough Skin
Level: 50
Jolly Nature
- Earthquake
- Dragon Claw
- Rock Slide
- Protect

Kingambit @ Black Glasses
Ability: Defiant
Level: 50
Adamant Nature
- Kowtow Cleave
- Sucker Punch
- Protect
- Low Kick

Glimmora @ Shuca Berry
Ability: Corrosion
Level: 50
Modest Nature
- Power Gem
- Sludge Bomb
- Earth Power
- Protect
"""

OPPONENT_TEAM = """
Pikachu @ Light Ball
Ability: Static
Level: 50
Jolly Nature
- Fake Out
- Protect
- Thunderbolt
- Electroweb

Charizard @ Charizardite Y
Ability: Blaze
Level: 50
Modest Nature
- Heat Wave
- Solar Beam
- Protect
- Weather Ball

Whimsicott @ Focus Sash
Ability: Prankster
Level: 50
Timid Nature
- Moonblast
- Tailwind
- Encore
- Protect

Garchomp @ Sitrus Berry
Ability: Rough Skin
Level: 50
Jolly Nature
- Earthquake
- Dragon Claw
- Rock Slide
- Protect

Kingambit @ Black Glasses
Ability: Defiant
Level: 50
Adamant Nature
- Kowtow Cleave
- Sucker Punch
- Protect
- Low Kick

Glimmora @ Shuca Berry
Ability: Corrosion
Level: 50
Modest Nature
- Power Gem
- Sludge Bomb
- Earth Power
- Protect
"""


class DittoTrackerPlayer(RandomPlayer):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.saw_transform = False
        self.transform_observations: list[tuple[str | None, str, str | None, bool]] = []
        self.transform_move_sets: list[frozenset[str]] = []
        self.post_mega_observations: list[tuple[str | None, str, str | None, bool]] = []
        self.error: Exception | None = None

    def teampreview(self, battle: AbstractBattle) -> str:
        return "/team 1234"

    def choose_move(self, battle: AbstractBattle) -> Any:
        try:
            view = battle_view(cast(DoubleBattle, battle))
            for position, active in enumerate(view.active_pokemon):
                if (
                    active is not None
                    and isinstance(active, TransformedPokemonView)
                    and active.species != "ditto"
                ):
                    self.transform_observations.append(
                        (
                            active.species,
                            active.base_species,
                            active.item,
                            view.decision.slots[position].can_mega,
                        )
                    )
                    self.transform_move_sets.append(frozenset(active.moves))
                    if any(
                        opponent is not None and opponent.species == "charizardmegay"
                        for opponent in view.opponent_active_pokemon
                    ):
                        self.post_mega_observations.append(self.transform_observations[-1])
                    assert active.moves
                    self.saw_transform = True
                    break
        except Exception as exc:
            # Preserve callback failures so the async battle can finish and report them in the test.
            if self.error is None:
                self.error = exc

        if battle.turn >= 2:
            return ForfeitBattleOrder()
        if isinstance(battle, DoubleBattle):
            orders = [
                next(
                    candidate
                    for candidate in slot
                    if getattr(candidate.order, "id", "") == "protect"
                )
                for slot in battle.valid_orders
            ]
            return DoubleBattleOrder(orders[0], orders[1])
        return super().choose_move(battle)


class MegaCharizardPlayer(RandomPlayer):
    def teampreview(self, battle: AbstractBattle) -> str:
        return "/team 1234"

    def choose_move(self, battle: AbstractBattle) -> Any:
        if not isinstance(battle, DoubleBattle):
            return super().choose_move(battle)
        orders = [
            next(
                candidate
                for candidate in slot
                if getattr(candidate.order, "id", "") == "protect" and not candidate.mega
            )
            for slot in battle.valid_orders
        ]

        for position, active in enumerate(battle.active_pokemon):
            if (
                active is None
                or active.base_species != "charizard"
                or not battle.can_mega_evolve[position]
            ):
                continue

            mega_order = next(
                (
                    candidate
                    for candidate in battle.valid_orders[position]
                    if candidate.mega and getattr(candidate.order, "id", "") == "protect"
                ),
                None,
            )
            if mega_order is None:
                continue

            orders[position] = mega_order
            return DoubleBattleOrder(orders[0], orders[1])

        return DoubleBattleOrder(orders[0], orders[1])


class TestDitto:
    @pytest.mark.integration
    @pytest.mark.heavy
    @pytest.mark.asyncio
    async def test_live_ditto_transform_proxy_integration(
        self,
        showdown_server: ServerConfiguration,
    ) -> None:
        """Verify live capture correctly wraps a transformed Ditto with the TransformedPokemonView on a real server."""
        poke_env_patches.install()

        first = DittoTrackerPlayer(
            account_configuration=AccountConfiguration("DittoTrackerA", None),
            battle_format=FORMAT.battle_format,
            server_configuration=showdown_server,
            team=DITTO_TEAM,
            max_concurrent_battles=1,
        )
        second = MegaCharizardPlayer(
            account_configuration=AccountConfiguration("DittoTrackerB", None),
            battle_format=FORMAT.battle_format,
            server_configuration=showdown_server,
            team=OPPONENT_TEAM,
            max_concurrent_battles=1,
        )

        try:
            await asyncio.wait_for(first.battle_against(second, n_battles=1), timeout=60.0)
        finally:
            await first.ps_client.stop_listening()
            await second.ps_client.stop_listening()
            poke_env_patches.uninstall_for_tests()

        if first.error is not None:
            raise first.error

        assert first.saw_transform, "Did not observe a transform proxy during the battle"
        assert ("charizard", "charizard", "choicescarf", False) in first.transform_observations
        assert ("charizard", "charizard", "choicescarf", False) in first.post_mega_observations
        assert frozenset({"heatwave", "solarbeam", "protect", "weatherball"}) in (
            first.transform_move_sets
        )
