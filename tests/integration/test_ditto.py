import asyncio
from typing import Any, cast

import pytest
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import RandomPlayer

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
        self.error: Exception | None = None

    def teampreview(self, battle: AbstractBattle) -> str:
        return "/team 1234"

    def choose_move(self, battle: AbstractBattle) -> Any:
        try:
            view = battle_view(cast(DoubleBattle, battle))
            for active in view.active_pokemon:
                if (
                    active is not None
                    and isinstance(active, TransformedPokemonView)
                    and active.species != "ditto"
                ):
                    assert active.base_species == active.species
                    assert active.moves
                    self.saw_transform = True
                    break
        except Exception as e:
            if self.error is None:
                self.error = e

        return super().choose_move(battle)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_ditto_transform_proxy_integration(
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
    second = RandomPlayer(
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

    if first.error:
        raise first.error

    assert first.saw_transform, "Did not observe a transform proxy during the battle"
