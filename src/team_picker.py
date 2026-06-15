import random

from poke_env.teambuilder import Teambuilder


class RandomTeamFromPool(Teambuilder):
    def __init__(self, teams):
        self.packed_teams = []

        for team in teams:
            parsed_team = self.parse_showdown_team(team)
            packed_team = self.join_team(parsed_team)
            self.packed_teams.append(packed_team)

    def yield_team(self):
        return random.choice(self.packed_teams)


if __name__ == "__main__":
    # running example from poke-env
    team_1 = """
    Goodra (M) @ Assault Vest
    Ability: Sap Sipper
    EVs: 248 HP / 252 SpA / 8 Spe
    Modest Nature
    IVs: 0 Atk
    - Dragon Pulse
    - Flamethrower
    - Sludge Wave
    - Thunderbolt

    Sylveon (M) @ Leftovers
    Ability: Pixilate
    EVs: 248 HP / 244 Def / 16 SpD
    Calm Nature
    IVs: 0 Atk
    - Hyper Voice
    - Mystical Fire
    - Protect
    - Wish

    Cinderace (M) @ Life Orb
    Ability: Blaze
    EVs: 252 Atk / 4 SpD / 252 Spe
    Jolly Nature
    - Pyro Ball
    - Sucker Punch
    - U-turn
    - High Jump Kick

    Toxtricity (M) @ Throat Spray
    Ability: Punk Rock
    EVs: 4 Atk / 252 SpA / 252 Spe
    Rash Nature
    - Overdrive
    - Boomburst
    - Shift Gear
    - Fire Punch

    Seismitoad (M) @ Leftovers
    Ability: Water Absorb
    EVs: 252 HP / 252 Def / 4 SpD
    Relaxed Nature
    - Stealth Rock
    - Scald
    - Earthquake
    - Toxic

    Corviknight (M) @ Leftovers
    Ability: Pressure
    EVs: 248 HP / 80 SpD / 180 Spe
    Impish Nature
    - Defog
    - Brave Bird
    - Roost
    - U-turn
    """

    teams = [team_1]
    builder = RandomTeamFromPool(teams)
    for _ in range(3):
        print(builder.yield_team())
