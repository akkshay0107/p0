import math
import random
from typing import List

from poke_env.battle import (
    AbstractBattle,
    DoubleBattle,
    Effect,
    Field,
    Move,
    Pokemon,
    PokemonType,
    SideCondition,
    Target,
    Weather,
)
from poke_env.battle.move_category import MoveCategory
from poke_env.calc.damage_calc_gen9 import calculate_damage as original_calculate_damage
from poke_env.player import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
    PassBattleOrder,
    Player,
    SingleBattleOrder,
)

DMG_FRACTION_WEIGHT = 18.0  # multiplied by dmg/max_hp (0..1+)
KO_BONUS = 10.0
THREAT_KO_MULTIPLIER = 12.0  # extra KO bonus scaled by the target's threat level
PRIORITY_KO_BONUS = 4.0
PRIORITY_LOW_HP_BONUS = (
    20.0  # bonus for using priority when at very low HP and slower than opponent
)
SPREAD_MOVE_BONUS = 5.0
WIDE_GUARD_PENALTY = -30.0  # penalty for spread moves vs Wide Guard user
SELF_SWITCH_THREATENED = 10.0
SELF_SWITCH_SAFE = 3.0
IMMUNE_TARGET_PENALTY = -40.0
FAINTED_TARGET_PENALTY = -100.0
ALLY_TARGET_PENALTY = -100.0
RECOIL_PENALTY_WEIGHT = -3.0  # × recoil fraction
DRAIN_BONUS_WEIGHT = 8.0  # x drain fraction
SUCKER_PUNCH_STATUS_PENALTY = -15.0  # penalty when target likely to use status
LAST_RESPECTS_PER_FAINT = 10.0  # extra score per fainted ally for Last Respects
SNARL_BONUS = 10.0
KNOCK_OFF_ITEM_BONUS = 20.0

PROTECT_THREATENED_FAST = 30.0  # protect when opponent is faster + SE
PROTECT_THREATENED_SLOW = 15.0  # protect when opponent is SE but slower
PROTECT_UNTHREATENED = 5.0
PROTECT_LOW_HP_BONUS = 10.0  # extra protect value when HP < 40%
PROTECT_REPEAT_PENALTY = -35.0  # per consecutive protect (× protect_counter)

SWITCH_BASE_PENALTY = -2.0  # inherent cost of losing momentum
SWITCH_INTO_RESIST = 18.0
SWITCH_AWAY_THREATENED = 22.0
SWITCH_POSITIVE_BOOST_PENALTY = -6.0
SWITCH_NEGATIVE_BOOST_BONUS = 4.0
SWITCH_HP_FACTOR = 5.0  # × switch_mon HP fraction (prefer healthy mons)
SWITCH_WEATHER_ABILITY_BONUS = 8.0
SWITCH_INTIMIDATE_BONUS = 10.0
SWITCH_SHADOW_TAG_BLOCKED = -100.0
SWITCH_INTIMIDATE_VS_DEFIANT_PENALTY = -12.0  # don't bring Intimidate vs Defiant/Competitive

TAILWIND_BONUS = 25.0
TAILWIND_ALREADY_ACTIVE = -100.0
TRICK_ROOM_FAVORABLE = 25.0  # tr when our side is slower
TRICK_ROOM_UNFAVORABLE = -20.0  # tr when our side is faster (or already active+favorable)
ICY_WIND_BONUS = 10.0  # have to fix for defiant / competitive

FAKE_OUT_BONUS = 30.0
FAKE_OUT_PRIORITY_TARGET = 8.0  # extra if targeting a setup threat
ENCORE_BONUS = 18.0
ENCORE_PROTECT_LOCK = 22.0
DISABLE_BONUS = 14.0  # disabling opponent's best move
WILL_O_WISP_PHYSICAL = 20.0
WILL_O_WISP_MIXED = 5.0
LEECH_SEED_BONUS = 14.0

HELPING_HAND_BONUS = 2.0
RAGE_POWDER_BONUS = 18.0
RAGE_POWDER_NO_THREAT = 3.0
WIDE_GUARD_SPREAD_THREAT = 18.0  # wide Guard when opponent has spread moves
ROOST_LOW_HP = 22.0
ROOST_HIGH_HP = 3.0
LIFE_DEW_BONUS = 15.0
AURORA_VEIL_BONUS = 22.0
AURORA_VEIL_NO_WEATHER = -50.0  # aurora Veil without snow
BULK_UP_BONUS = 12.0
CALM_MIND_BONUS = 12.0
CLANGOROUS_SOUL_BONUS = 16.0
RAIN_DANCE_BONUS = 15.0  # manual weather when beneficial
PARTING_SHOT_BONUS = 22.0

FAKE_OUT_SETUP_SYNERGY = 30.0  # fake out + setup
HELPING_HAND_ATTACK_SYNERGY = 10.0  # helping hand + attack
HELPING_HAND_KO_SYNERGY = 25.0
DOUBLE_PROTECT_PENALTY = (
    -20.0
)  # both mons protecting (usually wasted momentum), will add exceptions later
FOCUS_FIRE_KO_BONUS = 20.0  # both mons targeting same opponent for KO
SPREAD_PRESSURE_BONUS = 8.0  # spread + single target = good coverage
EQ_ALLY_HIT_PENALTY = -18.0  # earthquake hitting non-immune ally
REDIRECT_SETUP_SYNERGY = 22.0  # rage powder + partner setup/boost

SCORE_JITTER_RANGE = 4.0  # uniform noise ±this value added to each option
TEMP = 8.0
TOP_K = 3
FIRST_TURN_SWITCH_PENALTY = -30.0
MEGA_EVOLUTION_BONUS = 50.0

WEATHER_OVERRIDE_BONUS = 15.0
WEATHER_BENEFICIARY_ACTIVE_BONUS = 12.0
WEATHER_BENEFICIARY_BENCH_BONUS = 6.0
WEATHER_HOSTILE_PENALTY = -6.0
WEATHER_MOVE_POWER_BONUS = 8.0
WEATHER_MOVE_POWER_PENALTY = -6.0
SOLAR_BEAM_SUN_BONUS = 6.0
SOLAR_BEAM_NO_SUN_PENALTY = -15.0
HURRICANE_RAIN_BONUS = 8.0
HURRICANE_SUN_PENALTY = -10.0
ELECTRO_SHOT_RAIN_BONUS = 12.0
BLIZZARD_SNOW_BONUS = 8.0

# team preview scoring constants
TP_MEGA_BONUS = 30.0
TP_LEAD_WEATHER_SYNERGY = 20.0  # bonus for leading weather setter + beneficiary
TP_LEAD_TAILWIND_ATTACKER = 10.0
TP_LEAD_TR_SETTER = 10.0
TP_LEAD_FAKE_OUT = 8.0
TP_LEAD_INTIMIDATE = 6.0
TP_TYPE_MATCHUP_WEIGHT = 2.0  # multiplier for type coverage score


# stat increases assumed for calcs
ASSUMED_HP_BONUS = 107
ASSUMED_STAT_MAX_BONUS = 52
ASSUMED_STAT_MIN_BONUS = 20
DEFAULT_BASE_STAT = 100

THREAT_SPEED_BONUS = 0.3
THREAT_TYPE_MULT = 0.5
THREAT_KO_BONUS = 1.0
THREAT_STATS_BONUS = 0.2
THREAT_MAX_CAP = 1.5
HIGH_OFFENSE_THRESHOLD = 130


def calculate_damage(
    attacker_id: str, defender_id: str, move: Move, battle: DoubleBattle, *args, **kwargs
):
    # wrapper to apply fairy aura since poke_env damage calc doesn't support it
    dmg = original_calculate_damage(attacker_id, defender_id, move, battle, *args, **kwargs)
    if move.type == PokemonType.FAIRY:
        has_fairy_aura = False
        for mon in battle.all_active_pokemons:
            if mon and not mon.fainted:
                ability = getattr(mon, "ability", "")
                if ability:
                    ability = ability.lower().replace(" ", "").replace("-", "")
                species = getattr(mon, "species", "")
                if species:
                    species = species.lower().replace(" ", "").replace("-", "")
                if (
                    ability == "fairyaura"
                    or "floettemega" in species
                    or "floetteeternalmega" in species
                ):
                    has_fairy_aura = True
                    break
        if has_fairy_aura:
            # 1.33x multiplier to min and max damage
            if isinstance(dmg, tuple) and len(dmg) == 2:
                dmg = (dmg[0] * 1.33, dmg[1] * 1.33)
            else:
                dmg *= 1.33
    return dmg


class FuzzyHeuristic(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_stat(self, mon: Pokemon, stat_name: str, default: int = 100) -> int:
        if mon.stats and mon.stats.get(stat_name) is not None:
            val = mon.stats.get(stat_name)
            if val is not None:
                return val
        if hasattr(mon, "_stats") and mon._stats and mon._stats.get(stat_name) is not None:
            val = mon._stats.get(stat_name)
            if val is not None:
                return val
        return default

    def _get_active_identifier(self, mon: Pokemon, battle: DoubleBattle, is_opponent: bool) -> str:
        role = battle.opponent_role if is_opponent else battle.player_role
        active_list = battle.opponent_active_pokemon if is_opponent else battle.active_pokemon
        idx = 1 if len(active_list) > 1 and active_list[1] == mon else 0
        return f"{role}{'a' if idx == 0 else 'b'}: {mon.name}"

    def populate_pokemon_stats(self, pokemon: Pokemon):
        if not hasattr(pokemon, "_stats") or pokemon._stats is None:
            pokemon._stats = {}

        keys = ["hp", "atk", "def", "spa", "spd", "spe"]
        needs_populating = any(pokemon._stats.get(k) is None for k in keys)
        if needs_populating:
            base = pokemon.base_stats
            pokemon._stats["hp"] = base.get("hp", DEFAULT_BASE_STAT) + ASSUMED_HP_BONUS

            has_phys = any(m.category == MoveCategory.PHYSICAL for m in pokemon.moves.values())
            has_spec = any(m.category == MoveCategory.SPECIAL for m in pokemon.moves.values())
            if not has_phys and not has_spec:
                if base.get("atk", DEFAULT_BASE_STAT) > base.get("spa", DEFAULT_BASE_STAT):
                    has_phys = True
                else:
                    has_spec = True

            pokemon._stats["atk"] = base.get("atk", DEFAULT_BASE_STAT) + (
                ASSUMED_STAT_MAX_BONUS if has_phys else ASSUMED_STAT_MIN_BONUS
            )
            pokemon._stats["spa"] = base.get("spa", DEFAULT_BASE_STAT) + (
                ASSUMED_STAT_MAX_BONUS if has_spec else ASSUMED_STAT_MIN_BONUS
            )

            for k in ["def", "spd", "spe"]:
                pokemon._stats[k] = base.get(k, DEFAULT_BASE_STAT) + ASSUMED_STAT_MIN_BONUS

    def get_expected_opponent_types(self, opp_mon: Pokemon) -> List[str]:
        types = set()
        for move in opp_mon.moves.values():
            if move.type:
                types.add(move.type.name)
        if opp_mon.type_1:
            types.add(opp_mon.type_1.name)
        if opp_mon.type_2:
            types.add(opp_mon.type_2.name)
        return list(types)

    def is_spread_move(self, move: Move) -> bool:
        if move.target in [Target.ALL_ADJACENT_FOES, Target.ALL_ADJACENT, Target.ALL]:
            return True
        spread_names = {
            "blizzard",
            "clangingscales",
            "dazzlinggleam",
            "earthquake",
            "heatwave",
            "icywind",
            "rockslide",
            "snarl",
        }
        return move.id in spread_names

    def get_fainted_allies(self, battle: DoubleBattle) -> int:
        return sum(1 for mon in battle.team.values() if mon.fainted)

    def is_armor_tail_active(self, battle: DoubleBattle) -> bool:
        return any(
            mon and not mon.fainted and mon.ability == "armortail"
            for mon in battle.all_active_pokemons
        )

    def is_shadow_tag_active_for_opponent(self, battle: DoubleBattle) -> bool:
        return any(
            opp and not opp.fainted and opp.ability == "shadowtag"
            for opp in battle.opponent_active_pokemon
        )

    def get_actual_speed(self, mon: Pokemon, battle: DoubleBattle | None = None) -> float:
        spe = float(self._get_stat(mon, "spe", 100))
        boost = mon.boosts["spe"] or 0
        if boost > 0:
            spe *= (2.0 + boost) / 2.0
        elif boost < 0:
            spe *= 2.0 / (2.0 - boost)
        from poke_env.battle import Status

        if mon.status == Status.PAR:
            spe *= 0.5

        if battle is not None:
            # tailwind speed modifier
            # check if mon is ours or opponent's to apply correct tailwind
            is_opponent = False
            if battle.opponent_active_pokemon and mon in battle.opponent_active_pokemon:
                is_opponent = True
            elif battle.opponent_team and mon.species in battle.opponent_team:
                # fallback check
                is_opponent = True

            if is_opponent:
                if SideCondition.TAILWIND in battle.opponent_side_conditions:
                    spe *= 2.0
            else:
                if SideCondition.TAILWIND in battle.side_conditions:
                    spe *= 2.0

            # weather speed modifier
            if battle.weather:
                if mon.ability == "chlorophyll" and Weather.SUNNYDAY in battle.weather:
                    spe *= 2.0
                elif mon.ability == "sandrush" and Weather.SANDSTORM in battle.weather:
                    spe *= 2.0

        return spe

    def get_team_avg_speed(self, team_mons: List[Pokemon]) -> float:
        speeds = [self.get_actual_speed(mon) for mon in team_mons if mon and not mon.fainted]
        return sum(speeds) / max(1, len(speeds))

    def _get_mega_weather_ability(self, species: str) -> str | None:
        # returns the weather setting ability of a mega evolved species
        spec = species.lower().replace(" ", "").replace("-", "")
        if spec == "charizardmegay":
            return "drought"
        elif spec == "froslassmega":
            return "snowwarning"
        elif spec == "tyranitarmega":
            return "sandstream"
        return None

    def _weather_for_ability(self, ability: str) -> Weather | None:
        # returns the Weather enum corresponding to a weather setting ability
        if not ability:
            return None
        abil = ability.lower().replace(" ", "").replace("-", "")
        if abil == "drizzle":
            return Weather.RAINDANCE
        elif abil == "drought":
            return Weather.SUNNYDAY
        elif abil == "sandstream":
            return Weather.SANDSTORM
        elif abil == "snowwarning":
            return Weather.SNOW
        return None

    def _mon_benefits_from_weather(self, mon: Pokemon, weather: Weather) -> bool:
        if not mon or mon.fainted:
            return False

        ability = getattr(mon, "ability", "")
        if ability:
            ability = ability.lower().replace(" ", "").replace("-", "")
        if weather == Weather.SUNNYDAY and ability == "chlorophyll":
            return True
        if weather == Weather.SANDSTORM and ability == "sandrush":
            return True

        # check type
        types = [mon.type_1, mon.type_2]
        if weather == Weather.RAINDANCE and PokemonType.WATER in types:
            return True
        if weather == Weather.SUNNYDAY and PokemonType.FIRE in types:
            return True
        if weather == Weather.SANDSTORM and (
            PokemonType.ROCK in types or PokemonType.GROUND in types or PokemonType.STEEL in types
        ):
            return True
        if (weather == Weather.SNOW or weather == Weather.SNOWSCAPE) and PokemonType.ICE in types:
            return True

        # check moves
        for move in mon.moves.values():
            move_id = move.id.lower()
            if weather == Weather.SUNNYDAY:
                if move_id in ["solarbeam", "weatherball", "heatwave", "flareblitz", "overheat"]:
                    return True
            elif weather == Weather.RAINDANCE:
                if move_id in [
                    "hurricane",
                    "electroshot",
                    "weatherball",
                    "hydropump",
                    "wavecrash",
                    "aquajet",
                    "scald",
                    "flipturn",
                    "liquidation",
                ]:
                    return True
            elif weather == Weather.SNOW or weather == Weather.SNOWSCAPE:
                if move_id in ["blizzard"]:
                    return True

        return False

    def _count_weather_beneficiaries(self, team: dict, weather: Weather) -> int:
        count = 0
        for mon in team.values():
            if mon and not mon.fainted:
                if self._mon_benefits_from_weather(mon, weather):
                    count += 1
        return count

    def _is_weather_hostile(self, battle: DoubleBattle) -> bool:
        if not battle.weather or Weather.UNKNOWN in battle.weather:
            return False

        current_weather = None
        for w in [
            Weather.RAINDANCE,
            Weather.SUNNYDAY,
            Weather.SANDSTORM,
            Weather.SNOWSCAPE,
            Weather.SNOW,
        ]:
            if w in battle.weather:
                current_weather = w
                break

        if current_weather is None:
            return False

        our_count = self._count_weather_beneficiaries(battle.team, current_weather)
        opp_count = self._count_weather_beneficiaries(battle.opponent_team, current_weather)

        # if the opponent has more beneficiaries, then it is hostile
        if opp_count > our_count:
            return True
        if opp_count > 0 and our_count == 0:
            return True

        return False

    def get_threat_level(
        self, opp_mon: Pokemon, active_mons: List[Pokemon], battle: DoubleBattle
    ) -> float:
        # returns a threat score between 0.0 and roughly 2.5
        if not opp_mon or opp_mon.fainted:
            return 0.0

        threat = 0.0
        opp_speed = self.get_actual_speed(opp_mon, battle)
        is_tr = Field.TRICK_ROOM in battle.fields

        for ally in active_mons:
            if ally and not ally.fainted:
                ally_speed = self.get_actual_speed(ally, battle)
                is_faster = (opp_speed > ally_speed and not is_tr) or (
                    opp_speed < ally_speed and is_tr
                )
                if is_faster:
                    threat += THREAT_SPEED_BONUS

                max_mult = 1.0
                opp_types = self.get_expected_opponent_types(opp_mon)
                for t_name in opp_types:
                    try:
                        p_type = PokemonType.from_name(t_name)
                        mult = ally.damage_multiplier(p_type)
                        if mult > max_mult:
                            max_mult = mult
                    except Exception:
                        pass
                if max_mult > 1.0:
                    threat += THREAT_TYPE_MULT * (max_mult / 2.0)

                # check for immediate KO threat if faster
                if is_faster:
                    can_ko = False
                    for m in opp_mon.moves.values():
                        if m.category != MoveCategory.STATUS:
                            try:
                                attacker_id = self._get_active_identifier(opp_mon, battle, True)
                                defender_id = self._get_active_identifier(ally, battle, False)
                                min_dmg, _ = calculate_damage(attacker_id, defender_id, m, battle)
                                if min_dmg >= ally.current_hp:
                                    can_ko = True
                                    break
                            except Exception:
                                pass
                    if can_ko:
                        threat += THREAT_KO_BONUS

        # consider offensive presence (base stats roughly)
        opp_atk = self._get_stat(opp_mon, "atk", DEFAULT_BASE_STAT)
        opp_spa = self._get_stat(opp_mon, "spa", DEFAULT_BASE_STAT)
        if max(opp_atk, opp_spa) > HIGH_OFFENSE_THRESHOLD:
            threat += THREAT_STATS_BONUS

        return min(threat, THREAT_MAX_CAP)

    def score_attacking_move(
        self, move: Move, order: SingleBattleOrder, slot: int, battle: DoubleBattle
    ) -> float:
        score = 0.0
        target = order.move_target
        active_mon = battle.active_pokemon[slot]
        if active_mon is None or active_mon.fainted:
            return FAINTED_TARGET_PENALTY

        if move.priority > 0 and self.is_armor_tail_active(battle):
            return -100.0

        if target in [-1, -2]:
            return ALLY_TARGET_PENALTY

        opp_idx = target - 1 if target in [1, 2] else 0
        opp_active = battle.opponent_active_pokemon[opp_idx]

        if opp_active is None or opp_active.fainted:
            return FAINTED_TARGET_PENALTY

        attacker_id = self._get_active_identifier(active_mon, battle, False)

        fainted = self.get_fainted_allies(battle)
        if move.id == "lastrespects":
            score += fainted * LAST_RESPECTS_PER_FAINT

        targets = []
        if self.is_spread_move(move):
            targets = [0, 1]
        else:
            targets = [opp_idx]

        total_dmg_score = 0.0
        dmg_fraction = 0.0
        active_allies = [m for m in battle.active_pokemon if m and not m.fainted]

        for t_idx in targets:
            t_opp = battle.opponent_active_pokemon[t_idx]
            if t_opp is None or t_opp.fainted:
                continue

            t_defender_id = self._get_active_identifier(t_opp, battle, True)
            try:
                dmg, _ = calculate_damage(
                    attacker_id=attacker_id,
                    defender_id=t_defender_id,
                    move=move,
                    battle=battle,
                )
            except Exception:
                dmg = move.base_power * t_opp.damage_multiplier(move) * 0.3

            if dmg == 0:
                continue

            if (
                len(targets) > 1
                and len([o for o in battle.opponent_active_pokemon if o and not o.fainted]) > 1
            ):
                dmg *= 0.75

            dmg_fraction = dmg / max(1, t_opp.max_hp)
            total_dmg_score += dmg_fraction * DMG_FRACTION_WEIGHT

            if dmg >= t_opp.current_hp:
                threat_level = self.get_threat_level(t_opp, active_allies, battle)
                total_dmg_score += KO_BONUS + (THREAT_KO_MULTIPLIER * threat_level)
                if move.priority > 0:
                    total_dmg_score += PRIORITY_KO_BONUS

            if move.priority > 0 and active_mon.current_hp_fraction < 0.3:
                our_speed = self.get_actual_speed(active_mon, battle)
                opp_speed = self.get_actual_speed(t_opp, battle)
                if our_speed < opp_speed:
                    total_dmg_score += PRIORITY_LOW_HP_BONUS

        if total_dmg_score == 0.0:
            return IMMUNE_TARGET_PENALTY

        score += total_dmg_score

        if move.id == "suckerpunch":
            our_speed = self.get_actual_speed(active_mon, battle)
            opp_speed = self.get_actual_speed(opp_active, battle)

            if our_speed > opp_speed:
                score += -20.0

            if (
                opp_active.base_stats.get("atk", DEFAULT_BASE_STAT) < 60
                and opp_active.base_stats.get("spa", DEFAULT_BASE_STAT) < 80
            ):
                score += SUCKER_PUNCH_STATUS_PENALTY

        if self.is_spread_move(move):
            has_wide_guard_threat = False
            for opp in battle.opponent_active_pokemon:
                if opp and not opp.fainted:
                    if "wideguard" in opp.moves:
                        has_wide_guard_threat = True
            if has_wide_guard_threat:
                score += WIDE_GUARD_PENALTY
            else:
                score += SPREAD_MOVE_BONUS

        if move.self_switch:
            is_threatened = False
            if active_mon:
                our_speed = self._get_stat(active_mon, "spe", DEFAULT_BASE_STAT)
                for opp in battle.opponent_active_pokemon:
                    if opp and not opp.fainted:
                        opp_speed = self._get_stat(opp, "spe", DEFAULT_BASE_STAT)
                        opp_types = self.get_expected_opponent_types(opp)
                        for t_name in opp_types:
                            try:
                                p_type = PokemonType.from_name(t_name)
                                if (
                                    active_mon.damage_multiplier(p_type) > 1.0
                                    and opp_speed > our_speed
                                ):
                                    is_threatened = True
                            except Exception:
                                pass
                if is_threatened or active_mon.current_hp_fraction < 0.5:
                    score += SELF_SWITCH_THREATENED
                else:
                    score += SELF_SWITCH_SAFE

        if move.recoil:
            score += RECOIL_PENALTY_WEIGHT * move.recoil * dmg_fraction

        if move.drain:
            score += DRAIN_BONUS_WEIGHT * move.drain * dmg_fraction

        if move.id == "snarl":
            score += SNARL_BONUS

        if (
            move.id == "knockoff"
            and opp_active.item != "unknown_item"
            and opp_active.item is not None
        ):
            score += KNOCK_OFF_ITEM_BONUS

        # weather bonuses/penalties for moves
        weather = battle.weather
        if weather and Weather.UNKNOWN not in weather:
            is_rain = Weather.RAINDANCE in weather
            is_sun = Weather.SUNNYDAY in weather
            is_snow = Weather.SNOW in weather

            if move.id == "solarbeam":
                if is_sun:
                    score += SOLAR_BEAM_SUN_BONUS
                else:
                    score += SOLAR_BEAM_NO_SUN_PENALTY
            elif move.id == "hurricane":
                if is_rain:
                    score += HURRICANE_RAIN_BONUS
                elif is_sun:
                    score += HURRICANE_SUN_PENALTY
            elif move.id == "electroshot":
                if is_rain:
                    score += ELECTRO_SHOT_RAIN_BONUS
                else:
                    score += SOLAR_BEAM_NO_SUN_PENALTY
            elif move.id == "blizzard":
                if is_snow:
                    score += BLIZZARD_SNOW_BONUS
            elif move.id == "weatherball":
                score += WEATHER_MOVE_POWER_BONUS

            # check type-based weather boosts
            if move.type == PokemonType.WATER:
                if is_rain:
                    score += WEATHER_MOVE_POWER_BONUS
                elif is_sun:
                    score += WEATHER_MOVE_POWER_PENALTY
            elif move.type == PokemonType.FIRE:
                if is_sun:
                    score += WEATHER_MOVE_POWER_BONUS
                elif is_rain:
                    score += WEATHER_MOVE_POWER_PENALTY
        else:
            # no weather active
            if move.id in ["solarbeam", "electroshot"]:
                score += SOLAR_BEAM_NO_SUN_PENALTY

        acc = move.accuracy if move.accuracy is not None else 1.0
        # adjust accuracy dynamically for weather-dependent moves
        if weather and Weather.UNKNOWN not in weather:
            is_rain = Weather.RAINDANCE in weather
            is_snow = (
                Weather.SNOW in weather or Weather.SNOWSCAPE in weather or Weather.HAIL in weather
            )
            if move.id == "hurricane" and is_rain:
                acc = 1.0
            elif move.id == "blizzard" and is_snow:
                acc = 1.0

        score *= acc
        return score

    def score_status_move(
        self, move: Move, order: SingleBattleOrder, slot: int, battle: DoubleBattle
    ) -> float:
        score = 0.0
        active_mon = battle.active_pokemon[slot]
        if active_mon is None or active_mon.fainted:
            return 0.0

        target = order.move_target
        if target in [-1, -2]:
            if move.id not in ["helpinghand", "lifedew", "healpulse"]:
                return -100.0

        opp_idx = target - 1 if target in [1, 2] else 0
        opp_active = (
            battle.opponent_active_pokemon[opp_idx] if opp_idx >= 0 and opp_idx < 2 else None
        )

        is_prankster = active_mon.ability == "prankster"
        target_is_dark = (
            opp_active
            and opp_active.damage_multiplier(PokemonType.DARK) == 0.5
            and PokemonType.DARK in opp_active.types
        )

        if move.id in ["protect", "spikyshield"]:
            if active_mon.protect_counter > 0:
                score += PROTECT_REPEAT_PENALTY * active_mon.protect_counter
            else:
                is_threatened = False
                threat_is_faster = False
                our_speed = self._get_stat(active_mon, "spe", DEFAULT_BASE_STAT)

                for opp in battle.opponent_active_pokemon:
                    if opp and not opp.fainted:
                        opp_speed = self._get_stat(opp, "spe", DEFAULT_BASE_STAT)
                        opp_types = self.get_expected_opponent_types(opp)

                        has_se_threat = False
                        for t_name in opp_types:
                            try:
                                p_type = PokemonType.from_name(t_name)
                                if active_mon.damage_multiplier(p_type) > 1.0:
                                    has_se_threat = True
                            except Exception:
                                pass

                        if has_se_threat:
                            is_threatened = True
                            if opp_speed > our_speed:
                                threat_is_faster = True

                if is_threatened:
                    score += (
                        PROTECT_THREATENED_FAST if threat_is_faster else PROTECT_THREATENED_SLOW
                    )
                else:
                    score += PROTECT_UNTHREATENED

                if active_mon.current_hp_fraction < 0.4:
                    score += PROTECT_LOW_HP_BONUS

        elif move.id == "fakeout":
            if not active_mon.first_turn:
                return -100.0
            if self.is_armor_tail_active(battle):
                return -100.0
            if opp_active and PokemonType.GHOST in opp_active.types:
                return -100.0

            score += FAKE_OUT_BONUS
            if opp_active and (
                "trickroom" in opp_active.moves
                or "tailwind" in opp_active.moves
                or "calmmind" in opp_active.moves
            ):
                score += FAKE_OUT_PRIORITY_TARGET

        elif move.id == "tailwind":
            if Field.TRICK_ROOM in battle.fields:
                return -100.0

            if SideCondition.TAILWIND not in battle.side_conditions:
                helps = False
                for ally in battle.active_pokemon:
                    if ally and not ally.fainted:
                        ally_speed = self.get_actual_speed(ally, battle)
                        for opp in battle.opponent_active_pokemon:
                            if opp and not opp.fainted:
                                opp_speed = self.get_actual_speed(opp, battle)
                                if ally_speed < opp_speed or (ally_speed < opp_speed * 1.5):
                                    helps = True
                if helps:
                    score += TAILWIND_BONUS
                else:
                    score += TAILWIND_BONUS / 2.0
            else:
                score += TAILWIND_ALREADY_ACTIVE

        elif move.id == "trickroom":
            is_tr_active = Field.TRICK_ROOM in battle.fields
            slower_count = 0
            for ally in battle.active_pokemon:
                if ally and not ally.fainted:
                    ally_speed = self._get_stat(ally, "spe", DEFAULT_BASE_STAT)
                    for opp in battle.opponent_active_pokemon:
                        if opp and not opp.fainted:
                            opp_speed = self._get_stat(opp, "spe", DEFAULT_BASE_STAT)
                            if ally_speed < opp_speed:
                                slower_count += 1

            we_are_slower = slower_count > 0

            if not is_tr_active:
                score += TRICK_ROOM_FAVORABLE if we_are_slower else TRICK_ROOM_UNFAVORABLE
            else:
                score += TRICK_ROOM_UNFAVORABLE if we_are_slower else TRICK_ROOM_FAVORABLE

            if active_mon.ability == "armortail":
                score += 5.0  # farig only trick room setter so doesnt really matter

        elif move.id == "willowisp":
            if target not in [1, 2] or opp_active is None or opp_active.fainted:
                return -100.0
            if opp_active.status is not None or PokemonType.FIRE in opp_active.types:
                return -100.0
            if is_prankster and target_is_dark:
                return -100.0

            is_physical = self._get_stat(opp_active, "atk", DEFAULT_BASE_STAT) > self._get_stat(
                opp_active, "spa", DEFAULT_BASE_STAT
            )
            if is_physical:
                score += WILL_O_WISP_PHYSICAL
            else:
                score += WILL_O_WISP_MIXED

        elif move.id == "encore":
            if target not in [1, 2] or opp_active is None or opp_active.fainted:
                return -100.0
            if is_prankster and target_is_dark:
                return -100.0
            if Effect.ENCORE in opp_active.effects:
                return -100.0

            last_move = opp_active.last_move
            if last_move:
                if last_move.id in ["protect", "spikyshield", "fakeout"]:
                    score += ENCORE_PROTECT_LOCK
                elif last_move.category == MoveCategory.STATUS:
                    score += ENCORE_BONUS
                else:
                    mult = active_mon.damage_multiplier(last_move)
                    if mult < 1.0:
                        score += ENCORE_BONUS
            else:
                return -100.0

        elif move.id == "disable":
            if target not in [1, 2] or opp_active is None or opp_active.fainted:
                return -100.0
            if is_prankster and target_is_dark:
                return -100.0
            if Effect.DISABLE in opp_active.effects:
                return -100.0

            last_move = opp_active.last_move
            if last_move:
                mult = active_mon.damage_multiplier(last_move)
                if mult > 1.0:
                    score += DISABLE_BONUS
            else:
                return -100.0

        elif move.id == "leechseed":
            if target not in [1, 2] or opp_active is None or opp_active.fainted:
                return -100.0
            if PokemonType.GRASS in opp_active.types or Effect.LEECH_SEED in opp_active.effects:
                return -100.0
            else:
                score += LEECH_SEED_BONUS

        elif move.id == "helpinghand":
            partner = battle.active_pokemon[1 - slot]
            if partner is None or partner.fainted:
                return -100.0
            score += HELPING_HAND_BONUS

        elif move.id == "ragepowder":
            if PokemonType.GRASS in active_mon.types:
                pass
            partner = battle.active_pokemon[1 - slot]
            if partner and not partner.fainted:
                if partner.current_hp_fraction < 0.5 or any(
                    m.id in ["calmmind", "bulkup", "swordsdance", "dragonstrike"]
                    for m in partner.moves.values()
                ):
                    score += RAGE_POWDER_BONUS
                else:
                    score += RAGE_POWDER_NO_THREAT

        elif move.id == "wideguard":
            has_spread_threat = False
            for opp in battle.opponent_active_pokemon:
                if opp and not opp.fainted:
                    for opp_move in opp.moves.values():
                        if self.is_spread_move(opp_move):
                            has_spread_threat = True
            if has_spread_threat:
                score += WIDE_GUARD_SPREAD_THREAT

        elif move.id in ["roost", "lifedew"]:
            if active_mon.current_hp_fraction < 0.5:
                score += ROOST_LOW_HP
            else:
                score += ROOST_HIGH_HP
            if move.id == "lifedew":
                partner = battle.active_pokemon[1 - slot]
                if partner and partner.current_hp_fraction < 0.5:
                    score += ROOST_LOW_HP

        elif move.id == "auroraveil":
            if Weather.HAIL in battle.weather or Weather.SNOW in battle.weather:
                score += AURORA_VEIL_BONUS
            else:
                score += AURORA_VEIL_NO_WEATHER

        elif move.id == "bulkup":
            if active_mon.boosts.get("atk", 0) >= 2 or active_mon.boosts.get("def", 0) >= 2:
                return -50.0
            score += BULK_UP_BONUS

        elif move.id == "calmmind":
            if active_mon.boosts.get("spa", 0) >= 2 or active_mon.boosts.get("spd", 0) >= 2:
                return -50.0
            score += CALM_MIND_BONUS

        elif move.id == "clangoroussoul":
            if active_mon.boosts.get("atk", 0) >= 1 or active_mon.boosts.get("spe", 0) >= 1:
                return -50.0
            if active_mon.current_hp_fraction > 0.4:
                score += CLANGOROUS_SOUL_BONUS
            else:
                score -= 30.0

        elif move.id == "raindance":
            if Weather.RAINDANCE in battle.weather:
                return -50.0

            if self._is_weather_hostile(battle):
                score += WEATHER_OVERRIDE_BONUS
            else:
                score += RAIN_DANCE_BONUS

            our_bens = self._count_weather_beneficiaries(battle.team, Weather.RAINDANCE)
            score += our_bens * WEATHER_BENEFICIARY_ACTIVE_BONUS

            opp_bens = self._count_weather_beneficiaries(battle.opponent_team, Weather.RAINDANCE)
            score -= opp_bens * WEATHER_BENEFICIARY_ACTIVE_BONUS

            # redundant since sableye only setter
            if active_mon.ability == "prankster":
                score += 5.0

        elif move.id == "partingshot":
            score += PARTING_SHOT_BONUS

        elif move.id == "icywind":
            score += ICY_WIND_BONUS

        else:
            score += 10.0

        return score

    def score_single_order(
        self, order: SingleBattleOrder, slot: int, battle: DoubleBattle
    ) -> float:
        active_mon = battle.active_pokemon[slot]

        if isinstance(order, PassBattleOrder):
            return 0.0

        if isinstance(order.order, Pokemon):
            switch_mon = order.order
            self.populate_pokemon_stats(switch_mon)

            is_forced_switch = active_mon is None or active_mon.fainted

            if not is_forced_switch and self.is_shadow_tag_active_for_opponent(battle):
                return SWITCH_SHADOW_TAG_BLOCKED

            score = 0.0
            if active_mon is not None and not is_forced_switch:
                score += SWITCH_BASE_PENALTY
                if active_mon.first_turn and battle.turn > 1:
                    score += FIRST_TURN_SWITCH_PENALTY

                positive_boosts = sum(v for v in active_mon.boosts.values() if v > 0)
                negative_boosts = sum(abs(v) for v in active_mon.boosts.values() if v < 0)
                score += positive_boosts * SWITCH_POSITIVE_BOOST_PENALTY
                score += negative_boosts * SWITCH_NEGATIVE_BOOST_BONUS

            score += switch_mon.current_hp_fraction * SWITCH_HP_FACTOR

            if active_mon is not None and not is_forced_switch:
                is_threatened_with_ko = False
                for opp in battle.opponent_active_pokemon:
                    if opp and not opp.fainted:
                        for m in opp.moves.values():
                            if m.category != MoveCategory.STATUS:
                                try:
                                    attacker_id = self._get_active_identifier(opp, battle, True)
                                    defender_id = self._get_active_identifier(
                                        active_mon, battle, False
                                    )
                                    min_dmg, _ = calculate_damage(
                                        attacker_id, defender_id, m, battle
                                    )
                                    if min_dmg >= active_mon.current_hp:
                                        is_threatened_with_ko = True
                                        break
                                except Exception:
                                    pass
                        if is_threatened_with_ko:
                            break
                if is_threatened_with_ko:
                    score += SWITCH_AWAY_THREATENED

            for opp in battle.opponent_active_pokemon:
                if opp and not opp.fainted:
                    resists_opp = False
                    opp_types = self.get_expected_opponent_types(opp)
                    for t_name in opp_types:
                        try:
                            p_type = PokemonType.from_name(t_name)
                            if switch_mon.damage_multiplier(p_type) < 1.0:
                                resists_opp = True
                        except Exception:
                            pass
                    if resists_opp:
                        score += SWITCH_INTO_RESIST

            weather_abilities = {"drought", "sandstream", "drizzle", "snowwarning"}
            if switch_mon.ability in weather_abilities:
                target_weather = self._weather_for_ability(switch_mon.ability)
                if target_weather and (not battle.weather or target_weather not in battle.weather):
                    score += SWITCH_WEATHER_ABILITY_BONUS

                    if self._is_weather_hostile(battle):
                        score += WEATHER_OVERRIDE_BONUS

                    for m in battle.team.values():
                        if m and not m.fainted and m != switch_mon:
                            if self._mon_benefits_from_weather(m, target_weather):
                                if m in battle.active_pokemon:
                                    score += WEATHER_BENEFICIARY_ACTIVE_BONUS
                                else:
                                    score += WEATHER_BENEFICIARY_BENCH_BONUS

                    for m in battle.opponent_team.values():
                        if m and not m.fainted:
                            if self._mon_benefits_from_weather(m, target_weather):
                                if m in battle.opponent_active_pokemon:
                                    score -= WEATHER_BENEFICIARY_ACTIVE_BONUS
                                else:
                                    score -= WEATHER_BENEFICIARY_BENCH_BONUS

            if switch_mon.ability == "intimidate":
                has_defiant = any(
                    opp and not opp.fainted and opp.ability in ["defiant", "competitive"]
                    for opp in battle.opponent_active_pokemon
                )
                if has_defiant:
                    score += SWITCH_INTIMIDATE_VS_DEFIANT_PENALTY
                else:
                    score += SWITCH_INTIMIDATE_BONUS

            return score

        if isinstance(order.order, Move):
            if active_mon is None or active_mon.fainted:
                return 0.0

            original_species = active_mon.species
            original_temp_ability = active_mon.temporary_ability
            original_stats = (
                active_mon._stats.copy()
                if (hasattr(active_mon, "_stats") and active_mon._stats)
                else None
            )

            if order.mega and active_mon.item:
                active_mon.mega_evolve(active_mon.item)
                setattr(active_mon, "_stats", None)
                self.populate_pokemon_stats(active_mon)

            try:
                move = order.order
                if move.category != MoveCategory.STATUS:
                    score = self.score_attacking_move(move, order, slot, battle)
                else:
                    score = self.score_status_move(move, order, slot, battle)
                if order.mega:
                    score += MEGA_EVOLUTION_BONUS

                    mega_weather_abilities = {"drought", "sandstream", "drizzle", "snowwarning"}
                    if active_mon.ability in mega_weather_abilities:
                        target_weather = self._weather_for_ability(active_mon.ability)
                        if target_weather and (
                            not battle.weather or target_weather not in battle.weather
                        ):
                            if self._is_weather_hostile(battle):
                                score += WEATHER_OVERRIDE_BONUS

                            for m in battle.team.values():
                                if m and not m.fainted and m != active_mon:
                                    if self._mon_benefits_from_weather(m, target_weather):
                                        if m in battle.active_pokemon:
                                            score += WEATHER_BENEFICIARY_ACTIVE_BONUS
                                        else:
                                            score += WEATHER_BENEFICIARY_BENCH_BONUS

                            for m in battle.opponent_team.values():
                                if m and not m.fainted:
                                    if self._mon_benefits_from_weather(m, target_weather):
                                        if m in battle.opponent_active_pokemon:
                                            score -= WEATHER_BENEFICIARY_ACTIVE_BONUS
                                        else:
                                            score -= WEATHER_BENEFICIARY_BENCH_BONUS
                return score
            finally:
                if order.mega:
                    active_mon._update_from_pokedex(original_species, store_species=False)
                    active_mon.temporary_ability = original_temp_ability
                    if original_stats:
                        setattr(active_mon, "_stats", original_stats)

        return 0.0

    def evaluate_synergy(
        self, order0: SingleBattleOrder, order1: SingleBattleOrder, battle: DoubleBattle
    ) -> float:
        score = 0.0

        active0 = battle.active_pokemon[0]
        active1 = battle.active_pokemon[1]

        if active0 is None or active1 is None or active0.fainted or active1.fainted:
            return 0.0

        move0 = order0.order if isinstance(order0.order, Move) else None
        move1 = order1.order if isinstance(order1.order, Move) else None

        if move0 and move1:
            is_fake_out_0 = move0.id == "fakeout" and active0.first_turn
            is_fake_out_1 = move1.id == "fakeout" and active1.first_turn

            setup_moves = {
                "bulkup",
                "calmmind",
                "clangoroussoul",
                "auroraveil",
            }
            is_setup_0 = move0.id in setup_moves
            is_setup_1 = move1.id in setup_moves

            if (is_fake_out_0 and is_setup_1) or (is_fake_out_1 and is_setup_0):
                score += FAKE_OUT_SETUP_SYNERGY

            if move0.id == "helpinghand" and move1.category != MoveCategory.STATUS:
                target_idx = order1.move_target - 1 if order1.move_target in [1, 2] else 0
                target_mon = (
                    battle.opponent_active_pokemon[target_idx]
                    if target_idx >= 0 and target_idx < 2
                    else None
                )
                if target_mon and not target_mon.fainted:
                    try:
                        dmg, _ = calculate_damage(
                            attacker_id=self._get_active_identifier(active1, battle, False),
                            defender_id=self._get_active_identifier(target_mon, battle, True),
                            move=move1,
                            battle=battle,
                        )
                        if dmg < target_mon.current_hp and (dmg * 1.5) >= target_mon.current_hp:
                            score += HELPING_HAND_KO_SYNERGY
                        else:
                            score += HELPING_HAND_ATTACK_SYNERGY
                    except Exception:
                        score += HELPING_HAND_ATTACK_SYNERGY
                else:
                    score += HELPING_HAND_ATTACK_SYNERGY

            if move1.id == "helpinghand" and move0.category != MoveCategory.STATUS:
                target_idx = order0.move_target - 1 if order0.move_target in [1, 2] else 0
                target_mon = (
                    battle.opponent_active_pokemon[target_idx]
                    if target_idx >= 0 and target_idx < 2
                    else None
                )
                if target_mon and not target_mon.fainted:
                    try:
                        dmg, _ = calculate_damage(
                            attacker_id=self._get_active_identifier(active0, battle, False),
                            defender_id=self._get_active_identifier(target_mon, battle, True),
                            move=move0,
                            battle=battle,
                        )
                        if dmg < target_mon.current_hp and (dmg * 1.5) >= target_mon.current_hp:
                            score += HELPING_HAND_KO_SYNERGY
                        else:
                            score += HELPING_HAND_ATTACK_SYNERGY
                    except Exception:
                        score += HELPING_HAND_ATTACK_SYNERGY
                else:
                    score += HELPING_HAND_ATTACK_SYNERGY

            is_rage_powder_0 = move0.id == "ragepowder"
            is_rage_powder_1 = move1.id == "ragepowder"

            if (is_rage_powder_0 and is_setup_1) or (is_rage_powder_1 and is_setup_0):
                score += REDIRECT_SETUP_SYNERGY

            is_protect_0 = move0.id in ["protect", "spikyshield"]
            is_protect_1 = move1.id in ["protect", "spikyshield"]
            if is_protect_0 and is_protect_1:
                has_fake_out = any(
                    opp and "fakeout" in opp.moves for opp in battle.opponent_active_pokemon
                )
                has_setup = any(
                    opp and any(m.id in setup_moves for m in opp.moves.values())
                    for opp in battle.opponent_active_pokemon
                )
                fake_out_exception = has_fake_out and not has_setup

                is_last_tr = (
                    Field.TRICK_ROOM in battle.fields
                    and battle.turn == battle.fields[Field.TRICK_ROOM] + 4
                )
                is_last_weather = any(
                    battle.turn == start_turn + 4 for start_turn in battle.weather.values()
                )

                if not (fake_out_exception or is_last_tr or is_last_weather):
                    score += DOUBLE_PROTECT_PENALTY

            if move0.category != MoveCategory.STATUS and move1.category != MoveCategory.STATUS:
                if order0.move_target == order1.move_target and order0.move_target in [1, 2]:
                    target_idx = order0.move_target - 1
                    target_mon = battle.opponent_active_pokemon[target_idx]

                    if target_mon and not target_mon.fainted:
                        try:
                            dmg0, _ = calculate_damage(
                                attacker_id=self._get_active_identifier(active0, battle, False),
                                defender_id=self._get_active_identifier(target_mon, battle, True),
                                move=move0,
                                battle=battle,
                            )
                            dmg1, _ = calculate_damage(
                                attacker_id=self._get_active_identifier(active1, battle, False),
                                defender_id=self._get_active_identifier(target_mon, battle, True),
                                move=move1,
                                battle=battle,
                            )
                            total_dmg = dmg0 + dmg1
                        except Exception:
                            total_dmg = (move0.base_power + move1.base_power) * 0.3

                        if total_dmg >= target_mon.current_hp:
                            score += FOCUS_FIRE_KO_BONUS

            def check_eq_penalty(attacker_idx, move, partner_active, partner_move):
                is_immune = (
                    PokemonType.FLYING in partner_active.types
                    or partner_active.ability == "levitate"
                )
                if not is_immune and partner_move.id not in ["protect", "spikyshield"]:
                    kos = 0
                    alive_opps = 0
                    for _, opp in enumerate(battle.opponent_active_pokemon):
                        if opp and not opp.fainted:
                            alive_opps += 1
                            try:
                                attacker_mon = active0 if attacker_idx == 0 else active1
                                dmg, _ = calculate_damage(
                                    attacker_id=self._get_active_identifier(
                                        attacker_mon, battle, False
                                    ),
                                    defender_id=self._get_active_identifier(opp, battle, True),
                                    move=move,
                                    battle=battle,
                                )
                                if dmg * 0.75 >= opp.current_hp:
                                    kos += 1
                            except Exception:
                                pass
                    if alive_opps > 0 and kos == alive_opps:
                        return 0.0
                    return EQ_ALLY_HIT_PENALTY
                return 0.0

            if move0.id == "earthquake" and move1.category != MoveCategory.STATUS:
                score += check_eq_penalty(0, move0, active1, move1)

            if move1.id == "earthquake" and move0.category != MoveCategory.STATUS:
                score += check_eq_penalty(1, move1, active0, move0)

        if (
            move0
            and move1
            and move0.category != MoveCategory.STATUS
            and move1.category != MoveCategory.STATUS
        ):
            if order0.move_target != order1.move_target and (
                self.is_spread_move(move0) or self.is_spread_move(move1)
            ):
                score += SPREAD_PRESSURE_BONUS

        return score

    def score_joint_order(self, joint: DoubleBattleOrder, battle: DoubleBattle) -> float:
        score = 0.0
        score += self.score_single_order(joint.first_order, 0, battle)
        score += self.score_single_order(joint.second_order, 1, battle)
        score += self.evaluate_synergy(joint.first_order, joint.second_order, battle)
        return score

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if not isinstance(battle, DoubleBattle):
            return self.choose_random_move(battle)

        for mon in battle.active_pokemon:
            if mon is not None:
                self.populate_pokemon_stats(mon)
        for opp in battle.opponent_active_pokemon:
            if opp is not None:
                self.populate_pokemon_stats(opp)
        for s_list in battle.available_switches:
            for mon in s_list:
                if mon is not None:
                    self.populate_pokemon_stats(mon)

        orders0, orders1 = battle.valid_orders
        joint_orders = DoubleBattleOrder.join_orders(orders0, orders1)

        if not joint_orders:
            o0 = orders0[0] if orders0 else DefaultBattleOrder()
            o1 = orders1[0] if orders1 else DefaultBattleOrder()
            return DoubleBattleOrder(o0, o1)

        scored_orders = []
        for joint in joint_orders:
            score = self.score_joint_order(joint, battle)
            # add uniform jitter
            score += random.uniform(-SCORE_JITTER_RANGE, SCORE_JITTER_RANGE)
            scored_orders.append((score, joint))

        # top k into softmax
        scored_orders.sort(key=lambda x: x[0], reverse=True)
        top_k_orders = scored_orders[:TOP_K]
        max_score = top_k_orders[0][0]

        weights = []
        for s, _ in top_k_orders:
            # safe softmax, probably unnecessary given range of scores
            try:
                w = math.exp((s - max_score) / TEMP)
            except OverflowError:
                w = 0.0
            weights.append(w)

        total_w = sum(weights)
        if total_w == 0:
            return top_k_orders[0][1]

        probs = [w / total_w for w in weights]
        chosen_joint = random.choices([j for s, j in top_k_orders], weights=probs, k=1)[0]

        return chosen_joint

    def _score_team_combination(
        self, combination: List[Pokemon], opponent_team: List[Pokemon]
    ) -> float:
        score = 0.0

        has_mega = any(mon.item and "ite" in mon.item.lower() for mon in combination)
        if has_mega:
            score += TP_MEGA_BONUS

        def get_set_weather(mon: Pokemon) -> Weather | None:
            if mon.ability in ["drizzle", "drought", "sandstream", "snowwarning"]:
                return self._weather_for_ability(mon.ability)
            if mon.item and "ite" in mon.item.lower():
                spec = mon.species.lower().replace(" ", "").replace("-", "")
                mega_ability = None
                if spec == "charizard" and "charizarditey" in mon.item.lower():
                    mega_ability = "drought"
                elif spec == "froslass":
                    mega_ability = "snowwarning"
                elif spec == "tyranitar":
                    mega_ability = "sandstream"
                if mega_ability:
                    return self._weather_for_ability(mega_ability)
            return None

        opp_has_weather_setter = {}
        for w in [Weather.RAINDANCE, Weather.SUNNYDAY, Weather.SANDSTORM, Weather.SNOW]:
            opp_has_weather_setter[w] = False
            for opp_mon in opponent_team:
                if opp_mon.ability in ["drizzle", "drought", "sandstream", "snowwarning"]:
                    if self._weather_for_ability(opp_mon.ability) == w:
                        opp_has_weather_setter[w] = True
                if opp_mon.item and "ite" in opp_mon.item.lower():
                    spec = opp_mon.species.lower().replace(" ", "").replace("-", "")
                    mega_ability = None
                    if spec == "charizard" and "charizarditey" in opp_mon.item.lower():
                        mega_ability = "drought"
                    elif spec == "froslass":
                        mega_ability = "snowwarning"
                    elif spec == "tyranitar":
                        mega_ability = "sandstream"
                    if mega_ability and self._weather_for_ability(mega_ability) == w:
                        opp_has_weather_setter[w] = True

        set_weathers = set()
        for mon in combination:
            w = get_set_weather(mon)
            if w:
                set_weathers.add(w)

        def get_off_types(mon: Pokemon) -> List[PokemonType]:
            t_list = []
            for move in mon.moves.values():
                if move.type and move.category != MoveCategory.STATUS:
                    t_list.append(move.type)
            if not t_list:
                if mon.type_1:
                    t_list.append(mon.type_1)
                if mon.type_2:
                    t_list.append(mon.type_2)
            return list(set(t_list))

        def get_opp_off_types(opp_mon: Pokemon) -> List[PokemonType]:
            t_list = []
            for move in opp_mon.moves.values():
                if move.type and move.category != MoveCategory.STATUS:
                    t_list.append(move.type)
            if not t_list:
                if opp_mon.type_1:
                    t_list.append(opp_mon.type_1)
                if opp_mon.type_2:
                    t_list.append(opp_mon.type_2)
            return list(set(t_list))

        off_score = 0.0
        for opp_mon in opponent_team:
            max_mult = 0.0
            for our_mon in combination:
                for off_type in get_off_types(our_mon):
                    try:
                        mult = opp_mon.damage_multiplier(off_type)
                        if mult > max_mult:
                            max_mult = mult
                    except Exception:
                        pass
            if max_mult >= 2.0:
                off_score += 3.0
            elif max_mult >= 1.0:
                off_score += 1.0
            else:
                off_score -= 2.0

        def_score = 0.0
        for opp_mon in opponent_team:
            opp_types = get_opp_off_types(opp_mon)
            for our_mon in combination:
                for o_type in opp_types:
                    try:
                        mult = our_mon.damage_multiplier(o_type)
                        if mult < 1.0:
                            def_score += 1.0
                        elif mult > 1.0:
                            def_score -= 1.0
                    except Exception:
                        pass

        score += (off_score + def_score) * TP_TYPE_MATCHUP_WEIGHT
        return score

    def _score_lead_synergy(self, lead0: Pokemon, lead1: Pokemon) -> float:
        score = 0.0

        def get_set_weather(mon: Pokemon) -> Weather | None:
            if mon.ability in ["drizzle", "drought", "sandstream", "snowwarning"]:
                return self._weather_for_ability(mon.ability)
            if mon.item and "ite" in mon.item.lower():
                spec = mon.species.lower().replace(" ", "").replace("-", "")
                mega_ability = None
                if spec == "charizard" and "charizarditey" in mon.item.lower():
                    mega_ability = "drought"
                elif spec == "froslass":
                    mega_ability = "snowwarning"
                elif spec == "tyranitar":
                    mega_ability = "sandstream"
                if mega_ability:
                    return self._weather_for_ability(mega_ability)
            return None

        lead0_weather = get_set_weather(lead0)
        lead1_weather = get_set_weather(lead1)

        if lead0_weather and self._mon_benefits_from_weather(lead1, lead0_weather):
            score += TP_LEAD_WEATHER_SYNERGY
        if lead1_weather and self._mon_benefits_from_weather(lead0, lead1_weather):
            score += TP_LEAD_WEATHER_SYNERGY

        lead0_has_tw = "tailwind" in lead0.moves
        lead1_has_tw = "tailwind" in lead1.moves
        lead0_is_attacker = any(m.category != MoveCategory.STATUS for m in lead0.moves.values())
        lead1_is_attacker = any(m.category != MoveCategory.STATUS for m in lead1.moves.values())

        if (lead0_has_tw and lead1_is_attacker) or (lead1_has_tw and lead0_is_attacker):
            score += TP_LEAD_TAILWIND_ATTACKER

        lead0_has_tr = "trickroom" in lead0.moves
        lead1_has_tr = "trickroom" in lead1.moves
        if lead0_has_tr or lead1_has_tr:
            score += TP_LEAD_TR_SETTER

        lead0_has_fo = "fakeout" in lead0.moves
        lead1_has_fo = "fakeout" in lead1.moves
        if lead0_has_fo or lead1_has_fo:
            score += TP_LEAD_FAKE_OUT

        lead0_has_int = lead0.ability == "intimidate"
        lead1_has_int = lead1.ability == "intimidate"
        if lead0_has_int or lead1_has_int:
            score += TP_LEAD_INTIMIDATE

        return score

    def teampreview(self, battle: AbstractBattle) -> str:
        import itertools

        my_mons = list(battle.team.values())
        opp_mons = list(battle.opponent_team.values())

        # rank all 15 choices of 4 pokemon
        scored_combos = []
        for combo in itertools.combinations(my_mons, 4):
            score = self._score_team_combination(list(combo), opp_mons)
            scored_combos.append((score, list(combo)))

        scored_combos.sort(key=lambda x: x[0], reverse=True)
        top_3_combos = scored_combos[:3]

        # for each of top 3 combos, pick top 2 lead pairs
        selected_permutations = []
        for _, combo in top_3_combos:
            lead_scores = []
            for lead in itertools.combinations(combo, 2):
                l0, l1 = lead
                lead_score = self._score_lead_synergy(l0, l1)
                back = [m for m in combo if m != l0 and m != l1]
                lead_scores.append((lead_score, [l0, l1, back[0], back[1]]))

            # sort and pick the top 2 lead pairs
            lead_scores.sort(key=lambda x: x[0], reverse=True)
            selected_permutations.extend([perm for _, perm in lead_scores[:2]])

        if not selected_permutations:
            return self.random_teampreview(battle)

        chosen_perm = random.choice(selected_permutations)

        indices = []
        for mon in chosen_perm:
            idx = my_mons.index(mon) + 1
            indices.append(idx)
            mon._selected_in_teampreview = True

        return "/team " + "".join(str(i) for i in indices)
