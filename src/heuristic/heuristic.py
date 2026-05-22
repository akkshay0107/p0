import copy
import re

import numpy as np
import torch
from numpy.typing import NDArray
from poke_env.battle import AbstractBattle, DoubleBattle, Move, Pokemon
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.weather import Weather
from poke_env.player import (
    BattleOrder,
    DoubleBattleOrder,
    PassBattleOrder,
    Player,
)

import observation_builder
from env import Gen9VGCEnv


class FuzzyHeuristic(Player):
    """
    A heuristic player designed for Double Battles.
    Generates all valid pairs of actions, scores them using a set of rules,
    takes the top 5, and samples using softmax.
    """

    # Move + ability classes
    REDIRECTION = {"followme", "ragepowder"}
    SETUP = {"nastyplot", "bulkup"}
    SPEED_CONTROL = {"icywind", "drumbeating", "tailwind", "trickroom"}
    SPREAD = {
        "makeitrain",
        "eruption",
        "heatwave",
        "earthquake",
        "rockslide",
        "dazzlinggleam",
        "hypervoice",
        "muddywater",
        "blizzard",
        "icywind",
        "expandingforce",
    }
    PIVOT = {"uturn", "partingshot"}
    STAT_DROPPING = {"icywind", "partingshot", "drumbeating", "faketears"}
    PLUS_PRIORITY = {
        "extremespeed",
        "grassyglide",
        "aquajet",
        "fakeout",
        "feint",
        "vacuumwave",
    }
    GAMBLES = {"direclaw"}
    PROTECT = {"protect", "detect"}
    FIELD_SELF_MOVES = {
        "tailwind",
        "trickroom",
        "haze",
        "auroraveil",
        "sunnyday",
    }
    NO_TARGET_STATUS = {"followme", "ragepowder", "wideguard"}
    OPPONENT_TARGETED = {"taunt", "encore", "spore", "faketears", "partingshot"}
    ALLY_TARGETED = {"helpinghand", "coaching", "pollenpuff"}
    FIXED_DAMAGE = {"nightshade"}

    # for team preview heuristics
    TEAM_CORES = {
        "hatterene": {"hatterene", "indeedee-f"},
        "annihilape": {"annihilape", "maushold-four"},
        "pelipper": {"archaludon", "pelipper"},
        "sneasler": {"rillaboom", "gholdengo"},
        "typhlosion-hisui": {"typhlosion-hisui", "whimsicott"},
        "porygon2": {"porygon2", "ursaluna"},
    }

    LEAD_BIAS = {
        "incineroar": 0.5,  # fake-out + intimidate
        "rillaboom": 0.4,  # fake-out + terrain
        "maushold": 0.3,  # follow-me + taunt
        "maushold-four": 0.3,
        "indeedee-f": 0.4,  # psychic surge + follow-me + TR
        "hatterene": 0.2,  # TR setter
        "gallade": 0.1,  # TR setter
        "whimsicott": 0.4,  # prankster tailwind lead
        "pelipper": 0.25,  #  rain setter
        "ninetales-alola": 0.25,  # snow + aurora veil lead
        "arcanine-hisui": 0.3,  # intimidate
        "sneasler": 0.4,  # fake-out + unburden
        "typhlosion-hisui": 0.4,  # specs eruption
        "kilowattrel": 0.4,  # tailwind lead
        "farigiraf": 0.2,
        "porygon2": 0.2,
        "amoonguss": 0.2,
        "flamigo": 0.1,
        "annihilape": 0.3,  # anti incin lead
        "archaludon": 0.1,
        "gholdengo": 0.3,
        "dragonite": 0.0,
        "torkoal": -0.1,
        "ursaluna-bloodmoon": -0.2,
        "ursaluna": -0.4,
        "basculegion": -0.4,
    }

    WEATHER_ABILITIES = {"FIRE": "drought", "WATER": "drizzle", "ICE": "snowwarning"}
    TERRAIN_ABILITIES = {"GRASS": "grassysurge", "PSYCHIC": "psychicsurge"}
    ABILITY_TO_WEATHER = {v: k for k, v in WEATHER_ABILITIES.items()}
    ABILITY_TO_TERRAIN = {v: k for k, v in TERRAIN_ABILITIES.items()}
    BOOST_MULT = (
        0.25,
        2.0 / 7.0,
        2.0 / 6.0,
        0.4,
        0.5,
        2.0 / 3.0,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
    )
    WEATHER_TYPE = {
        Weather.SUNNYDAY: "FIRE",
        Weather.RAINDANCE: "WATER",
        Weather.SNOW: "ICE",
    }
    TERRAIN_TYPE = {
        Field.GRASSY_TERRAIN: "GRASS",
        Field.PSYCHIC_TERRAIN: "PSYCHIC",
    }

    def __init__(self, k: int = 4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self._defensive_cache = {}

    def _get_actions(self, mask: torch.Tensor, poke_no: int) -> NDArray[np.int64]:
        valid_indices = torch.where(mask[poke_no] > 0)[0]
        return valid_indices.cpu().numpy()

    def _is_valid_pair(self, a0: int, a1: int) -> bool:
        # Both switching to same slot
        if (1 <= a0 <= 6) and a0 == a1:
            return False
        # Both terastallizing
        if (27 <= a0 <= 46) and (27 <= a1 <= 46):
            return False
        # Both passing
        if a0 == 0 and a1 == 0:
            return False

        return True

    @staticmethod
    def _safe_stat(mon: Pokemon, stat: str, default: int = 100) -> int:
        if mon.stats is None:
            return default

        val = mon.stats[stat]
        return default if val is None else val

    @staticmethod
    def _safe_boost(mon: Pokemon, stat: str, default: int = 0) -> int:
        return mon.boosts.get(stat, default) if mon.boosts else default

    @staticmethod
    def _safe_hp_fraction(mon: Pokemon) -> float:
        if mon.current_hp is None or mon.max_hp is None or mon.max_hp <= 0:
            return 1.0
        return mon.current_hp / mon.max_hp

    def _calculate_damage(
        self,
        move: Move,
        attacker: Pokemon,
        defender: Pokemon,
        battle: DoubleBattle,
        hits_multiple: bool = False,
    ) -> float:
        if move.category == MoveCategory.STATUS:
            return 0.0

        type_mult = self._type_multiplier(attacker, defender, move)
        if type_mult == 0:
            return 0.0

        if move.id == "superfang":
            return 0.5 * self._safe_hp_fraction(defender)

        if move.id in self.FIXED_DAMAGE:
            return attacker.level / (
                defender.max_hp if (defender.max_hp and defender.max_hp > 0) else 100
            )

        power = move.base_power
        if move.id in {"eruption", "waterspout"}:
            power = 150 * self._safe_hp_fraction(attacker)
            is_slower = self._is_slower_than_opponents(battle, attacker)
            tr_active = self._trickroom_turns(battle) > 0
            if (is_slower and not tr_active) or (not is_slower and tr_active):
                power *= 0.4
        elif power <= 0:
            power = 60

        level = attacker.level
        move_type = self._effective_move_type(attacker, move)
        category = self._effective_move_category(attacker, move)

        if category == MoveCategory.PHYSICAL:
            a_stat = self._safe_stat(attacker, "atk")
            d_stat = self._safe_stat(defender, "def")
            a_boost = self._safe_boost(attacker, "atk")
            d_boost = self._safe_boost(defender, "def")
            if attacker.ability == "guts" and attacker.status:
                a_stat = int(a_stat * 1.5)
        else:
            a_stat = self._safe_stat(attacker, "spa")
            d_stat = self._safe_stat(defender, "spd")
            a_boost = self._safe_boost(attacker, "spa")
            d_boost = self._safe_boost(defender, "spd")

        a_stat = int(a_stat * self._get_boost_mult(a_boost))
        d_stat = int(d_stat * self._get_boost_mult(d_boost))

        if (
            PokemonType.ICE in defender.types
            and Weather.SNOW in battle.weather
            and category == MoveCategory.PHYSICAL
        ):
            d_stat *= 1.5

        damage = (((2 * level / 5) + 2) * power * a_stat / d_stat) / 50 + 2
        if hits_multiple:
            damage *= 0.75

        weather_type, _ = self._active_weather_type(battle)
        if weather_type == "FIRE":
            if move_type == PokemonType.FIRE:
                damage *= 1.5
            elif move_type == PokemonType.WATER:
                damage *= 0.5
        elif weather_type == "WATER":
            if move_type == PokemonType.WATER:
                damage *= 1.5
            elif move_type == PokemonType.FIRE:
                damage *= 0.5

        damage *= self._stab_multiplier(attacker, move, move_type)
        damage *= type_mult

        if attacker.status == "brn" and category == MoveCategory.PHYSICAL:
            if attacker.ability != "guts" and move.id != "facade":
                damage *= 0.5

        return damage / (defender.max_hp if (defender.max_hp and defender.max_hp > 0) else 100)

    def _get_accuracy(
        self, move: Move, attacker: Pokemon, defender: Pokemon, battle: DoubleBattle
    ) -> float:
        acc = move.accuracy
        if acc is True or acc == 0:
            return 1.0

        # Weather based perfect accuracy
        weather, _ = self._active_weather_type(battle)
        if move.id == "blizzard" and weather == "ICE":
            return 1.0
        if move.id in ["hurricane", "thunder"] and weather == "WATER":
            return 1.0

        # Accuracy/Evasion boosts
        acc_stage = self._safe_boost(attacker, "accuracy")
        eva_stage = self._safe_boost(defender, "evasion")

        acc_mult = (max(3.0, 3.0 + acc_stage)) / (max(3.0, 3.0 - acc_stage))
        eva_mult = (max(3.0, 3.0 + eva_stage)) / (max(3.0, 3.0 - eva_stage))

        return min(1.0, acc * acc_mult / eva_mult)

    def _offensive_rating(
        self,
        move: Move,
        attacker: Pokemon,
        recepient: Pokemon,
        battle: DoubleBattle,
        hits_multiple: bool = False,
    ) -> float:
        dmg_pct = self._calculate_damage(move, attacker, recepient, battle, hits_multiple)
        rating = dmg_pct

        # ko bonus
        current_hp_pct = (
            recepient.current_hp / recepient.max_hp
            if (recepient.max_hp and recepient.max_hp > 0)
            else 1.0
        )
        if dmg_pct >= current_hp_pct:
            rating += 0.4

        # outspeed bonus
        if move.priority > 0:
            rating += 0.2
        else:
            # Speed comparison
            a_is_op = attacker in battle.opponent_active_pokemon
            r_is_op = recepient in battle.opponent_active_pokemon

            a_spe = self._safe_stat(attacker, "spe") * self._get_boost_mult(
                self._safe_boost(attacker, "spe")
            )
            if attacker.ability == "unburden" and not attacker.item:
                a_spe *= 2.0
            if self._tailwind_turns(battle, a_is_op) > 0:
                a_spe *= 2.0

            r_spe = self._safe_stat(recepient, "spe") * self._get_boost_mult(
                self._safe_boost(recepient, "spe")
            )
            if recepient.ability == "unburden" and not recepient.item:
                r_spe *= 2.0
            if self._tailwind_turns(battle, r_is_op) > 0:
                r_spe *= 2.0

            if self._trickroom_turns(battle) > 0:
                if a_spe < r_spe:
                    rating += 0.1
            else:
                if a_spe > r_spe:
                    rating += 0.1

        return rating * self._get_accuracy(move, attacker, recepient, battle)

    def _get_proba(self, battle: DoubleBattle, mon: Pokemon) -> NDArray[np.float64] | None:
        ors = []
        for op in battle.opponent_active_pokemon:
            if op is None or op.fainted:
                continue
            for move in op.moves.values():
                if move.category == MoveCategory.STATUS:
                    continue
                ors.append(self._offensive_rating(move, op, mon, battle))

        if len(ors) == 0:
            return None

        ors_arr = np.array(ors, dtype=np.float64)
        total = np.sum(ors_arr)
        if total == 0:
            return np.ones(len(ors_arr), dtype=np.float64) / len(ors_arr)
        return ors_arr / total

    def _get_dmg(self, battle: DoubleBattle, mon: Pokemon) -> NDArray[np.float64]:
        dmgs = []
        for op in battle.opponent_active_pokemon:
            if op is None or op.fainted:
                continue
            for move in op.moves.values():
                if move.category == MoveCategory.STATUS:
                    continue
                dmgs.append(self._calculate_damage(move, op, mon, battle))

        return np.array(dmgs, dtype=np.float64)

    def _get_defensive_rating(self, battle: DoubleBattle, mon: Pokemon) -> float:
        if mon not in self._defensive_cache:
            proba = self._get_proba(battle, mon)
            if proba is None:
                self._defensive_cache[mon] = 0.0
            else:
                dmg = self._get_dmg(battle, mon)
                self._defensive_cache[mon] = float(np.dot(proba, dmg))
        return self._defensive_cache[mon]

    def _score_switch(self, battle: DoubleBattle, action: int, pos: int) -> float:
        active_mon = battle.active_pokemon[pos]
        if active_mon is None:
            return 0.0

        score = 0.0
        # SWITCHING BONUSES
        pref_atk = "atk" if active_mon.base_stats["atk"] >= active_mon.base_stats["spa"] else "spa"
        mult = self._get_boost_mult(self._safe_boost(active_mon, pref_atk))

        # If mult < 1.0 (drop/bad speed), 1.0 - mult > 0 -> positive score (incentivize switch)
        # If mult > 1.0 (boost/good speed), 1 / mult - 1 < 0 -> negative score (penalize switch)
        if mult < 1.0:
            score += 1.0 - mult
        else:
            obm = 1.0 / mult
            score += (obm - 1.0) * 0.4  # reduced penalty for switching out with positive stats

        # incentivize switching out if affected by negative effects
        if Effect.ENCORE in active_mon.effects and battle.available_moves[pos]:
            locked_move = battle.available_moves[pos][0]
            if locked_move.category == MoveCategory.STATUS:
                score += 0.6
            else:
                type_effectiveness = [
                    opp.damage_multiplier(locked_move)
                    for opp in battle.opponent_active_pokemon
                    if opp is not None
                ]
                score += max(0.0, 0.8 - sum(type_effectiveness) / len(type_effectiveness))

        if Effect.TAUNT in active_mon.effects:
            status_moves = [
                m for m in active_mon.moves.values() if m.category == MoveCategory.STATUS
            ]
            score += len(status_moves) / 5.0

        if Effect.CONFUSION in active_mon.effects:
            score += 0.15

        defensive_rating_active = self._get_defensive_rating(battle, active_mon)
        switch_mon = list(battle.team.values())[action - 1]
        defensive_rating_switch = self._get_defensive_rating(battle, switch_mon)

        score += defensive_rating_active - defensive_rating_switch

        weather_type = self.ABILITY_TO_WEATHER.get(switch_mon.ability)
        if weather_type and self._active_weather_type(battle)[0] != weather_type:
            score += 0.4

        terrain_type = self.ABILITY_TO_TERRAIN.get(switch_mon.ability)
        if terrain_type and self._active_terrain_type(battle)[0] != terrain_type:
            score += 0.3

        if switch_mon.ability == "intimidate":
            for op in battle.opponent_active_pokemon:
                if op and not op.fainted:
                    if op.ability in ["defiant", "competitive"]:
                        score -= 0.3
                    elif self._safe_stat(op, "atk") > self._safe_stat(op, "spa"):
                        score += 0.15

        return score

    def _score_tera(
        self,
        battle: DoubleBattle,
        active_mon: Pokemon,
        dummy_tera: Pokemon,
        move: Move | None = None,
        target: Pokemon | None = None,
    ) -> float:
        proba = self._get_proba(battle, active_mon)
        defensive_gain = 0.0
        if proba is not None:
            defensive_rating_active = self._get_defensive_rating(battle, active_mon)
            dmg_tera = self._get_dmg(battle, dummy_tera)
            defensive_rating_tera = float(np.dot(proba, dmg_tera))
            defensive_gain = defensive_rating_active - defensive_rating_tera

        offensive_gain = 0.0
        if move is not None and move.category != MoveCategory.STATUS and target is not None:
            pre = self._calculate_damage(move, active_mon, target, battle)
            post = self._calculate_damage(move, dummy_tera, target, battle)
            offensive_gain = 1.2 * (post - pre)
            hp_pct = (
                target.current_hp / target.max_hp if (target.max_hp and target.max_hp > 0) else 1.0
            )
            if pre < hp_pct and post >= hp_pct:
                offensive_gain += 0.3

        return (
            defensive_gain + offensive_gain - 0.1
        )  # base negative score to not tera for small upsides

    def _score_move(self, battle: DoubleBattle, action: int, mon: Pokemon, pos: int) -> float:
        # Action mapping from env.py:
        move_idx = (action - 7) % 20 // 5
        target_idx = (action - 7) % 5 - 2
        mvs = (
            battle.available_moves[pos]
            if len(battle.available_moves[pos]) == 1
            and battle.available_moves[pos][0].id in ["struggle", "recharge"]
            else list(mon.moves.values())
        )

        if move_idx >= len(mvs):
            return -10.0

        move = mvs[move_idx]
        if move.category == MoveCategory.STATUS:
            return self._score_status_move(battle, move, mon, target_idx, pos)

        # determine targets
        targets = []
        if target_idx == 1:
            targets = [battle.opponent_active_pokemon[0]]
        elif target_idx == 2:
            targets = [battle.opponent_active_pokemon[1]]
        elif target_idx == -1:
            targets = [battle.active_pokemon[0]]
        elif target_idx == -2:
            targets = [battle.active_pokemon[1]]
        elif target_idx == 0:
            if move.id in self.SPREAD:
                targets = [op for op in battle.opponent_active_pokemon if op and not op.fainted]
            else:
                # Pick first non-fainted opponent
                for op in battle.opponent_active_pokemon:
                    if op and not op.fainted:
                        targets = [op]
                        break

        if not targets:
            return -10.0

        if move.id == "pollenpuff" and len(targets) == 1:
            t = targets[0]
            if t is None or t.fainted:
                return -10.0
            if t in battle.active_pokemon:
                return self._healing_value(t, battle)

        total_rating = 0.0
        hits_multiple = len(targets) > 1
        for t in targets:
            if t is None or t.fainted:
                continue
            rating = self._offensive_rating(move, mon, t, battle, hits_multiple)

            if move.drain > 0 and mon.current_hp_fraction < 0.7:
                rating *= 1.25

            # Discourage stat drops on Defiant / Competitive
            if move.id in self.STAT_DROPPING:
                if t.ability == "defiant" or t.ability == "competitive":
                    rating -= 0.15

            if t in battle.active_pokemon:
                rating *= -2.0  # penalize damage to our own teammates

            if move.id in self.PIVOT:
                defensive_rating_active = self._get_defensive_rating(battle, mon)
                rating += max(0.0, 0.4 - defensive_rating_active)

            if move.id == "fakeout":
                if not mon.first_turn:
                    return -5.0
                if t is None or t.fainted or t in battle.active_pokemon:
                    return -10.0
                if self._is_immune_to_fake_out(t):
                    return -5.0
                if self._blocks_priority(mon, t, battle):
                    return -5.0
                rating += 0.5

            total_rating += rating

        if hits_multiple:
            total_rating *= 0.8

        if move.id in {"earthquake"}:
            partner = battle.active_pokemon[1 - pos]
            if partner and not partner.fainted:
                partner_dmg = self._calculate_damage(move, mon, partner, battle, hits_multiple=True)
                # Skip or reduce penalty if partner is immune or protecting
                if partner.damage_multiplier(move) > 0 and partner.protect_counter == 0:
                    total_rating -= 0.80 * partner_dmg

        return total_rating

    def _score_status_move(
        self, battle: DoubleBattle, move: Move, mon: Pokemon, target_idx: int, pos: int
    ) -> float:
        target = None
        if target_idx == 1:
            target = battle.opponent_active_pokemon[0]
        elif target_idx == 2:
            target = battle.opponent_active_pokemon[1]
        elif target_idx == -1:
            target = battle.active_pokemon[0]
        elif target_idx == -2:
            target = battle.active_pokemon[1]

        # Branch by move family for target validation
        if move.id in self.OPPONENT_TARGETED:
            if target is None or target.fainted or target in battle.active_pokemon:
                return -10.0
        elif move.id in self.ALLY_TARGETED:
            if target is None or target.fainted or target in battle.opponent_active_pokemon:
                return -10.0
        elif (
            move.id in self.FIELD_SELF_MOVES
            or move.id in self.PROTECT
            or move.id in self.SETUP
            or move.id in self.NO_TARGET_STATUS
        ):
            # target is irrelevant for these
            pass
        else:
            # Default fallback
            if target is None and move.target != "self":
                pass

        score = 0.0

        if move.id == "tailwind":
            if self._tailwind_turns(battle) > 0:
                return -5.0
            if self._trickroom_turns(battle) > 0:
                if self._trickroom_turns(battle) == 1:
                    score += 0.2
                else:
                    return -5.0
            if self._is_slower_than_opponents(battle, mon):
                score += 0.3

        elif move.id == "trickroom":
            tr_turns = self._trickroom_turns(battle)
            we_are_slower = self._is_slower_than_opponents(battle, mon)
            if tr_turns > 0:
                if not we_are_slower:  # We are faster, TR is bad for us
                    score += 0.4  # Encourage reversing
                else:
                    return -5.0  # TR is good for us, don't reverse
            else:
                if we_are_slower:
                    score += 0.6  # Encourage setting TR
                else:
                    return -5.0

        elif move.id in self.SETUP:
            boost_sum = 0
            if move.boosts:
                for boost in move.boosts:
                    boost_sum += self._safe_boost(mon, boost)

            if boost_sum <= -3 or boost_sum >= 2:
                return 0.05
            else:
                defensive_rating = self._get_defensive_rating(battle, mon)
                # scale setup score by defensive safety
                return (0.3 - (abs(boost_sum) / 10.0)) + (0.3 - defensive_rating)

        elif move.id == "taunt":
            if target is None:
                return -10.0
            if self._is_status_immune_target(target, mon, battle):
                return -5.0
            if Effect.TAUNT in target.effects:
                return -5.0
            if self._is_immune_to_prankster(target) and mon.ability == "prankster":
                return -5.0

            has_tr = any(m.id == "trickroom" for m in target.moves.values())
            has_setup = any(m.id in self.SETUP for m in target.moves.values())
            if has_tr:
                return 0.3
            elif has_setup:
                return 0.15

        elif move.id == "encore":
            if target is None:
                return -10.0
            if self._is_status_immune_target(target, mon, battle):
                return -5.0
            if Effect.ENCORE in target.effects:
                return -5.0
            if self._is_immune_to_prankster(target) and mon.ability == "prankster":
                return -5.0

            # Find the last move used by the target
            last_move = self._get_last_move(battle, target)
            if not last_move and target.protect_counter > 0:
                score += 0.6
                return score

            if last_move:
                if last_move.category == MoveCategory.STATUS:
                    score += 0.5
                else:
                    avg_dmg = 0.0
                    count = 0
                    for our_mon in battle.active_pokemon:
                        if our_mon and not our_mon.fainted:
                            avg_dmg += self._calculate_damage(last_move, target, our_mon, battle)
                            count += 1
                    if count > 0:
                        avg_dmg /= count

                    score += 2.0 * (0.25 - avg_dmg**2)
            else:
                return -1.0

        elif move.id == "haze":
            # see logic for switching out
            def get_mon_boost_value(m):
                pref_atk = "atk" if m.base_stats["atk"] >= m.base_stats["spa"] else "spa"
                mult = self._get_boost_mult(self._safe_boost(m, pref_atk))
                if mult > 1.0:
                    return (mult - 1.0) * 0.4
                else:
                    return mult - 1.0

            our_value = sum(get_mon_boost_value(p) for p in battle.active_pokemon if p)
            opp_value = sum(get_mon_boost_value(p) for p in battle.opponent_active_pokemon if p)

            score += 0.2 * (opp_value - our_value)

        elif move.id in self.PROTECT:
            if mon.protect_counter > 0:
                return -0.3
            score += 0.25

        elif move.id == "helpinghand":
            if target is None or target.fainted:
                return -10.0
            best_damage = 0.0
            for mv in target.moves.values():
                if mv.category == MoveCategory.STATUS:
                    continue
                for opp in battle.opponent_active_pokemon:
                    if opp and not opp.fainted:
                        best_damage = max(
                            best_damage, self._calculate_damage(mv, target, opp, battle)
                        )
            return 0.10 + 0.35 * best_damage

        elif move.id in {"followme", "ragepowder"}:
            pressure = sum(
                self._get_defensive_rating(battle, ally) for ally in battle.active_pokemon if ally
            )
            bonus = 0.0
            partner = battle.active_pokemon[1 - pos]
            if partner and not partner.fainted:
                partner_moves = {mv.id for mv in partner.moves.values()}
                if "trickroom" in partner_moves:
                    bonus += 0.35
                elif self.SETUP.intersection(partner_moves):
                    bonus += 0.25
            score = 0.18 + 0.18 * pressure + bonus
            if move.id == "ragepowder" and any(
                op and op.ability == "overcoat" for op in battle.opponent_active_pokemon
            ):
                score -= 0.10
            return score

        elif move.id == "spore":
            if target is None:
                return -10.0
            if self._is_status_immune_target(target, mon, battle):
                return -5.0
            if self._is_immune_to_spore(target, mon, battle):
                return -5.0
            if target.status is not None:
                return -5.0

            threat = 0.0
            for mv in target.moves.values():
                if mv.category == MoveCategory.STATUS:
                    continue
                for ally in battle.active_pokemon:
                    if ally and not ally.fainted:
                        threat = max(
                            threat,
                            self._calculate_damage(
                                mv, target, ally, battle, hits_multiple=(mv.id in self.SPREAD)
                            ),
                        )
            return 0.35 + 0.5 * threat

        elif move.id == "faketears":
            if target is None:
                return -10.0
            if self._is_status_immune_target(target, mon, battle):
                return -5.0

            special_pressure = 0.0
            for ally in battle.active_pokemon:
                if ally is None or ally.fainted or ally == mon:
                    continue
                if self._safe_stat(ally, "spa") <= self._safe_stat(ally, "atk"):
                    continue
                for mv in ally.moves.values():
                    if mv.category == MoveCategory.SPECIAL:
                        special_pressure = max(
                            special_pressure, self._calculate_damage(mv, ally, target, battle)
                        )

            self_pressure = 0.0
            if self._safe_stat(mon, "spa") > self._safe_stat(mon, "atk"):
                for mv in mon.moves.values():
                    if mv.category == MoveCategory.SPECIAL:
                        self_pressure = max(
                            self_pressure, self._calculate_damage(mv, mon, target, battle)
                        )

            return 0.12 + 0.22 * max(special_pressure, self_pressure)

        elif move.id == "coaching":
            if target is None or target.fainted:
                return -10.0
            if self._safe_stat(target, "atk") < self._safe_stat(target, "spa"):
                return -1.0
            atk_boost = self._safe_boost(target, "atk")
            def_boost = self._safe_boost(target, "def")
            if atk_boost >= 3 and def_boost >= 3:
                return -2.0
            partner_pressure = self._get_defensive_rating(battle, target)
            return 0.18 + 0.08 * max(0, 2 - atk_boost) + 0.12 * (0.8 - partner_pressure)

        elif move.id == "wideguard":
            pressure = 0.0
            has_spread = [False, False]
            for i, op in enumerate(battle.opponent_active_pokemon):
                if op and not op.fainted:
                    for mv in op.moves.values():
                        if mv.id in self.SPREAD and mv.category != MoveCategory.STATUS:
                            has_spread[i] = True
                            for ally in battle.active_pokemon:
                                if ally and not ally.fainted:
                                    pressure += self._calculate_damage(
                                        mv, op, ally, battle, hits_multiple=True
                                    )
            score = 0.3 * pressure
            if all(has_spread):
                score += 0.10

        elif move.id == "auroraveil":
            if self._active_weather_type(battle)[0] != "ICE":
                return -5.0
            if self._veil_turns(battle) > 0:
                return -5.0
            allies = [p for p in battle.active_pokemon if p and not p.fainted]
            pressure = sum(self._get_defensive_rating(battle, p) for p in allies)
            score = 0.4 + 0.1 * pressure

        elif move.id == "recover":
            hp_frac = self._safe_hp_fraction(mon)
            if hp_frac > 0.8:
                return -5.0
            score = self._healing_value(mon, battle)
            if self._get_defensive_rating(battle, mon) > 0.9 and hp_frac < 0.35:
                score -= 0.2
            return score

        elif move.id == "sunnyday":
            weather_type, turns = self._active_weather_type(battle)
            if weather_type == "FIRE" and turns > 1:
                return -5.0

            ally_fire = 0.0
            ally_sun_bonus = 0.0
            for ally in battle.active_pokemon:
                if ally is None or ally.fainted:
                    continue
                for mv in ally.moves.values():
                    if mv.category == MoveCategory.STATUS:
                        continue
                    if mv.type == PokemonType.FIRE:
                        ally_fire = 0.16

            opp_water = 0.0
            for op in battle.opponent_active_pokemon:
                if op is None or op.fainted:
                    continue
                for mv in op.moves.values():
                    if mv.category != MoveCategory.STATUS and mv.type == PokemonType.WATER:
                        opp_water = 0.16

            score = 0.15 + ally_fire + ally_sun_bonus + opp_water
            if weather_type == "WATER":
                score += 0.25
            return score

        elif move.id == "partingshot":
            if target is None or target.fainted or target in battle.active_pokemon:
                return -10.0
            if self._is_status_immune_target(target, mon, battle):
                return -5.0

            if target.ability in {"defiant", "competitive"}:
                return -5.0
            if target.ability in {"clearbody", "whitesmoke", "fullmetalbody"}:
                return -2.0
            if getattr(target, "item", "") == "clearamulet":
                return -2.0

            threat = 0.0
            count = 0
            for mv in target.moves.values():
                if mv.category == MoveCategory.STATUS:
                    continue
                for ally in battle.active_pokemon:
                    if ally and not ally.fainted:
                        threat += self._calculate_damage(
                            mv, target, ally, battle, hits_multiple=(mv.id in self.SPREAD)
                        )
                        count += 1
            if count > 0:
                threat /= count

            pivot = min(0.35, self._get_defensive_rating(battle, mon))
            score = 0.20 + 0.8 * threat + pivot
            score += 0.08 * max(0, self._safe_boost(target, "atk"))
            score += 0.08 * max(0, self._safe_boost(target, "spa"))
            return score

        return score

    def _is_slower_than_opponents(self, battle: DoubleBattle, mon: Pokemon) -> bool:
        mon_spe = self._safe_stat(mon, "spe") * self._get_boost_mult(self._safe_boost(mon, "spe"))
        partner = None
        for p in battle.active_pokemon:
            if p and p != mon:
                partner = p

        partner_spe = 0
        if partner:
            partner_spe = self._safe_stat(partner, "spe") * self._get_boost_mult(
                self._safe_boost(partner, "spe")
            )

        for op in battle.opponent_active_pokemon:
            if op and not op.fainted:
                op_spe = self._safe_stat(op, "spe") * self._get_boost_mult(
                    self._safe_boost(op, "spe")
                )
                if mon_spe < op_spe or (partner_spe > 0 and partner_spe < op_spe):
                    return True
        return False

    def _score_single_action(self, battle: DoubleBattle, action: int, pos: int) -> float:
        if 1 <= action <= 6:
            return self._score_switch(battle, action, pos)

        active_mon = battle.active_pokemon[pos]
        if active_mon is None:
            return 0.0

        if action <= 26:
            return self._score_move(battle, action, active_mon, pos)

        dummy_tera = copy.deepcopy(active_mon)
        dummy_tera._terastallized = True
        move, target = self._get_move_from_action(battle, action, pos)
        tera_score = self._score_tera(battle, active_mon, dummy_tera, move, target)
        move_score = self._score_move(battle, action, dummy_tera, pos)
        return tera_score + move_score

    def _get_move_from_action(
        self, battle: DoubleBattle, action: int, pos: int
    ) -> tuple[Move | None, Pokemon | None]:
        if not (7 <= action <= 46):
            return None, None

        mon = battle.active_pokemon[pos]
        if mon is None:
            return None, None

        move_idx = (action - 7) % 20 // 5
        target_idx = (action - 7) % 5 - 2

        mvs = (
            battle.available_moves[pos]
            if len(battle.available_moves[pos]) == 1
            and battle.available_moves[pos][0].id in ["struggle", "recharge"]
            else list(mon.moves.values())
        )

        if move_idx >= len(mvs):
            return None, None

        move = mvs[move_idx]
        target = None
        if target_idx == 1:
            target = battle.opponent_active_pokemon[0]
        elif target_idx == 2:
            target = battle.opponent_active_pokemon[1]
        elif target_idx == -1:
            target = battle.active_pokemon[0]
        elif target_idx == -2:
            target = battle.active_pokemon[1]

        return move, target

    def _score_synergy(self, battle: DoubleBattle, a0: int, a1: int) -> float:
        score = 0.0
        m0, t0 = self._get_move_from_action(battle, a0, 0)
        m1, t1 = self._get_move_from_action(battle, a1, 1)

        if not m0 or not m1:
            return 0.0

        opp_has_spread = any(
            any(m_id in self.SPREAD for m_id in op.moves.keys())
            for op in battle.opponent_active_pokemon
            if op
        )

        is_m0_redirection = m0.id in self.REDIRECTION
        is_m1_redirection = m1.id in self.REDIRECTION
        redirection_multiplier = 0.5 if opp_has_spread else 1.0

        if (m0.id == "fakeout" and m1.id in self.SETUP) or (
            m1.id == "fakeout" and m0.id in self.SETUP
        ):
            score += 0.60
        if (m0.id == "fakeout" and m1.id == "trickroom") or (
            m1.id == "fakeout" and m0.id == "trickroom"
        ):
            score += 0.80

        if (is_m0_redirection and m1.id in self.SETUP) or (
            is_m1_redirection and m0.id in self.SETUP
        ):
            score += 0.60 * redirection_multiplier
        if (is_m0_redirection and m1.id == "trickroom") or (
            is_m1_redirection and m0.id == "trickroom"
        ):
            score += 0.80 * redirection_multiplier

        if m0.id == "helpinghand" and m1.category != MoveCategory.STATUS and t1:
            dmg = self._calculate_damage(m1, battle.active_pokemon[1], t1, battle)  # type: ignore
            score += 0.3 * dmg
            hp_pct = t1.current_hp / t1.max_hp if (t1.max_hp and t1.max_hp > 0) else 1.0
            if dmg < hp_pct and (dmg * 1.5) >= hp_pct:
                score += 0.3
        elif m1.id == "helpinghand" and m0.category != MoveCategory.STATUS and t0:
            dmg = self._calculate_damage(m0, battle.active_pokemon[0], t0, battle)  # type: ignore
            score += 0.3 * dmg
            hp_pct = t0.current_hp / t0.max_hp if (t0.max_hp and t0.max_hp > 0) else 1.0
            if dmg < hp_pct and (dmg * 1.5) >= hp_pct:
                score += 0.3

        is_m0_speed = m0.id in self.SPEED_CONTROL
        is_m1_speed = m1.id in self.SPEED_CONTROL
        if (is_m0_speed and m1.category != MoveCategory.STATUS) or (
            is_m1_speed and m0.category != MoveCategory.STATUS
        ):
            score += 0.25

        if m0.id == "beatup" and t0 == battle.active_pokemon[1]:
            p1 = battle.active_pokemon[1]
            if p1 and (p1.ability == "stamina" or any(m == "ragefist" for m in p1.moves)):
                score += 0.60
        if m1.id == "beatup" and t1 == battle.active_pokemon[0]:
            p0 = battle.active_pokemon[0]
            if p0 and (p0.ability == "stamina" or any(m == "ragefist" for m in p0.moves)):
                score += 0.60

        if m0.id == "feint" and t0 and t0 == t1:
            score += 0.3
        if m1.id == "feint" and t1 and t1 == t0:
            score += 0.3

        if (
            m0.id == "faketears"
            and t0
            and t0 == t1
            and battle.active_pokemon[1]
            and self._safe_stat(battle.active_pokemon[1], "spa")
            > self._safe_stat(battle.active_pokemon[1], "atk")
        ):
            score += 0.2
        if (
            m1.id == "faketears"
            and t1
            and t1 == t0
            and battle.active_pokemon[0]
            and self._safe_stat(battle.active_pokemon[0], "spa")
            > self._safe_stat(battle.active_pokemon[0], "atk")
        ):
            score += 0.2

        opp_has_redirection = any(
            any(m in self.REDIRECTION for m in op.moves.keys())
            for op in battle.opponent_active_pokemon
            if op
        )
        if opp_has_redirection:
            if m0.id in self.SPREAD:
                score += 0.2
            if m1.id in self.SPREAD:
                score += 0.2

        if t0 and t0 == t1 and any(t0 == op for op in battle.opponent_active_pokemon if op):
            hp_pct = t0.current_hp / t0.max_hp if (t0.max_hp and t0.max_hp > 0) else 1.0
            dmg0 = self._calculate_damage(m0, battle.active_pokemon[0], t0, battle)  # type: ignore
            dmg1 = self._calculate_damage(m1, battle.active_pokemon[1], t1, battle)  # type: ignore
            if dmg0 < hp_pct and dmg1 < hp_pct and (dmg0 + dmg1) >= hp_pct:
                score += 0.35

        if m0.id in self.PROTECT and m1.id in self.PROTECT:
            opp_has_fake_out = any(
                "fakeout" in op.moves for op in battle.opponent_active_pokemon if op
            )
            opp_has_setup = any(
                self.SETUP.intersection(op.moves.keys())
                for op in battle.opponent_active_pokemon
                if op
            )

            if opp_has_fake_out and not opp_has_setup:
                score += 0.3
            else:
                score -= 0.4

        return score

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if not isinstance(battle, DoubleBattle):
            return self.choose_random_move(battle)

        self._defensive_cache.clear()

        mask = observation_builder.get_action_mask(battle)
        actions0 = self._get_actions(mask, 0)
        actions1 = self._get_actions(mask, 1)

        scores_0 = {a0: self._score_single_action(battle, a0, pos=0) for a0 in actions0}
        scores_1 = {a1: self._score_single_action(battle, a1, pos=1) for a1 in actions1}

        scores = []
        pairs = []
        for a0 in actions0:
            for a1 in actions1:
                if self._is_valid_pair(a0, a1):
                    score = scores_0[a0] + scores_1[a1] + self._score_synergy(battle, a0, a1)
                    scores.append(score)
                    pairs.append((a0, a1))

        if not scores:
            o0 = (
                Gen9VGCEnv._action_to_order_individual(actions0[0], battle, fake=False, pos=0)
                if actions0
                else PassBattleOrder()
            )
            o1 = (
                Gen9VGCEnv._action_to_order_individual(actions1[0], battle, fake=False, pos=1)
                if actions1
                else PassBattleOrder()
            )
            return DoubleBattleOrder(o0, o1)  # type: ignore

        scores_t = torch.tensor(scores, dtype=torch.float32)
        kc = min(len(scores), self.k)
        top_scores, top_indices = torch.topk(scores_t, kc)
        # print(f"\n {top_scores} \n")

        bias, temp = 0.5, 2.0
        logits = (top_scores - bias) / temp
        if torch.isnan(logits).any():
            # fallback if something goes wrong
            idx = 0
        else:
            idx = torch.distributions.Categorical(logits=logits).sample().item()

        chosen_pair = pairs[top_indices[idx].item()]  # type: ignore
        o0 = Gen9VGCEnv._action_to_order_individual(chosen_pair[0], battle, fake=False, pos=0)
        o1 = Gen9VGCEnv._action_to_order_individual(chosen_pair[1], battle, fake=False, pos=1)
        return DoubleBattleOrder(o0, o1)

    @staticmethod
    def _to_id_str(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    def _get_last_move(self, battle: DoubleBattle, pokemon: Pokemon) -> Move | None:
        # Check current turn's events first
        for event in reversed(battle.current_observation.events):
            if event[1] == "move":
                try:
                    event_mon = battle.get_pokemon(event[2])
                    if event_mon == pokemon:
                        move_name = event[3]
                        move_id = self._to_id_str(move_name)
                        return pokemon.moves.get(move_id)
                except Exception:
                    continue

        # Check observations from previous turns
        for turn in range(battle.turn, 0, -1):
            if turn not in battle.observations:
                continue
            obs = battle.observations[turn]
            for event in reversed(obs.events):
                if event[1] == "move":
                    try:
                        event_mon = battle.get_pokemon(event[2])
                        if event_mon == pokemon:
                            move_name = event[3]
                            move_id = self._to_id_str(move_name)
                            return pokemon.moves.get(move_id)
                    except Exception:
                        continue
        return None

    # Helper checks
    def _is_immune_to_fake_out(self, opp: Pokemon | None) -> bool:
        if opp is None:
            return False
        if PokemonType.GHOST in opp.types:
            return True
        if getattr(opp, "item", "") == "covertcloak":
            return True
        return False

    def _blocks_priority(
        self, attacker: Pokemon, target: Pokemon | None, battle: DoubleBattle
    ) -> bool:
        if target is None:
            return False
        if Field.PSYCHIC_TERRAIN in battle.fields and not battle.is_grounded(attacker):
            return False
        if Field.PSYCHIC_TERRAIN in battle.fields and battle.is_grounded(target):
            return True

        target_side = (
            battle.opponent_active_pokemon
            if target in battle.opponent_active_pokemon
            else battle.active_pokemon
        )
        for mon in target_side:
            if (
                mon
                and not mon.fainted
                and mon.ability in {"armortail", "queenlymajesty", "dazzling"}
            ):
                return True
        return False

    def _is_immune_to_prankster(self, opp: Pokemon | None) -> bool:
        if opp is None:
            return False
        if PokemonType.DARK in opp.types:
            return True
        return False

    def _is_status_immune_target(
        self, target: Pokemon | None, attacker: Pokemon, battle: DoubleBattle
    ) -> bool:
        if target is None:
            return False
        if target.ability == "goodasgold":
            return True
        if self._is_immune_to_prankster(target) and attacker.ability == "prankster":
            return True
        return False

    def _is_immune_to_spore(
        self, target: Pokemon | None, attacker: Pokemon, battle: DoubleBattle
    ) -> bool:
        if target is None:
            return False
        if PokemonType.GRASS in target.types:
            return True
        if target.ability in {"overcoat", "insomnia", "vitalspirit", "sweetveil"}:
            return True
        if getattr(target, "item", "") == "safetygoggles":
            return True
        if Field.ELECTRIC_TERRAIN in battle.fields and battle.is_grounded(target):
            return True
        return False

    def _can_hit_ghost_with_normal_fighting(self, attacker: Pokemon) -> bool:
        ab = getattr(attacker, "ability", "")
        if ab == "mindseye" or ab == "scrappy":
            return True
        return False

    def _active_weather_type(self, battle: DoubleBattle) -> tuple[str, int]:
        for w, start_turn in battle.weather.items():
            if w in self.WEATHER_TYPE:
                turns = max(0, 5 - (battle.turn - start_turn))
                return self.WEATHER_TYPE[w], turns
        return "NONE", 0

    def _active_terrain_type(self, battle: DoubleBattle) -> tuple[str, int]:
        for field, start_turn in battle.fields.items():
            if field in self.TERRAIN_TYPE:
                turns = max(0, 5 - (battle.turn - start_turn))
                return self.TERRAIN_TYPE[field], turns
        return "NONE", 0

    def _trickroom_turns(self, battle: DoubleBattle) -> int:
        start_turn = battle.fields.get(Field.TRICK_ROOM, -1)
        if start_turn >= 0:
            return max(0, 5 - (battle.turn - start_turn))
        return 0

    def _tailwind_turns(self, battle: DoubleBattle, opponent: bool = False) -> int:
        side_conds = battle.opponent_side_conditions if opponent else battle.side_conditions
        start_turn = side_conds.get(SideCondition.TAILWIND, -1)
        if start_turn >= 0:
            return max(0, 4 - (battle.turn - start_turn))
        return 0

    def _veil_turns(self, battle: DoubleBattle) -> int:
        start_turn = battle.side_conditions.get(SideCondition.AURORA_VEIL, -1)
        if start_turn >= 0:
            return max(0, 5 - (battle.turn - start_turn))
        return 0

    def _stab_multiplier(
        self,
        mon: Pokemon,
        move: Move,
        move_type: PokemonType | None = None,
    ) -> float:
        move_type = move.type if move_type is None else move_type
        is_original_stab = move_type in mon.original_types

        if not mon.is_terastallized:
            return mon.stab_multiplier if is_original_stab else 1.0

        if move_type == mon.tera_type:
            return mon.stab_multiplier

        if is_original_stab:
            return 2.0 if mon.ability == "adaptability" else 1.5

        return 1.0

    def _effective_move_type(self, mon: Pokemon, move: Move) -> PokemonType:
        if move.id == "terablast" and mon.is_terastallized and mon.tera_type is not None:
            return mon.tera_type
        return move.type

    def _effective_move_category(self, mon: Pokemon, move: Move) -> MoveCategory:
        if move.id == "terablast" and mon.is_terastallized:
            atk = self._safe_stat(mon, "atk") * self._get_boost_mult(self._safe_boost(mon, "atk"))
            spa = self._safe_stat(mon, "spa") * self._get_boost_mult(self._safe_boost(mon, "spa"))
            return MoveCategory.PHYSICAL if atk > spa else MoveCategory.SPECIAL
        return move.category

    def _type_multiplier(self, attacker: Pokemon, defender: Pokemon, move: Move) -> float:
        move_type = self._effective_move_type(attacker, move)
        type_mult = defender.damage_multiplier(move_type)
        if (
            type_mult == 0
            and move_type in {PokemonType.NORMAL, PokemonType.FIGHTING}
            and PokemonType.GHOST in defender.types
            and self._can_hit_ghost_with_normal_fighting(attacker)
        ):
            return 1.0
        return type_mult

    def _healing_value(self, target: Pokemon, battle: DoubleBattle) -> float:
        missing = max(0.0, 1.0 - self._safe_hp_fraction(target))
        if missing <= 0:
            return -0.2
        heal_pct = min(0.5, missing)
        pressure = self._get_defensive_rating(battle, target)
        return heal_pct * (0.6 + 0.3 * pressure)

    def _get_boost_mult(self, boost: int) -> float:
        return self.BOOST_MULT[int(boost) + 6]


if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path

    from poke_env import AccountConfiguration, LocalhostServerConfiguration

    sys.path.append(str(Path(__file__).parent))
    from teams import RandomTeamFromPool

    async def main():
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
            print(f"No team files found in {teams_dir}.")
            return

        team = RandomTeamFromPool(team_files)
        fmt = "gen9vgc2025regh"

        bot_player = FuzzyHeuristic(
            account_configuration=AccountConfiguration("FuzzyBot", None),
            battle_format=fmt,
            server_configuration=LocalhostServerConfiguration,
            team=team,
            accept_open_team_sheet=True,
            max_concurrent_battles=1,
        )

        print("FuzzyBot is listening for challenges on localhost...")
        await bot_player.accept_challenges(None, 1000)

    asyncio.run(main())
