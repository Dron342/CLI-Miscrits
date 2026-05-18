from __future__ import annotations

import random
import re
from typing import Any

from .battle_learning import damage_kind_for_action, damage_multiplier, learned_bonus, matchup_memory, reason_tags


BATTLE_OPCODES = {
    "ERROR": -1,
    "START": 0,
    "SWITCH": 1,
    "ABILITY": 2,
    "POTION": 3,
    "FLEE": 4,
    "HANDICAP": 5,
    "SLEEP": 6,
    "CONFUSE": 7,
    "READY": 8,
    "END": 9,
    "CAPTURE": 10,
    "CAPTURE_CONTINUE": 11,
    "TERMINATE": 12,
    "MISSED": 13,
    "INVALID": 14,
    "DISCONNECT": 15,
}

RANDOM_OPCODES = {
    "ERROR": -1,
    "READY": 0,
    "START": 1,
    "BAN": 2,
    "PICK": 3,
    "TERMINATE": 4,
    "UPDATE": 5,
    "SUCCESS": 6,
    "MATCH": 7,
    "PREPARE": 8,
}

ELEMENT_WEAKNESS = {
    "Fire": "Water",
    "Water": "Nature",
    "Nature": "Fire",
    "Wind": "Lightning",
    "Lightning": "Earth",
    "Earth": "Wind",
    "Physical": "",
}
BASE_ELEMENTS = tuple(key for key in ELEMENT_WEAKNESS if key != "Physical")

CONTROL_TYPES = {"Sleep", "Confuse", "Paralyze"}
TURN_DENIAL_TYPES = {"Sleep"}
SOFT_CONTROL_TYPES = {"Confuse"}
SWITCH_LOCK_TYPES = {"Paralyze"}
DOT_TYPES = {"Poison", "Dot", "Disease", "Bleed", "SwitchCurse"}
DEFENSIVE_TYPES = {"Negate", "Block", "Ethereal", "Barbed", "AI", "CI", "SI", "PI"}
STATUS_TYPES = CONTROL_TYPES | DOT_TYPES | DEFENSIVE_TYPES | {"Buff", "Bot", "Hot"}
NON_STACKING_STATUS_TYPES = CONTROL_TYPES | DOT_TYPES | DEFENSIVE_TYPES | {"Antiheal", "Hot", "TimeBomb", "SwitchCurse"}
REFRESHABLE_STATUS_TYPES = DOT_TYPES
ATTACK_CONSUMED_EFFECTS = {"block", "barbed"}
IMMUNITY_BY_STATUS = {"Antiheal": "AI", "Confuse": "CI", "Sleep": "SI", "Paralyze": "PI"}
STAT_KEYS = {"ea", "pa", "ed", "pd", "spd", "acc"}
SPECIAL_COOLDOWNS = {"Sleep": (5, 4), "Confuse": (5, 4)}
SWITCH_HP_THRESHOLD = 0.18
PROACTIVE_SWITCH_GAIN = 45.0
SWITCH_ACTION_COST = 86.0
SWITCH_RECENT_TURN_LOCKOUT = 2
SWITCH_LETHAL_PENALTY = 180.0
SWITCH_SAFE_HP_BONUS = 28.0
LETHAL_BONUS = 900.0
NEAR_LETHAL_BONUS = 350.0
LAST_FOE_FINISH_BONUS = 90.0
LAST_ALLY_LONG_BUFF_PENALTY = 120.0
MAX_UTILITY_STREAK = 2
ATTACK_FOLLOWUP_RATIO = 0.72
ABILITY_EXPLORATION_RATE = 0.08
ABILITY_EXPLORATION_GAP = 18.0
LOOKAHEAD_OPPONENT_WEIGHT = 0.62
LOOKAHEAD_FINISH_BONUS = 34.0
LOOKAHEAD_DEATH_PENALTY = 72.0
LEVEL_KEYS = [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 30]


class BattleState:
    def __init__(self, user_id: str = "") -> None:
        self.user_id = user_id
        self.battle_type = ""
        self.pvp = False
        self.player: dict[str, Any] = {}
        self.foe: dict[str, Any] = {}
        self.winner = ""
        self.turns = 0
        self.next_turn = ""
        self.pending = False
        self._last_turn_seen = 0
        self._player_switch_turns: list[int] = []
        self._utility_streak_by_pair: dict[str, int] = {}

    def apply(self, opcode: int, data: Any) -> None:
        if not isinstance(data, dict):
            return
        new_turns = int(data.get("turns", self.turns) or self.turns)
        if new_turns > self._last_turn_seen:
            turn_delta = new_turns - self._last_turn_seen
            self._tick_recharges(turn_delta)
            self._tick_active_effects(str(data.get("next_turn", self.next_turn) or self.next_turn), turn_delta)
            self._last_turn_seen = new_turns
        if opcode == BATTLE_OPCODES["START"]:
            self._start(data)
        elif opcode == BATTLE_OPCODES["SWITCH"]:
            self._switch(data)
        elif opcode in (BATTLE_OPCODES["ABILITY"], BATTLE_OPCODES["CONFUSE"], BATTLE_OPCODES["POTION"]):
            self._mark_used_ability(data)
            self._apply_actions(data)
        elif opcode == BATTLE_OPCODES["END"]:
            self.winner = str(data.get("winner", self.winner) or self.winner)
        elif opcode == BATTLE_OPCODES["TERMINATE"]:
            loser = str(data.get("loser", ""))
            if loser:
                self.winner = self.foe_id if loser == self.player_id else self.player_id
        action_winner = winner_from_actions(data)
        if action_winner:
            self.winner = action_winner
        self.next_turn = str(data.get("next_turn", self.next_turn) or self.next_turn)
        self.pending = bool(data.get("pending", self.pending))
        self.turns = int(data.get("turns", self.turns) or self.turns)

    def should_act(self, data: Any) -> bool:
        return isinstance(data, dict) and str(data.get("next_turn", "")) == self.player_id and not data.get("pending", False)

    def damage_samples(self, opcode: int, data: Any) -> list[dict[str, Any]]:
        if opcode not in (BATTLE_OPCODES["ABILITY"], BATTLE_OPCODES["CONFUSE"], BATTLE_OPCODES["POTION"]):
            return []
        if not isinstance(data, dict):
            return []
        actor = self._actor(data)
        if not actor:
            return []
        actor_side = "player" if actor is self.player else "foe" if actor is self.foe else ""
        attacker = actor.get("miscrits", [{}])[0]
        if not isinstance(attacker, dict):
            return []
        ability_id = int(data.get("id", 0) or 0)
        ability = find_ability(attacker, ability_id)
        if not ability:
            return []
        actions = data.get("actions", [])
        if not isinstance(actions, list):
            return []
        samples = []
        for action in actions:
            if not isinstance(action, dict) or action.get("damage") is None:
                continue
            target = self._player_by_id(str(action.get("target", "")))
            defender = target.get("miscrits", [{}])[0] if isinstance(target, dict) else {}
            if not isinstance(defender, dict) or not defender:
                continue
            details = estimate_ability_damage_details(effective_ability(ability, attacker), attacker, defender, action)
            actual = max(0.0, float(action.get("damage", 0) or 0))
            if actual <= 0.0 and not bool(action.get("crit", False)):
                continue
            action_type = str(action.get("type", "") or "")
            damage_kind = damage_kind_for_action(action_type)
            details["features"]["action_type"] = action_type or "Unknown"
            details["features"]["damage_kind"] = damage_kind
            samples.append(
                {
                    "turns": self.turns,
                    "opcode": opcode,
                    "side": actor_side,
                    "user_id": str(data.get("user_id", "")),
                    "ability": {
                        "id": ability_id,
                        "name": str(ability.get("name", "")),
                        "type": str(ability.get("type", "")),
                        "element": str(ability.get("element", "")),
                    },
                    "attacker": compact_battle_miscrit(attacker, True),
                    "defender": compact_battle_miscrit(defender, True),
                    "action": summarize_damage_action(action),
                    "action_type": action_type,
                    "damage_kind": damage_kind,
                    "base_damage": round(float(details["base_damage"]), 4),
                    "predicted_damage": round(float(details["damage"]), 4),
                    "actual_damage": round(actual, 4),
                    "ratio": round(actual / max(1.0, float(details["base_damage"])), 5),
                    "error": round(actual - float(details["damage"]), 4),
                    "features": details["features"],
                }
            )
        return samples

    def decide(self) -> dict[str, Any]:
        active = self.active_player_miscrit
        foe = self.active_foe_miscrit
        plan = self.battle_plan()
        if not active:
            return {"type": "none", "reason": "no_active_miscrit"}
        if is_dead(active):
            switch_choice = self._best_switch()
            if switch_choice["id"]:
                return {"type": "switch", "id": switch_choice["id"], "reason": "active_dead", "debug": {**switch_choice, "plan": plan}}
        if not foe:
            ability_id = self._best_ability_id(active, plan)
            if ability_id:
                return {"type": "ability", "id": ability_id, "reason": "best_available_attack"}

        ability_choice = self._best_ability(active, foe, plan)
        switch_choice = self._best_switch(foe)
        if should_switch(active, foe, ability_choice, switch_choice, plan):
            return {"type": "switch", "id": switch_choice["id"], "reason": switch_choice.get("reason", "better_matchup"), "debug": {**switch_choice, "plan": plan}}
        if ability_choice["id"]:
            self._record_ability_choice(active, foe, ability_choice)
            return {
                "type": "ability",
                "id": ability_choice["id"],
                "reason": ability_choice.get("reason", "best_scored_ability"),
                "reason_tags": ability_choice.get("reason_tags", reason_tags(ability_choice.get("reason", ""))),
                "action_type": ability_choice.get("type", "ability"),
                "chosen_probability": ability_choice.get("chosen_probability", 1.0),
                "candidate_rank": ability_choice.get("candidate_rank", 0),
                "debug": ability_choice,
            }
        if switch_choice["id"] and not bool(switch_choice.get("lethal_incoming", False)):
            return {"type": "switch", "id": switch_choice["id"], "reason": "no_usable_ability", "debug": {**switch_choice, "plan": plan}}
        return {"type": "none", "reason": "no_action_available"}

    def battle_plan(self) -> dict[str, Any]:
        player_state = team_condition(self.player.get("miscrits", []))
        foe_state = team_condition(self.foe.get("miscrits", []))
        roster_advantage = int(player_state["alive"]) - int(foe_state["alive"])
        hp_advantage = float(player_state["hp_ratio"]) - float(foe_state["hp_ratio"])
        last_ally = int(player_state["alive"]) <= 1
        last_foe = int(foe_state["alive"]) <= 1
        ahead = not last_foe and not last_ally and (roster_advantage >= 1 or hp_advantage >= 0.22)
        behind = not last_foe and (last_ally or roster_advantage <= -1 or hp_advantage <= -0.22)
        mode = "finish" if last_foe else "last_stand" if last_ally else "ahead" if ahead else "behind" if behind else "even"
        last_switch_turn = max(self._player_switch_turns) if self._player_switch_turns else -999
        turns_since_switch = self.turns - last_switch_turn if last_switch_turn >= 0 else 999
        return {
            "mode": mode,
            "ahead": ahead,
            "behind": behind,
            "last_ally": last_ally,
            "last_foe": last_foe,
            "player_alive": player_state["alive"],
            "foe_alive": foe_state["alive"],
            "player_hp_ratio": round(float(player_state["hp_ratio"]), 4),
            "foe_hp_ratio": round(float(foe_state["hp_ratio"]), 4),
            "roster_advantage": roster_advantage,
            "hp_advantage": round(hp_advantage, 4),
            "player_switch_count": len(self._player_switch_turns),
            "last_player_switch_turn": last_switch_turn,
            "turns_since_player_switch": turns_since_switch,
        }

    def snapshot(self) -> dict[str, Any]:
        active_side = ""
        if self.next_turn == self.player_id:
            active_side = "player"
        elif self.next_turn == self.foe_id:
            active_side = "foe"
        return {
            "battle_type": self.battle_type,
            "pvp": self.pvp,
            "turns": self.turns,
            "winner": self.winner,
            "next_turn": self.next_turn,
            "pending": self.pending,
            "active_side": active_side,
            "player": compact_player(self.player),
            "foe": compact_player(self.foe),
        }

    @property
    def player_id(self) -> str:
        return str(self.player.get("user_id", self.user_id) or self.user_id)

    @property
    def foe_id(self) -> str:
        return str(self.foe.get("user_id", ""))

    @property
    def active_player_miscrit(self) -> dict[str, Any]:
        team = self.player.get("miscrits", [])
        return team[0] if isinstance(team, list) and team and isinstance(team[0], dict) else {}

    @property
    def active_foe_miscrit(self) -> dict[str, Any]:
        team = self.foe.get("miscrits", [])
        return team[0] if isinstance(team, list) and team and isinstance(team[0], dict) else {}

    def _start(self, data: dict[str, Any]) -> None:
        self.battle_type = str(data.get("type", ""))
        self.pvp = bool(data.get("pvp", False))
        p1 = data.get("player1", {}) if isinstance(data.get("player1"), dict) else {}
        p2 = data.get("player2", {}) if isinstance(data.get("player2"), dict) else {}
        if not self.user_id:
            self.user_id = str(p1.get("user_id", ""))
        if str(p1.get("user_id", "")) == self.user_id:
            self.player, self.foe = p1, p2
        else:
            self.player, self.foe = p2, p1

    def hydrate_metadata(self, metadata: dict[int, dict[str, Any]]) -> None:
        for side in (self.player, self.foe):
            team = side.get("miscrits", [])
            if not isinstance(team, list):
                continue
            for item in team:
                if isinstance(item, dict):
                    hydrate_miscrit(item, metadata, self.pvp, self.battle_type)

    def _switch(self, data: dict[str, Any]) -> None:
        target_player = self._actor(data)
        if not target_player:
            return
        switch_to = int(data.get("id", 0) or 0)
        team = target_player.get("miscrits", [])
        if not switch_to or not isinstance(team, list):
            return
        for index, item in enumerate(team):
            if isinstance(item, dict) and int(item.get("id", 0) or 0) == switch_to:
                leaving = team[0] if team and isinstance(team[0], dict) else {}
                if leaving and leaving is not item:
                    clear_switch_transient_effects(leaving)
                team.insert(0, team.pop(index))
                if target_player is self.player:
                    self._player_switch_turns.append(self.turns)
                    self._player_switch_turns = self._player_switch_turns[-8:]
                break

    def _apply_actions(self, data: dict[str, Any]) -> None:
        actions = data.get("actions", [])
        if not isinstance(actions, list):
            return
        for action in actions:
            if not isinstance(action, dict):
                continue
            target_id = str(action.get("target", ""))
            target = self._player_by_id(target_id)
            if not target:
                continue
            dead_id = int(action.get("dead", 0) or 0)
            active = target.get("miscrits", [{}])[0]
            if isinstance(active, dict):
                remember_action_effect(active, action)
                if action_causes_wake(action):
                    remove_effect(active, "sleep")
                if action.get("chp") is not None:
                    active["chp"] = clamp_int(int(action.get("chp") or 0), 0, max_hp(active))
                elif action.get("hp") is not None and str(action.get("type", "")).casefold() not in {"attack", "heal", "hot", "dot", "poison", "bleed", "disease"}:
                    active["chp"] = clamp_int(int(action.get("hp") or 0), 0, max_hp(active))
                elif action.get("damage") is not None:
                    active["chp"] = clamp_int(current_hp(active) - int(action.get("damage") or 0), 0, max_hp(active))
                elif action.get("ap") is not None and str(action.get("type", "")).casefold() in {"heal", "hot"}:
                    active["chp"] = clamp_int(current_hp(active) + int(action.get("ap") or 0), 0, max_hp(active))
            if dead_id:
                for item in target.get("miscrits", []):
                    if isinstance(item, dict) and int(item.get("id", 0) or 0) == dead_id:
                        item["chp"] = 0
                switch_to = int(action.get("id", 0) or 0)
                if switch_to:
                    switch_active_miscrit(target, switch_to)

    def _mark_used_ability(self, data: dict[str, Any]) -> None:
        actor = self._actor(data)
        if not actor:
            return
        ability_id = int(data.get("id", 0) or 0)
        if not ability_id:
            return
        active = actor.get("miscrits", [{}])[0]
        if not isinstance(active, dict):
            return
        ability = find_ability(active, ability_id)
        if not ability:
            return
        miss = False
        actions = data.get("actions", [])
        if isinstance(actions, list) and actions and isinstance(actions[0], dict):
            miss = str(actions[0].get("type", "")).casefold() == "miss"
        recharge = next_recharge(ability, self.pvp, miss)
        if recharge:
            ability["recharge"] = recharge
            ability["cooldown_remaining"] = recharge
        effective = effective_ability(ability, active)
        if not miss and removes_elemental_weakness(effective) and ability_targets_self(ability):
            add_elemental_weakness_removed(active, int(ability.get("turns", ability.get("duration", 4)) or 4))
        if not miss and cleanses_self(effective) and ability_targets_self(ability):
            clear_dangerous_dots(active)

    def _tick_recharges(self, turns: int) -> None:
        for player in (self.player, self.foe):
            team = player.get("miscrits", [])
            if not isinstance(team, list):
                continue
            for miscrit in team:
                if not isinstance(miscrit, dict):
                    continue
                for ability in abilities_for(miscrit):
                    recharge = int(ability.get("recharge", ability.get("cooldown_remaining", ability.get("cd", 0))) or 0)
                    if recharge > 0:
                        value = max(0, recharge - turns)
                        ability["recharge"] = value
                        ability["cooldown_remaining"] = value

    def _tick_active_effects(self, user_id: str, turns: int) -> None:
        player = self._player_by_id(user_id)
        team = player.get("miscrits", []) if isinstance(player, dict) else []
        if not isinstance(team, list) or not team or not isinstance(team[0], dict):
            return
        tick_effects(team[0], turns)

    def _actor(self, data: dict[str, Any]) -> dict[str, Any]:
        return self._player_by_id(str(data.get("user_id", "")))

    def _player_by_id(self, user_id: str) -> dict[str, Any]:
        if user_id == self.player_id:
            return self.player
        if user_id == self.foe_id:
            return self.foe
        return {}

    def _best_switch_id(self) -> int:
        return int(self._best_switch().get("id", 0) or 0)

    def _best_switch(self, foe: dict[str, Any] | None = None) -> dict[str, Any]:
        foe = foe or self.active_foe_miscrit
        active = self.active_player_miscrit
        candidates = [item for item in self.player.get("miscrits", [])[1:] if isinstance(item, dict) and int(item.get("id", 0) or 0) != 0 and not is_dead(item)]
        if not candidates:
            return {"id": 0, "reason": "no_alive_bench"}
        ranked = []
        for item in candidates:
            ranked.append(switch_evaluation(item, foe))
        ranked.sort(key=lambda item: float(item["score"]), reverse=True)
        active_eval = switch_evaluation(active, foe) if active and foe else {"score": 0.0}
        return {
            "id": int(ranked[0]["id"] or 0),
            "reason": ranked[0].get("reason", "better_matchup"),
            "type": "switch",
            "candidate_rank": 1,
            "chosen_probability": 1.0,
            "score": ranked[0]["score"],
            "active_score": round(float(active_eval.get("score", 0.0) or 0.0), 3),
            "gain": round(float(ranked[0]["score"]) - float(active_eval.get("score", 0.0) or 0.0), 3),
            "incoming": ranked[0].get("incoming", 0.0),
            "incoming_ratio": ranked[0].get("incoming_ratio", 0.0),
            "after_incoming_ratio": ranked[0].get("after_incoming_ratio", 0.0),
            "survives": ranked[0].get("survives", True),
            "lethal_incoming": ranked[0].get("lethal_incoming", False),
            "elemental_score": ranked[0].get("elemental_score", 0.0),
            "incoming_element_multiplier": ranked[0].get("incoming_element_multiplier", 1.0),
            "outgoing_element_multiplier": ranked[0].get("outgoing_element_multiplier", 1.0),
            "best_damage": ranked[0].get("best_damage", 0.0),
            "candidates": ranked[:5],
        }

    def _best_ability_id(self, miscrit: dict[str, Any], plan: dict[str, Any] | None = None) -> int:
        return int(self._best_ability(miscrit, self.active_foe_miscrit, plan).get("id", 0) or 0)

    def _best_ability(self, miscrit: dict[str, Any], foe: dict[str, Any], plan: dict[str, Any] | None = None) -> dict[str, Any]:
        candidates = [item for item in abilities_for(miscrit) if isinstance(item, dict) and is_usable_ability(item)]
        if not candidates:
            return {"id": 0, "reason": "no_usable_ability"}
        ranked = []
        for ability in candidates:
            score_data = score_ability(ability, miscrit, foe, self.pvp, plan)
            ranked.append(score_data)
        ranked.sort(key=lambda item: float(item["score"]), reverse=True)
        for index, item in enumerate(ranked, start=1):
            item["candidate_rank"] = index
        finishers = [item for item in ranked if bool(item.get("lethal", False)) or bool(item.get("near_lethal", False))]
        if finishers:
            finishers.sort(key=lambda item: (bool(item.get("lethal", False)), float(item.get("damage", 0.0) or 0.0), float(item.get("score", 0.0) or 0.0)), reverse=True)
            best = finishers[0]
            best["reason"] = "lethal_finish" if bool(best.get("lethal", False)) else "near_lethal_pressure"
            best["reason_tags"] = [best["reason"]]
            best["chosen_probability"] = 1.0
            return {**best, "candidates": ranked[:6]}
        best = ranked[0]
        best_attack = next((item for item in ranked if str(item.get("type", "") or "") == "Attack"), None)
        if best_attack and int(best_attack.get("id", 0) or 0) != int(best.get("id", 0) or 0):
            utility_streak = self._utility_streak(miscrit, foe)
            followup_ratio = 0.58 if bool((plan or {}).get("last_ally", False)) else ATTACK_FOLLOWUP_RATIO
            best_score = float(best.get("score", 0.0) or 0.0)
            attack_score = float(best_attack.get("score", 0.0) or 0.0)
            attack_is_close = attack_score >= (best_score * followup_ratio if best_score > 0 else best_score)
            if utility_streak >= MAX_UTILITY_STREAK or attack_is_close:
                forced = dict(best_attack)
                forced["reason"] = "forced_attack_after_utility" if utility_streak >= MAX_UTILITY_STREAK else "attack_followup"
                forced["reason_tags"] = [forced["reason"]]
                forced["chosen_probability"] = 1.0
                forced["utility_streak_before"] = utility_streak
                forced["attack_followup_ratio"] = followup_ratio
                return {**forced, "candidates": ranked[:6]}
        explored = exploratory_ability_choice(ranked, plan or {})
        if explored:
            previous_tags = explored.get("reason_tags", reason_tags(explored.get("reason", "")))
            explored["reason"] = "explore_" + str(explored.get("reason", "alternate_ability"))
            explored["reason_tags"] = ["explore", *previous_tags]
            explored["exploration"] = True
            return {**explored, "candidates": ranked[:6]}
        best["chosen_probability"] = best.get("chosen_probability", deterministic_ability_probability(ranked, plan or {}))
        return {**best, "reason": best.get("reason", "best_scored_ability"), "candidates": ranked[:6]}

    def _utility_streak(self, active: dict[str, Any], foe: dict[str, Any]) -> int:
        return int(self._utility_streak_by_pair.get(self._pair_key(active, foe), 0) or 0)

    def _record_ability_choice(self, active: dict[str, Any], foe: dict[str, Any], choice: dict[str, Any]) -> None:
        key = self._pair_key(active, foe)
        previous = self._utility_streak_by_pair.get(key, 0)
        is_attack = str(choice.get("type", "") or "") == "Attack"
        self._utility_streak_by_pair[key] = 0 if is_attack else int(previous or 0) + 1

    @staticmethod
    def _pair_key(active: dict[str, Any], foe: dict[str, Any]) -> str:
        return f"{int(active.get('id', 0) or 0)}:{int(foe.get('id', 0) or 0)}"


def exploratory_ability_choice(ranked: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    rate, pool, weights = ability_exploration_policy(ranked, plan)
    if not pool or random.random() >= rate:
        return {}
    chosen = dict(random.choices(pool, weights=weights, k=1)[0])
    total_weight = sum(weights)
    chosen_weight = weights[pool.index(next(item for item in pool if int(item.get("id", 0) or 0) == int(chosen.get("id", 0) or 0)))]
    chosen["chosen_probability"] = rate * chosen_weight / max(0.000001, total_weight)
    return chosen


def ability_exploration_policy(ranked: list[dict[str, Any]], plan: dict[str, Any]) -> tuple[float, list[dict[str, Any]], list[float]]:
    if len(ranked) < 2:
        return 0.0, [], []
    best = ranked[0]
    best_score = float(best.get("score", 0.0) or 0.0)
    if bool(best.get("lethal", False)) or bool(best.get("near_lethal", False)):
        return 0.0, [], []
    rate = ABILITY_EXPLORATION_RATE
    if bool(plan.get("ahead", False)):
        rate *= 0.55
    if bool(plan.get("behind", False)):
        rate *= 1.7
    if bool(plan.get("last_ally", False)) or bool(plan.get("last_foe", False)):
        rate *= 0.6
    rate = min(0.18, max(0.0, rate))
    pool = []
    for item in ranked[1:6]:
        score = float(item.get("score", 0.0) or 0.0)
        if bool(item.get("immune_blocked", False)) or bool(item.get("redundant", False)):
            continue
        if best_score - score > ABILITY_EXPLORATION_GAP:
            continue
        if score < best_score * 0.72 and best_score > 0:
            continue
        pool.append(item)
    if not pool:
        return 0.0, [], []
    floor = min(float(item.get("score", 0.0) or 0.0) for item in pool)
    weights = [max(0.2, float(item.get("score", 0.0) or 0.0) - floor + 1.0) for item in pool]
    return rate, pool, weights


def deterministic_ability_probability(ranked: list[dict[str, Any]], plan: dict[str, Any]) -> float:
    rate, pool, _ = ability_exploration_policy(ranked, plan)
    return 1.0 - rate if pool else 1.0


def winner_from_actions(data: dict[str, Any]) -> str:
    actions = data.get("actions", []) if isinstance(data, dict) else []
    if not isinstance(actions, list):
        return ""
    for action in actions:
        if not isinstance(action, dict):
            continue
        winner = str(action.get("winner", "") or "")
        if winner:
            return winner
    return ""


def should_switch(active: dict[str, Any], foe: dict[str, Any], ability_choice: dict[str, Any], switch_choice: dict[str, Any], plan: dict[str, Any] | None = None) -> bool:
    switch_id = int(switch_choice.get("id", 0) or 0)
    if not switch_id:
        return False
    if not foe:
        return False
    best_ability_score = float(ability_choice.get("score", 0.0) or 0.0)
    if bool(ability_choice.get("lethal", False)):
        return False
    if bool(ability_choice.get("near_lethal", False)):
        return False
    if bool(switch_choice.get("lethal_incoming", False)):
        return False
    plan = plan or {}
    if bool(plan.get("last_ally", False)):
        return False
    if bool(plan.get("last_foe", False)) and float(ability_choice.get("damage", 0.0) or 0.0) >= current_hp(foe) * 0.68:
        return False
    if int(plan.get("turns_since_player_switch", 999) or 999) <= SWITCH_RECENT_TURN_LOCKOUT:
        return False

    active_hp = hp_ratio(active)
    incoming = estimate_incoming_damage(foe, active)
    incoming_ratio = incoming / max(1.0, float(max_hp(active)))
    gain = float(switch_choice.get("gain", 0.0) or 0.0)
    adjusted_gain = gain - SWITCH_ACTION_COST - float(plan.get("player_switch_count", 0) or 0) * 10.0
    switch_incoming_ratio = float(switch_choice.get("incoming_ratio", 0.0) or 0.0)
    switch_after_ratio = float(switch_choice.get("after_incoming_ratio", 0.0) or 0.0)
    switch_best_damage = float(switch_choice.get("best_damage", 0.0) or 0.0)
    ability_damage = float(ability_choice.get("damage", 0.0) or 0.0)
    foe_hp = max(1.0, float(current_hp(foe)))
    meaningful_attack = bool(ability_choice.get("id")) and (ability_damage >= foe_hp * 0.42 or best_ability_score >= 48.0)
    active_endangered = incoming_ratio >= active_hp * 0.75
    switch_reduces_damage = switch_incoming_ratio <= max(0.22, incoming_ratio * 0.55)
    switch_softens_damage = switch_incoming_ratio <= max(0.30, incoming_ratio * 0.78)
    switch_creates_finish = switch_best_damage >= foe_hp * 0.82
    switch_creates_pressure = switch_best_damage >= foe_hp * 0.55 and switch_after_ratio >= 0.45
    active_bad_element = is_elemental_threat(active, foe)
    active_incoming_element_multiplier = best_element_multiplier(foe, active)
    switch_incoming_element_multiplier = float(switch_choice.get("incoming_element_multiplier", 1.0) or 1.0)
    switch_outgoing_element_multiplier = float(switch_choice.get("outgoing_element_multiplier", 1.0) or 1.0)
    protected_incoming = estimate_incoming_damage_with_weakness_removed(foe, active) if has_usable_weakness_cover(active) else incoming
    protected_after_ratio = clamp((float(current_hp(active)) - protected_incoming) / max(1.0, float(max_hp(active))), 0.0, 1.0)
    weakness_cover_is_better = (
        bool(ability_choice.get("removes_elemental_weakness", False))
        and protected_after_ratio >= max(0.20, switch_after_ratio - 0.08)
        and protected_incoming < float(current_hp(active))
    )
    current_can_pressure = ability_damage >= foe_hp * 0.55 or best_ability_score >= 60.0

    if switch_incoming_ratio >= 0.70 or switch_after_ratio <= 0.18:
        return False
    if weakness_cover_is_better:
        return False
    hard_counter_escape = (
        active_incoming_element_multiplier >= 2.0
        and switch_incoming_element_multiplier <= 1.0
        and bool(switch_choice.get("survives", True))
        and switch_after_ratio >= 0.26
        and not current_can_pressure
        and (
            switch_best_damage >= max(1.0, ability_damage) * 0.82
            or switch_outgoing_element_multiplier >= 1.0
        )
        and adjusted_gain >= -20.0
    )
    if hard_counter_escape:
        switch_choice["reason"] = "escape_hard_counter"
        return True
    if (
        active_bad_element
        and not current_can_pressure
        and incoming_ratio >= (0.18 if active_hp <= 0.38 else 0.24)
        and bool(switch_choice.get("survives", True))
        and switch_after_ratio >= 0.26
        and (switch_creates_pressure or adjusted_gain >= 25.0)
        and adjusted_gain >= (0.0 if active_hp <= 0.28 else 10.0)
    ):
        switch_choice["reason"] = "escape_bad_element"
        return True
    if meaningful_attack and not active_endangered:
        return False
    if ability_damage >= foe_hp * 0.72 and active_hp > 0.16:
        return False

    if active_hp <= 0.08 and active_endangered and adjusted_gain >= 28.0 and switch_creates_finish:
        return True
    if active_hp <= SWITCH_HP_THRESHOLD and active_endangered and adjusted_gain >= 42.0 and best_ability_score < 24.0 and switch_creates_pressure:
        return True
    if bool(plan.get("ahead", False)) and not active_endangered:
        return adjusted_gain >= PROACTIVE_SWITCH_GAIN + 45.0 and best_ability_score < 20.0
    if bool(plan.get("behind", False)) and active_endangered and bool(switch_choice.get("survives", True)) and adjusted_gain >= 42.0 and best_ability_score < 30.0 and switch_creates_finish:
        return True
    if active_endangered and adjusted_gain >= PROACTIVE_SWITCH_GAIN + 16.0 and best_ability_score < 18.0 and switch_creates_finish:
        return True
    return False


def hydrate_miscrit(miscrit: dict[str, Any], metadata: dict[int, dict[str, Any]], pvp: bool, battle_type: str) -> None:
    mid = normalized_mid(miscrit)
    meta = metadata.get(mid, {})
    sync_effects_from_raw_statuses(miscrit)
    if meta:
        miscrit.setdefault("element", str(meta.get("element", "") or ""))
        miscrit.setdefault("rarity", str(meta.get("rarity", "") or ""))
        evo = battle_evo(miscrit)
        miscrit.setdefault("evo", evo)
        if not str(miscrit.get("name", "")).strip():
            names = meta.get("names", [])
            if isinstance(names, list) and names:
                miscrit["name"] = str(names[min(evo, len(names) - 1)] or names[0])
            else:
                miscrit["name"] = str(meta.get("name", f"#{mid}"))
    if abilities_for(miscrit):
        return
    raw_abilities = meta.get("abilities", [])
    if not isinstance(raw_abilities, list) or not raw_abilities:
        return
    level = int(miscrit.get("level", miscrit.get("l", 1)) or 1)
    count = sum(1 for key in LEVEL_KEYS if key <= level)
    selected = [dict(item) for item in raw_abilities[:count] if isinstance(item, dict)]
    selected.reverse()
    for ability in selected:
        cooldown = int(ability.get("cooldown", 0) or 0)
        ability["og_cooldown"] = cooldown
        if cooldown > 0:
            ability["cooldown"] = cooldown + 1
        elif pvp:
            ability["cooldown"] = 2
        ability.setdefault("recharge", 0)
    miscrit["abilities"] = selected
    if battle_type == "Random":
        existing = miscrit.get("enchants", [])
        if not isinstance(existing, list):
            existing = []
        all_ids = [int(item.get("id", 0) or 0) for item in selected if int(item.get("id", 0) or 0)]
        miscrit["enchants"] = list(dict.fromkeys([*existing, *all_ids]))


def switch_active_miscrit(player: dict[str, Any], switch_to: int) -> None:
    team = player.get("miscrits", [])
    if not isinstance(team, list) or not switch_to:
        return
    for index, item in enumerate(team):
        if isinstance(item, dict) and int(item.get("id", 0) or 0) == switch_to:
            if index == 0:
                return
            current = team.pop(0)
            chosen = team.pop(index - 1)
            team.insert(0, chosen)
            if isinstance(current, dict):
                insert_at = len(team)
                for pos in range(1, len(team)):
                    if is_dead(team[pos]):
                        insert_at = pos
                        break
                team.insert(insert_at, current)
            return


def score_ability(ability: dict[str, Any], attacker: dict[str, Any], defender: dict[str, Any], pvp: bool, plan: dict[str, Any] | None = None) -> dict[str, Any]:
    effective = effective_ability(ability, attacker)
    kind = str(effective.get("type", "") or ability.get("type", "")).strip()
    damage = estimate_ability_damage(effective, attacker, defender)
    accuracy = accuracy_factor(effective)
    score = damage * accuracy
    reasons = []

    defender_hp = current_hp(defender)
    if damage >= defender_hp > 0:
        score += LETHAL_BONUS
        reasons.append("lethal")
    elif damage >= defender_hp * 0.92 > 0 and is_damage_ability(effective):
        score += NEAR_LETHAL_BONUS
        reasons.append("near_lethal")

    incoming = estimate_incoming_damage(defender, attacker)
    self_hp = current_hp(attacker)
    if self_hp > 0 and incoming >= self_hp and damage < defender_hp:
        score -= 16.0 if damage >= defender_hp * 0.55 > 0 else 28.0
        reasons.append("incoming_lethal")
    if is_damage_ability(effective) and defender_hp > 0:
        pressure = damage / max(1.0, float(defender_hp))
        if pressure >= 0.70:
            score += 34.0
            reasons.append("kill_pressure")
        elif pressure >= 0.45:
            score += 15.0
            reasons.append("damage_pressure")
        if has_effect(defender, "sleep") and damage < defender_hp:
            wake_penalty = 11.0 + min(10.0, defender_hp * 0.06)
            if pressure >= 0.75:
                wake_penalty *= 0.55
            elif pressure <= 0.25:
                wake_penalty *= 1.15
            score -= wake_penalty
            reasons.append("wakes_sleep")

    utility = ability_utility(effective, attacker, defender, pvp)
    score += utility
    recovery = expected_self_recovery(effective, attacker, damage)
    if recovery > 0 and (is_damage_ability(effective) or has_lifesteal(effective) or cleanses_self(effective)):
        missing = max(0.0, float(max_hp(attacker) - current_hp(attacker)))
        effective_recovery = min(recovery, missing)
        if effective_recovery > 0:
            survives_without = self_hp > incoming
            survives_with = self_hp + effective_recovery > incoming
            if not survives_without and survives_with:
                score += 22.0 + min(12.0, effective_recovery * 0.35)
                reasons.append("recovery_saves_life")
            elif incoming >= self_hp + effective_recovery and not clears_dangerous_dot(effective, attacker):
                score -= min(28.0, effective_recovery * 0.8)
                reasons.append("recovery_too_small")
            else:
                score += min(24.0, effective_recovery * 0.42)
                reasons.append("recovery_value")
    elif recovery < 0:
        score += recovery * 1.2
        reasons.append("antiheal_risk")
    if clears_dangerous_dot(effective, attacker):
        score += 36.0
        reasons.append("cleanse_dot")
    force_value = force_switch_utility(effective, attacker, defender)
    if force_value:
        score += force_value
        reasons.append("force_switch")
    redundant = redundant_status_effect(effective, attacker, defender)
    if redundant and not (damage >= defender_hp > 0):
        score -= 180.0
        reasons.append("redundant_status")
    immune_blocked = debuff_blocked_by_immunity(effective, defender)
    if immune_blocked and not (damage >= defender_hp > 0):
        score -= 999.0
        reasons.append("immune_blocked")

    cooldown = int(effective.get("cooldown", effective.get("og_cooldown", 0)) or 0)
    if cooldown > 0:
        score -= min(8.0, float(cooldown) * 1.1)
    if int(effective.get("max_uses", 0) or 0) == 1:
        score += 8.0 if damage >= defender_hp > 0 else -4.0

    if kind == "Attack" and damage <= 0:
        score -= 20.0
    if not reasons:
        if damage > 0:
            reasons.append("highest_expected_damage")
        elif utility > 0:
            reasons.append("highest_utility")
        else:
            reasons.append("fallback")
    reason = "_".join(reasons)
    score += learned_bonus("ability", int(ability.get("id", 0) or 0), attacker, defender, reasons)
    score += matchup_memory(attacker, defender) * 0.35
    win_adjustment, win_reasons = win_condition_adjustment(effective, kind, damage, accuracy, utility, attacker, defender, plan or {})
    if win_adjustment:
        score += win_adjustment
        reasons.extend(win_reasons)
        reason = "_".join(dict.fromkeys(reasons))
    lookahead_adjustment, lookahead, lookahead_reasons = ability_lookahead_adjustment(effective, kind, damage, accuracy, attacker, defender, pvp, plan or {})
    if lookahead_adjustment:
        score += lookahead_adjustment
        reasons.extend(lookahead_reasons)
        reason = "_".join(dict.fromkeys(reasons))

    return {
        "id": int(ability.get("id", 0) or 0),
        "name": str(ability.get("name", "")),
        "type": kind,
        "element": str(effective.get("element", "")),
        "score": round(score, 3),
        "damage": round(damage, 3),
        "accuracy": round(accuracy, 3),
        "utility": round(utility, 3),
        "lethal": damage >= defender_hp > 0,
        "near_lethal": damage >= defender_hp * 0.92 > 0 and is_damage_ability(effective),
        "redundant": redundant,
        "immune_blocked": immune_blocked,
        "removes_elemental_weakness": removes_elemental_weakness(effective),
        "win_adjustment": round(win_adjustment, 3),
        "lookahead_adjustment": round(lookahead_adjustment, 3),
        "lookahead": lookahead,
        "win_plan": (plan or {}).get("mode", "even"),
        "reason": reason,
        "reason_tags": list(dict.fromkeys(reasons)),
    }


def win_condition_adjustment(
    ability: dict[str, Any],
    kind: str,
    damage: float,
    accuracy: float,
    utility: float,
    attacker: dict[str, Any],
    defender: dict[str, Any],
    plan: dict[str, Any],
) -> tuple[float, list[str]]:
    if not plan:
        return 0.0, []
    adjustment = 0.0
    reasons: list[str] = []
    damage_ability = is_damage_ability(ability)
    defender_hp = current_hp(defender)
    incoming = estimate_incoming_damage(defender, attacker)
    self_hp = current_hp(attacker)
    target_self = ability_targets_self(ability)

    if bool(plan.get("last_foe", False)):
        if damage_ability:
            pressure = damage / max(1.0, float(defender_hp))
            adjustment += min(LAST_FOE_FINISH_BONUS, pressure * 48.0)
            reasons.append("finish_last_foe")
            if damage >= defender_hp > 0:
                adjustment += 80.0
                reasons.append("secure_final_kill")
            elif damage >= defender_hp * 0.7 > 0:
                adjustment += 28.0
                reasons.append("final_pressure")
        if kind in {"Buff", "Bot"} and target_self:
            adjustment -= 95.0
            reasons.append("avoid_setup_last_foe")
        elif kind in STATUS_TYPES and not damage_ability:
            adjustment -= 42.0
            reasons.append("avoid_slow_status_last_foe")

    if bool(plan.get("last_ally", False)):
        if kind in {"Buff", "Bot"} and target_self and not damage_ability:
            adjustment -= LAST_ALLY_LONG_BUFF_PENALTY
            reasons.append("last_ally_no_long_buff")
        if kind in DOT_TYPES and damage < defender_hp:
            adjustment -= 24.0
            reasons.append("last_ally_avoid_slow_dot")
        if damage_ability:
            adjustment += min(68.0, damage * 0.24)
            reasons.append("last_ally_damage_now")
        if self_hp > 0 and incoming >= self_hp:
            if kind in TURN_DENIAL_TYPES or kind in DEFENSIVE_TYPES or kind in {"Heal", "Hot"}:
                adjustment += 18.0
                reasons.append("last_ally_survival")
            elif kind in SOFT_CONTROL_TYPES:
                adjustment += 6.0
                reasons.append("last_ally_soft_control")
            elif damage < defender_hp:
                adjustment -= 18.0
                reasons.append("last_ally_no_time")

    if bool(plan.get("ahead", False)):
        if damage_ability and accuracy >= 0.95:
            adjustment += 12.0
            reasons.append("ahead_safe_damage")
        if accuracy < 0.75:
            adjustment -= 26.0
            reasons.append("ahead_avoid_low_accuracy")
        if kind in CONTROL_TYPES and not damage_ability and incoming < self_hp * 0.6:
            adjustment -= 12.0
            reasons.append("ahead_avoid_unneeded_status")
        if kind in {"Heal", "Hot"} and target_self and hp_ratio(attacker) <= 0.55:
            adjustment += 16.0
            reasons.append("ahead_preserve_hp")

    if bool(plan.get("behind", False)):
        if kind in CONTROL_TYPES and not redundant_status_effect(ability, attacker, defender):
            if kind == "Sleep":
                adjustment += 42.0
                reasons.append("behind_sleep_window")
            elif kind == "Confuse":
                adjustment += 20.0
                reasons.append("behind_confuse_pressure")
            elif kind == "Paralyze":
                adjustment += 7.0
                reasons.append("behind_switch_lock")
        if kind in DOT_TYPES:
            adjustment += 13.0
            reasons.append("behind_dot_pressure")
        if damage_ability and damage >= defender_hp * 0.65 > 0:
            adjustment += 26.0
            reasons.append("behind_high_damage_risk")
        crit_factor = ability_crit_factor(ability)
        if crit_factor > 0:
            adjustment += min(34.0, crit_factor * 22.0)
            reasons.append("behind_crit_out")
        if utility > 0 and kind in {"Buff", "Bot"} and target_self and not has_effect(defender, "sleep"):
            adjustment -= 18.0
            reasons.append("behind_setup_needs_window")

    return adjustment, reasons


def ability_lookahead_adjustment(
    ability: dict[str, Any],
    kind: str,
    damage: float,
    accuracy: float,
    attacker: dict[str, Any],
    defender: dict[str, Any],
    pvp: bool,
    plan: dict[str, Any],
) -> tuple[float, dict[str, Any], list[str]]:
    defender_hp = current_hp(defender)
    attacker_hp = current_hp(attacker)
    if attacker_hp <= 0 or defender_hp <= 0:
        return 0.0, {}, []
    if damage >= defender_hp > 0:
        return 0.0, {"foe_after_hp": 0, "opponent_damage": 0.0, "our_followup_damage": 0.0}, []

    accuracy_factor_value = clamp(float(accuracy or 1.0), 0.0, 1.0)
    expected_damage = max(0.0, float(damage) * accuracy_factor_value)
    raw_damage_details = estimate_ability_damage_details(ability, attacker, defender)
    expected_raw_damage = max(
        0.0,
        float(raw_damage_details.get("base_damage", 0.0) or 0.0)
        * float(raw_damage_details.get("multiplier", 1.0) or 1.0)
        * accuracy_factor_value,
    )
    attacker_after = dict(attacker)
    defender_after = dict(defender)
    defender_was_sleeping = has_effect(defender, "sleep")
    recovery = expected_self_recovery(ability, attacker, damage)
    reflected = barbed_reflect_damage(defender) if is_damage_ability(ability) and expected_damage > 0 else 0.0
    recovery_cap = max(0.0, max_hp(attacker) - attacker_hp)
    applied_recovery = min(recovery, recovery_cap) if recovery >= 0 else recovery
    attacker_after_hp = clamp_int(int(round(attacker_hp + applied_recovery - reflected)), 0, max_hp(attacker))
    defender_after_hp = clamp_int(int(round(defender_hp - expected_damage)), 0, max_hp(defender))
    attacker_after["chp"] = attacker_after_hp
    defender_after["chp"] = defender_after_hp
    if defender_was_sleeping and is_damage_ability(ability) and expected_damage > 0:
        remove_effect(defender_after, "sleep")
    if is_damage_ability(ability) and expected_raw_damage > 0:
        consume_attack_defenses(defender_after, expected_raw_damage)

    if defender_after_hp <= 0:
        return LOOKAHEAD_FINISH_BONUS, {"foe_after_hp": 0, "opponent_damage": 0.0, "our_followup_damage": 0.0}, ["lookahead_expected_ko"]

    response = predict_opponent_response(defender_after, attacker_after, pvp)
    opponent_damage = float(response.get("damage", 0.0) or 0.0)
    if defender_was_sleeping and not is_damage_ability(ability):
        opponent_damage = 0.0
        response["controlled"] = "sleep_existing"
    elif kind in TURN_DENIAL_TYPES and not redundant_status_effect(ability, attacker, defender):
        opponent_damage = 0.0
        response["controlled"] = "sleep"
    elif kind in SOFT_CONTROL_TYPES and not redundant_status_effect(ability, attacker, defender):
        opponent_damage *= 0.62
        response["controlled"] = "confuse"

    attacker_after_response_hp = clamp_int(int(round(attacker_after_hp - opponent_damage)), 0, max_hp(attacker))
    attacker_after_response = dict(attacker_after)
    attacker_after_response["chp"] = attacker_after_response_hp
    our_followup_damage = estimate_incoming_damage(attacker_after_response, defender_after) if attacker_after_response_hp > 0 else 0.0

    adjustment = 0.0
    reasons: list[str] = []
    if opponent_damage >= attacker_after_hp > 0:
        if our_followup_damage >= defender_after_hp > 0:
            adjustment -= LOOKAHEAD_DEATH_PENALTY * 0.38
            reasons.append("lookahead_trade_risk")
        else:
            adjustment -= LOOKAHEAD_DEATH_PENALTY
            reasons.append("lookahead_enemy_kills")
    else:
        pressure = opponent_damage / max(1.0, float(max_hp(attacker)))
        adjustment -= min(38.0, pressure * 44.0 * LOOKAHEAD_OPPONENT_WEIGHT)
        if pressure <= 0.12:
            adjustment += 6.0
            reasons.append("lookahead_safe")
    if our_followup_damage >= defender_after_hp > 0:
        adjustment += LOOKAHEAD_FINISH_BONUS
        reasons.append("lookahead_next_finish")
    elif defender_after_hp > 0 and our_followup_damage >= defender_after_hp * 0.72:
        adjustment += 14.0
        reasons.append("lookahead_next_pressure")
    if bool(plan.get("last_ally", False)) and opponent_damage > 0:
        adjustment -= min(22.0, opponent_damage / max(1.0, float(max_hp(attacker))) * 28.0)
        reasons.append("lookahead_last_ally_risk")

    return (
        adjustment,
        {
            "foe_after_hp": defender_after_hp,
            "self_after_hp": attacker_after_hp,
            "self_after_response_hp": attacker_after_response_hp,
            "opponent_ability": response.get("name", ""),
            "opponent_ability_id": response.get("id", 0),
            "opponent_damage": round(opponent_damage, 3),
            "our_followup_damage": round(our_followup_damage, 3),
            "reflected_damage": round(reflected, 3),
        },
        reasons,
    )


def predict_opponent_response(attacker: dict[str, Any], defender: dict[str, Any], pvp: bool) -> dict[str, Any]:
    best: dict[str, Any] = {"id": 0, "name": "", "damage": 0.0, "score": 0.0}
    for ability in abilities_for(attacker):
        if not isinstance(ability, dict) or not is_usable_ability(ability):
            continue
        effective = effective_ability(ability, attacker)
        kind = str(effective.get("type", ability.get("type", "")) or "").strip()
        damage = estimate_ability_damage(effective, attacker, defender)
        accuracy = accuracy_factor(effective)
        utility = ability_utility(effective, attacker, defender, pvp) * 0.45
        score = damage * accuracy + utility
        if damage >= current_hp(defender) > 0:
            score += 80.0
        score += learned_bonus("ability", int(ability.get("id", 0) or 0), attacker, defender, "opponent_predict") * 0.65
        if score > float(best.get("score", 0.0) or 0.0):
            best = {
                "id": int(ability.get("id", 0) or 0),
                "name": str(ability.get("name", "")),
                "type": kind,
                "damage": round(damage * accuracy, 3),
                "score": round(score, 3),
            }
    if not best.get("id"):
        damage = estimate_incoming_damage(attacker, defender)
        best = {"id": 0, "name": "basic_pressure", "type": "Attack", "damage": round(damage, 3), "score": round(damage, 3)}
    return best


def estimate_ability_damage(ability: dict[str, Any], attacker: dict[str, Any], defender: dict[str, Any]) -> float:
    return float(estimate_ability_damage_details(ability, attacker, defender)["damage"])


def estimate_ability_damage_details(
    ability: dict[str, Any],
    attacker: dict[str, Any],
    defender: dict[str, Any],
    action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kind = str(ability.get("type", "")).strip()
    if kind not in {"Attack", "Bleed"} and kind not in DOT_TYPES and not bool(ability.get("true_dmg", ability.get("true_damage", False))):
        additional = estimate_additional_damage(ability, attacker, defender)
        features = damage_features(ability, attacker, defender, kind, "Misc", 1.0, False, 1.0, action)
        multiplier = damage_multiplier(features)
        damage = apply_block_reduction(max(0.0, additional * multiplier), defender)
        return {"damage": damage, "base_damage": additional, "multiplier": multiplier, "features": features}
    ap = abs(float(ability.get("ap", ability.get("power", 0)) or 0))
    times = max(1.0, float(ability.get("times", 1) or 1))
    if ap <= 0:
        additional = estimate_additional_damage(ability, attacker, defender)
        features = damage_features(ability, attacker, defender, kind, str(ability.get("element", "Physical") or "Physical"), times, False, 1.0, action)
        multiplier = damage_multiplier(features)
        damage = apply_block_reduction(max(0.0, additional * multiplier), defender)
        return {"damage": damage, "base_damage": additional, "multiplier": multiplier, "features": features}
    true_damage = bool(ability.get("true_dmg", ability.get("true_damage", False)))
    if bool(ability.get("true_dmg", ability.get("true_damage", False))):
        element = str(ability.get("element", "True") or "True")
        ratio = 1.0
        base = ap * times
    else:
        element = str(ability.get("element", "Physical") or "Physical")
        attack_key = "pa" if element == "Physical" else "ea"
        defense_key = "pd" if element == "Physical" else "ed"
        attack = max(1.0, stat_value(attacker, attack_key))
        defense = max(1.0, stat_value(defender, defense_key))
        ratio = clamp(attack / defense, 0.42, 2.25)
        base = ap * times * ratio * element_multiplier(element, defender)
    crit_multiplier = 1.5 if action and bool(action.get("crit", False)) else 1.0
    base *= crit_multiplier
    if kind in DOT_TYPES:
        turns = max(1.0, float(ability.get("turns", 1) or 1))
        base *= 0.38 + min(1.1, turns * 0.18)
    base = max(0.0, base + estimate_additional_damage(ability, attacker, defender))
    features = damage_features(ability, attacker, defender, kind, element, times, true_damage, ratio, action)
    multiplier = damage_multiplier(features)
    damage = apply_block_reduction(max(0.0, base * multiplier), defender)
    return {"damage": damage, "base_damage": base, "multiplier": multiplier, "features": features}


def estimate_additional_damage(ability: dict[str, Any], attacker: dict[str, Any], defender: dict[str, Any]) -> float:
    total = 0.0
    additional = ability.get("additional", [])
    if not isinstance(additional, list):
        return 0.0
    for item in additional:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "")).strip()
        if item_type not in {"Attack"} and item_type not in DOT_TYPES and not bool(item.get("true_dmg", False)):
            continue
        merged = expanded_additional_ability(ability, item)
        merged.setdefault("type", item_type)
        merged.setdefault("element", item.get("element", ability.get("element", "Physical")))
        total += (
            float(estimate_ability_damage_details({key: value for key, value in merged.items() if key != "additional"}, attacker, defender)["base_damage"])
            * additional_probability(ability, item)
        )
    return total


def expanded_additional_ability(ability: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    merged = {**ability, **item}
    merged["_parent_desc"] = str(ability.get("desc", ability.get("description", "")) or "")
    item_type = str(item.get("type", "") or "").strip()
    parent_type = str(ability.get("type", "") or "").strip()
    if "ap" not in item and item_type and item_type != parent_type:
        merged.pop("ap", None)
    if item_type == "Attack" and "ap" not in item:
        inferred = inferred_additional_attack_power(ability)
        if inferred > 0:
            merged["ap"] = inferred
    elif item_type in {"Heal", "Hot"} and "ap" not in item:
        merged["ap"] = inferred_direct_heal_amount(merged)
    elif item_type == "LifeSteal" and "ap" not in item:
        merged["ap"] = inferred_lifesteal_amount(ability)
    elif bool(item.get("true_dmg", False)) and "ap" not in item:
        fixed = inferred_fixed_damage(ability)
        if fixed > 0:
            merged["ap"] = fixed
    return merged


def additional_probability(ability: dict[str, Any], item: dict[str, Any]) -> float:
    text = ability_text(ability)
    chance_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s+chance", text)
    if chance_match:
        return clamp(float(chance_match.group(1)) / 100.0, 0.05, 1.0)
    if str(item.get("type", "")).strip() in {"Heal", "Hot", "LifeSteal", "ForceSwitch", "Cleanser"}:
        return 1.0
    return 0.65


def ability_text(ability: dict[str, Any]) -> str:
    return " ".join(
        str(ability.get(key, "") or "")
        for key in ("name", "desc", "description", "tooltip", "effect", "_parent_desc")
    ).casefold()


def inferred_additional_attack_power(ability: dict[str, Any]) -> float:
    text = ability_text(ability)
    patterns = [
        r"additional\s+(?:one|two|three|four|\d+)\s+(\d+(?:\.\d+)?)\s*ap",
        r"additional[^.]*?(\d+(?:\.\d+)?)\s*(?:ap|power|fixed damage)",
        r"triggers? an additional attack with\s+(\d+(?:\.\d+)?)\s*(?:fixed damage|power)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return 0.0


def inferred_fixed_damage(ability: dict[str, Any]) -> float:
    text = ability_text(ability)
    match = re.search(r"(\d+(?:\.\d+)?)\s*fixed damage", text)
    return float(match.group(1)) if match else 0.0


def inferred_direct_heal_amount(ability: dict[str, Any]) -> float:
    ap = abs(float(ability.get("ap", 0) or 0))
    if ap > 0:
        return ap
    text = ability_text(ability)
    patterns = [
        r"heals? yourself by\s+(\d+(?:\.\d+)?)\s*hp",
        r"heals? yourself by\s+(\d+(?:\.\d+)?)",
        r"grants?\s+(\d+(?:\.\d+)?)\s+healing",
        r"heals? for\s+(\d+(?:\.\d+)?)\s+hp",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return 0.0


def inferred_lifesteal_amount(ability: dict[str, Any]) -> float:
    text = ability_text(ability)
    match = re.search(r"steals?\s+(\d+(?:\.\d+)?)\s*hp", text)
    return float(match.group(1)) if match else 0.0


def has_lifesteal(ability: dict[str, Any]) -> bool:
    additional = ability.get("additional", [])
    if isinstance(additional, list) and any(isinstance(item, dict) and str(item.get("type", "")) == "LifeSteal" for item in additional):
        return True
    return inferred_lifesteal_amount(ability) > 0


def expected_self_recovery(ability: dict[str, Any], attacker: dict[str, Any], damage: float) -> float:
    kind = str(ability.get("type", "")).strip()
    total = 0.0
    if ability_targets_self(ability) and kind in {"Heal", "Hot", "Cleanser"}:
        total += inferred_direct_heal_amount(ability)
    additional = ability.get("additional", [])
    has_lifesteal_additional = isinstance(additional, list) and any(
        isinstance(item, dict) and str(item.get("type", "")).strip() == "LifeSteal" for item in additional
    )
    if inferred_lifesteal_amount(ability) > 0 and not has_lifesteal_additional:
        total += min(max(0.0, float(damage)), inferred_lifesteal_amount(ability) or float(damage) * 0.25)
    if isinstance(additional, list):
        for item in additional:
            if not isinstance(item, dict):
                continue
            expanded = expanded_additional_ability(ability, item)
            item_type = str(expanded.get("type", "")).strip()
            chance = additional_probability(ability, item)
            if item_type in {"Heal", "Hot"}:
                total += inferred_direct_heal_amount(expanded) * chance
            elif item_type == "LifeSteal":
                total += inferred_lifesteal_amount(expanded) * chance
    if total > 0 and healing_cursed(attacker):
        return -total
    return max(0.0, total)


def cleanses_self(ability: dict[str, Any]) -> bool:
    kind = str(ability.get("type", "")).strip()
    text = ability_text(ability)
    return kind == "Cleanser" or "cleanses yourself" in text or "cleanses user" in text


def has_dangerous_dot(miscrit: dict[str, Any]) -> bool:
    return has_any_effect(miscrit, ["bleed", "poison", "disease", "dot", "switchcurse"])


def clears_dangerous_dot(ability: dict[str, Any], attacker: dict[str, Any]) -> bool:
    return cleanses_self(ability) and has_dangerous_dot(attacker)


def forces_switch(ability: dict[str, Any]) -> bool:
    if str(ability.get("type", "")).strip() == "ForceSwitch":
        return True
    additional = ability.get("additional", [])
    if isinstance(additional, list) and any(isinstance(item, dict) and str(item.get("type", "")).strip() == "ForceSwitch" for item in additional):
        return True
    text = ability_text(ability)
    return "force switches" in text or "forcefully switches" in text or "forces you to switch" in text


def force_switch_target(ability: dict[str, Any]) -> str:
    text = ability_text(ability)
    if "foe" in text:
        return "foe"
    if "user" in text or "yourself" in text or "you to switch" in text:
        return "self"
    return "unknown"


def force_switch_utility(ability: dict[str, Any], attacker: dict[str, Any], defender: dict[str, Any]) -> float:
    if not forces_switch(ability):
        return 0.0
    target = force_switch_target(ability)
    incoming = estimate_incoming_damage(defender, attacker)
    outgoing = estimate_incoming_damage(attacker, defender)
    if target == "foe":
        value = 3.0
        if is_elemental_threat(attacker, defender):
            value += 3.0
        value += min(8.0, max(0.0, incoming - outgoing * 0.9) * 0.06)
        if incoming >= current_hp(attacker) * 0.45:
            value += 3.0
        if hp_ratio(defender) <= 0.25:
            value -= 4.0
        return max(0.0, value)
    if target == "self":
        value = 2.0
        if is_elemental_threat(attacker, defender) or hp_ratio(attacker) <= 0.32:
            value += 4.0
        else:
            value -= 2.0
        return max(0.0, value)
    return 2.0


def damage_features(
    ability: dict[str, Any],
    attacker: dict[str, Any],
    defender: dict[str, Any],
    kind: str,
    element: str,
    times: float,
    true_damage: bool,
    ratio: float,
    action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    defense_state = "normal"
    if bool(defender.get("negated", defender.get("negate", False))):
        defense_state = "negate"
    elif has_effect(defender, "block") or block_amount(defender) > 0:
        defense_state = "block"
    elif bool(defender.get("ethereal", False)):
        defense_state = "ethereal"
    return {
        "ability_id": int(ability.get("id", 0) or 0),
        "kind": str(kind or "Unknown"),
        "element": str(element or "Physical"),
        "true_damage": bool(true_damage),
        "multi_hit": float(times or 1.0) > 1.0,
        "times": round(float(times or 1.0), 3),
        "crit": bool(action.get("crit", False)) if isinstance(action, dict) else False,
        "ratio_bucket": ratio_bucket(float(ratio or 1.0)),
        "element_multiplier": round(element_multiplier(str(element or ""), defender), 3),
        "defense_state": defense_state,
        "attacker_mid": normalized_mid(attacker),
        "defender_mid": normalized_mid(defender),
    }


def ratio_bucket(ratio: float) -> str:
    if ratio < 0.75:
        return "low"
    if ratio > 1.35:
        return "high"
    return "even"


def summarize_damage_action(action: dict[str, Any]) -> dict[str, Any]:
    return {key: action[key] for key in ("type", "target", "id", "damage", "ap", "hp", "chp", "crit", "miss", "dead") if key in action}


def action_causes_wake(action: dict[str, Any]) -> bool:
    action_type = str(action.get("type", "")).strip()
    return action_type in {
        "Attack",
        "PoisonDamage",
        "DotDamage",
        "BleedDamage",
        "SwitchCurseDamage",
        "DiseaseDamage",
        "AntihealDamage",
        "BarbedDamage",
    } and float(action.get("damage", 0) or 0) > 0


def remember_block_value(target: dict[str, Any], action: dict[str, Any]) -> None:
    ap = block_value_from_action(action)
    if ap <= 0:
        remove_effect(target, "block")
        return
    duration = int(action.get("turns", action.get("duration", action.get("rounds", 1))) or 1)
    add_effect(target, "block", max(1, duration))
    set_effect_value(target, "block", {"ap": ap})


def remember_barbed_value(target: dict[str, Any], action: dict[str, Any]) -> None:
    damage = max(0.0, float(action.get("damage", action.get("ap", 0)) or 0))
    charges = max(1, effect_turns(target, "barbed") + 1)
    add_effect(target, "barbed", charges)
    set_effect_value(target, "barbed", {"damage": damage, "charges": charges})


def block_value_from_action(action: dict[str, Any]) -> float:
    for key in ("ap", "block", "remaining", "value"):
        try:
            value = float(action.get(key, 0) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0.0


def block_amount(miscrit: dict[str, Any]) -> float:
    values = miscrit.get("_ai_effect_values", {})
    if isinstance(values, dict):
        block = values.get("block", {})
        if isinstance(block, dict):
            try:
                value = float(block.get("ap", 0) or 0)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                return value
    statuses = miscrit.get("statuses", miscrit.get("status", miscrit.get("effects", [])))
    if isinstance(statuses, list):
        for item in statuses:
            if not isinstance(item, dict) or status_key(item.get("key", item.get("name", item.get("type", "")))) != "block":
                continue
            return block_value_from_action(item)
    elif isinstance(statuses, dict):
        raw = statuses.get("block", statuses.get("Block"))
        if isinstance(raw, dict):
            return block_value_from_action(raw)
        try:
            return max(0.0, float(raw or 0))
        except (TypeError, ValueError):
            return 0.0
    try:
        return max(0.0, float(miscrit.get("block", 0) or 0))
    except (TypeError, ValueError):
        return 0.0


def apply_block_reduction(damage: float, defender: dict[str, Any]) -> float:
    return max(0.0, float(damage or 0.0) - block_amount(defender))


def consume_attack_defenses(defender: dict[str, Any], raw_damage: float) -> None:
    current_block = block_amount(defender)
    if current_block > 0:
        remaining = max(0.0, current_block - max(0.0, float(raw_damage or 0.0)))
        if remaining > 0:
            set_effect_value(defender, "block", {"ap": remaining})
        else:
            remove_effect(defender, "block")
    consume_barbed_charge(defender)


def barbed_reflect_damage(defender: dict[str, Any]) -> float:
    if not has_effect(defender, "barbed"):
        return 0.0
    values = defender.get("_ai_effect_values", {})
    if isinstance(values, dict):
        raw = values.get("barbed", {})
        if isinstance(raw, dict):
            try:
                return max(0.0, float(raw.get("damage", 0) or 0))
            except (TypeError, ValueError):
                return 0.0
    return 10.0


def consume_barbed_charge(defender: dict[str, Any]) -> None:
    if not has_effect(defender, "barbed"):
        return
    charges = max(0, effect_turns(defender, "barbed") - 1)
    if charges > 0:
        set_effect_turns(defender, "barbed", charges)
        set_effect_value(defender, "barbed", {"charges": charges})
    else:
        remove_effect(defender, "barbed")


def remember_action_effect(target: dict[str, Any], action: dict[str, Any]) -> None:
    action_type = str(action.get("type", "")).strip()
    if action_type == "Woken":
        remove_effect(target, "sleep")
        return
    if action_type == "BlockDamage":
        remember_block_value(target, action)
        return
    if action_type == "Barbed":
        remember_barbed_value(target, action)
        return
    if action_type == "Immune":
        immunity = immunity_key_from_status(str(action.get("immunity", "")))
        if immunity:
            add_effect(target, status_key(immunity), 2)
        return
    if action_type in {"Cleanser", "Cleanse", "Purged", "Purge"}:
        clear_dangerous_dots(target)
        return
    if action_type not in STATUS_TYPES:
        return
    duration = int(action.get("turns", action.get("duration", action.get("rounds", 0))) or 0)
    if duration == 0:
        remove_effect(target, status_key(action_type))
        return
    if duration <= 0:
        duration = 3 if action_type in CONTROL_TYPES else 4
    add_effect(target, status_key(action_type), duration)
    if action_type == "Block":
        remember_block_value(target, action)
    if action_type == "Negate":
        add_elemental_weakness_removed(target, duration)
    keys = action.get("keys", [])
    if action_type in {"Buff", "Bot"} and isinstance(keys, list):
        for key in keys:
            add_effect(target, f"{status_key(action_type)}:{str(key).casefold()}", duration)


def redundant_status_effect(ability: dict[str, Any], attacker: dict[str, Any], defender: dict[str, Any]) -> bool:
    kind = str(ability.get("type", "")).strip()
    if kind not in NON_STACKING_STATUS_TYPES:
        return False
    target = attacker if ability_targets_self(ability) else defender
    base_key = status_key(kind)
    if kind in REFRESHABLE_STATUS_TYPES:
        return has_effect(target, base_key) and not should_refresh_status(ability, target, base_key)
    if has_effect(target, base_key):
        return True
    return False


def should_refresh_status(ability: dict[str, Any], target: dict[str, Any], base_key: str) -> bool:
    if not has_effect(target, base_key):
        return False
    current_turns = effect_turns(target, base_key)
    refresh_turns = int(ability.get("turns", ability.get("duration", ability.get("rounds", 0))) or 0)
    if refresh_turns <= 0:
        refresh_turns = 4
    return current_turns <= max(1, refresh_turns // 2)


def debuff_blocked_by_immunity(ability: dict[str, Any], defender: dict[str, Any]) -> bool:
    if ability_targets_self(ability):
        return False
    kind = str(ability.get("type", "")).strip()
    immunity = immunity_key_from_status(kind)
    if immunity and has_effect(defender, status_key(immunity)):
        return True
    if kind in {"Buff", "Bot"} and float(ability.get("ap", 0) or 0) < 0:
        return stat_debuff_blocked_by_immunity(ability, defender)
    return False


def stat_debuff_blocked_by_immunity(ability: dict[str, Any], defender: dict[str, Any]) -> bool:
    if has_any_effect(defender, ["debuff_immune", "debuffimmune", "stat_immune", "statimmune"]):
        return True
    keys = ability.get("keys", [])
    if not isinstance(keys, list) or not keys:
        return False
    return all(has_any_effect(defender, [f"{str(key).casefold()}_immune", f"immune:{str(key).casefold()}"]) for key in keys)


def immunity_key_from_status(kind: str) -> str:
    normalized = normalize_status_name(kind)
    return IMMUNITY_BY_STATUS.get(normalized, "")


def normalize_status_name(kind: str) -> str:
    value = str(kind or "").strip().casefold()
    mapping = {
        "antiheal": "Antiheal",
        "confuse": "Confuse",
        "sleep": "Sleep",
        "paralyze": "Paralyze",
        "ai": "AI",
        "ci": "CI",
        "si": "SI",
        "pi": "PI",
    }
    return mapping.get(value, str(kind or "").strip())


def ability_targets_self(ability: dict[str, Any]) -> bool:
    return str(ability.get("target", "Foe")).casefold() != "foe"


def is_damage_ability(ability: dict[str, Any]) -> bool:
    kind = str(ability.get("type", "")).strip()
    if kind in {"Attack", "Bleed"} or kind in DOT_TYPES:
        return True
    if bool(ability.get("true_dmg", ability.get("true_damage", False))):
        return True
    additional = ability.get("additional", [])
    return isinstance(additional, list) and any(isinstance(item, dict) and is_damage_ability(item) for item in additional)


def status_key(kind: str) -> str:
    return str(kind or "").strip().casefold()


def add_effect(miscrit: dict[str, Any], key: str, duration: int) -> None:
    if not key:
        return
    clear_expired_effect(miscrit, key)
    effects = miscrit.get("_ai_effects", {})
    if not isinstance(effects, dict):
        effects = {}
    effects[key] = max(int(effects.get(key, 0) or 0), int(duration or 0))
    miscrit["_ai_effects"] = effects


def set_effect_value(miscrit: dict[str, Any], key: str, values: dict[str, Any]) -> None:
    if not key:
        return
    effect_values = miscrit.get("_ai_effect_values", {})
    if not isinstance(effect_values, dict):
        effect_values = {}
    current = effect_values.get(key, {})
    if not isinstance(current, dict):
        current = {}
    current.update(values)
    effect_values[key] = current
    miscrit["_ai_effect_values"] = effect_values


def set_effect_turns(miscrit: dict[str, Any], key: str, turns: int) -> None:
    if not key:
        return
    effects = miscrit.get("_ai_effects", {})
    if not isinstance(effects, dict):
        effects = {}
    effects[key] = max(0, int(turns or 0))
    miscrit["_ai_effects"] = effects


def add_elemental_weakness_removed(miscrit: dict[str, Any], duration: int) -> None:
    duration = max(1, int(duration or 4))
    add_effect(miscrit, "elemental_weakness_removed", duration)
    add_effect(miscrit, "negate", duration)


def remove_effect(miscrit: dict[str, Any], key: str) -> None:
    effects = miscrit.get("_ai_effects", {})
    if isinstance(effects, dict):
        effects.pop(key, None)
    effect_values = miscrit.get("_ai_effect_values", {})
    if isinstance(effect_values, dict):
        effect_values.pop(key, None)
    mark_expired_effect(miscrit, key)
    for raw_key in (key, key.replace("_", "")):
        if raw_key in miscrit:
            miscrit[raw_key] = False


def expired_effect_keys(miscrit: dict[str, Any]) -> set[str]:
    raw = miscrit.get("_ai_expired_effects", [])
    if isinstance(raw, list):
        return {status_key(str(item)) for item in raw if item}
    if isinstance(raw, set):
        return {status_key(str(item)) for item in raw if item}
    return set()


def mark_expired_effect(miscrit: dict[str, Any], key: str) -> None:
    normalized = status_key(key)
    if not normalized:
        return
    expired = expired_effect_keys(miscrit)
    expired.add(normalized)
    miscrit["_ai_expired_effects"] = sorted(expired)


def clear_expired_effect(miscrit: dict[str, Any], key: str) -> None:
    normalized = status_key(key)
    if not normalized:
        return
    expired = expired_effect_keys(miscrit)
    if normalized in expired:
        expired.remove(normalized)
        if expired:
            miscrit["_ai_expired_effects"] = sorted(expired)
        else:
            miscrit.pop("_ai_expired_effects", None)


def clear_dangerous_dots(miscrit: dict[str, Any]) -> None:
    for key in ("bleed", "poison", "disease", "dot", "switchcurse"):
        remove_effect(miscrit, key)


def clear_switch_transient_effects(miscrit: dict[str, Any]) -> None:
    effects = miscrit.get("_ai_effects", {})
    if isinstance(effects, dict):
        for key in list(effects):
            normalized = status_key(str(key))
            if normalized == "switchcurse" or normalized == "buff" or normalized == "bot" or normalized.startswith("buff:") or normalized.startswith("bot:"):
                remove_effect(miscrit, str(key))
    remove_effect(miscrit, "switchcurse")


def healing_cursed(miscrit: dict[str, Any]) -> bool:
    return has_effect(miscrit, "antiheal")


def has_effect(miscrit: dict[str, Any], key: str) -> bool:
    normalized_key = status_key(key)
    expired = expired_effect_keys(miscrit)
    effects = miscrit.get("_ai_effects", {})
    if isinstance(effects, dict):
        for effect_key, value in effects.items():
            if status_key(str(effect_key)) == normalized_key and int(value or 0) > 0:
                return True
    for raw_key in (normalized_key, normalized_key.replace("_", ""), normalized_key.replace("switchcurse", "curse")):
        if status_key(raw_key) in expired:
            continue
        value = miscrit.get(raw_key)
        if isinstance(value, bool) and value:
            return True
        if isinstance(value, (int, float)) and value > 0:
            return True
    statuses = miscrit.get("statuses", miscrit.get("status", miscrit.get("effects", [])))
    if isinstance(statuses, list):
        for item in statuses:
            if isinstance(item, dict):
                raw_key = str(item.get("key", item.get("name", item.get("type", ""))) or "")
                if raw_key and status_key(raw_key) == normalized_key and status_key(raw_key) not in expired:
                    return True
            elif status_key(str(item)) == normalized_key and status_key(str(item)) not in expired:
                return True
        return False
    if isinstance(statuses, dict):
        for raw_key, value in statuses.items():
            normalized = status_key(str(raw_key))
            if normalized == normalized_key:
                if normalized in expired:
                    return False
                return int(value or 0) > 0 if isinstance(value, (int, float)) else bool(value)
    return False


def effect_turns(miscrit: dict[str, Any], key: str) -> int:
    normalized_key = status_key(key)
    effects = miscrit.get("_ai_effects", {})
    if isinstance(effects, dict):
        for effect_key, value in effects.items():
            if status_key(str(effect_key)) == normalized_key:
                try:
                    return max(0, int(value or 0))
                except (TypeError, ValueError):
                    return 0
    for raw_key, turns, _payload in raw_status_entries(miscrit):
        if status_key(raw_key) == normalized_key:
            try:
                return max(0, int(turns or 0))
            except (TypeError, ValueError):
                return 0
    return 0


def sync_effects_from_raw_statuses(miscrit: dict[str, Any]) -> None:
    effects = miscrit.get("_ai_effects", {})
    if not isinstance(effects, dict):
        effects = {}
    expired = expired_effect_keys(miscrit)
    for key, turns, payload in raw_status_entries(miscrit):
        normalized = status_key(key)
        if not normalized or normalized in effects or normalized in expired:
            continue
        try:
            value = int(turns)
        except (TypeError, ValueError):
            value = 1 if turns else 0
        if value > 0:
            effects[normalized] = value
            if isinstance(payload, dict) and normalized == "block":
                remember_block_value(miscrit, payload)
    if effects:
        miscrit["_ai_effects"] = effects


def raw_status_entries(miscrit: dict[str, Any]) -> list[tuple[str, Any, dict[str, Any]]]:
    statuses = miscrit.get("statuses", miscrit.get("status", miscrit.get("effects", [])))
    entries: list[tuple[str, Any, dict[str, Any]]] = []
    if isinstance(statuses, dict):
        for key, value in statuses.items():
            if value:
                payload = value if isinstance(value, dict) else {}
                turns = payload.get("turns", payload.get("duration", value)) if isinstance(payload, dict) else value
                entries.append((str(key), turns, payload))
    elif isinstance(statuses, list):
        for item in statuses:
            if isinstance(item, dict):
                key = item.get("key", item.get("name", item.get("type", "")))
                if key:
                    entries.append((str(key), item.get("turns", item.get("duration", 1)), item))
            elif item:
                entries.append((str(item), 1, {}))
    return entries


def has_any_effect(miscrit: dict[str, Any], keys: list[str]) -> bool:
    return any(has_effect(miscrit, key) for key in keys)


def tick_effects(miscrit: dict[str, Any], turns: int) -> None:
    effects = miscrit.get("_ai_effects", {})
    if not isinstance(effects, dict):
        return
    next_effects = {}
    for key, value in effects.items():
        if status_key(str(key)) in ATTACK_CONSUMED_EFFECTS:
            next_effects[key] = value
            continue
        remaining = int(value or 0) - max(1, int(turns or 1))
        if remaining > 0:
            next_effects[key] = remaining
        else:
            mark_expired_effect(miscrit, key)
    miscrit["_ai_effects"] = next_effects


def team_condition(team: Any) -> dict[str, Any]:
    items = [item for item in team if isinstance(item, dict)] if isinstance(team, list) else []
    if not items:
        return {"alive": 0, "dead": 0, "hp_ratio": 0.0}
    alive = 0
    dead = 0
    total_hp = 0.0
    total_max = 0.0
    for item in items:
        item_max = max(1, max_hp(item))
        item_hp = min(float(current_hp(item)), float(item_max))
        total_hp += item_hp
        total_max += float(item_max)
        if is_dead(item):
            dead += 1
        else:
            alive += 1
    return {"alive": alive, "dead": dead, "hp_ratio": total_hp / max(1.0, total_max)}


def ability_utility(ability: dict[str, Any], attacker: dict[str, Any], defender: dict[str, Any], pvp: bool) -> float:
    kind = str(ability.get("type", "")).strip()
    target_self = ability_targets_self(ability)
    ap = float(ability.get("ap", 0) or 0)
    score = 0.0
    if debuff_blocked_by_immunity(ability, defender):
        return 0.0
    if redundant_status_effect(ability, attacker, defender):
        return 0.0
    if target_self and removes_elemental_weakness(ability):
        incoming = estimate_incoming_damage(defender, attacker)
        protected_incoming = estimate_incoming_damage_with_weakness_removed(defender, attacker)
        prevented = max(0.0, incoming - protected_incoming)
        if is_elemental_threat(attacker, defender) and not elemental_weakness_removed(attacker):
            score += 8.0 + min(12.0, prevented * 0.12)
            if protected_incoming < current_hp(attacker) <= incoming:
                score += 14.0
            if hp_ratio(attacker) <= 0.45:
                score += 4.0
        elif elemental_weakness_removed(attacker):
            score -= 10.0
    if kind in {"Heal", "Hot"} and target_self:
        if healing_cursed(attacker):
            score -= inferred_direct_heal_amount(ability) * (1.15 if kind == "Heal" else 1.35)
            return score * accuracy_factor(ability)
        missing = max(0.0, float(max_hp(attacker) - current_hp(attacker)))
        if missing:
            turns = max(1.0, float(ability.get("turns", 1) or 1))
            heal_value = inferred_direct_heal_amount(ability) * (1.0 if kind == "Heal" else min(2.1, 0.65 + turns * 0.24))
            incoming = estimate_incoming_damage(defender, attacker)
            useful_heal = min(missing, heal_value)
            if useful_heal:
                if current_hp(attacker) + useful_heal <= incoming and not clears_dangerous_dot(ability, attacker):
                    score += useful_heal * 0.12
                else:
                    score += useful_heal * (0.95 if hp_ratio(attacker) <= 0.35 else 0.45)
            if clears_dangerous_dot(ability, attacker):
                score += 32.0
    elif kind in CONTROL_TYPES and not target_self:
        incoming = estimate_incoming_damage(defender, attacker)
        turns = max(1.0, float(ability.get("turns", 1) or 1))
        duration_bonus = min(3.0, max(0.0, turns - 1.0) * 0.85)
        if kind in TURN_DENIAL_TYPES:
            score += 5.3 if pvp else 4.8
        elif kind in SOFT_CONTROL_TYPES:
            score += 4.6 if pvp else 4.2
        elif kind in SWITCH_LOCK_TYPES:
            score += 4.9 if pvp else 4.4
            if hp_ratio(defender) > 0.35:
                score += 1.2
        score += min(7.0, incoming * 0.05)
        score += duration_bonus
        if hp_ratio(defender) <= 0.25:
            score -= 6.0
    elif kind in DOT_TYPES and not target_self:
        turns = max(1.0, float(ability.get("turns", 1) or 1))
        times = max(1.0, float(ability.get("times", 1) or 1))
        score += 4.4 + min(12.0, estimated_dot_pressure(kind, abs(ap), turns, times) * 0.16)
        if kind == "Bleed":
            score += 4.0 + min(6.0, max(0.0, turns - 1.0) * 1.1)
        elif kind == "SwitchCurse":
            score += 3.6 + min(5.0, max(0.0, turns - 1.0) * 1.0)
        if hp_ratio(defender) <= 0.25:
            score -= 4.0
    elif kind in {"Buff", "Bot"}:
        stat_score = score_stat_effect(ability, target_self, attacker, defender)
        if stat_score > 0 and has_effect(defender, "sleep"):
            stat_score = stat_score * 1.45 + 8.0
        elif stat_score > 0 and not target_self:
            stat_score *= 0.9
        score += stat_score
    elif kind in DEFENSIVE_TYPES:
        incoming = estimate_incoming_damage(defender, attacker)
        if target_self:
            if kind == "Negate":
                if is_elemental_threat(attacker, defender):
                    score += 2.1 + min(4.0, incoming * 0.08)
                else:
                    score += 0.45 + min(0.8, incoming * 0.04)
            else:
                score += 2.0 + min(10.0, incoming * 0.16)
            if hp_ratio(attacker) <= 0.35:
                score += 2.0
        else:
            score += 4.0 if kind in {"AI", "CI", "SI", "PI", "Negate"} else 2.0
    elif cleanses_self(ability):
        score += 36.0 if has_dangerous_dot(attacker) else 3.0
    elif forces_switch(ability):
        score += force_switch_utility(ability, attacker, defender)
    for item in ability.get("additional", []) if isinstance(ability.get("additional", []), list) else []:
        if isinstance(item, dict):
            score += ability_utility({**expanded_additional_ability(ability, item), "additional": []}, attacker, defender, pvp) * additional_probability(ability, item)
    return max(0.0, score) * accuracy_factor(ability)


def score_stat_effect(ability: dict[str, Any], target_self: bool, attacker: dict[str, Any], defender: dict[str, Any]) -> float:
    keys = ability.get("keys", [])
    if not isinstance(keys, list):
        keys = []
    ap = float(ability.get("ap", 0) or 0)
    if not keys or ap == 0:
        return 2.0
    total = 0.0
    for raw_key in keys:
        key = str(raw_key).casefold()
        if key not in STAT_KEYS:
            continue
        weight = {"pa": 1.1, "ea": 1.1, "spd": 0.85, "pd": 0.85, "ed": 0.85, "acc": 0.75}.get(key, 0.7)
        beneficial = (ap > 0 and target_self) or (ap < 0 and not target_self)
        if beneficial:
            total += min(35.0, abs(ap)) * weight * (0.14 if key in {"pa", "ea"} else 0.11)
        else:
            total -= min(18.0, abs(ap)) * 0.12
    if hp_ratio(defender) <= 0.25 and not target_self:
        total *= 0.55
    return total


def estimated_dot_pressure(kind: str, ap: float, turns: float, times: float) -> float:
    base = max(0.0, float(ap or 0.0)) * max(1.0, float(times or 1.0))
    if kind in {"Bleed", "SwitchCurse"}:
        # Logs show growing ticks for these effects, e.g. SwitchCurse 4 -> 8 -> 12.
        return base * max(1.0, float(turns or 1.0)) * (max(1.0, float(turns or 1.0)) + 1.0) * 0.25
    return base * max(1.0, float(turns or 1.0))


def switch_evaluation(candidate: dict[str, Any], foe: dict[str, Any]) -> dict[str, Any]:
    if not candidate:
        return {"id": 0, "score": -999.0, "reason": "no_candidate"}
    score = miscrit_power(candidate) * 0.28
    candidate_hp_ratio = hp_ratio(candidate)
    score += candidate_hp_ratio * 48.0
    incoming = 0.0
    best_damage = 0.0
    elemental_score = 0.0
    incoming_element_multiplier = 1.0
    outgoing_element_multiplier = 1.0
    lethal_incoming = False
    survives = True
    after_incoming_ratio = candidate_hp_ratio
    reason = "better_matchup"
    if foe:
        incoming_element_multiplier = best_element_multiplier(foe, candidate)
        outgoing_element_multiplier = best_element_multiplier(candidate, foe)
        elemental_score = elemental_matchup_score(candidate, foe)
        score += elemental_score
        score += matchup_memory(candidate, foe) * 0.75
        for ability in abilities_for(candidate):
            if isinstance(ability, dict) and is_usable_ability(ability):
                best_damage = max(best_damage, estimate_ability_damage(effective_ability(ability, candidate), candidate, foe) * accuracy_factor(ability))
        incoming = estimate_incoming_damage(foe, candidate)
        lethal_incoming = incoming >= current_hp(candidate) > 0
        survives = not lethal_incoming
        score += min(90.0, best_damage * 0.55)
        score -= min(80.0, incoming * 0.42)
        if survives:
            remaining_ratio = clamp((current_hp(candidate) - incoming) / max(1.0, float(max_hp(candidate))), 0.0, 1.0)
            after_incoming_ratio = remaining_ratio
            score += remaining_ratio * SWITCH_SAFE_HP_BONUS
        else:
            after_incoming_ratio = 0.0
            score -= SWITCH_LETHAL_PENALTY
            reason = "unsafe_switch"
        if has_super_effective_ability(candidate, foe):
            score += 22.0
        if is_elemental_threat(candidate, foe):
            score -= 12.0 if has_usable_weakness_cover(candidate) else 26.0
    if candidate_hp_ratio <= 0.18:
        score -= 34.0
        if reason == "better_matchup":
            reason = "low_hp_switch"
    score += learned_bonus("switch", int(candidate.get("id", 0) or 0), candidate, foe, "switch")
    return {
        "id": int(candidate.get("id", 0) or 0),
        "mid": normalized_mid(candidate),
        "name": miscrit_name(candidate),
        "score": round(score, 3),
        "hp_ratio": round(candidate_hp_ratio, 3),
        "incoming": round(incoming, 3),
        "incoming_ratio": round(incoming / max(1.0, float(max_hp(candidate))), 3),
        "after_incoming_ratio": round(after_incoming_ratio, 3),
        "survives": survives,
        "lethal_incoming": lethal_incoming,
        "elemental_score": round(elemental_score, 3),
        "incoming_element_multiplier": round(incoming_element_multiplier, 3),
        "outgoing_element_multiplier": round(outgoing_element_multiplier, 3),
        "elemental_threat": is_elemental_threat(candidate, foe) if foe else False,
        "has_weakness_cover": has_usable_weakness_cover(candidate),
        "best_damage": round(best_damage, 3),
        "reason": reason,
    }


def switch_score(candidate: dict[str, Any], foe: dict[str, Any]) -> float:
    return float(switch_evaluation(candidate, foe).get("score", -999.0) or -999.0)


def elemental_matchup_score(candidate: dict[str, Any], foe: dict[str, Any]) -> float:
    if not candidate or not foe:
        return 0.0
    best_offense = 1.0
    for ability in abilities_for(candidate):
        if not isinstance(ability, dict) or not is_usable_ability(ability):
            continue
        effective = effective_ability(ability, candidate)
        if not is_damage_ability(effective):
            continue
        best_offense = max(best_offense, element_multiplier(str(effective.get("element", "")), foe))
    worst_defense = 1.0
    for ability in abilities_for(foe):
        if not isinstance(ability, dict) or not is_usable_ability(ability):
            continue
        effective = effective_ability(ability, foe)
        if not is_damage_ability(effective):
            continue
        worst_defense = max(worst_defense, element_multiplier(str(effective.get("element", "")), candidate))
    score = 0.0
    if best_offense > 1.0:
        score += 34.0
    elif best_offense < 1.0:
        score -= 18.0
    if worst_defense > 1.0:
        score -= 42.0
    elif worst_defense < 1.0:
        score += 18.0
    return score


def best_element_multiplier(attacker: dict[str, Any], defender: dict[str, Any]) -> float:
    if not attacker or not defender:
        return 1.0
    best = 1.0
    for ability in abilities_for(attacker):
        if not isinstance(ability, dict) or not is_usable_ability(ability):
            continue
        effective = effective_ability(ability, attacker)
        if not is_damage_ability(effective):
            continue
        best = max(best, element_multiplier(str(effective.get("element", "")), defender))
    if best > 1.0:
        return best
    for element in miscrit_elements(attacker):
        best = max(best, element_multiplier(element, defender))
    return best


def estimate_incoming_damage(attacker: dict[str, Any], defender: dict[str, Any]) -> float:
    if not attacker or not defender:
        return 0.0
    best = 0.0
    for ability in abilities_for(attacker):
        if not isinstance(ability, dict) or not is_usable_ability(ability):
            continue
        best = max(best, estimate_ability_damage(effective_ability(ability, attacker), attacker, defender) * accuracy_factor(ability))
    if best <= 0.0:
        offense = max(stat_value(attacker, "pa"), stat_value(attacker, "ea"))
        defense = max(1.0, max(stat_value(defender, "pd"), stat_value(defender, "ed")))
        best = max(4.0, offense / defense * 20.0)
    return best


def effective_ability(ability: dict[str, Any], miscrit: dict[str, Any]) -> dict[str, Any]:
    result = dict(ability)
    enchants = miscrit.get("enchants", [])
    ability_id = int(ability.get("id", 0) or 0)
    enchanted = False
    if isinstance(enchants, list):
        enchanted = ability_id in {int(item) for item in enchants if str(item).lstrip("-").isdigit()}
    elif isinstance(enchants, dict):
        enchanted = str(ability_id) in enchants or ability_id in enchants
    if enchanted:
        enchant = ability.get("enchant", {})
        if isinstance(enchant, dict):
            for key, value in enchant.items():
                if key in {"ap", "accuracy", "times", "turns"}:
                    result[key] = float(result.get(key, 0) or 0) + float(value or 0)
                elif key == "additional" and isinstance(value, list):
                    base = result.get("additional", [])
                    result["additional"] = (base if isinstance(base, list) else []) + value
    return result


def abilities_for(miscrit: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("curr_abilities", "abilities"):
        if key not in miscrit:
            continue
        value = miscrit.get(key, [])
        if isinstance(value, list) and value:
            return [item for item in value if isinstance(item, dict)]
    return []


def find_ability(miscrit: dict[str, Any], ability_id: int) -> dict[str, Any]:
    for ability in abilities_for(miscrit):
        if int(ability.get("id", 0) or 0) == ability_id:
            return ability
    return {}


def next_recharge(ability: dict[str, Any], pvp: bool, miss: bool) -> int:
    cooldown = int(ability.get("cooldown", ability.get("og_cooldown", 0)) or 0)
    kind = str(ability.get("type", ""))
    if cooldown == -1:
        return -1 if not miss else 0
    if pvp and kind in SPECIAL_COOLDOWNS:
        return SPECIAL_COOLDOWNS[kind][1 if miss else 0]
    if pvp and cooldown == 0:
        cooldown = 2
    return cooldown if not miss else 0


def is_dead(miscrit: dict[str, Any]) -> bool:
    if "chp" in miscrit:
        return int(miscrit.get("chp", 0) or 0) <= 0
    if "c" in miscrit:
        return int(miscrit.get("c", 0) or 0) <= 0
    return False


def is_usable_ability(ability: dict[str, Any]) -> bool:
    if int(ability.get("recharge", ability.get("cooldown_remaining", ability.get("cd", 0))) or 0) != 0:
        return False
    if bool(ability.get("disabled", False)):
        return False
    return int(ability.get("id", 0) or 0) > 0


def current_hp(miscrit: dict[str, Any]) -> int:
    for key in ("chp", "c", "hp"):
        if key in miscrit:
            return max(0, int(miscrit.get(key, 0) or 0))
    return max(0, battle_max_hp(miscrit))


def max_hp(miscrit: dict[str, Any]) -> int:
    value = battle_max_hp(miscrit)
    if value > 0:
        return value
    fallback = 0
    for key in ("chp", "c", "hp"):
        if key in miscrit:
            fallback = int(miscrit.get(key, 0) or 0)
            break
    return max(1, fallback)


def hp_ratio(miscrit: dict[str, Any]) -> float:
    return clamp(float(current_hp(miscrit)) / max(1.0, float(max_hp(miscrit))), 0.0, 1.0)


def stat_value(miscrit: dict[str, Any], key: str) -> float:
    stats = miscrit.get("stats", {}) if isinstance(miscrit.get("stats"), dict) else {}
    direct = miscrit.get(key, 0)
    return float(stats.get(key, direct) or 0.0)


def normalized_mid(miscrit: dict[str, Any]) -> int:
    return int(miscrit.get("mid", miscrit.get("mId", miscrit.get("m", 0))) or 0)


def miscrit_name(miscrit: dict[str, Any]) -> str:
    mid = normalized_mid(miscrit)
    return str(miscrit.get("name", miscrit.get("miscrit", f"#{mid or miscrit.get('id', '?')}")))


def accuracy_factor(ability: dict[str, Any]) -> float:
    accuracy = float(ability.get("accuracy", 100) or 100)
    if accuracy == 170:
        return 1.0
    return clamp(accuracy / 100.0, 0.08, 1.35)


def ability_crit_factor(ability: dict[str, Any]) -> float:
    for key in ("crit", "critical", "crit_chance", "critical_chance"):
        if key not in ability:
            continue
        value = ability.get(key)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        try:
            number = float(value or 0)
        except (TypeError, ValueError):
            continue
        if number > 1.0:
            return clamp(number / 100.0, 0.0, 1.0)
        return clamp(number, 0.0, 1.0)
    return 0.0


def element_multiplier(attack_element: str, defender: dict[str, Any]) -> float:
    element = str(attack_element or "").strip()
    if element in {"", "Misc", "Physical"}:
        return 1.0
    defender_elements = miscrit_elements(defender)
    if any(ELEMENT_WEAKNESS.get(defender_element, "") == element for defender_element in defender_elements):
        return 1.0 if elemental_weakness_removed(defender) else 2.0
    if ELEMENT_WEAKNESS.get(element, "") in defender_elements:
        return 0.5
    return 1.0


def miscrit_elements(miscrit: dict[str, Any]) -> list[str]:
    raw = ""
    metadata = miscrit.get("metadata", {})
    if isinstance(metadata, dict):
        raw = str(metadata.get("element", ""))
    raw = str(miscrit.get("element", raw) or raw)
    raw = raw.replace("\\", "/").replace("|", "/").replace(",", "/")
    parts: list[str] = []
    for chunk in (part.strip() for part in raw.split("/") if part.strip()):
        if chunk in BASE_ELEMENTS or chunk == "Physical":
            parts.append(chunk)
            continue
        remaining = chunk
        split_parts: list[str] = []
        for element in BASE_ELEMENTS:
            if remaining.startswith(element):
                split_parts.append(element)
                remaining = remaining[len(element) :]
        if split_parts and not remaining:
            parts.extend(split_parts)
        else:
            parts.append(chunk)
    return parts or ["Physical"]


def is_elemental_threat(curr: dict[str, Any], opp: dict[str, Any]) -> bool:
    if elemental_weakness_removed(curr):
        return False
    for ability in abilities_for(opp):
        if not isinstance(ability, dict):
            continue
        elements = [str(ability.get("element", ""))]
        for item in ability.get("additional", []) if isinstance(ability.get("additional", []), list) else []:
            if isinstance(item, dict) and str(item.get("element", "Misc")) != "Misc":
                elements.append(str(item.get("element", "")))
        for element in elements:
            if element_multiplier(element, curr) > 1.0:
                return True
    return any(ELEMENT_WEAKNESS.get(element, "") in miscrit_elements(opp) for element in miscrit_elements(curr))


def has_super_effective_ability(attacker: dict[str, Any], defender: dict[str, Any]) -> bool:
    return any(element_multiplier(str(ability.get("element", "")), defender) > 1.0 for ability in abilities_for(attacker))


def removes_elemental_weakness(ability: dict[str, Any]) -> bool:
    text = " ".join(str(ability.get(key, "") or "") for key in ("name", "desc", "description", "tooltip", "effect")).casefold()
    if "elemental weakness" not in text:
        return False
    return "remove" in text or "removes" in text


def elemental_weakness_removed(miscrit: dict[str, Any]) -> bool:
    return bool(miscrit.get("negated", False)) or has_any_effect(miscrit, ["elemental_weakness_removed", "weakness_removed", "negate"])


def has_usable_weakness_cover(miscrit: dict[str, Any]) -> bool:
    if elemental_weakness_removed(miscrit):
        return True
    return any(
        isinstance(ability, dict)
        and is_usable_ability(ability)
        and ability_targets_self(ability)
        and removes_elemental_weakness(effective_ability(ability, miscrit))
        for ability in abilities_for(miscrit)
    )


def estimate_incoming_damage_with_weakness_removed(attacker: dict[str, Any], defender: dict[str, Any]) -> float:
    protected = dict(defender)
    effects = protected.get("_ai_effects", {})
    protected["_ai_effects"] = dict(effects) if isinstance(effects, dict) else {}
    add_elemental_weakness_removed(protected, 2)
    return estimate_incoming_damage(attacker, protected)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def ability_power(ability: dict[str, Any]) -> float:
    kind = str(ability.get("type", "")).lower()
    ap = abs(float(ability.get("ap", ability.get("power", 0)) or 0))
    accuracy = float(ability.get("accuracy", 100) or 100) / 100.0
    score = ap * max(0.25, min(1.0, accuracy))
    if kind == "attack":
        score += 30.0
    elif kind in {"buff", "bot", "negate", "block", "ethereal"}:
        score += 16.0
    elif kind in {"bleed", "dot", "poison", "disease"}:
        score += 12.0
    score += len(ability.get("additional", []) or []) * 4.0
    return score


def miscrit_power(miscrit: dict[str, Any]) -> float:
    stats = miscrit.get("stats", {}) if isinstance(miscrit.get("stats"), dict) else {}
    if stats:
        return sum(float(stats.get(key, 0) or 0) for key in ("hp", "spd", "ea", "pa", "ed", "pd"))
    return sum(float(miscrit.get(key, 0) or 0) for key in ("hp", "spd", "ea", "pa", "ed", "pd"))


def compact_player(player: dict[str, Any]) -> dict[str, Any]:
    team = player.get("miscrits", [])
    if not isinstance(team, list):
        team = []
    return {
        "user_id": player.get("user_id", ""),
        "username": player.get("username", player.get("name", "")),
        "team": [compact_battle_miscrit(item, index == 0) for index, item in enumerate(team) if isinstance(item, dict)],
    }


def compact_battle_miscrit(miscrit: dict[str, Any], active: bool) -> dict[str, Any]:
    mid = int(miscrit.get("mid", miscrit.get("m", miscrit.get("mId", 0))) or 0)
    max_hp = battle_max_hp(miscrit)
    chp = int(miscrit.get("chp", miscrit.get("c", miscrit.get("hp", max_hp))) or 0)
    evo = battle_evo(miscrit)
    return {
        "id": int(miscrit.get("id", 0) or 0),
        "mid": mid,
        "evo": evo,
        "name": str(miscrit.get("name", miscrit.get("miscrit", f"#{mid or miscrit.get('id', '?')}"))),
        "level": miscrit.get("level", miscrit.get("l")),
        "rating": miscrit.get("rating", ""),
        "element": miscrit.get("element", ""),
        "chp": max(0, chp),
        "max_hp": max(0, max_hp),
        "active": active,
        "dead": max(0, chp) <= 0,
        "statuses": compact_statuses(miscrit),
    }


def compact_statuses(miscrit: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    ai_effects = miscrit.get("_ai_effects", {})
    known: set[str] = set()
    expired = expired_effect_keys(miscrit)
    if isinstance(ai_effects, dict):
        for key, value in ai_effects.items():
            if value:
                row = {"key": str(key), "turns": int(value) if isinstance(value, int | float) else value}
                if status_key(str(key)) == "block":
                    ap = block_amount(miscrit)
                    if ap > 0:
                        row["ap"] = ap
                out.append(row)
                known.add(status_key(str(key)))
    for key, value, payload in raw_status_entries(miscrit):
        normalized = status_key(key)
        if normalized and normalized not in known and normalized not in expired and value:
            row = {"key": str(key), "turns": int(value) if isinstance(value, int | float) else value}
            if normalized == "block":
                ap = block_value_from_action(payload)
                if ap > 0:
                    row["ap"] = ap
            out.append(row)
            known.add(normalized)
    return out[:8]


def battle_max_hp(miscrit: dict[str, Any]) -> int:
    stats = miscrit.get("stats", {}) if isinstance(miscrit.get("stats"), dict) else {}
    value = stats.get("hp", miscrit.get("max_hp", miscrit.get("mh", miscrit.get("h", miscrit.get("hp", 0)))))
    return int(value or 0)


def battle_evo(miscrit: dict[str, Any]) -> int:
    explicit = miscrit.get("evo", miscrit.get("evo_id", miscrit.get("evolution")))
    try:
        if explicit is not None and str(explicit) != "":
            return max(0, min(3, int(explicit)))
    except (TypeError, ValueError):
        pass
    try:
        level = int(miscrit.get("level", miscrit.get("l", 1)) or 1)
    except (TypeError, ValueError):
        level = 1
    return max(0, min(3, level // 10))
