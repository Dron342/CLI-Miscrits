from __future__ import annotations

import base64
import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any

from .battle_ai import BATTLE_OPCODES, LEVEL_KEYS, RANDOM_OPCODES, BattleState, effective_ability
from .battle_learning import BattleRecorder, load_weights
from .breeding import evo_from_level, metadata_by_mid, normalize_player
from .data_cache import DataCache
from .nakama import MiscritsClient, MiscritsError
from .realtime import NakamaRealtime, decode_match_data


ARENA_MODES = {
    "battle": "Default",
    "default": "Default",
    "random": "Random",
    "platinum": "Platinum",
    "daily": "Daily",
}

RATING_POINTS = {
    "Legendary": 5,
    "Exotic": 4,
    "Epic": 3,
    "Rare": 2,
    "Common": 1,
}

SPECIES_STAT_POINTS = {
    "Weak": 1.0,
    "Moderate": 2.0,
    "Strong": 3.0,
    "Elite": 4.0,
    "Max": 5.0,
}

MAX_LEVEL = 35
RANDOM_BAN_COUNT = 6
RANDOM_PICK_COUNT = 4
AUTO_ENCHANT_MAX_PER_PREPARE = 4
AUTO_ENCHANT_MIN_STRENGTH_SCORE = 18.0
AUTO_ENCHANT_GOLD_RESERVE = 1500
TEAM_EXPLORATION_RATE = 0.12
TEAM_EXPLORATION_GAP = 55.0
DRAFT_EXPLORATION_RATE = 0.16
DRAFT_EXPLORATION_GAP = 45.0
TEAM_POOL_LIMIT = 32
TEAM_POOL_CORE_LIMIT = 20

RECOVERABLE_REALTIME_MARKERS = (
    "timed out",
    "timed out waiting",
    "websocket handshake failed: empty response",
    "websocket handshake failed: http 401 unauthorized",
    "realtime socket closed by server",
    "server-side session disconnect",
)


@dataclass(frozen=True)
class ArenaRunConfig:
    mode: str
    timeout_seconds: float = 300.0
    dry_run: bool = False
    max_turns: int = 150
    ready_retry_seconds: float = 2.0
    prepare: bool = True
    target_location_id: int = 1
    target_area_id: int = 2
    repeat_count: int = 1
    repeat_delay_seconds: float = 3.0
    stop_on_error: bool = True


class ArenaRunner:
    def __init__(self, client: MiscritsClient, progress: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.client = client
        self.progress = progress
        self.cache = DataCache(client)
        raw_miscrits = self.cache.get_json("miscrits.json", refresh_index=False) or []
        self.metadata = metadata_by_mid(raw_miscrits)
        self.battle_metadata = battle_metadata_by_mid(raw_miscrits)

    def status(self, mode: str = "battle") -> dict[str, Any]:
        mode_name = canonical_mode(mode)
        player_result = self.client.get_player()
        if not player_result.success:
            return {"ok": False, "error": player_result.raw}
        player = normalize_player(player_result.data)
        enrich_team(player, self.metadata)
        player["location"] = normalize_location(player_result.data)
        arena_info = self._arena_info(mode_name)
        validation = validate_arena_mode(mode_name, player, arena_info)
        return {
            "ok": True,
            "mode": mode_name,
            "validation": validation,
            "location": player.get("location", {}),
            "location_valid": is_battle_arena_location(player),
            "recommended_team": self.recommend_team(mode_name, player),
            "team": [compact_miscrit(item) for item in team_items(player)],
            "team_rating": team_rating_points(player),
            "arena_info": arena_info,
            "reward_progress": arena_reward_progress(mode_name, arena_info, self.metadata),
            "rules": arena_rules(mode_name),
        }

    def run(self, config: ArenaRunConfig) -> dict[str, Any]:
        mode_name = canonical_mode(config.mode)
        self._emit("start", phase="preparing", mode=mode_name)
        prepare_log: list[dict[str, Any]] = []
        if config.prepare:
            prepare_result = self.prepare_for_battle(mode_name, config, prepare_log, apply_changes=not config.dry_run)
            if not prepare_result.get("ok"):
                self._emit("prepare_failed", phase="error", mode=mode_name, prepare=prepare_result)
                return {"ok": False, "mode": mode_name, "prepare": prepare_result, "log": prepare_log}
        status = self.status(mode_name)
        if not status.get("ok"):
            self._emit("status_failed", phase="error", mode=mode_name, status=status)
            return status
        self._emit("status_ready", phase="ready", mode=mode_name, status=status)
        if config.dry_run:
            self._emit("dry_run_ready", phase="ready", mode=mode_name, status=status)
            return {"ok": True, "dry_run": True, "mode": mode_name, "status": status, "prepare_log": prepare_log}
        if not status["validation"]["ok"]:
            self._emit("validation_failed", phase="error", mode=mode_name, validation=status["validation"])
            return {
                "ok": False,
                "mode": mode_name,
                "error": status["validation"],
                "rules": status["rules"],
                "prepare_log": prepare_log,
            }

        self.client.ensure_realtime_session(force_login=True)
        user_id = token_user_id(str(self.client.session.get("token", "")))
        log: list[dict[str, Any]] = []
        try:
            battle = self._run_realtime_queue(mode_name, user_id, log, config)
        except MiscritsError as exc:
            if not is_recoverable_arena_error(exc):
                self._emit("error", phase="error", mode=mode_name, error=str(exc))
                return {
                    "ok": False,
                    "mode": mode_name,
                    "error": str(exc),
                    "prepare_log": prepare_log,
                    "log": log,
                }
            if not should_relogin_for_realtime_error(exc):
                return self._recoverable_result(mode_name, prepare_log, log, user_id, exc, "realtime_disconnect")
            self.client.login_saved()
            user_id = token_user_id(str(self.client.session.get("token", "")))
            self._log(log, "realtime_relogin_retry", phase="connecting", reason=str(exc))
            try:
                battle = self._run_realtime_queue(mode_name, user_id, log, config)
            except MiscritsError as retry_exc:
                if is_recoverable_arena_error(retry_exc):
                    return self._recoverable_result(mode_name, prepare_log, log, user_id, retry_exc, "realtime_disconnect")
                self._emit("error", phase="error", mode=mode_name, error=str(retry_exc))
                return {"ok": False, "mode": mode_name, "error": str(retry_exc), "prepare_log": prepare_log, "log": log}
            except (TimeoutError, OSError) as retry_exc:
                return self._recoverable_result(mode_name, prepare_log, log, user_id, retry_exc, "timeout")
        except (TimeoutError, OSError) as exc:
            return self._recoverable_result(mode_name, prepare_log, log, user_id, exc, "timeout")
        result = {"ok": True, "mode": mode_name, "battle": battle, "prepare_log": prepare_log, "log": log}
        self._emit("complete", phase="finished", mode=mode_name, result=result)
        return result

    def run_loop(
        self,
        config: ArenaRunConfig,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        repeat_count = int(config.repeat_count)
        infinite = repeat_count <= 0
        max_runs = None if infinite else max(1, repeat_count)
        single_config = replace(config, repeat_count=1)
        results: list[dict[str, Any]] = []
        stopped = False
        failed = False
        index = 0
        mode_name = canonical_mode(config.mode)

        while infinite or index < int(max_runs or 0):
            if should_stop and should_stop():
                stopped = True
                break
            index += 1
            self._emit(
                "loop_start",
                phase="preparing",
                mode=mode_name,
                loop={"current": index, "target": 0 if infinite else max_runs, "infinite": infinite, "completed": len(results)},
            )
            try:
                result = self.run(single_config)
            except Exception as exc:
                result = {"ok": False, "mode": mode_name, "error": str(exc) or exc.__class__.__name__, "exception_type": exc.__class__.__name__}
                self._emit(
                    "loop_battle_exception",
                    phase="error",
                    mode=mode_name,
                    loop={"current": index, "target": 0 if infinite else max_runs, "infinite": infinite, "completed": len(results)},
                    error=result["error"],
                    exception_type=result["exception_type"],
                )
            results.append(result)
            recoverable = bool(result.get("recoverable"))
            failed = failed or (not bool(result.get("ok")) and not recoverable)
            self._emit(
                "loop_battle_complete",
                phase="recovering" if recoverable else "finished",
                mode=mode_name,
                loop={"current": index, "target": 0 if infinite else max_runs, "infinite": infinite, "completed": len(results)},
                result=result,
            )
            if failed and config.stop_on_error:
                break
            if should_stop and should_stop():
                stopped = True
                break
            if not infinite and index >= int(max_runs or 0):
                break
            if config.repeat_delay_seconds > 0:
                self._emit(
                    "loop_cooldown",
                    phase="cooldown",
                    mode=mode_name,
                    loop={
                        "current": index + 1,
                        "target": 0 if infinite else max_runs,
                        "infinite": infinite,
                        "completed": len(results),
                        "delay_seconds": config.repeat_delay_seconds,
                    },
                )
                deadline = time.monotonic() + config.repeat_delay_seconds
                while time.monotonic() < deadline:
                    if should_stop and should_stop():
                        stopped = True
                        break
                    time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
                if stopped:
                    break

        ok = bool(results) and not failed
        summary = summarize_loop_results(results)
        final_phase = "stopped" if stopped else "finished"
        payload = {
            "ok": ok,
            "mode": mode_name,
            "stopped": stopped,
            "failed": failed,
            "loop": {
                "requested": 0 if infinite else max_runs,
                "infinite": infinite,
                "completed": len(results),
                "summary": summary,
            },
            "battles": results,
        }
        self._emit("loop_complete", phase=final_phase, mode=mode_name, loop=payload["loop"], result=payload)
        return payload

    def _run_realtime_queue(
        self,
        mode_name: str,
        user_id: str,
        log: list[dict[str, Any]],
        config: ArenaRunConfig,
    ) -> dict[str, Any]:
        rt = NakamaRealtime(str(self.client.session["token"]), self.client.config)
        self._log(
            log,
            "socket_connecting",
            phase="connecting",
            scheme=self.client.config.socket_scheme,
            host=self.client.config.socket_host,
            port=self.client.config.socket_port,
        )
        rt.connect()
        self._log(log, "socket_connected", phase="connecting")
        try:
            active_match_id = self._current_active_battle_id(log)
            if active_match_id:
                self._log(log, "active_battle_found", phase="battle", match_id=active_match_id)
                try:
                    joined = rt.join_match(match_id=active_match_id)
                    match_id = str(joined.get("match_id", active_match_id) or active_match_id)
                    self._log(log, "active_battle_joined", phase="battle", match_id=match_id)
                    return self._run_battle(rt, match_id, user_id, log, config)
                except MiscritsError as exc:
                    self._log(log, "active_battle_join_failed", phase="searching", match_id=active_match_id, error=str(exc))
            self._log(log, "matchmaker_add_send", phase="searching", mode=mode_name)
            ticket = rt.add_matchmaker(mode_name)
            self._log(log, "queue_started", phase="searching", mode=mode_name, ticket=ticket.get("ticket"))
            self._log(log, "matchmaker_wait", phase="searching", timeout_seconds=config.timeout_seconds)
            matched = rt.wait_for("matchmaker_matched", config.timeout_seconds)
            self._log(
                log,
                "matched",
                phase="draft" if mode_name == "Random" else "battle",
                match_id=matched.get("match_id"),
                has_token=bool(matched.get("token")),
            )
            if mode_name == "Random":
                match_id = self._complete_random_stage(rt, matched, log, config.timeout_seconds)
            else:
                match_id = str(matched.get("match_id", ""))
                self._log(log, "match_join_send", phase="battle", match_id=match_id, has_token=bool(matched.get("token")))
                joined = rt.join_match(match_id=match_id or None, token=matched.get("token"))
                match_id = match_id or str(joined.get("match_id", ""))
                self._log(log, "match_joined", phase="battle", match_id=match_id)
            return self._run_battle(rt, match_id, user_id, log, config)
        finally:
            rt.close()

    def _recoverable_result(
        self,
        mode_name: str,
        prepare_log: list[dict[str, Any]],
        log: list[dict[str, Any]],
        user_id: str,
        exc: BaseException,
        reason: str,
    ) -> dict[str, Any]:
        error = str(exc) or exc.__class__.__name__
        snapshot = last_battle_snapshot(log)
        if snapshot:
            outcome = battle_outcome("", user_id, snapshot)
            if outcome in {"victory", "defeat"}:
                battle = {
                    "ended": False,
                    "winner": "",
                    "outcome": outcome,
                    "turns_sent": count_sent_turns(log),
                    "reason": reason,
                    "recoverable": True,
                }
                self._emit("recoverable_battle_disconnect", phase="finished", mode=mode_name, error=error, outcome=outcome, battle=snapshot)
                return {"ok": True, "mode": mode_name, "battle": battle, "recoverable": True, "error": error, "prepare_log": prepare_log, "log": log}
        self._emit("recoverable_realtime_error", phase="recovering", mode=mode_name, error=error, reason=reason, log=log)
        return {
            "ok": False,
            "mode": mode_name,
            "recoverable": True,
            "reason": reason,
            "error": error,
            "prepare_log": prepare_log,
            "log": log,
        }

    def prepare_for_battle(
        self,
        mode_name: str,
        config: ArenaRunConfig,
        log: list[dict[str, Any]],
        apply_changes: bool,
    ) -> dict[str, Any]:
        player_result = self.client.get_player()
        if not player_result.success:
            return {"ok": False, "reason": "get_player_failed", "raw": player_result.raw}
        player = normalize_player(player_result.data)
        enrich_team(player, self.metadata)
        player["location"] = normalize_location(player_result.data)

        if not is_battle_arena_location(player):
            log.append(
                {
                    "event": "move_to_arena",
                    "from": player["location"],
                    "to": {"location_id": config.target_location_id, "area_id": config.target_area_id},
                    "applied": apply_changes,
                }
            )
            if apply_changes:
                moved = self.client.update_location(config.target_location_id, config.target_area_id)
                if not moved.success:
                    return {"ok": False, "reason": "update_location_failed", "raw": moved.raw}
                player_result = self.client.get_player()
                if not player_result.success:
                    return {"ok": False, "reason": "get_player_after_location_failed", "raw": player_result.raw}
                player = normalize_player(player_result.data)
                enrich_team(player, self.metadata)
                player["location"] = normalize_location(player_result.data)

        if mode_name != "Random":
            recommendation = self.recommend_team(mode_name, player, explore=apply_changes)
            if not recommendation.get("ok"):
                return {"ok": False, "reason": "team_build_failed", "team_result": recommendation}
            desired_team = [int(item) for item in recommendation["team"]]
            updated_player = self.apply_team_if_needed(
                player,
                desired_team,
                recommendation,
                log,
                apply_changes,
                reason="best_team",
            )
            if not updated_player.get("ok"):
                return updated_player
            player = updated_player["player"]

            log.append({"event": "heal_team", "reason": "pre_queue_full_heal", "applied": apply_changes})
            if apply_changes:
                healed = self.client.heal_team()
                if not healed.success:
                    fallback = self.recommend_team(mode_name, player, explore=False, alive_only=True)
                    if not fallback.get("ok"):
                        return {
                            "ok": False,
                            "reason": "heal_team_failed",
                            "raw": healed.raw,
                            "fallback_team": fallback,
                        }
                    fallback_team = [int(item) for item in fallback["team"]]
                    log.append(
                        {
                            "event": "heal_team_unavailable",
                            "reason": "fallback_alive_team",
                            "raw": healed.raw,
                            "fallback_team": fallback_team,
                        }
                    )
                    updated_player = self.apply_team_if_needed(
                        player,
                        fallback_team,
                        fallback,
                        log,
                        apply_changes,
                        reason="alive_fallback_team",
                    )
                    if not updated_player.get("ok"):
                        return updated_player
                    player = updated_player["player"]
                    desired_team = fallback_team
                else:
                    player_result = self.client.get_player()
                    if not player_result.success:
                        return {"ok": False, "reason": "get_player_after_heal_failed", "raw": player_result.raw}
                    player = normalize_player(player_result.data)
                    enrich_team(player, self.metadata)
                    player["location"] = normalize_location(player_result.data)

            enchant_result = self.enchant_recommended_team(player, desired_team, mode_name, log, apply_changes)
            if not enchant_result.get("ok"):
                return {"ok": False, "reason": "enchant_failed", "enchant": enchant_result}
        else:
            if team_needs_heal(player):
                log.append({"event": "skip_heal_random", "reason": "random_uses_draft_team"})

        return {"ok": True}

    def enchant_recommended_team(
        self,
        player: dict[str, Any],
        team_ids: list[int],
        mode_name: str,
        log: list[dict[str, Any]],
        apply_changes: bool,
    ) -> dict[str, Any]:
        if mode_name == "Random" or not team_ids:
            return {"ok": True, "reason": "not_needed"}
        upgraded = []
        skipped_reason = ""
        for _ in range(AUTO_ENCHANT_MAX_PER_PREPARE):
            candidate = pick_best_gold_enchant_candidate(player, team_ids, mode_name)
            if not candidate.get("ok"):
                skipped_reason = str(candidate.get("reason", "none"))
                break
            log.append({"event": "enchant_ability", **candidate, "applied": apply_changes})
            if apply_changes:
                try:
                    result = self.client.enchant_ability(int(candidate["miscrit_id"]), int(candidate["ability_id"]), "gold")
                except MiscritsError as exc:
                    log.append({"event": "enchant_ability_failed", **candidate, "error": str(exc), "applied": True})
                    return {"ok": True, "upgraded": len(upgraded), "reason": "enchant_rpc_failed", "failed": candidate}
                if not result.success:
                    log.append({"event": "enchant_ability_failed", **candidate, "raw": result.raw, "applied": True})
                    return {"ok": True, "upgraded": len(upgraded), "reason": "enchant_rpc_failed", "failed": candidate}
                apply_local_enchant(player, int(candidate["miscrit_id"]), int(candidate["ability_id"]), int(candidate["gold_price"]))
            upgraded.append(candidate)
        if not upgraded:
            log.append({"event": "auto_enchant_skip", "reason": skipped_reason or "no_candidate", "applied": False})
        return {"ok": True, "upgraded": len(upgraded), "reason": skipped_reason or "done"}

    def apply_team_if_needed(
        self,
        player: dict[str, Any],
        desired_team: list[int],
        recommendation: dict[str, Any],
        log: list[dict[str, Any]],
        apply_changes: bool,
        *,
        reason: str,
    ) -> dict[str, Any]:
        if desired_team == player.get("team", []):
            return {"ok": True, "player": player}
        log.append(
            {
                "event": "update_team",
                "reason": reason,
                "from": player.get("team", []),
                "to": desired_team,
                "score": recommendation.get("score"),
                "applied": apply_changes,
            }
        )
        if not apply_changes:
            dry_run_player = dict(player)
            dry_run_player["team"] = list(desired_team)
            return {"ok": True, "player": dry_run_player}
        updated = self.client.update_team(desired_team)
        if not updated.success:
            return {"ok": False, "reason": "update_team_failed", "raw": updated.raw}
        player_result = self.client.get_player()
        if not player_result.success:
            return {"ok": False, "reason": "get_player_after_team_failed", "raw": player_result.raw}
        fresh_player = normalize_player(player_result.data)
        enrich_team(fresh_player, self.metadata)
        fresh_player["location"] = normalize_location(player_result.data)
        return {"ok": True, "player": fresh_player}

    def recommend_team(
        self,
        mode_name: str,
        player: dict[str, Any],
        explore: bool = False,
        alive_only: bool = False,
    ) -> dict[str, Any]:
        if mode_name == "Random":
            return {"ok": True, "team": player.get("team", []), "reason": "random_uses_draft_team"}
        candidates = [
            item
            for item in player.get("miscrits", [])
            if team_candidate_eligible(item, mode_name, alive_only=alive_only)
        ]
        if len(candidates) < 4:
            return {
                "ok": False,
                "reason": "not_enough_alive_miscrits" if alive_only else "not_enough_eligible_miscrits",
                "count": len(candidates),
            }
        profiles = [team_candidate_profile(item, mode_name) for item in candidates]
        profiles.sort(key=lambda item: float(item["base_score"]), reverse=True)
        pool = select_team_candidate_pool(profiles)
        profile_by_id = {int(profile["id"]): profile for profile in pool}
        best: dict[str, Any] | None = None
        ranked_combos: list[dict[str, Any]] = []
        for combo in combinations(pool, 4):
            evaluation = evaluate_team_combo(list(combo), mode_name, include_members=False)
            if not evaluation.get("ok"):
                continue
            ranked_combos.append(evaluation)
            if best is None or float(evaluation["score"]) > float(best["score"]):
                best = evaluation
        if not best:
            return {"ok": False, "reason": "no_valid_team_combo"}
        if explore:
            ranked_combos.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
            explored = exploratory_team_choice(ranked_combos)
            if explored:
                explored = evaluate_team_combo(
                    [profile_by_id[int(item)] for item in explored.get("team", []) if int(item) in profile_by_id],
                    mode_name,
                )
                explored["exploration"] = True
                explored["best_score"] = best.get("score")
                explored["reason"] = "explore_team_combo"
                return explored
        return evaluate_team_combo(
            [profile_by_id[int(item)] for item in best.get("team", []) if int(item) in profile_by_id],
            mode_name,
        )

    def _arena_info(self, mode_name: str) -> dict[str, Any]:
        rpc_kind = {
            "Default": "battle",
            "Random": "random",
            "Platinum": "platinum",
            "Daily": "daily",
        }.get(mode_name)
        if not rpc_kind:
            return {}
        result = self.client.get_arena(rpc_kind)
        return result.data if result.success and isinstance(result.data, dict) else {}

    def _complete_random_stage(
        self,
        rt: NakamaRealtime,
        matched: dict[str, Any],
        log: list[dict[str, Any]],
        timeout_seconds: float,
    ) -> str:
        stage_match_id = str(matched.get("match_id", ""))
        self._log(log, "random_stage_join_send", phase="draft", match_id=stage_match_id, has_token=bool(matched.get("token")))
        rt.join_match(match_id=stage_match_id or None, token=matched.get("token"))
        self._log(log, "random_stage_joined", phase="draft", match_id=stage_match_id)
        self._log(log, "random_ready_send", phase="draft")
        rt.send_match_state(stage_match_id, RANDOM_OPCODES["READY"], {})
        pool: list[int] = []
        banned: set[int] = set()
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            state = decode_match_data(rt.wait_for("match_data", max(1.0, deadline - time.monotonic())))
            opcode = state["op_code"]
            data = state["data"]
            if opcode == RANDOM_OPCODES["START"]:
                pool = [int(item) for item in data] if isinstance(data, list) else []
                bans = choose_random_bans(pool, self.metadata)
                self._log(
                    log,
                    "random_ban",
                    phase="draft",
                    draft={"pool": species_cards(pool, self.metadata), "bans": species_cards(bans, self.metadata), "picks": []},
                )
                rt.send_match_state(stage_match_id, RANDOM_OPCODES["BAN"], {"bans": bans})
            elif opcode == RANDOM_OPCODES["UPDATE"]:
                banned = {int(item) for item in data} if isinstance(data, list) else set()
                picks = choose_random_picks([item for item in pool if item not in banned], self.metadata)
                self._log(
                    log,
                    "random_pick",
                    phase="draft",
                    draft={
                        "pool": species_cards(pool, self.metadata),
                        "bans": species_cards(sorted(banned), self.metadata),
                        "picks": species_cards(picks, self.metadata),
                    },
                )
                rt.send_match_state(stage_match_id, RANDOM_OPCODES["PICK"], {"picks": picks})
            elif opcode == RANDOM_OPCODES["PREPARE"]:
                self._log(log, "random_prepare", phase="draft", draft={"prepare": data})
            elif opcode == RANDOM_OPCODES["MATCH"]:
                match_id = str(data.get("matchId", "")) if isinstance(data, dict) else ""
                if not match_id:
                    raise MiscritsError("Random Arena did not return a battle match id.")
                self._log(log, "random_battle_match", phase="battle", match_id=match_id)
                rt.join_match(match_id=match_id)
                return match_id
            elif opcode == RANDOM_OPCODES["TERMINATE"]:
                raise MiscritsError("Random Arena draft terminated by the server.")
        raise TimeoutError("Timed out during Random Arena draft.")

    def _run_battle(
        self,
        rt: NakamaRealtime,
        match_id: str,
        user_id: str,
        log: list[dict[str, Any]],
        config: ArenaRunConfig,
    ) -> dict[str, Any]:
        state_model = BattleState(user_id)
        recorder = BattleRecorder(mode=config.mode, match_id=match_id, player_id=user_id)
        deadline = time.monotonic() + config.timeout_seconds
        turns_sent = 0
        reconnects = 0
        while turns_sent <= config.max_turns:
            try:
                wait_seconds = max(1.0, min(30.0, deadline - time.monotonic()))
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for battle state.")
                state = decode_match_data(rt.wait_for("match_data", wait_seconds))
                opcode = int(state["op_code"])
                data = state["data"]
                damage_samples = state_model.damage_samples(opcode, data)
                state_model.apply(opcode, data)
                state_model.hydrate_metadata(self.battle_metadata)
                snapshot = enrich_battle_snapshot(state_model.snapshot(), self.metadata)
                recorder.acknowledge_decision(opcode, data)
                recorder.observe_state(opcode, data, snapshot)
                recorder.observe_damage_samples(damage_samples)
                self._log(log, "battle_state", phase="battle", opcode=opcode, data=summarize_battle_data(data), battle=snapshot)
                if opcode == BATTLE_OPCODES["END"]:
                    outcome = battle_outcome(state_model.winner, state_model.player_id, snapshot)
                    grade = recorder.finish(outcome, turns_sent, snapshot)
                    self._emit("battle_end", phase="finished", outcome=outcome, grade=grade, battle=snapshot)
                    return {"ended": True, "winner": state_model.winner, "outcome": outcome, "turns_sent": turns_sent, "grade": grade, "battle_log_id": recorder.battle_id}
                if opcode == BATTLE_OPCODES["TERMINATE"]:
                    outcome = battle_outcome(state_model.winner, state_model.player_id, snapshot)
                    grade = recorder.finish(outcome, turns_sent, snapshot)
                    self._emit("battle_terminate", phase="finished", outcome=outcome, grade=grade, battle=snapshot)
                    return {"terminated": True, "winner": state_model.winner, "outcome": outcome, "turns_sent": turns_sent, "grade": grade, "battle_log_id": recorder.battle_id}
                if opcode != BATTLE_OPCODES["READY"]:
                    rt.send_match_state(match_id, BATTLE_OPCODES["READY"], {})
                if state_model.should_act(data):
                    decision = state_model.decide()
                    sent = send_decision(rt, match_id, decision)
                    recorder.record_decision(decision, sent, snapshot)
                    self._log(log, "auto_decision", phase="battle", decision=decision, sent=sent, battle=snapshot)
                    if sent:
                        turns_sent += 1
            except (MiscritsError, TimeoutError, OSError) as exc:
                snapshot = enrich_battle_snapshot(state_model.snapshot(), self.metadata)
                outcome = battle_outcome(state_model.winner, state_model.player_id, snapshot)
                if not is_recoverable_arena_error(exc) or reconnects >= 8:
                    recorder.events.append({"timestamp": time.time(), "event": "interrupted", "turns": snapshot.get("turns", 0), "summary": {"error": str(exc) or exc.__class__.__name__}})
                    grade = recorder.finish(outcome, turns_sent, snapshot)
                    self._emit("battle_interrupted", phase="recovering", outcome=outcome, grade=grade, error=str(exc), battle=snapshot)
                    raise
                reconnects += 1
                self._log(
                    log,
                    "battle_socket_lost",
                    phase="recovering",
                    error=str(exc) or exc.__class__.__name__,
                    reconnect=reconnects,
                    match_id=match_id,
                    battle=snapshot,
                )
                try:
                    match_id = self._reconnect_active_battle(rt, match_id, user_id, log)
                    deadline = max(deadline, time.monotonic() + min(float(config.timeout_seconds), 120.0))
                except (MiscritsError, TimeoutError, OSError) as reconnect_exc:
                    recorder.events.append(
                        {
                            "timestamp": time.time(),
                            "event": "reconnect_failed",
                            "turns": snapshot.get("turns", 0),
                            "summary": {"error": str(reconnect_exc) or reconnect_exc.__class__.__name__},
                        }
                    )
                    grade = recorder.finish(outcome, turns_sent, snapshot)
                    self._emit("battle_interrupted", phase="recovering", outcome=outcome, grade=grade, error=str(reconnect_exc), battle=snapshot)
                    raise reconnect_exc
        snapshot = enrich_battle_snapshot(state_model.snapshot(), self.metadata)
        outcome = battle_outcome(state_model.winner, state_model.player_id, snapshot)
        grade = recorder.finish(outcome, turns_sent, snapshot)
        self._emit("battle_timeout", phase="finished", outcome=outcome, grade=grade, battle=snapshot)
        return {"ended": False, "winner": state_model.winner, "outcome": outcome, "turns_sent": turns_sent, "reason": "timeout_or_turn_limit", "grade": grade, "battle_log_id": recorder.battle_id}

    def _current_active_battle_id(self, log: list[dict[str, Any]] | None = None) -> str:
        try:
            result = self.client.get_player()
        except MiscritsError as exc:
            if log is not None:
                self._log(log, "active_battle_check_failed", phase="connecting", error=str(exc))
            return ""
        if not result.success or not isinstance(result.data, dict):
            if log is not None:
                self._log(log, "active_battle_check_failed", phase="connecting", raw=result.raw)
            return ""
        match_id = active_battle_match_id(result.data)
        if log is not None:
            self._log(log, "active_battle_check", phase="connecting", active=bool(match_id), match_id=match_id)
        return match_id

    def _reconnect_active_battle(
        self,
        rt: NakamaRealtime,
        current_match_id: str,
        user_id: str,
        log: list[dict[str, Any]],
    ) -> str:
        rt.close()
        self.client.ensure_realtime_session()
        user_id = token_user_id(str(self.client.session.get("token", ""))) or user_id
        rt.token = str(self.client.session["token"])
        active_match_id = self._current_active_battle_id(log)
        match_id = active_match_id or current_match_id
        if not match_id:
            raise MiscritsError("Cannot reconnect battle: no active battle match id.")
        self._log(log, "battle_reconnect_connecting", phase="recovering", match_id=match_id, active=bool(active_match_id), user_id=user_id)
        rt.connect()
        try:
            joined = rt.join_match(match_id=match_id)
        except MiscritsError:
            if not active_match_id or not current_match_id or current_match_id == match_id:
                raise
            self._log(log, "battle_rejoin_active_failed", phase="recovering", match_id=match_id, fallback_match_id=current_match_id)
            joined = rt.join_match(match_id=current_match_id)
            match_id = current_match_id
        joined_match_id = str(joined.get("match_id", match_id) or match_id)
        self._log(log, "battle_rejoined", phase="battle", match_id=joined_match_id, active=bool(active_match_id), user_id=user_id)
        return joined_match_id

    def _emit(self, event: str, **payload: Any) -> None:
        if not self.progress:
            return
        update = {"event": event, "timestamp": time.time(), **payload}
        self.progress(update)

    def _log(self, log: list[dict[str, Any]], event: str, **payload: Any) -> None:
        entry = {"event": event, **payload}
        log.append(entry)
        self._emit(event, **payload)


def canonical_mode(mode: str) -> str:
    key = mode.strip().lower()
    if key not in ARENA_MODES:
        raise ValueError(f"Unknown arena mode: {mode}. Use battle, random, platinum, or daily.")
    return ARENA_MODES[key]


def is_recoverable_arena_error(exc: BaseException | str) -> bool:
    text = str(exc).strip().lower()
    if isinstance(exc, TimeoutError):
        return True
    return any(marker in text for marker in RECOVERABLE_REALTIME_MARKERS)


def should_relogin_for_realtime_error(exc: BaseException | str) -> bool:
    text = str(exc).strip().lower()
    return "websocket handshake failed: empty response" in text or "websocket handshake failed: http 401 unauthorized" in text


def last_battle_snapshot(log: list[dict[str, Any]]) -> dict[str, Any]:
    for entry in reversed(log):
        if isinstance(entry, dict) and isinstance(entry.get("battle"), dict):
            return entry["battle"]
    return {}


def active_battle_match_id(player_data: dict[str, Any]) -> str:
    battle_keys = ("battle", "activeBattle", "active_battle", "battleId", "battle_id", "matchId", "match_id", "currentBattle")
    for key in battle_keys:
        match_id = match_id_from_value(player_data.get(key))
        if match_id:
            return match_id
    for key in ("state", "status", "arena"):
        value = player_data.get(key)
        if isinstance(value, dict) and set(battle_keys).intersection(value):
            match_id = active_battle_match_id(value)
            if match_id:
                return match_id
    return ""


def match_id_from_value(value: Any) -> str:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned and cleaned.lower() not in {"none", "null", "false", "0"}:
            return cleaned
    if isinstance(value, dict):
        for key in ("match_id", "matchId", "id", "battle", "battleId"):
            match_id = match_id_from_value(value.get(key))
            if match_id:
                return match_id
    return ""


def count_sent_turns(log: list[dict[str, Any]]) -> int:
    count = 0
    for entry in log:
        if not isinstance(entry, dict) or entry.get("event") != "auto_decision":
            continue
        sent = entry.get("sent")
        if sent is True or (isinstance(sent, dict) and sent.get("ok") is not False):
            count += 1
    return count


def validate_arena_mode(mode_name: str, player: dict[str, Any], arena_info: dict[str, Any] | None = None) -> dict[str, Any]:
    if mode_name in {"Default", "Random", "Platinum", "Daily"} and not is_battle_arena_location(player):
        return {"ok": False, "reason": "not_at_battle_arena", "location": player.get("location", {})}
    if mode_name in {"Default", "Platinum", "Daily"}:
        team = team_items(player)
        if len(team) != 4:
            return {"ok": False, "reason": "team_must_have_4_miscrits", "team_size": len(team)}
        if not all_team_unique(player):
            return {"ok": False, "reason": "team_miscrits_must_be_unique"}
        if all(is_dead_from_player(item) for item in team):
            return {"ok": False, "reason": "all_team_miscrits_dead"}
        if mode_name in {"Platinum", "Daily"} and any(int(item.get("level", 1) or 1) < MAX_LEVEL for item in team):
            return {"ok": False, "reason": "team_must_be_level_35"}
        if mode_name == "Platinum" and team_rating_points(player) > 12:
            return {"ok": False, "reason": "platinum_rating_cap", "team_rating": team_rating_points(player), "cap": 12}
    random_info = arena_info.get("info", arena_info) if isinstance(arena_info, dict) else {}
    if mode_name == "Random" and random_info and bool(random_info.get("banned", False)):
        return {"ok": False, "reason": "random_arena_temp_ban", "arena_info": arena_info}
    return {"ok": True}


def arena_rules(mode_name: str) -> list[str]:
    if mode_name == "Random":
        return ["queue type Random", "must be at battle arena location", "draft stage: READY, ban 6, pick 4", "server then returns normal battle match id"]
    rules = [
        "queue type " + mode_name,
        "must be at battle arena location",
        "team size 4",
        "all team species must be unique",
        "team must not be fully dead",
        "full heal is attempted before queue; if unavailable, a living fallback team is used",
    ]
    if mode_name in {"Platinum", "Daily"}:
        rules.append("all team miscrits must be level 35")
    if mode_name == "Platinum":
        rules.append("team rarity points must be <= 12")
    return rules


def arena_reward_progress(mode_name: str, arena_info: dict[str, Any], metadata: dict[int, dict[str, Any]]) -> dict[str, Any]:
    info = arena_info.get("info", arena_info) if isinstance(arena_info, dict) else {}
    if not isinstance(info, dict):
        info = {}
    raw_rewards = info.get("rewards", {})
    rewards = raw_rewards if isinstance(raw_rewards, dict) else {}
    stats = info.get("stats", {})
    stats = stats if isinstance(stats, dict) else {}
    progress = arena_reward_wins(mode_name, info, stats)
    items: list[dict[str, Any]] = []
    totals: dict[str, int] = {}
    claimed_totals: dict[str, int] = {}
    remaining_totals: dict[str, int] = {}
    for raw_threshold, reward in rewards.items():
        try:
            threshold = int(raw_threshold)
        except (TypeError, ValueError):
            continue
        if threshold <= 0 or not isinstance(reward, dict):
            continue
        currency = str(reward.get("currency", "") or "").strip().lower()
        amount = arena_reward_amount(currency, reward)
        mid = 0
        name = ""
        rarity = ""
        element = ""
        if currency == "miscrit":
            data = reward.get("data", {})
            data = data if isinstance(data, dict) else {}
            mid = int(data.get("id", 0) or 0)
            meta = metadata.get(mid, {})
            names = meta.get("names", [])
            if isinstance(names, list) and names:
                name = str(names[0] or "")
            if not name:
                name = str(meta.get("name", f"#{mid}" if mid else "Miscrit") or "Miscrit")
            rarity = str(meta.get("rarity", "") or "")
            element = str(meta.get("element", "") or "")
        claimed = progress >= threshold
        totals[currency] = totals.get(currency, 0) + amount
        if claimed:
            claimed_totals[currency] = claimed_totals.get(currency, 0) + amount
        else:
            remaining_totals[currency] = remaining_totals.get(currency, 0) + amount
        items.append(
            {
                "threshold": threshold,
                "claimed": claimed,
                "remaining_wins": max(0, threshold - progress),
                "currency": currency,
                "amount": amount,
                "mid": mid,
                "name": name,
                "rarity": rarity,
                "element": element,
                "raw": reward,
            }
        )
    items.sort(key=lambda item: int(item["threshold"]))
    return {
        "ok": bool(items),
        "mode": mode_name,
        "progress": progress,
        "target": max((int(item["threshold"]) for item in items), default=0),
        "items": items,
        "totals": totals,
        "claimed_totals": claimed_totals,
        "remaining_totals": remaining_totals,
    }


def arena_reward_wins(mode_name: str, info: dict[str, Any], stats: dict[str, Any]) -> int:
    if mode_name == "Default":
        return int(stats.get("ba_weekly_wins", info.get("ba_weekly_wins", 0)) or 0)
    if mode_name == "Random":
        return int(stats.get("ra_monthly_wins", info.get("ra_monthly_wins", 0)) or 0)
    if mode_name == "Daily":
        return int(stats.get("da_daily_wins", info.get("da_daily_wins", 0)) or 0)
    return 0


def arena_reward_amount(currency: str, reward: dict[str, Any]) -> int:
    if currency == "miscrit":
        return 1
    try:
        return max(0, int(reward.get("amount", 1) or 0))
    except (TypeError, ValueError):
        return 1


def team_items(player: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {int(item["id"]): item for item in player.get("miscrits", [])}
    return [by_id[item_id] for item_id in player.get("team", []) if item_id in by_id]


def normalize_location(data: dict[str, Any]) -> dict[str, Any]:
    location = data.get("location", {}) if isinstance(data.get("location"), dict) else {}
    area = data.get("area", {}) if isinstance(data.get("area"), dict) else {}
    return {
        "location_id": int(location.get("id", data.get("locationId", data.get("location_id", 0))) or 0),
        "location_name": str(location.get("name", data.get("location", "")) or ""),
        "area_id": int(area.get("id", data.get("areaId", data.get("area_id", 0))) or 0),
        "area_name": str(area.get("name", data.get("area", "")) or ""),
    }


def is_battle_arena_location(player: dict[str, Any]) -> bool:
    loc = player.get("location", {}) if isinstance(player.get("location"), dict) else {}
    area_id = int(loc.get("area_id", 0) or 0)
    area_name = str(loc.get("area_name", "")).casefold()
    return area_id == 2 or "arena" in area_name


def all_team_unique(player: dict[str, Any]) -> bool:
    mids = [int(item.get("mid", 0) or 0) for item in team_items(player)]
    return len(mids) == len(set(mids))


def team_needs_heal(player: dict[str, Any]) -> bool:
    for item in team_items(player):
        max_hp = max_hp_value(item)
        chp = int(item.get("chp", max_hp) or 0)
        if max_hp > 0 and chp < max_hp:
            return True
    return False


def max_hp_value(item: dict[str, Any]) -> int:
    stats = item.get("stats", {}) if isinstance(item.get("stats"), dict) else {}
    return int(stats.get("hp", item.get("max_hp", item.get("hp", 0))) or 0)


def team_rating_points(player: dict[str, Any]) -> int:
    total = 0
    for item in team_items(player):
        total += RATING_POINTS.get(str(item.get("rarity", "")), 1)
    return total


def is_dead_from_player(item: dict[str, Any]) -> bool:
    if "chp" in item:
        return int(item.get("chp", 0) or 0) <= 0
    return False


def enrich_team(player: dict[str, Any], metadata: dict[int, dict[str, Any]]) -> None:
    for item in player.get("miscrits", []):
        meta = metadata.get(int(item.get("mid", 0) or 0), {})
        evo = int(item.get("evo", evo_from_level(int(item.get("level", 1) or 1))) or 0)
        item["evo"] = max(0, min(3, evo))
        names = meta.get("names", [])
        if isinstance(names, list) and names:
            item["name"] = str(names[min(item["evo"], len(names) - 1)] or names[0])
        else:
            item["name"] = meta.get("name", f"#{item.get('mid', 0)}")
        item["element"] = meta.get("element", "")
        item["rarity"] = meta.get("rarity", "")
        raw_abilities = meta.get("abilities", [])
        if isinstance(raw_abilities, list):
            item["abilities"] = current_level_abilities(raw_abilities, int(item.get("level", 1) or 1))


def compact_miscrit(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "mid": item.get("mid"),
        "evo": item.get("evo", evo_from_level(int(item.get("level", 1) or 1))),
        "name": item.get("name"),
        "level": item.get("level"),
        "rating": item.get("rating"),
        "rarity": item.get("rarity"),
        "element": item.get("element"),
        "chp": item.get("chp"),
        "max_hp": max_hp_value(item),
        "ability_kit_score": round(team_ability_kit_score(item), 3),
        "has_relics": has_active_relics(item),
    }


def team_candidate_eligible(item: dict[str, Any], mode_name: str, *, alive_only: bool = False) -> bool:
    if int(item.get("id", 0) or 0) <= 0:
        return False
    if alive_only and is_dead_from_player(item):
        return False
    if mode_name in {"Platinum", "Daily"} and int(item.get("level", 1) or 1) != MAX_LEVEL:
        return False
    return True


def score_team_candidate(item: dict[str, Any], mode_name: str) -> float:
    level = float(item.get("level", 1) or 1)
    rating_sum = float(item.get("rating_sum", 0) or 0)
    hp = float(max_hp_value(item))
    spd = float(item.get("spd", 0) or 0)
    offense = max(float(item.get("pa", 0) or 0), float(item.get("ea", 0) or 0))
    defense = max(float(item.get("pd", 0) or 0), float(item.get("ed", 0) or 0))
    rarity = RATING_POINTS.get(str(item.get("rarity", "")), 1)
    score = level * 4.4 + rating_sum * 7.5 + hp * 0.45 + spd * 1.35 + offense * 1.5 + defense * 1.05
    score += float(rarity - 1) * (12.0 if mode_name != "Platinum" else 8.0)
    if str(item.get("element", "")).strip():
        score += 3.5
    score += team_ability_kit_score(item) * (1.0 if mode_name != "Platinum" else 0.72)
    if mode_name in {"Platinum", "Daily"}:
        score += 25.0 if int(level) == MAX_LEVEL else -1000.0
    return score


def team_candidate_profile(item: dict[str, Any], mode_name: str) -> dict[str, Any]:
    abilities = [effective_ability(ability, item) for ability in item.get("abilities", []) if isinstance(ability, dict)]
    ability_types = {str(ability.get("type", "")).strip() for ability in abilities}
    attack_abilities = [ability for ability in abilities if str(ability.get("type", "")).strip() == "Attack"]
    attack_elements = {
        str(ability.get("element", "")).strip()
        for ability in attack_abilities
        if str(ability.get("element", "")).strip() and str(ability.get("element", "")).strip() != "Misc"
    }
    attack_values = [arena_ability_value(ability) for ability in attack_abilities]
    strongest_attack = max(attack_values, default=0.0)
    coverage = {
        str(ability.get("element", "")).strip()
        for ability in abilities
        if str(ability.get("element", "")).strip() and str(ability.get("element", "")).strip() != "Misc"
    }
    kit_score = team_ability_kit_score(item)
    return {
        "item": item,
        "id": int(item.get("id", 0) or 0),
        "mid": int(item.get("mid", 0) or 0),
        "element": str(item.get("element", "")).strip(),
        "base_score": score_team_candidate(item, mode_name),
        "kit_score": kit_score,
        "attack_elements": attack_elements,
        "coverage": coverage,
        "strongest_attack": strongest_attack,
        "has_sleep": "Sleep" in ability_types,
        "has_negate": "Negate" in ability_types,
        "has_cleanser": any(
            str(ability.get("type", "")).strip() == "Cleanser" or "cleanse" in ability_text(ability)
            for ability in abilities
        ),
        "has_force_switch": "ForceSwitch" in ability_types or any(ability_forces_switch(ability) for ability in abilities),
        "has_lifesteal": any(ability_lifesteals(ability) for ability in abilities),
        "has_dot": any(str(ability.get("type", "")).strip() in {"Poison", "Dot", "Disease", "Bleed", "SwitchCurse"} for ability in abilities),
        "has_relics": has_active_relics(item),
    }


def select_team_candidate_pool(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(profiles) <= TEAM_POOL_LIMIT:
        return profiles

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    def add(profile: dict[str, Any]) -> None:
        profile_id = int(profile.get("id", 0) or 0)
        if profile_id <= 0 or profile_id in selected_ids or len(selected) >= TEAM_POOL_LIMIT:
            return
        selected.append(profile)
        selected_ids.add(profile_id)

    for profile in profiles[:TEAM_POOL_CORE_LIMIT]:
        add(profile)

    by_element: dict[str, list[dict[str, Any]]] = {}
    for profile in profiles:
        element = str(profile.get("element", "")).casefold()
        if element:
            by_element.setdefault(element, []).append(profile)
    for element_profiles in by_element.values():
        for profile in element_profiles[:2]:
            add(profile)

    for profile in profiles:
        add(profile)
        if len(selected) >= TEAM_POOL_LIMIT:
            break
    selected.sort(key=lambda item: float(item["base_score"]), reverse=True)
    return selected


def evaluate_team_combo(combo: list[dict[str, Any]], mode_name: str, include_members: bool = True) -> dict[str, Any]:
    if len(combo) != 4:
        return {"ok": False, "reason": "team_not_full"}
    profiles = [item if "item" in item else team_candidate_profile(item, mode_name) for item in combo]
    members = [profile["item"] for profile in profiles]
    mids = [int(item.get("mid", 0) or 0) for item in members]
    if len(mids) != len(set(mids)):
        return {"ok": False, "reason": "duplicate_mid"}
    if mode_name in {"Platinum", "Daily"} and any(int(item.get("level", 1) or 1) != MAX_LEVEL for item in members):
        return {"ok": False, "reason": "not_all_level_35"}
    relic_member_count = sum(1 for profile in profiles if bool(profile.get("has_relics", False)))
    if 0 < relic_member_count < len(profiles):
        return {"ok": False, "reason": "mixed_relic_team", "relic_member_count": relic_member_count}
    rating_points = sum(RATING_POINTS.get(str(item.get("rarity", "")), 1) for item in members)
    if mode_name == "Platinum" and rating_points > 12:
        return {"ok": False, "reason": "platinum_rating_cap", "team_rating": rating_points}
    elements = {str(item.get("element", "")).casefold() for item in members if str(item.get("element", "")).strip()}
    attack_elements = {
        str(element).casefold()
        for profile in profiles
        for element in profile.get("attack_elements", set())
        if str(element).strip()
    }
    level_spread = max(int(item.get("level", 1) or 1) for item in members) - min(int(item.get("level", 1) or 1) for item in members)
    score = sum(float(profile["base_score"]) for profile in profiles)
    kit_score = sum(float(profile["kit_score"]) for profile in profiles)
    strong_attackers = sum(1 for profile in profiles if float(profile.get("strongest_attack", 0.0) or 0.0) >= 70.0)
    utility_counts = {
        "sleep": sum(1 for profile in profiles if bool(profile.get("has_sleep", False))),
        "negate": sum(1 for profile in profiles if bool(profile.get("has_negate", False))),
        "cleanser": sum(1 for profile in profiles if bool(profile.get("has_cleanser", False))),
        "force_switch": sum(1 for profile in profiles if bool(profile.get("has_force_switch", False))),
        "lifesteal": sum(1 for profile in profiles if bool(profile.get("has_lifesteal", False))),
        "dot": sum(1 for profile in profiles if bool(profile.get("has_dot", False))),
    }
    score += len(elements) * 16.0
    score += len(attack_elements) * 10.0
    score -= level_spread * 9.0
    score += min(strong_attackers, 4) * 10.0
    score += 8.0 if utility_counts["sleep"] else 0.0
    score += 7.0 if utility_counts["negate"] else 0.0
    score += 4.0 if utility_counts["cleanser"] else 0.0
    score += 3.0 if utility_counts["force_switch"] else 0.0
    score += 2.0 if utility_counts["lifesteal"] else 0.0
    score += 2.0 if utility_counts["dot"] else 0.0
    if len(elements) <= 2:
        score -= (3 - len(elements)) * 14.0
    if strong_attackers < 3:
        score -= (3 - strong_attackers) * 12.0
    if mode_name == "Platinum":
        score += rating_points * 14.0
        if rating_points == 12:
            score += 30.0
    team = [int(profile["id"]) for profile in sorted(profiles, key=lambda item: float(item["base_score"]), reverse=True)]
    result = {
        "ok": True,
        "team": team,
        "score": round(score, 3),
        "team_rating": rating_points,
        "relic_member_count": relic_member_count,
        "ability_kit_score": round(kit_score, 3),
        "utility_counts": utility_counts,
        "element_count": len(elements),
        "attack_element_count": len(attack_elements),
        "strong_attackers": strong_attackers,
    }
    if include_members:
        result["members"] = [compact_team_profile(profile) for profile in profiles]
    return result


def compact_team_profile(profile: dict[str, Any]) -> dict[str, Any]:
    item = profile["item"]
    compact = compact_miscrit(item)
    compact["ability_kit_score"] = round(float(profile.get("kit_score", compact["ability_kit_score"]) or 0.0), 3)
    return compact


def has_active_relics(item: dict[str, Any]) -> bool:
    relics = item.get("active_relics", [])
    if isinstance(relics, dict):
        return any(value is not None for value in relics.values())
    if isinstance(relics, list):
        return any(value is not None for value in relics)
    return any(item.get(key) is not None for key in ("r1", "r2", "r3", "r4"))


def exploratory_team_choice(ranked_combos: list[dict[str, Any]]) -> dict[str, Any]:
    if len(ranked_combos) < 2 or random.random() >= TEAM_EXPLORATION_RATE:
        return {}
    best_score = float(ranked_combos[0].get("score", 0.0) or 0.0)
    pool = [
        item
        for item in ranked_combos[1:10]
        if best_score - float(item.get("score", 0.0) or 0.0) <= TEAM_EXPLORATION_GAP
    ]
    if not pool:
        return {}
    floor = min(float(item.get("score", 0.0) or 0.0) for item in pool)
    weights = [max(0.2, float(item.get("score", 0.0) or 0.0) - floor + 1.0) for item in pool]
    return dict(random.choices(pool, weights=weights, k=1)[0])


def choose_random_bans(pool: list[int], metadata: dict[int, dict[str, Any]]) -> list[int]:
    weights = load_weights()
    ranked = sorted(set(pool), key=lambda mid: species_score(mid, metadata, weights), reverse=True)
    if len(ranked) > RANDOM_BAN_COUNT and random.random() < DRAFT_EXPLORATION_RATE * 0.5:
        head = ranked[: min(len(ranked), RANDOM_BAN_COUNT + 4)]
        chosen = list(ranked[: max(0, RANDOM_BAN_COUNT - 1)])
        alternatives = [item for item in head if item not in set(chosen)]
        if alternatives:
            chosen.append(random.choice(alternatives))
            return list(chosen)[:RANDOM_BAN_COUNT]
    return ranked[:RANDOM_BAN_COUNT]


def choose_random_picks(pool: list[int], metadata: dict[int, dict[str, Any]]) -> list[int]:
    clean = list(dict.fromkeys(int(item) for item in pool if int(item) > 0))
    if len(clean) <= RANDOM_PICK_COUNT:
        return clean[:RANDOM_PICK_COUNT]
    weights = load_weights()
    best: tuple[float, tuple[int, ...]] | None = None
    scored: list[tuple[float, tuple[int, ...]]] = []
    profiles = {mid: species_draft_profile(mid, metadata, weights) for mid in clean}
    for combo in combinations(clean, RANDOM_PICK_COUNT):
        combo_profiles = [profiles[mid] for mid in combo]
        elements = {str(profile.get("element", "")).casefold() for profile in combo_profiles if str(profile.get("element", "")).strip()}
        attack_elements = {
            str(element).casefold()
            for profile in combo_profiles
            for element in profile.get("attack_elements", set())
            if str(element).strip()
        }
        score = sum(float(profile.get("duel_score", 0.0) or 0.0) for profile in combo_profiles)
        score += len(elements) * 9.0
        score += len(attack_elements) * 4.0
        if len(elements) <= 2:
            score -= (3 - len(elements)) * 10.0
        scored.append((score, combo))
        if best is None or score > best[0]:
            best = (score, combo)
    explored = exploratory_draft_combo(scored)
    if explored:
        return list(explored)
    return list(best[1]) if best else clean[:RANDOM_PICK_COUNT]


def exploratory_draft_combo(scored: list[tuple[float, tuple[int, ...]]]) -> tuple[int, ...]:
    if len(scored) < 2 or random.random() >= DRAFT_EXPLORATION_RATE:
        return ()
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score = scored[0][0]
    pool = [item for item in scored[1:12] if best_score - item[0] <= DRAFT_EXPLORATION_GAP]
    if not pool:
        return ()
    floor = min(item[0] for item in pool)
    weights = [max(0.2, item[0] - floor + 1.0) for item in pool]
    return random.choices(pool, weights=weights, k=1)[0][1]


def species_score(mid: int, metadata: dict[int, dict[str, Any]], weights: dict[str, Any] | None = None) -> float:
    return float(species_draft_profile(mid, metadata, weights).get("duel_score", 0.0) or 0.0)


def species_draft_profile(mid: int, metadata: dict[int, dict[str, Any]], weights: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = metadata.get(int(mid), {})
    rarity = str(meta.get("rarity", ""))
    abilities = current_level_abilities(meta.get("abilities", []) if isinstance(meta.get("abilities"), list) else [], MAX_LEVEL)
    attacks = [ability for ability in abilities if str(ability.get("type", "")).strip() == "Attack"]
    attack_values = sorted((arena_ability_value(ability) for ability in attacks), reverse=True)
    strongest_attack = max(attack_values, default=0.0)
    top_attack_value = sum(attack_values[:3])
    attack_elements = {
        str(ability.get("element", "")).strip()
        for ability in attacks
        if str(ability.get("element", "")).strip() and str(ability.get("element", "")).strip() != "Misc"
    }
    stats = {key: species_stat_points(meta.get(key, "")) for key in ("hp", "spd", "ea", "pa", "ed", "pd")}
    offense = max(stats["ea"], stats["pa"])
    defense = max(stats["ed"], stats["pd"])
    raw_kit_holder = {"abilities": abilities}
    kit_score = team_ability_kit_score(raw_kit_holder)
    learned = species_learning_score(mid, weights or load_weights())
    duel_score = 0.0
    duel_score += RATING_POINTS.get(rarity, 1) * 10.0
    duel_score += stats["hp"] * 8.0
    duel_score += stats["spd"] * 6.0
    duel_score += offense * 10.0
    duel_score += defense * 7.0
    duel_score += top_attack_value * 0.16
    duel_score += strongest_attack * 0.10
    duel_score += kit_score * 0.50
    duel_score += learned
    return {
        "mid": int(mid),
        "element": str(meta.get("element", "")),
        "rarity": rarity,
        "stats": stats,
        "attack_elements": attack_elements,
        "strongest_attack": round(strongest_attack, 3),
        "kit_score": round(kit_score, 3),
        "learned_score": round(learned, 3),
        "duel_score": round(duel_score, 3),
    }


def species_stat_points(value: Any) -> float:
    return SPECIES_STAT_POINTS.get(str(value or "").strip(), 0.0)


def species_learning_score(mid: int, weights: dict[str, Any]) -> float:
    total_weight = 0.0
    total_count = 0
    for bucket_name, scale in (("pair_matchups", 0.45), ("opponent_pair_matchups", 0.28)):
        bucket = weights.get(bucket_name, {}) if isinstance(weights, dict) else {}
        if not isinstance(bucket, dict):
            continue
        prefix = f"{int(mid)}>"
        for key, value in bucket.items():
            if not str(key).startswith(prefix) or not isinstance(value, dict):
                continue
            count = max(1, int(value.get("count", 0) or 0))
            total_weight += float(value.get("weight", 0.0) or 0.0) * count * scale
            total_count += count
    if total_count <= 0:
        return 0.0
    return max(-10.0, min(10.0, total_weight / total_count))


def current_level_abilities(raw_abilities: list[Any], level: int) -> list[dict[str, Any]]:
    count = sum(1 for key in LEVEL_KEYS if key <= int(level or 1))
    selected = [dict(item) for item in raw_abilities[:count] if isinstance(item, dict)]
    for index, ability in enumerate(selected):
        ability.setdefault("level", LEVEL_KEYS[index] if index < len(LEVEL_KEYS) else MAX_LEVEL)
        ability.setdefault("recharge", 0)
    selected.reverse()
    return selected


def team_ability_kit_score(miscrit: dict[str, Any]) -> float:
    abilities = [item for item in miscrit.get("abilities", []) if isinstance(item, dict)]
    if not abilities:
        return 0.0
    scores = sorted((arena_ability_value(effective_ability(item, miscrit)) for item in abilities), reverse=True)
    top = scores[:4]
    utility_types = {str(item.get("type", "")).strip() for item in abilities}
    coverage = {str(item.get("element", "")).strip() for item in abilities if str(item.get("element", "")).strip() and str(item.get("element", "")).strip() != "Misc"}
    utility_bonus = 0.0
    if "Sleep" in utility_types:
        utility_bonus += 16.0
    if "Confuse" in utility_types:
        utility_bonus += 9.0
    if "Paralyze" in utility_types:
        utility_bonus += 4.0
    if "Negate" in utility_types:
        utility_bonus += 24.0
    if "ForceSwitch" in utility_types or any(ability_forces_switch(item) for item in abilities):
        utility_bonus += 14.0
    if any(ability_lifesteals(item) for item in abilities):
        utility_bonus += 10.0
    if any(str(item.get("type", "")).strip() == "Cleanser" or "cleanse" in ability_text(item) for item in abilities):
        utility_bonus += 10.0
    return sum(top) * 0.22 + len(coverage) * 5.0 + utility_bonus


def arena_ability_value(ability: dict[str, Any]) -> float:
    kind = str(ability.get("type", "")).strip()
    ap = abs(float(ability.get("ap", ability.get("power", 0)) or 0))
    times = max(1.0, float(ability.get("times", 1) or 1))
    accuracy = accuracy_value(ability)
    additional = ability.get("additional", [])
    additional_count = len(additional) if isinstance(additional, list) else 0
    cooldown = max(0.0, float(ability.get("cooldown", ability.get("og_cooldown", 0)) or 0))
    value = 0.0
    if kind == "Attack":
        value = ap * times * accuracy * 2.4
        if str(ability.get("element", "")) not in {"", "Misc"}:
            value += 12.0
        if bool(ability.get("true_dmg", ability.get("true_damage", False))):
            value += 16.0
    elif kind == "Negate":
        value = 122.0 + accuracy * 12.0
    elif kind == "Sleep":
        value = 76.0 + accuracy * 28.0 + max(0.0, float(ability.get("turns", 0) or 0)) * 8.0
    elif kind == "Confuse":
        value = 50.0 + accuracy * 18.0 + max(0.0, float(ability.get("turns", 0) or 0)) * 5.0
    elif kind == "Paralyze":
        value = 24.0 + accuracy * 10.0 + max(0.0, float(ability.get("turns", 0) or 0)) * 2.0
    elif kind in {"Poison", "Dot", "Disease", "Bleed", "SwitchCurse"}:
        value = 62.0 + ap * 1.7 + max(0.0, float(ability.get("turns", 0) or 0)) * 9.0 + accuracy * 10.0
    elif kind in {"Buff", "Bot"}:
        value = 48.0 + ap * 2.3 + accuracy * 6.0
    elif kind in {"Heal", "Hot", "Cleanser"}:
        value = 28.0 + ap * 1.15 + max(0.0, float(ability.get("turns", 0) or 0)) * 5.0
    else:
        value = 30.0 + ap * 1.25 + accuracy * 5.0
    value += additional_count * 9.0
    if ability_lifesteals(ability):
        value += 12.0
    if ability_forces_switch(ability):
        value += 12.0
    return max(0.0, value - cooldown * 6.0)


def pick_best_gold_enchant_candidate(player: dict[str, Any], team_ids: list[int], mode_name: str) -> dict[str, Any]:
    by_id = {int(item.get("id", 0) or 0): item for item in player.get("miscrits", []) if isinstance(item, dict)}
    gold = int(player.get("gold", 0) or 0)
    best: dict[str, Any] = {}
    best_value = -1.0
    any_upgradeable = False
    cheapest = 999999999
    for miscrit_id in team_ids:
        miscrit = by_id.get(int(miscrit_id))
        if not miscrit:
            continue
        carrier_bonus = min(45.0, max(0.0, score_team_candidate(miscrit, mode_name) * 0.04))
        for ability in miscrit.get("abilities", []):
            if not isinstance(ability, dict) or ability_already_enchanted(miscrit, ability):
                continue
            enchant = ability.get("enchant", {})
            if not isinstance(enchant, dict) or not enchant:
                continue
            any_upgradeable = True
            price = enchant_price(ability)
            cheapest = min(cheapest, price)
            if gold < price + AUTO_ENCHANT_GOLD_RESERVE:
                continue
            base_value = arena_ability_value(ability)
            enchanted_value = arena_ability_value(apply_enchant_preview(ability))
            strength = max(0.0, enchanted_value - base_value) + carrier_bonus
            if strength < AUTO_ENCHANT_MIN_STRENGTH_SCORE:
                continue
            value = strength * 0.75 + (strength / max(1.0, float(price))) * 850.0
            if value > best_value:
                best_value = value
                best = {
                    "ok": True,
                    "miscrit_id": int(miscrit_id),
                    "miscrit_name": str(miscrit.get("name", f"#{miscrit.get('mid', miscrit_id)}")),
                    "ability_id": int(ability.get("id", 0) or 0),
                    "ability_name": str(ability.get("name", "")),
                    "ability_type": str(ability.get("type", "")),
                    "gold_price": price,
                    "strength_score": round(strength, 3),
                    "value_score": round(value, 3),
                    "gold": gold,
                }
    if best:
        return best
    if not any_upgradeable:
        return {"ok": False, "reason": "no_upgradeable_abilities"}
    if cheapest < 999999999 and gold < cheapest + AUTO_ENCHANT_GOLD_RESERVE:
        return {"ok": False, "reason": "insufficient_gold", "gold": gold, "cheapest_price": cheapest, "reserve": AUTO_ENCHANT_GOLD_RESERVE}
    return {"ok": False, "reason": "no_strong_candidates"}


def apply_local_enchant(player: dict[str, Any], miscrit_id: int, ability_id: int, price: int) -> None:
    player["gold"] = max(0, int(player.get("gold", 0) or 0) - int(price))
    for item in player.get("miscrits", []):
        if not isinstance(item, dict) or int(item.get("id", 0) or 0) != int(miscrit_id):
            continue
        enchants = item.get("enchants", [])
        if not isinstance(enchants, list):
            enchants = []
        if int(ability_id) not in {int(value) for value in enchants if str(value).lstrip("-").isdigit()}:
            enchants.append(int(ability_id))
        item["enchants"] = enchants
        return


def ability_already_enchanted(miscrit: dict[str, Any], ability: dict[str, Any]) -> bool:
    ability_id = int(ability.get("id", 0) or 0)
    enchants = miscrit.get("enchants", [])
    if isinstance(enchants, list):
        return ability_id in {int(item) for item in enchants if str(item).lstrip("-").isdigit()}
    if isinstance(enchants, dict):
        return str(ability_id) in enchants or ability_id in enchants
    return False


def apply_enchant_preview(ability: dict[str, Any]) -> dict[str, Any]:
    result = dict(ability)
    enchant = ability.get("enchant", {})
    if not isinstance(enchant, dict):
        return result
    for key, value in enchant.items():
        if key in {"ap", "accuracy", "times", "turns"}:
            result[key] = float(result.get(key, 0) or 0) + float(value or 0)
        elif key == "additional" and isinstance(value, list):
            base = result.get("additional", [])
            result["additional"] = (base if isinstance(base, list) else []) + value
    return result


def enchant_price(ability: dict[str, Any]) -> int:
    level = int(ability.get("level", 30) or 30)
    price = max(1, min(level, 30)) * 100
    if str(ability.get("type", "")) in {"Confuse", "Sleep", "Heal", "Stun", "Paralyze"}:
        price += 500
    additional = ability.get("additional", [])
    if isinstance(additional, list) and additional:
        price += 500
    enchant = ability.get("enchant", {})
    if isinstance(enchant, dict) and isinstance(enchant.get("additional"), list):
        price += 1000
    return int(price)


def accuracy_value(ability: dict[str, Any]) -> float:
    accuracy = float(ability.get("accuracy", 100) or 100)
    if accuracy == 170:
        return 1.0
    return max(0.35, min(1.0, accuracy / 100.0))


def ability_text(ability: dict[str, Any]) -> str:
    return " ".join(str(ability.get(key, "") or "") for key in ("name", "desc", "description", "tooltip")).casefold()


def ability_lifesteals(ability: dict[str, Any]) -> bool:
    additional = ability.get("additional", [])
    if isinstance(additional, list) and any(isinstance(item, dict) and str(item.get("type", "")) == "LifeSteal" for item in additional):
        return True
    return "steals" in ability_text(ability) and "hp" in ability_text(ability)


def ability_forces_switch(ability: dict[str, Any]) -> bool:
    if str(ability.get("type", "")) == "ForceSwitch":
        return True
    additional = ability.get("additional", [])
    if isinstance(additional, list) and any(isinstance(item, dict) and str(item.get("type", "")) == "ForceSwitch" for item in additional):
        return True
    text = ability_text(ability)
    return "force switches" in text or "forcefully switches" in text or "forces you to switch" in text


def names_for_ids(ids: list[int], metadata: dict[int, dict[str, Any]]) -> list[str]:
    return [str(metadata.get(int(mid), {}).get("name", f"#{mid}")) for mid in ids]


def species_cards(ids: list[int], metadata: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    cards = []
    for mid in ids:
        meta = metadata.get(int(mid), {})
        names = meta.get("names", [])
        evo = 3
        name = str(names[min(evo, len(names) - 1)] or names[0]) if isinstance(names, list) and names else str(meta.get("name", f"#{mid}"))
        cards.append(
            {
                "mid": int(mid),
                "evo": evo,
                "name": name,
                "element": str(meta.get("element", "")),
                "rarity": str(meta.get("rarity", "")),
            }
        )
    return cards


def battle_metadata_by_mid(items: list[Any]) -> dict[int, dict[str, Any]]:
    metadata: dict[int, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            mid = int(item.get("mid", item.get("id", 0)) or 0)
        except (TypeError, ValueError):
            continue
        if not mid:
            continue
        names = item.get("names", [])
        name = ""
        if isinstance(names, list) and names:
            name = str(names[0] or "")
        metadata[mid] = {
            "mid": mid,
            "name": name or str(item.get("name", item.get("display_name", f"#{mid}"))),
            "names": [str(name) for name in names] if isinstance(names, list) else [],
            "element": item.get("element", ""),
            "rarity": item.get("rarity", ""),
            "abilities": item.get("abilities", []),
        }
    return metadata


def enrich_battle_snapshot(snapshot: dict[str, Any], metadata: dict[int, dict[str, Any]]) -> dict[str, Any]:
    for side in ("player", "foe"):
        player = snapshot.get(side, {})
        team = player.get("team", []) if isinstance(player, dict) else []
        if not isinstance(team, list):
            continue
        for item in team:
            if not isinstance(item, dict):
                continue
            meta = metadata.get(int(item.get("mid", 0) or 0), {})
            if meta:
                evo = int(item.get("evo", evo_from_level(int(item.get("level", 1) or 1))) or 0)
                item["evo"] = max(0, min(3, evo))
                names = meta.get("names", [])
                if isinstance(names, list) and names:
                    item["name"] = str(names[min(item["evo"], len(names) - 1)] or names[0])
                else:
                    item["name"] = str(meta.get("name", item.get("name", "")))
                item["element"] = str(meta.get("element", item.get("element", "")))
                item["rarity"] = str(meta.get("rarity", item.get("rarity", "")))
    return snapshot


def battle_outcome(winner: str, player_id: str, snapshot: dict[str, Any] | None = None) -> str:
    if not winner:
        inferred = infer_battle_outcome(snapshot or {})
        if inferred:
            return inferred
        return "unknown"
    return "victory" if str(winner) == str(player_id) else "defeat"


def infer_battle_outcome(snapshot: dict[str, Any]) -> str:
    player_team = snapshot_side_team(snapshot, "player")
    foe_team = snapshot_side_team(snapshot, "foe")
    if player_team:
        player_dead = sum(1 for item in player_team if battle_item_dead(item))
        if player_dead >= len(player_team) and len(player_team) >= 4:
            return "defeat"
        if player_dead >= 4:
            return "defeat"
    if foe_team:
        foe_dead = sum(1 for item in foe_team if battle_item_dead(item))
        if foe_dead >= len(foe_team) and len(foe_team) >= 4:
            return "victory"
        if foe_dead >= 4:
            return "victory"
    return ""


def snapshot_side_team(snapshot: dict[str, Any], side: str) -> list[dict[str, Any]]:
    side_data = snapshot.get(side, {}) if isinstance(snapshot, dict) else {}
    team = side_data.get("team", []) if isinstance(side_data, dict) else []
    return [item for item in team if isinstance(item, dict)] if isinstance(team, list) else []


def battle_item_dead(item: dict[str, Any]) -> bool:
    if bool(item.get("dead", False)):
        return True
    for key in ("chp", "c"):
        if key in item:
            try:
                return int(item.get(key, 0) or 0) <= 0
            except (TypeError, ValueError):
                return False
    return False


def summarize_loop_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    victories = 0
    defeats = 0
    unknown = 0
    errors = 0
    recoverable_errors = 0
    scores: list[float] = []
    for result in results:
        if not result.get("ok"):
            if result.get("recoverable"):
                recoverable_errors += 1
            else:
                errors += 1
            unknown += 1
            continue
        battle = result.get("battle", {}) if isinstance(result.get("battle"), dict) else {}
        outcome = str(battle.get("outcome", "unknown") or "unknown")
        if outcome == "victory":
            victories += 1
        elif outcome == "defeat":
            defeats += 1
        else:
            unknown += 1
        grade = battle.get("grade", {}) if isinstance(battle.get("grade"), dict) else {}
        if "score" in grade:
            try:
                scores.append(float(grade["score"]))
            except (TypeError, ValueError):
                pass
    total = len(results)
    return {
        "total": total,
        "victories": victories,
        "defeats": defeats,
        "unknown": unknown,
        "errors": errors,
        "recoverable_errors": recoverable_errors,
        "win_rate": victories / total if total else 0.0,
        "average_score": sum(scores) / len(scores) if scores else 0.0,
    }


def send_decision(rt: NakamaRealtime, match_id: str, decision: dict[str, Any]) -> bool:
    kind = str(decision.get("type", "none"))
    action_id = int(decision.get("id", 0) or 0)
    if kind == "ability" and action_id > 0:
        rt.send_match_state(match_id, BATTLE_OPCODES["ABILITY"], {"id": action_id})
        return True
    if kind == "switch" and action_id != 0:
        rt.send_match_state(match_id, BATTLE_OPCODES["SWITCH"], {"id": action_id})
        return True
    return False


def summarize_battle_data(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    keep = {}
    for key in ("type", "pvp", "user_id", "next_turn", "turns", "winner", "loser", "id", "pending", "time_left"):
        if key in data:
            keep[key] = data[key]
    actions = data.get("actions", [])
    if isinstance(actions, list):
        keep["actions_count"] = len(actions)
    return keep


def token_user_id(token: str) -> str:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8"))
        return str(data.get("uid", data.get("user_id", data.get("sub", ""))))
    except (IndexError, ValueError, json.JSONDecodeError):
        return ""
