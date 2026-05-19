from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .arena import (
    ArenaRunConfig,
    ArenaRunner,
    arena_reward_progress,
    arena_time_left,
    is_recoverable_arena_error,
    optional_arena_int,
    platinum_cap,
    platinum_streak_rewards,
)
from .breeding import metadata_by_mid
from .config import DATA_DIR
from .data_cache import DataCache
from .event_log import log_event
from .nakama import MiscritsClient, MiscritsError
from .storage import load_json, save_json


PLAN_FILE = DATA_DIR / "account_plans.json"
MOSCOW_TZ = ZoneInfo("Europe/Moscow")
RESET_HOUR = 4
DEFAULT_PLAN = {
    "enabled": True,
    "repeat_daily": True,
    "tick_seconds": 20,
    "battle_delay_seconds": 3,
    "stop_on_error": False,
    "blocks": [
        {"id": "wish_sk", "type": "wish", "kind": "sk", "enabled": True},
        {"id": "wish_vi", "type": "wish", "kind": "vi", "enabled": True},
        {
            "id": "daily_arena",
            "type": "arena",
            "mode": "daily",
            "enabled": True,
            "goal_mode": "arena_counter",
            "target_arena_wins": 5,
        },
        {
            "id": "platinum_arena",
            "type": "arena",
            "mode": "platinum",
            "enabled": True,
            "goal_mode": "arena_counter",
            "target_cycle_platinum": 300,
            "target_platinum": 300,
        },
        {
            "id": "random_arena",
            "type": "arena",
            "mode": "random",
            "enabled": True,
            "goal_mode": "cycle_wins",
            "target_cycle_wins": 6,
        },
    ],
    "steps": {
        "wish_sk": {"enabled": True},
        "wish_vi": {"enabled": True},
        "daily_arena": {"enabled": True, "target_wins": 5},
        "platinum_arena": {"enabled": True, "target_platinum": 300},
        "random_arena": {"enabled": True, "target_wins": 6},
    },
    "fallback": {
        "random_when_heal_blocked": True,
        "random_batch_wins": 1,
    },
}

_PLAN_LOCK = threading.RLock()


@dataclass
class PlanRunResult:
    ok: bool
    status: str = "idle"
    reason: str = ""
    state: dict[str, Any] = field(default_factory=dict)


def load_all_plans() -> dict[str, Any]:
    data = load_json(PLAN_FILE, {})
    return data if isinstance(data, dict) else {}


def save_all_plans(data: dict[str, Any]) -> None:
    save_json(PLAN_FILE, data)


def default_plan_config() -> dict[str, Any]:
    return normalize_plan_config(deep_merge({}, DEFAULT_PLAN), explicit_blocks=True)


def get_account_plan(account_id: str) -> dict[str, Any]:
    account_id = str(account_id or "").strip()
    with _PLAN_LOCK:
        data = load_all_plans()
        entry = data.get(account_id, {}) if isinstance(data.get(account_id), dict) else {}
        raw_config = entry.get("config", {}) if isinstance(entry.get("config"), dict) else {}
        config = normalize_plan_config(deep_merge(default_plan_config(), raw_config), explicit_blocks="blocks" in raw_config)
        state = entry.get("state", {}) if isinstance(entry.get("state"), dict) else {}
        reset_state_if_needed(state, {})
        return {"ok": True, "account_id": account_id, "config": config, "state": state}


def save_account_plan(account_id: str, config: dict[str, Any]) -> dict[str, Any]:
    account_id = str(account_id or "").strip()
    raw_config = config if isinstance(config, dict) else {}
    merged = normalize_plan_config(deep_merge(default_plan_config(), raw_config), explicit_blocks="blocks" in raw_config)
    with _PLAN_LOCK:
        data = load_all_plans()
        entry = data.get(account_id, {}) if isinstance(data.get(account_id), dict) else {}
        entry["config"] = merged
        entry.setdefault("state", {})
        data[account_id] = entry
        save_all_plans(data)
    log_event("plan_config_saved", category="plan", source="web", initiator="user", account_id=account_id, payload=merged)
    return {"ok": True, "account_id": account_id, "config": merged, "state": entry["state"]}


class AccountPlanRunner:
    def __init__(
        self,
        account_id: str,
        client: MiscritsClient,
        *,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.account_id = str(account_id or "")
        self.client = client
        self.progress = progress
        raw_miscrits = DataCache(client).get_json("miscrits.json", refresh_index=False) or []
        self.metadata = metadata_by_mid(raw_miscrits)

    def run_forever(self, should_stop: Callable[[], bool] | None = None) -> PlanRunResult:
        while True:
            if should_stop and should_stop():
                return PlanRunResult(True, "stopped", state=self.load_state())
            try:
                result = self.run_once(should_stop=should_stop)
            except (MiscritsError, TimeoutError, OSError) as exc:
                if not is_recoverable_plan_error(exc):
                    raise
                self.emit("plan_recoverable_error", "recovering", reason=str(exc))
                config = self.load_config()
                sleep_with_stop(float(config.get("tick_seconds", 20) or 20), should_stop)
                continue
            if result.status == "stopped":
                return result
            config = self.load_config()
            if result.status == "complete":
                if not bool(config.get("repeat_daily", True)):
                    return result
                wait_seconds = seconds_until_next_reset() + 2
                self.emit("plan_wait_next_reset", "waiting", wait_seconds=round(wait_seconds, 1), state=result.state)
                sleep_with_stop(wait_seconds, should_stop)
                continue
            config = self.load_config()
            sleep_with_stop(float(config.get("tick_seconds", 20) or 20), should_stop)

    def refresh_arena_states(self) -> dict[str, Any]:
        snapshots: dict[str, dict[str, Any]] = {}
        arena_state_updated: set[str] = set()
        for block in plan_blocks(self.load_config()):
            if not bool(block.get("enabled", True)) or str(block.get("type", "")) != "arena":
                continue
            mode = str(block.get("mode", "")).strip().casefold()
            if mode not in {"battle", "daily", "platinum", "random"}:
                continue
            snapshot = snapshots.get(mode)
            if snapshot is None:
                snapshot = self.load_arena_snapshot(mode)
                snapshots[mode] = snapshot
            if mode == "platinum":
                progress = arena_progress(snapshot, mode)
                goal_mode, goal_progress, target = self.arena_goal(block, snapshot)
                if mode not in arena_state_updated:
                    self.update_arena_state(
                        mode,
                        snapshot,
                        progress,
                        target=target,
                        goal_mode=goal_mode,
                        goal_progress=goal_progress,
                        rewards_complete=arena_rewards_complete(snapshot),
                    )
                    arena_state_updated.add(mode)
                self.update_arena_block_state(block, goal_mode, goal_progress, target, snapshot)
            else:
                goal_mode, goal_progress, target = self.arena_goal(block, snapshot)
                if mode not in arena_state_updated:
                    self.update_arena_state(
                        mode,
                        snapshot,
                        arena_progress(snapshot, mode),
                        target=target,
                        goal_mode=goal_mode,
                        goal_progress=goal_progress,
                        rewards_complete=arena_rewards_complete(snapshot),
                    )
                    arena_state_updated.add(mode)
                self.update_arena_block_state(block, goal_mode, goal_progress, target, snapshot)
        return self.load_state()

    def run_once(self, should_stop: Callable[[], bool] | None = None) -> PlanRunResult:
        config = self.load_config()
        if not bool(config.get("enabled", True)):
            self.emit("plan_disabled", "idle")
            return PlanRunResult(True, "disabled", state=self.load_state())

        player = self.get_player()
        with _PLAN_LOCK:
            state = self.load_state_unlocked()
            reset_state_if_needed(state, player)
            self.save_state_unlocked(state)

        self.emit("plan_tick", "planning", state=state)
        for block in plan_blocks(config):
            if not bool(block.get("enabled", True)):
                continue
            result = self.run_block(block, player, should_stop)
            if result is not None:
                return result

        self.emit("plan_complete", "finished", state=self.load_state())
        return PlanRunResult(True, "complete", state=self.load_state())

    def run_block(
        self,
        block: dict[str, Any],
        player: dict[str, Any],
        should_stop: Callable[[], bool] | None = None,
    ) -> PlanRunResult | None:
        block_type = str(block.get("type", "")).strip()
        if block_type == "wish":
            kind = str(block.get("kind", "")).strip().casefold()
            if kind not in {"sk", "vi"}:
                return None
            state_key = f"wish_{kind}"
            field_name = "last_wish_SK" if kind == "sk" else "last_wish_VI"
            if not self.state_done(state_key) and self.wish_due(player, field_name):
                return self.collect_wish(kind, state_key)
            return None

        if block_type != "arena":
            return None
        mode = str(block.get("mode", "")).strip().casefold()
        if mode not in {"battle", "daily", "platinum", "random"}:
            return None
        snapshot = self.load_arena_snapshot(mode)
        if mode == "platinum":
            goal_mode, goal_progress, target = self.arena_goal(block, snapshot)
            rewards_complete = arena_rewards_complete(snapshot)
            self.update_arena_state(
                mode,
                snapshot,
                arena_progress(snapshot, mode),
                target=target,
                goal_mode=goal_mode,
                goal_progress=goal_progress,
                rewards_complete=rewards_complete,
            )
            self.update_arena_block_state(block, goal_mode, goal_progress, target, snapshot)
            if rewards_complete:
                self.emit("plan_arena_rewards_complete", "planning", mode=mode, block_id=block.get("id"))
                return None
            if goal_progress < target:
                return self.play_platinum_until(block, should_stop)
            return None
        goal_mode, goal_progress, target = self.arena_goal(block, snapshot)
        rewards_complete = arena_rewards_complete(snapshot)
        self.update_arena_state(
            mode,
            snapshot,
            arena_progress(snapshot, mode),
            target=target,
            goal_mode=goal_mode,
            goal_progress=goal_progress,
            rewards_complete=rewards_complete,
        )
        self.update_arena_block_state(block, goal_mode, goal_progress, target, snapshot)
        if rewards_complete:
            self.emit("plan_arena_rewards_complete", "planning", mode=mode, block_id=block.get("id"))
            return None
        if goal_progress < target:
            return self.play_arena_until(block, should_stop)
        return None

    def collect_wish(self, kind: str, state_key: str) -> PlanRunResult:
        self.emit("plan_step_start", "action", step=state_key)
        try:
            result = self.client.wish(kind)
        except MiscritsError as exc:
            self.emit("plan_step_failed", "error", step=state_key, reason=str(exc))
            return PlanRunResult(False, "error", reason=str(exc), state=self.load_state())
        with _PLAN_LOCK:
            state = self.load_state_unlocked()
            state[state_key] = {"done": True, "success": bool(result.success), "raw": result.raw, "timestamp": time.time()}
            self.save_state_unlocked(state)
        self.emit("plan_step_complete", "action", step=state_key, success=result.success, result=result.raw)
        return PlanRunResult(True, "running", state=self.load_state())

    def play_platinum_until(self, block: dict[str, Any], should_stop: Callable[[], bool] | None = None) -> PlanRunResult:
        snapshot = self.load_arena_snapshot("platinum")
        goal_mode, goal_progress, target = self.arena_goal(block, snapshot)
        self.update_arena_state("platinum", snapshot, arena_progress(snapshot, "platinum"), target=target, goal_mode=goal_mode, goal_progress=goal_progress)
        self.update_arena_block_state(block, goal_mode, goal_progress, target, snapshot)
        if goal_progress >= target:
            return PlanRunResult(True, "running", state=self.load_state())
        result = self.play_one_arena("platinum", should_stop)
        next_snapshot = self.load_arena_snapshot("platinum")
        next_goal_mode, next_goal_progress, next_target = self.arena_goal(block, next_snapshot)
        self.update_arena_state(
            "platinum",
            next_snapshot,
            arena_progress(next_snapshot, "platinum"),
            target=next_target,
            goal_mode=next_goal_mode,
            goal_progress=next_goal_progress,
        )
        self.update_arena_block_state(block, next_goal_mode, next_goal_progress, next_target, next_snapshot)
        self.emit("plan_platinum_progress", "planning", earned=next_goal_progress, target=next_target, goal_mode=next_goal_mode)
        return PlanRunResult(bool(result.get("ok") or result.get("recoverable")), "running", state=self.load_state())

    def play_arena_until(self, block: dict[str, Any], should_stop: Callable[[], bool] | None = None) -> PlanRunResult:
        mode = str(block.get("mode", "")).strip().casefold()
        snapshot = self.load_arena_snapshot(mode)
        goal_mode, goal_progress, target = self.arena_goal(block, snapshot)
        self.update_arena_state(
            mode,
            snapshot,
            arena_progress(snapshot, mode),
            target=target,
            goal_mode=goal_mode,
            goal_progress=goal_progress,
            rewards_complete=arena_rewards_complete(snapshot),
        )
        self.update_arena_block_state(block, goal_mode, goal_progress, target, snapshot)
        if arena_rewards_complete(snapshot) or goal_progress >= target:
            return PlanRunResult(True, "running", state=self.load_state())
        result = self.play_one_arena(mode, should_stop)
        if self.heal_blocked(result) and mode != "random" and self.fallback_random_enabled():
            self.emit("plan_fallback_random", "fallback", reason="heal_blocked", blocked_mode=mode)
            self.play_one_arena("random", should_stop)
            return PlanRunResult(True, "running", state=self.load_state())
        next_snapshot = self.load_arena_snapshot(mode)
        before_counter = arena_progress(snapshot, mode)
        next_counter = arena_progress(next_snapshot, mode)
        if arena_battle_won(result, before_counter, next_counter):
            self.record_arena_block_win(block)
        next_goal_mode, next_goal_progress, next_target = self.arena_goal(block, next_snapshot)
        self.update_arena_state(
            mode,
            next_snapshot,
            next_counter,
            target=next_target,
            goal_mode=next_goal_mode,
            goal_progress=next_goal_progress,
            rewards_complete=arena_rewards_complete(next_snapshot),
        )
        self.update_arena_block_state(block, next_goal_mode, next_goal_progress, next_target, next_snapshot)
        return PlanRunResult(bool(result.get("ok")), "running", state=self.load_state())

    def play_one_arena(self, mode: str, should_stop: Callable[[], bool] | None = None) -> dict[str, Any]:
        if should_stop and should_stop():
            return {"ok": False, "stopped": True}
        config = self.load_config()
        self.emit("plan_arena_start", "battle", mode=mode)
        runner = ArenaRunner(self.client, progress=lambda event: self.emit("plan_arena_event", event.get("phase", "battle"), mode=mode, arena_event=event))
        result = runner.run(
            ArenaRunConfig(
                mode=mode,
                timeout_seconds=float(config.get("timeout_seconds", 300) or 300),
                max_turns=int(config.get("max_turns", 150) or 150),
                prepare=True,
                repeat_count=1,
                repeat_delay_seconds=float(config.get("battle_delay_seconds", 3) or 3),
                stop_on_error=bool(config.get("stop_on_error", False)),
            )
        )
        outcome = result.get("battle", {}).get("outcome") if isinstance(result.get("battle"), dict) else ""
        self.emit("plan_arena_complete", "battle", mode=mode, ok=result.get("ok"), outcome=outcome, result=compact_plan_result(result))
        sleep_with_stop(float(config.get("battle_delay_seconds", 3) or 3), should_stop)
        return result

    def get_player(self, silent: bool = False) -> dict[str, Any]:
        result = self.client.get_player()
        if not result.success or not isinstance(result.data, dict):
            raise RuntimeError(f"get_player failed: {result.raw}")
        if not silent:
            self.emit("plan_player_loaded", "planning")
        return result.data

    def load_arena_snapshot(self, mode: str) -> dict[str, Any]:
        result = self.client.get_arena(mode)
        if not result.success or not isinstance(result.data, dict):
            raise RuntimeError(f"get_{mode}_arena failed: {result.raw}")
        snapshot = normalize_arena_snapshot(mode, result.data, self.metadata)
        self.emit(
            "plan_arena_info",
            "planning",
            mode=mode,
            progress=arena_progress(snapshot, mode),
            rewards=snapshot.get("reward_steps", []),
            stats=snapshot.get("stats", {}),
        )
        return snapshot

    def update_arena_state(
        self,
        mode: str,
        snapshot: dict[str, Any],
        progress: int,
        target: int = 0,
        *,
        goal_mode: str = "",
        goal_progress: int | None = None,
        rewards_complete: bool | None = None,
    ) -> None:
        key = {"daily": "daily_wins", "random": "random_wins", "platinum": "platinum_earned"}.get(mode, f"{mode}_progress")
        with _PLAN_LOCK:
            state = self.load_state_unlocked()
            state[key] = int(progress)
            arenas = state.setdefault("arenas", {})
            arenas[mode] = {
                "progress": int(progress),
                "target": int(target or 0),
                "goal_mode": str(goal_mode or ""),
                "goal_progress": int(goal_progress if goal_progress is not None else progress),
                "stats": snapshot.get("stats", {}),
                "reward_steps": snapshot.get("reward_steps", []),
                "reward_progress": snapshot.get("reward_progress", {}),
                "rewards_complete": bool(arena_rewards_complete(snapshot) if rewards_complete is None else rewards_complete),
                "time_left": snapshot.get("time_left"),
                "banned": snapshot.get("banned", False),
                "cap_target": snapshot.get("cap_target"),
                "pa_cap": snapshot.get("pa_cap"),
                "pa_streak": snapshot.get("pa_streak"),
                "pa_max_streak": snapshot.get("pa_max_streak"),
                "streak_rewards": snapshot.get("streak_rewards", []),
            }
            self.save_state_unlocked(state)

    def update_arena_block_state(
        self,
        block: dict[str, Any],
        goal_mode: str,
        goal_progress: int,
        target: int,
        snapshot: dict[str, Any],
    ) -> None:
        block_id = str(block.get("id", "") or "").strip()
        if not block_id:
            return
        with _PLAN_LOCK:
            state = self.load_state_unlocked()
            arena_blocks = state.setdefault("arena_blocks", {})
            arena_blocks[block_id] = {
                "mode": str(block.get("mode", "") or ""),
                "goal_mode": goal_mode,
                "progress": int(goal_progress),
                "target": int(target),
                "cycle_wins": self.arena_block_cycle_wins(state, block_id),
                "arena_counter": arena_progress(snapshot, str(block.get("mode", "") or "")),
                "rewards_complete": arena_rewards_complete(snapshot),
            }
            self.save_state_unlocked(state)

    def arena_goal(self, block: dict[str, Any], snapshot: dict[str, Any]) -> tuple[str, int, int]:
        mode = str(block.get("mode", "") or "").strip().casefold()
        if mode == "platinum":
            goal_mode = normalize_arena_goal_mode(block.get("goal_mode"))
            if goal_mode == "cycle_wins":
                target = max(0, int(block.get("target_cycle_platinum", block.get("target_platinum", 300)) or 0))
                progress = self.arena_block_cycle_platinum(self.load_state(), str(block.get("id", "") or ""), snapshot)
                return goal_mode, progress, target
            target = max(0, int(block.get("target_platinum", 300) or 0))
            return "arena_counter", arena_progress(snapshot, mode), target
        fallback_target = 5 if mode == "daily" else 6
        goal_mode = normalize_arena_goal_mode(block.get("goal_mode"))
        if goal_mode == "cycle_wins":
            target = max(0, int(block.get("target_cycle_wins", block.get("target_wins", fallback_target)) or 0))
            progress = self.arena_block_cycle_wins(self.load_state(), str(block.get("id", "") or ""))
            return goal_mode, progress, target
        target = max(0, int(block.get("target_arena_wins", block.get("target_wins", fallback_target)) or 0))
        return "arena_counter", arena_progress(snapshot, mode), target

    def arena_block_cycle_wins(self, state: dict[str, Any], block_id: str) -> int:
        wins = state.get("arena_cycle_wins", {})
        if not isinstance(wins, dict):
            return 0
        return max(0, int(wins.get(block_id, 0) or 0))

    def arena_block_cycle_platinum(self, state: dict[str, Any], block_id: str, snapshot: dict[str, Any]) -> int:
        progress = arena_progress(snapshot, "platinum")
        baselines = state.setdefault("arena_cycle_platinum_start", {})
        if not isinstance(baselines, dict):
            baselines = {}
            state["arena_cycle_platinum_start"] = baselines
        if block_id not in baselines:
            baselines[block_id] = progress
            self.save_state_unlocked(state)
        try:
            baseline = int(baselines.get(block_id, progress) or 0)
        except (TypeError, ValueError):
            baseline = progress
            baselines[block_id] = progress
            self.save_state_unlocked(state)
        return max(0, progress - baseline)

    def record_arena_block_win(self, block: dict[str, Any]) -> None:
        block_id = str(block.get("id", "") or "").strip()
        if not block_id:
            return
        with _PLAN_LOCK:
            state = self.load_state_unlocked()
            wins = state.setdefault("arena_cycle_wins", {})
            wins[block_id] = self.arena_block_cycle_wins(state, block_id) + 1
            self.save_state_unlocked(state)

    def load_config(self) -> dict[str, Any]:
        return get_account_plan(self.account_id)["config"]

    def load_state(self) -> dict[str, Any]:
        with _PLAN_LOCK:
            return self.load_state_unlocked()

    def load_state_unlocked(self) -> dict[str, Any]:
        data = load_all_plans()
        entry = data.get(self.account_id, {}) if isinstance(data.get(self.account_id), dict) else {}
        state = entry.get("state", {}) if isinstance(entry.get("state"), dict) else {}
        return state

    def save_state_unlocked(self, state: dict[str, Any]) -> None:
        data = load_all_plans()
        entry = data.get(self.account_id, {}) if isinstance(data.get(self.account_id), dict) else {}
        entry.setdefault("config", default_plan_config())
        entry["state"] = state
        data[self.account_id] = entry
        save_all_plans(data)

    def wish_due(self, player: dict[str, Any], field_name: str) -> bool:
        raw = player.get("raw", player)
        last = raw.get(field_name) if isinstance(raw, dict) else player.get(field_name)
        if not last:
            return True
        parsed = parse_game_time(str(last))
        return not parsed or parsed < current_reset_start_utc()

    def state_value(self, key: str) -> int:
        state = self.load_state()
        return int(state.get(key, 0) or 0)

    def state_done(self, key: str) -> bool:
        value = self.load_state().get(key, {})
        return bool(value.get("done")) if isinstance(value, dict) else bool(value)

    def heal_blocked(self, result: dict[str, Any]) -> bool:
        text = str(result.get("error", "")) + " " + str(result.get("prepare", ""))
        return "heal_team_failed" in text or "energy" in text.casefold() or "virtue" in text.casefold()

    def fallback_random_enabled(self) -> bool:
        config = self.load_config()
        return bool(config.get("fallback", {}).get("random_when_heal_blocked", True))

    def emit(self, event: str, phase: str, **payload: Any) -> None:
        update = {"event": event, "phase": phase, "timestamp": time.time(), **payload}
        if self.progress:
            self.progress(update)
        log_event(
            event,
            category="plan",
            source="plan_runner",
            initiator="cli",
            account_id=self.account_id,
            mode=str(payload.get("mode", "")),
            phase=phase,
            reason=str(payload.get("reason", "")),
            payload=compact_plan_payload(payload),
        )


def reset_state_if_needed(state: dict[str, Any], player: dict[str, Any]) -> None:
    key = current_reset_key()
    if state.get("reset_key") == key:
        return
    state.clear()
    state.update(
        {
            "reset_key": key,
            "started_at": time.time(),
            "wish_sk": {"done": False},
            "wish_vi": {"done": False},
            "daily_wins": 0,
            "platinum_wins": 0,
            "platinum_earned": 0,
            "random_wins": 0,
            "arenas": {},
            "arena_blocks": {},
            "arena_cycle_wins": {},
            "arena_cycle_platinum_start": {},
        }
    )


def normalize_plan_config(config: dict[str, Any], *, explicit_blocks: bool) -> dict[str, Any]:
    out = deep_merge({}, config if isinstance(config, dict) else {})
    blocks = out.get("blocks", []) if explicit_blocks else blocks_from_legacy_steps(out.get("steps", {}))
    out["blocks"] = normalize_plan_blocks(blocks)
    out["steps"] = legacy_steps_from_blocks(out["blocks"])
    return out


def is_recoverable_plan_error(exc: BaseException | str) -> bool:
    text = str(exc).strip().lower()
    return is_recoverable_arena_error(exc) or "network error:" in text


def plan_blocks(config: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = config.get("blocks", [])
    return normalize_plan_blocks(blocks if isinstance(blocks, list) else [])


def normalize_plan_blocks(blocks: Any) -> list[dict[str, Any]]:
    if not isinstance(blocks, list):
        return []
    normalized: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    for raw in blocks:
        if not isinstance(raw, dict):
            continue
        block_type = str(raw.get("type", "")).strip().casefold()
        if block_type == "wish":
            kind = str(raw.get("kind", "")).strip().casefold()
            if kind not in {"sk", "vi"}:
                continue
            base_id = f"wish_{kind}"
            normalized.append(
                {
                    "id": normalized_block_id(raw, base_id, counters),
                    "type": "wish",
                    "kind": kind,
                    "enabled": bool(raw.get("enabled", True)),
                }
            )
            continue
        if block_type != "arena":
            continue
        mode = str(raw.get("mode", "")).strip().casefold()
        if mode not in {"battle", "daily", "platinum", "random"}:
            continue
        base_id = f"{mode}_arena"
        block = {
            "id": normalized_block_id(raw, base_id, counters),
            "type": "arena",
            "mode": mode,
            "enabled": bool(raw.get("enabled", True)),
        }
        if mode == "platinum":
            goal_mode = normalize_arena_goal_mode(raw.get("goal_mode"))
            block["goal_mode"] = goal_mode
            block["target_cycle_platinum"] = max(
                0,
                int(raw.get("target_cycle_platinum", raw.get("target_platinum", 300)) or 0),
            )
            block["target_platinum"] = max(0, int(raw.get("target_platinum", 300) or 0))
        else:
            fallback_target = 5 if mode == "daily" else 6
            goal_mode = normalize_arena_goal_mode(raw.get("goal_mode"))
            block["goal_mode"] = goal_mode
            block["target_cycle_wins"] = max(
                0,
                int(raw.get("target_cycle_wins", raw.get("target_wins", fallback_target)) or 0),
            )
            block["target_arena_wins"] = max(
                0,
                int(raw.get("target_arena_wins", raw.get("target_wins", fallback_target)) or 0),
            )
            block["target_wins"] = (
                block["target_cycle_wins"] if goal_mode == "cycle_wins" else block["target_arena_wins"]
            )
        normalized.append(block)
    return normalized


def normalized_block_id(raw: dict[str, Any], base_id: str, counters: dict[str, int]) -> str:
    raw_id = str(raw.get("id", "") or "").strip()
    if raw_id:
        return raw_id
    counters[base_id] = counters.get(base_id, 0) + 1
    suffix = counters[base_id]
    return base_id if suffix == 1 else f"{base_id}_{suffix}"


def blocks_from_legacy_steps(steps: Any) -> list[dict[str, Any]]:
    source = steps if isinstance(steps, dict) else {}
    return [
        {"id": "wish_sk", "type": "wish", "kind": "sk", "enabled": source.get("wish_sk", {}).get("enabled", True)},
        {"id": "wish_vi", "type": "wish", "kind": "vi", "enabled": source.get("wish_vi", {}).get("enabled", True)},
        {
            "id": "daily_arena",
            "type": "arena",
            "mode": "daily",
            "enabled": source.get("daily_arena", {}).get("enabled", True),
            "goal_mode": "arena_counter",
            "target_arena_wins": int(source.get("daily_arena", {}).get("target_wins", 5) or 5),
            "target_wins": int(source.get("daily_arena", {}).get("target_wins", 5) or 5),
        },
        {
            "id": "platinum_arena",
            "type": "arena",
            "mode": "platinum",
            "enabled": source.get("platinum_arena", {}).get("enabled", True),
            "goal_mode": normalize_arena_goal_mode(source.get("platinum_arena", {}).get("goal_mode")),
            "target_cycle_platinum": int(
                source.get("platinum_arena", {}).get(
                    "target_cycle_platinum",
                    source.get("platinum_arena", {}).get("target_platinum", 300),
                )
                or 300
            ),
            "target_platinum": int(source.get("platinum_arena", {}).get("target_platinum", 300) or 300),
        },
        {
            "id": "random_arena",
            "type": "arena",
            "mode": "random",
            "enabled": source.get("random_arena", {}).get("enabled", True),
            "goal_mode": "arena_counter",
            "target_arena_wins": int(source.get("random_arena", {}).get("target_wins", 6) or 6),
            "target_wins": int(source.get("random_arena", {}).get("target_wins", 6) or 6),
        },
    ]


def legacy_steps_from_blocks(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    steps: dict[str, Any] = {}
    for block in blocks:
        if str(block.get("type", "")) == "wish":
            kind = str(block.get("kind", ""))
            if kind in {"sk", "vi"} and f"wish_{kind}" not in steps:
                steps[f"wish_{kind}"] = {"enabled": bool(block.get("enabled", True))}
        elif str(block.get("type", "")) == "arena":
            mode = str(block.get("mode", ""))
            key = f"{mode}_arena"
            if mode == "platinum" and key not in steps:
                steps[key] = {
                    "enabled": bool(block.get("enabled", True)),
                    "goal_mode": normalize_arena_goal_mode(block.get("goal_mode")),
                    "target_cycle_platinum": int(block.get("target_cycle_platinum", block.get("target_platinum", 300)) or 300),
                    "target_platinum": int(block.get("target_platinum", 300) or 300),
                }
            elif mode in {"battle", "daily", "random"} and key not in steps:
                fallback_target = 5 if mode == "daily" else 6
                goal_mode = normalize_arena_goal_mode(block.get("goal_mode"))
                target_wins = (
                    int(block.get("target_cycle_wins", fallback_target) or fallback_target)
                    if goal_mode == "cycle_wins"
                    else int(block.get("target_arena_wins", block.get("target_wins", fallback_target)) or fallback_target)
                )
                steps[key] = {
                    "enabled": bool(block.get("enabled", True)),
                    "target_wins": target_wins,
                }
    return steps


def normalize_arena_snapshot(mode: str, data: dict[str, Any], metadata: dict[int, dict[str, Any]] | None = None) -> dict[str, Any]:
    info = data.get("info", data)
    if not isinstance(info, dict):
        info = {}
    stats = info.get("stats", {})
    if not isinstance(stats, dict):
        stats = {}
    rewards = info.get("rewards", {})
    if not isinstance(rewards, dict):
        rewards = {}
    return {
        "mode": mode,
        "ok": True,
        "team": data.get("team", []),
        "info": info,
        "stats": stats,
        "rewards": rewards,
        "reward_steps": reward_steps(rewards),
        "reward_progress": arena_reward_progress(
            {"battle": "Default", "random": "Random", "daily": "Daily", "platinum": "Platinum"}.get(mode, mode),
            data,
            metadata or {},
        ),
        "cap_target": platinum_cap_target(info),
        "pa_cap": platinum_cap(info, stats),
        "pa_streak": optional_arena_int(info, "pa_streak", "paStreak", "streak"),
        "pa_max_streak": optional_arena_int(info, "pa_max_streak", "paMaxStreak", "max_streak", "maxStreak"),
        "streak_rewards": platinum_streak_rewards(info),
        "time_left": arena_time_left(info),
        "banned": bool(info.get("banned", False)),
        "raw": data,
    }


def arena_progress(snapshot: dict[str, Any], mode: str) -> int:
    info = snapshot.get("info", {}) if isinstance(snapshot.get("info"), dict) else {}
    stats = snapshot.get("stats", {}) if isinstance(snapshot.get("stats"), dict) else {}
    if mode == "daily":
        return int(stats.get("da_daily_wins", info.get("da_daily_wins", 0)) or 0)
    if mode == "random":
        return int(stats.get("ra_monthly_wins", info.get("ra_monthly_wins", 0)) or 0)
    if mode == "platinum":
        return int(info.get("pa_cap", stats.get("pa_cap", 0)) or 0)
    if mode == "battle":
        return int(stats.get("ba_weekly_wins", info.get("ba_weekly_wins", 0)) or 0)
    return 0


def normalize_arena_goal_mode(raw: Any) -> str:
    value = str(raw or "").strip().casefold()
    return value if value in {"cycle_wins", "arena_counter"} else "arena_counter"


def arena_rewards_complete(snapshot: dict[str, Any]) -> bool:
    reward_progress = snapshot.get("reward_progress", {}) if isinstance(snapshot.get("reward_progress"), dict) else {}
    items = reward_progress.get("items", [])
    return bool(items) and all(bool(item.get("claimed")) for item in items if isinstance(item, dict))


def arena_battle_won(result: dict[str, Any], before_counter: int, after_counter: int) -> bool:
    battle = result.get("battle", {}) if isinstance(result.get("battle"), dict) else {}
    outcome = str(battle.get("outcome", "") or "").strip().casefold()
    return outcome == "victory" or after_counter > before_counter


def reward_steps(rewards: dict[str, Any]) -> list[int]:
    steps: list[int] = []
    for key in rewards:
        try:
            steps.append(int(key))
        except (TypeError, ValueError):
            continue
    return sorted(set(steps))


def platinum_cap_target(info: dict[str, Any]) -> int:
    for key in ("pa_cap_max", "paMaxCap", "cap_max", "max_cap", "capMax", "pa_cap_limit", "pa_limit", "daily_cap"):
        try:
            value = int(info.get(key, 0) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return 300


def current_reset_key(now: datetime | None = None) -> str:
    local = (now or datetime.now(timezone.utc)).astimezone(MOSCOW_TZ)
    if local.time() < dt_time(RESET_HOUR, 0):
        local = local - timedelta(days=1)
    return local.date().isoformat()


def current_reset_start_utc(now: datetime | None = None) -> datetime:
    local = (now or datetime.now(timezone.utc)).astimezone(MOSCOW_TZ)
    reset = local.replace(hour=RESET_HOUR, minute=0, second=0, microsecond=0)
    if local < reset:
        reset = reset - timedelta(days=1)
    return reset.astimezone(timezone.utc)


def seconds_until_next_reset(now: datetime | None = None) -> float:
    current = now or datetime.now(timezone.utc)
    local = current.astimezone(MOSCOW_TZ)
    reset = local.replace(hour=RESET_HOUR, minute=0, second=0, microsecond=0)
    if local >= reset:
        reset = reset + timedelta(days=1)
    return max(0.0, (reset.astimezone(timezone.utc) - current).total_seconds())


def parse_game_time(value: str) -> datetime | None:
    if not value or value.startswith("1970-"):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def sleep_with_stop(seconds: float, should_stop: Callable[[], bool] | None = None) -> bool:
    deadline = time.monotonic() + max(0.0, seconds)
    while time.monotonic() < deadline:
        if should_stop and should_stop():
            return True
        time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
    return False


def compact_plan_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": result.get("ok"),
        "mode": result.get("mode"),
        "error": result.get("error"),
        "prepare": result.get("prepare"),
        "battle": {
            key: result.get("battle", {}).get(key)
            for key in ("outcome", "winner", "turns_sent", "grade", "reason")
            if isinstance(result.get("battle"), dict) and key in result.get("battle", {})
        },
    }


def compact_plan_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "result" and isinstance(value, dict):
            compact[key] = compact_plan_result(value)
        elif key == "arena_event" and isinstance(value, dict):
            compact[key] = compact_plan_arena_event(value)
        else:
            compact[key] = value
    return compact


def compact_plan_arena_event(event: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in event.items():
        if key == "result" and isinstance(value, dict):
            compact[key] = compact_plan_result(value)
        elif key == "log" and isinstance(value, list):
            compact["log_count"] = len(value)
        elif key == "battle" and isinstance(value, dict):
            compact[key] = compact_plan_battle(value)
        elif key == "data" and isinstance(value, dict):
            compact[key] = {item: value.get(item) for item in ("turns", "next_turn", "winner", "state") if item in value}
        else:
            compact[key] = value
    return compact


def compact_plan_battle(battle: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: battle.get(key)
        for key in ("turns", "active_side", "winner", "outcome", "ended", "terminated")
        if key in battle
    }
    for side_key in ("player", "foe"):
        side = battle.get(side_key)
        if not isinstance(side, dict):
            continue
        team = side.get("team") if isinstance(side.get("team"), list) else []
        active = next((item for item in team if isinstance(item, dict) and item.get("active")), {})
        compact[side_key] = {
            "team_count": len(team),
            "deaths": sum(1 for item in team if isinstance(item, dict) and item.get("dead")),
            "active": {
                item: active.get(item)
                for item in ("id", "mid", "name", "chp", "max_hp", "element", "level")
                if item in active
            },
        }
    return compact


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out
