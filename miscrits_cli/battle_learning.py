from __future__ import annotations

import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .config import DATA_DIR
from .storage import load_json, save_json


BATTLE_HISTORY_FILE = DATA_DIR / "battle_history.json"
BATTLE_WEIGHTS_FILE = DATA_DIR / "battle_ai_weights.json"
BATTLE_LOG_DIR = DATA_DIR / "battle_logs"
BATTLE_LOG_INDEX_FILE = BATTLE_LOG_DIR / "index.json"
MAX_HISTORY = 0
MAX_EVENTS_PER_BATTLE = 5000
MAX_DECISIONS_PER_BATTLE = 1000
MAX_DAMAGE_SAMPLES_PER_BATTLE = 2000
LEARNING_RATE = 0.08
MAX_WEIGHT = 24.0
DAMAGE_LEARNING_RATE = 0.06
IMITATION_LEARNING_RATE = 0.045
IMITATION_MAX_REWARD = 0.75


def default_weights() -> dict[str, Any]:
    return {
        "version": 1,
        "updated_at": 0,
        "battles": 0,
        "actions": {},
        "reasons": {},
        "matchups": {},
        "pair_matchups": {},
        "opponent_actions": {},
        "opponent_matchups": {},
        "opponent_pair_matchups": {},
        "damage_model": default_damage_model(),
    }


def default_damage_model() -> dict[str, Any]:
    return {
        "version": 1,
        "samples": 0,
        "mae": 0.0,
        "mape": 0.0,
        "global": {"count": 0, "avg": 1.0, "scale": 1.0},
        "buckets": {},
    }


def load_weights() -> dict[str, Any]:
    data = load_json(BATTLE_WEIGHTS_FILE, default_weights())
    if not isinstance(data, dict):
        return default_weights()
    base = default_weights()
    base.update(data)
    for key in ("actions", "reasons", "matchups", "pair_matchups", "opponent_actions", "opponent_matchups", "opponent_pair_matchups"):
        if not isinstance(base.get(key), dict):
            base[key] = {}
    if not isinstance(base.get("damage_model"), dict):
        base["damage_model"] = default_damage_model()
    else:
        damage_model = default_damage_model()
        damage_model.update(base["damage_model"])
        if not isinstance(damage_model.get("buckets"), dict):
            damage_model["buckets"] = {}
        if not isinstance(damage_model.get("global"), dict):
            damage_model["global"] = {"count": 0, "avg": 1.0, "scale": 1.0}
        base["damage_model"] = damage_model
    return base


def save_weights(weights: dict[str, Any]) -> None:
    weights["updated_at"] = time.time()
    save_json(BATTLE_WEIGHTS_FILE, weights)


def learned_bonus(kind: str, action_id: int, attacker: dict[str, Any], defender: dict[str, Any], reason: str = "") -> float:
    weights = load_weights()
    bonus = 0.0
    action_entry = weights.get("actions", {}).get(action_key(kind, action_id), {})
    if isinstance(action_entry, dict):
        bonus += float(action_entry.get("weight", 0.0) or 0.0)
    if reason:
        for part in str(reason).split("_"):
            reason_entry = weights.get("reasons", {}).get(part, {})
            if isinstance(reason_entry, dict):
                bonus += float(reason_entry.get("weight", 0.0) or 0.0) * 0.35
    matchup_entry = weights.get("matchups", {}).get(matchup_key(attacker, defender, kind, action_id), {})
    if isinstance(matchup_entry, dict):
        bonus += float(matchup_entry.get("weight", 0.0) or 0.0)
    opponent_action = opponent_action_entry(weights.get("opponent_actions", {}), kind, action_id)
    if isinstance(opponent_action, dict):
        bonus += float(opponent_action.get("weight", 0.0) or 0.0) * 0.65
    opponent_matchup = opponent_matchup_entry(weights.get("opponent_matchups", {}), attacker, defender, kind, action_id)
    if isinstance(opponent_matchup, dict):
        bonus += float(opponent_matchup.get("weight", 0.0) or 0.0) * 0.7
    return clamp(bonus, -MAX_WEIGHT, MAX_WEIGHT)


def matchup_memory(attacker: dict[str, Any], defender: dict[str, Any]) -> float:
    weights = load_weights()
    entry = weights.get("pair_matchups", {}).get(pair_matchup_key(attacker, defender), {})
    if not isinstance(entry, dict):
        return 0.0
    bonus = float(entry.get("weight", 0.0) or 0.0)
    opponent_entry = weights.get("opponent_pair_matchups", {}).get(pair_matchup_key(attacker, defender), {})
    if isinstance(opponent_entry, dict):
        bonus += float(opponent_entry.get("weight", 0.0) or 0.0) * 0.65
    return clamp(bonus, -MAX_WEIGHT, MAX_WEIGHT)


def damage_multiplier(features: dict[str, Any]) -> float:
    weights = load_weights()
    model = weights.get("damage_model", {}) if isinstance(weights.get("damage_model"), dict) else {}
    global_scale = scale_from_bucket(model.get("global", {}))
    bucket_scales = []
    buckets = model.get("buckets", {}) if isinstance(model.get("buckets"), dict) else {}
    for key in damage_feature_keys(features):
        value = buckets.get(key, {})
        if isinstance(value, dict) and int(value.get("count", 0) or 0) >= 2:
            bucket_scales.append(scale_from_bucket(value))
    if not bucket_scales:
        return clamp(global_scale, 0.35, 2.75)
    adjustment = sum(scale - 1.0 for scale in bucket_scales) / max(2.0, len(bucket_scales) ** 0.5 * 2.0)
    return clamp(global_scale * (1.0 + adjustment), 0.35, 2.75)


def record_battle(history_entry: dict[str, Any]) -> None:
    history = load_json(BATTLE_HISTORY_FILE, [])
    if not isinstance(history, list):
        history = []
    history.append(history_entry)
    if MAX_HISTORY > 0:
        history = history[-MAX_HISTORY:]
    save_json(BATTLE_HISTORY_FILE, history)
    save_battle_log(history_entry)


def load_battle_history(limit: int | None = 50) -> list[dict[str, Any]]:
    history = load_json(BATTLE_HISTORY_FILE, [])
    if not isinstance(history, list):
        return []
    if limit is None or int(limit) <= 0:
        return history
    return history[-max(1, limit) :]


def save_battle_log(entry: dict[str, Any]) -> None:
    battle_id = safe_battle_id(str(entry.get("id", "") or ""))
    if not battle_id:
        return
    log_file = BATTLE_LOG_DIR / f"{battle_id}.json"
    save_json(log_file, entry)
    index = load_json(BATTLE_LOG_INDEX_FILE, [])
    if not isinstance(index, list):
        index = []
    summary = battle_log_summary(entry)
    summary["file"] = str(log_file)
    index = [item for item in index if not isinstance(item, dict) or str(item.get("id", "")) != battle_id]
    index.append(summary)
    index.sort(key=lambda item: float(item.get("finished_at", item.get("started_at", 0)) or 0))
    save_json(BATTLE_LOG_INDEX_FILE, index)


def load_battle_log_index(limit: int | None = 100, mode: str = "", outcome: str = "", text: str = "") -> list[dict[str, Any]]:
    index = load_json(BATTLE_LOG_INDEX_FILE, [])
    if not isinstance(index, list):
        index = []
    known = {str(item.get("id", "")) for item in index if isinstance(item, dict)}
    for entry in load_battle_history(None):
        if not isinstance(entry, dict):
            continue
        battle_id = str(entry.get("id", ""))
        if battle_id and battle_id not in known:
            index.append(battle_log_summary(entry))
            known.add(battle_id)
    mode = str(mode or "").strip().lower()
    outcome = str(outcome or "").strip().lower()
    text = str(text or "").strip().casefold()
    items = [item for item in index if isinstance(item, dict)]
    if mode:
        items = [item for item in items if battle_mode_key(item) == mode]
    if outcome:
        items = [item for item in items if str(item.get("outcome", "")).lower() == outcome]
    if text:
        items = [item for item in items if text in " ".join(str(item.get(key, "")) for key in ("id", "mode", "match_id", "outcome", "grade_label")).casefold()]
    items = sorted(items, key=lambda item: float(item.get("finished_at", item.get("started_at", 0)) or 0), reverse=True)
    if limit is None or int(limit) <= 0:
        return items
    return items[: max(1, int(limit))]


def load_battle_log(battle_id: str) -> dict[str, Any]:
    battle_id = safe_battle_id(battle_id)
    if not battle_id:
        return {}
    log_file = BATTLE_LOG_DIR / f"{battle_id}.json"
    data = load_json(log_file, {}) if log_file.exists() else {}
    if isinstance(data, dict) and data:
        data["timeline"] = battle_timeline(data)
        return data
    for item in reversed(load_battle_history(None)):
        if isinstance(item, dict) and str(item.get("id", "")) == battle_id:
            item = dict(item)
            item["timeline"] = battle_timeline(item)
            return item
    return {}


def battle_log_summary(entry: dict[str, Any]) -> dict[str, Any]:
    grade = entry.get("grade", {}) if isinstance(entry.get("grade"), dict) else {}
    post = entry.get("post_battle", {}) if isinstance(entry.get("post_battle"), dict) else {}
    summary = post.get("summary", {}) if isinstance(post.get("summary"), dict) else {}
    return {
        "id": str(entry.get("id", "")),
        "mode": entry.get("mode", ""),
        "match_id": entry.get("match_id", ""),
        "started_at": entry.get("started_at", 0),
        "finished_at": entry.get("finished_at", 0),
        "outcome": entry.get("outcome", "unknown"),
        "turns_sent": entry.get("turns_sent", 0),
        "grade_label": grade.get("label", ""),
        "grade_score": grade.get("score", 0),
        "ally_deaths": grade.get("ally_deaths", 0),
        "foe_deaths": grade.get("foe_deaths", 0),
        "decision_count": len(entry.get("decisions", [])) if isinstance(entry.get("decisions"), list) else 0,
        "event_count": len(entry.get("events", [])) if isinstance(entry.get("events"), list) else 0,
        "damage_sample_count": len(entry.get("damage_samples", [])) if isinstance(entry.get("damage_samples"), list) else 0,
        "missed_finishes": summary.get("missed_finishes", 0),
        "useless_switches": summary.get("useless_switches", 0),
        "lost_damage": summary.get("lost_damage", 0),
    }


def safe_battle_id(value: str) -> str:
    clean = "".join(ch for ch in str(value) if ch.isalnum() or ch in {"-", "_"})
    return clean[:80]


def battle_timeline(entry: dict[str, Any]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for event in entry.get("events", []) if isinstance(entry.get("events"), list) else []:
        if not isinstance(event, dict):
            continue
        name = str(event.get("event", "event"))
        row: dict[str, Any] = {
            "timestamp": event.get("timestamp", 0),
            "turns": event.get("turns", 0),
            "event": name,
        }
        if name == "decision":
            decision = event.get("decision", {}) if isinstance(event.get("decision"), dict) else {}
            debug = decision.get("debug", {}) if isinstance(decision.get("debug"), dict) else {}
            row.update(
                {
                    "actor": miscrit_label(event.get("active", {})),
                    "target": miscrit_label(event.get("foe", {})),
                    "action": decision_label(decision),
                    "reason": debug.get("reason", ""),
                    "sent": event.get("sent", False),
                    "hp": event.get("hp", {}),
                }
            )
        elif name == "damage_sample":
            row.update(
                {
                    "actor": miscrit_label(event.get("attacker", {})),
                    "target": miscrit_label(event.get("defender", {})),
                    "action": ability_label(event.get("ability", {})),
                    "actual_damage": event.get("actual", 0),
                    "expected_damage": event.get("expected", 0),
                    "error": event.get("error", 0),
                }
            )
        else:
            row["summary"] = event.get("summary", {})
            row["hp"] = event.get("hp", {})
            if "opcode" in event:
                row["opcode"] = event.get("opcode")
        timeline.append(row)
    timeline.sort(key=lambda item: (float(item.get("timestamp", 0) or 0), int(item.get("turns", 0) or 0)))
    return timeline


def miscrit_label(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    name = str(value.get("name", "") or "")
    mid = value.get("mid", "")
    return f"{name} #{mid}".strip() if mid else name


def ability_label(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    name = str(value.get("name", "") or "")
    ability_id = value.get("id", "")
    return f"{name} ({ability_id})".strip() if ability_id else name


def decision_label(decision: dict[str, Any]) -> str:
    kind = str(decision.get("type", "action"))
    debug = decision.get("debug", {}) if isinstance(decision.get("debug"), dict) else {}
    name = str(debug.get("name", "") or "")
    action_id = decision.get("id", "")
    if name:
        return f"{kind}: {name}"
    return f"{kind} {action_id}".strip()


def learning_status() -> dict[str, Any]:
    weights = load_weights()
    history = load_battle_history(None)
    recent = history[-20:]
    return {
        "ok": True,
        "weights_file": str(BATTLE_WEIGHTS_FILE),
        "history_file": str(BATTLE_HISTORY_FILE),
        "battles": weights.get("battles", 0),
        "history_total": len(history),
        "arena_stats": arena_history_stats(history),
        "actions": top_entries(weights.get("actions", {}), 12),
        "reasons": top_entries(weights.get("reasons", {}), 12),
        "matchups": top_entries(weights.get("matchups", {}), 12),
        "pair_matchups": top_entries(weights.get("pair_matchups", {}), 12),
        "opponent_actions": top_entries(weights.get("opponent_actions", {}), 12),
        "opponent_matchups": top_entries(weights.get("opponent_matchups", {}), 12),
        "damage_model": damage_model_status(weights.get("damage_model", {})),
        "recent_battles": list(reversed(recent)),
    }


EDITABLE_WEIGHT_BUCKETS = (
    "actions",
    "reasons",
    "matchups",
    "pair_matchups",
    "opponent_actions",
    "opponent_matchups",
    "opponent_pair_matchups",
)


def ai_dashboard() -> dict[str, Any]:
    weights = load_weights()
    history = load_battle_history(None)
    status = learning_status()
    recent = history[-80:]
    reason_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    score_series: list[dict[str, Any]] = []
    post_metrics = {
        "missed_finishes": 0,
        "useless_switches": 0,
        "lost_damage": 0.0,
        "decided": 0,
    }
    for entry in recent:
        grade = entry.get("grade", {}) if isinstance(entry.get("grade"), dict) else {}
        post = entry.get("post_battle", {}) if isinstance(entry.get("post_battle"), dict) else {}
        summary = post.get("summary", {}) if isinstance(post.get("summary"), dict) else {}
        outcome = str(entry.get("outcome", "") or "unknown").lower()
        score_series.append(
            {
                "id": entry.get("id", ""),
                "mode": battle_mode_key(entry),
                "outcome": outcome,
                "score": float(grade.get("score", 0.0) or 0.0),
                "ally_deaths": int(grade.get("ally_deaths", 0) or 0),
                "foe_deaths": int(grade.get("foe_deaths", 0) or 0),
                "finished_at": entry.get("finished_at", 0),
            }
        )
        if outcome in {"victory", "defeat"}:
            post_metrics["decided"] += 1
        post_metrics["missed_finishes"] += int(summary.get("missed_finishes", 0) or 0)
        post_metrics["useless_switches"] += int(summary.get("useless_switches", 0) or 0)
        post_metrics["lost_damage"] += float(summary.get("lost_damage", 0.0) or 0.0)
        for row in entry.get("decisions", []) if isinstance(entry.get("decisions"), list) else []:
            if not isinstance(row, dict) or not decision_applied(row):
                continue
            decision = row.get("decision", {}) if isinstance(row.get("decision"), dict) else {}
            debug = decision.get("debug", {}) if isinstance(decision.get("debug"), dict) else {}
            reason = str(debug.get("reason", decision.get("reason", "")) or "").strip()
            if reason:
                reason_counts[reason] += 1
            kind = str(decision.get("type", "action") or "action")
            action_id = int(decision.get("id", 0) or 0)
            if action_id:
                action_counts[action_key(kind, action_id)] += 1
    decided = max(1, int(post_metrics["decided"]))
    post_metrics["lost_damage"] = round(float(post_metrics["lost_damage"]), 3)
    post_metrics["missed_finish_rate"] = round(float(post_metrics["missed_finishes"]) / decided, 4)
    post_metrics["useless_switch_rate"] = round(float(post_metrics["useless_switches"]) / decided, 4)
    return {
        **status,
        "editable_weights": {key: top_entries(weights.get(key, {}), 10_000) for key in EDITABLE_WEIGHT_BUCKETS},
        "recent_series": score_series,
        "decision_reasons": counter_rows(reason_counts, 16),
        "decision_actions": counter_rows(action_counts, 16),
        "post_metrics": post_metrics,
        "logic": {
            "player_learning_buckets": sum(len(weights.get(key, {})) for key in ("actions", "reasons", "matchups", "pair_matchups")),
            "opponent_learning_buckets": sum(len(weights.get(key, {})) for key in ("opponent_actions", "opponent_matchups", "opponent_pair_matchups")),
            "damage_buckets": len(weights.get("damage_model", {}).get("buckets", {})) if isinstance(weights.get("damage_model"), dict) else 0,
        },
    }


def set_ai_weight(category: str, key: str, weight: float) -> dict[str, Any]:
    category = str(category or "").strip()
    key = str(key or "").strip()
    if category not in EDITABLE_WEIGHT_BUCKETS:
        raise ValueError(f"Unknown AI weight category: {category}")
    if not key:
        raise ValueError("AI weight key cannot be empty.")
    weights = load_weights()
    bucket = weights.setdefault(category, {})
    if not isinstance(bucket, dict):
        bucket = {}
        weights[category] = bucket
    entry = bucket.get(key, {})
    if not isinstance(entry, dict):
        entry = {}
    entry["weight"] = round(clamp(float(weight), -MAX_WEIGHT, MAX_WEIGHT), 5)
    entry.setdefault("count", 0)
    entry.setdefault("total", 0.0)
    entry.setdefault("avg", 0.0)
    entry["manual"] = True
    entry["updated_at"] = time.time()
    bucket[key] = entry
    save_weights(weights)
    return {"category": category, "key": key, **entry}


def counter_rows(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    return [{"key": key, "count": count} for key, count in counter.most_common(max(1, limit))]


def arena_history_stats(history: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, Any] = {"total": summarize_history_scope(history)}
    for mode in ("battle", "daily", "platinum", "random"):
        scoped = [item for item in history if battle_mode_key(item) == mode]
        stats[mode] = summarize_history_scope(scoped)
    return stats


def summarize_history_scope(history: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(1 for item in history if str(item.get("outcome", "")).lower() == "victory")
    losses = sum(1 for item in history if str(item.get("outcome", "")).lower() == "defeat")
    unknown = max(0, len(history) - wins - losses)
    total_decided = wins + losses
    return {
        "battles": len(history),
        "wins": wins,
        "losses": losses,
        "unknown": unknown,
        "win_rate": wins / total_decided if total_decided else 0.0,
    }


def battle_mode_key(item: dict[str, Any]) -> str:
    mode = str(item.get("mode", "") or "").strip().lower()
    if mode in {"default", "battle"}:
        return "battle"
    if mode in {"daily", "platinum", "random"}:
        return mode
    return mode


@dataclass
class BattleRecorder:
    mode: str
    match_id: str
    player_id: str
    battle_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: float = field(default_factory=time.time)
    start_snapshot: dict[str, Any] = field(default_factory=dict)
    last_snapshot: dict[str, Any] = field(default_factory=dict)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    damage_samples: list[dict[str, Any]] = field(default_factory=list)
    acknowledged_actions: set[tuple[int, str, int, int]] = field(default_factory=set)

    def observe_state(self, opcode: int, data: Any, snapshot: dict[str, Any]) -> None:
        if not self.start_snapshot and snapshot.get("player"):
            self.start_snapshot = compact_snapshot(snapshot)
        self.last_snapshot = compact_snapshot(snapshot)
        self.events.append(
            {
                "timestamp": time.time(),
                "event": "state",
                "opcode": opcode,
                "turns": snapshot.get("turns", 0),
                "summary": summarize_event_data(data),
                "hp": hp_summary(snapshot),
            }
        )
        self.events = self.events[-MAX_EVENTS_PER_BATTLE:]

    def acknowledge_decision(self, opcode: int, data: Any) -> bool:
        if not isinstance(data, dict) or str(data.get("user_id", "")) != str(self.player_id):
            return False
        action_id = int(data.get("id", 0) or 0)
        decision_type = decision_type_from_opcode(opcode)
        if not decision_type or not action_id:
            return False
        fingerprint = (int(opcode), str(data.get("user_id", "")), action_id, int(data.get("turns", 0) or 0))
        if fingerprint in self.acknowledged_actions:
            return False
        for row in self.decisions:
            if decision_applied(row) or not bool(row.get("sent", False)):
                continue
            decision = row.get("decision", {}) if isinstance(row.get("decision"), dict) else {}
            if str(decision.get("type", "")) != decision_type or int(decision.get("id", 0) or 0) != action_id:
                continue
            row["executed"] = True
            row["executed_at"] = time.time()
            row["executed_opcode"] = int(opcode)
            self.acknowledged_actions.add(fingerprint)
            self.events.append(
                {
                    "timestamp": row["executed_at"],
                    "event": "decision_acknowledged",
                    "turns": row.get("turns", 0),
                    "opcode": int(opcode),
                    "decision": row.get("decision", {}),
                }
            )
            self.events = self.events[-MAX_EVENTS_PER_BATTLE:]
            return True
        return False

    def observe_damage_samples(self, samples: list[dict[str, Any]]) -> None:
        if not samples:
            return
        self.damage_samples.extend(samples)
        self.damage_samples = self.damage_samples[-MAX_DAMAGE_SAMPLES_PER_BATTLE:]
        for sample in samples:
            self.events.append(
                {
                    "timestamp": time.time(),
                    "event": "damage_sample",
                    "turns": sample.get("turns", 0),
                    "ability": sample.get("ability", {}),
                    "attacker": sample.get("attacker", {}),
                    "defender": sample.get("defender", {}),
                    "expected": sample.get("predicted_damage", 0),
                    "actual": sample.get("actual_damage", 0),
                    "error": sample.get("error", 0),
                    "features": sample.get("features", {}),
                }
            )
        self.events = self.events[-MAX_EVENTS_PER_BATTLE:]

    def record_decision(self, decision: dict[str, Any], sent: bool, snapshot: dict[str, Any]) -> None:
        active = active_miscrit(snapshot, "player")
        foe = active_miscrit(snapshot, "foe")
        row = {
            "timestamp": time.time(),
            "turns": snapshot.get("turns", 0),
            "sent": sent,
            "executed": False,
            "decision": clean_decision(decision),
            "active": compact_miscrit(active),
            "foe": compact_miscrit(foe),
            "hp": hp_summary(snapshot),
        }
        self.decisions.append(row)
        self.decisions = self.decisions[-MAX_DECISIONS_PER_BATTLE:]
        self.events.append({"timestamp": row["timestamp"], "event": "decision", **row})
        self.events = self.events[-MAX_EVENTS_PER_BATTLE:]

    def finish(self, outcome: str, turns_sent: int, final_snapshot: dict[str, Any]) -> dict[str, Any]:
        if final_snapshot:
            self.last_snapshot = compact_snapshot(final_snapshot)
        turns_executed = sum(1 for row in self.decisions if decision_applied(row))
        grade = grade_battle(outcome, self.start_snapshot, self.last_snapshot, turns_executed)
        entry = {
            "id": self.battle_id,
            "mode": self.mode,
            "match_id": self.match_id,
            "started_at": self.started_at,
            "finished_at": time.time(),
            "outcome": outcome,
            "turns_sent": turns_sent,
            "turns_executed": turns_executed,
            "grade": grade,
            "start": self.start_snapshot,
            "finish": self.last_snapshot,
            "decisions": self.decisions,
            "damage_samples": self.damage_samples,
            "events": self.events,
        }
        entry["post_battle"] = post_battle_analysis(entry)
        record_battle(entry)
        if outcome in {"victory", "defeat"}:
            update_weights_from_battle(entry)
        return grade


def update_weights_from_battle(entry: dict[str, Any]) -> None:
    grade = entry.get("grade", {})
    reward = float(grade.get("learning_reward", 0.0) or 0.0)
    weights = load_weights()
    changed = update_damage_model(weights, entry.get("damage_samples", []))
    if abs(reward) >= 0.01:
        weights["battles"] = int(weights.get("battles", 0) or 0) + 1
        for row in entry.get("decisions", []):
            if not isinstance(row, dict) or not decision_applied(row):
                continue
            decision = row.get("decision", {})
            if not isinstance(decision, dict):
                continue
            kind = str(decision.get("type", "none"))
            action_id = int(decision.get("id", 0) or 0)
            reason = str(decision.get("reason", ""))
            update_bucket(weights["actions"], action_key(kind, action_id), reward)
            for part in reason.split("_"):
                if part:
                    update_bucket(weights["reasons"], part, reward * 0.55)
            update_bucket(weights["matchups"], matchup_key(row.get("active", {}), row.get("foe", {}), kind, action_id), reward * 0.75)
            update_bucket(weights["pair_matchups"], pair_matchup_key(row.get("active", {}), row.get("foe", {})), reward * 0.9)
        changed = True
    if str(entry.get("outcome", "")) == "defeat":
        changed = update_opponent_imitation(weights, entry.get("damage_samples", [])) or changed
    if changed:
        save_weights(weights)


def update_opponent_imitation(weights: dict[str, Any], samples: Any) -> bool:
    if not isinstance(samples, list):
        return False
    changed = False
    for sample in samples:
        if not isinstance(sample, dict) or str(sample.get("side", "")) != "foe":
            continue
        ability = sample.get("ability", {}) if isinstance(sample.get("ability"), dict) else {}
        attacker = sample.get("attacker", {}) if isinstance(sample.get("attacker"), dict) else {}
        defender = sample.get("defender", {}) if isinstance(sample.get("defender"), dict) else {}
        action_id = int(ability.get("id", 0) or 0)
        if not action_id or not attacker or not defender:
            continue
        reward = opponent_sample_reward(sample)
        if reward <= 0:
            continue
        update_bucket(weights["opponent_actions"], action_key("ability", action_id), reward)
        update_bucket(weights["opponent_matchups"], matchup_key(attacker, defender, "ability", action_id), reward * 0.85)
        update_bucket(weights["opponent_pair_matchups"], pair_matchup_key(attacker, defender), reward * 0.65)
        changed = True
    return changed


def opponent_sample_reward(sample: dict[str, Any]) -> float:
    actual = max(0.0, float(sample.get("actual_damage", sample.get("actual", 0.0)) or 0.0))
    defender = sample.get("defender", {}) if isinstance(sample.get("defender"), dict) else {}
    defender_max = max(1.0, float(defender.get("max_hp", defender.get("hp", 1)) or 1))
    pressure = actual / defender_max
    action = sample.get("action", {}) if isinstance(sample.get("action"), dict) else {}
    reward = 0.16 + min(IMITATION_MAX_REWARD, pressure * 1.8)
    if bool(action.get("dead", False)) or actual >= float(defender.get("chp", 0) or 0) > 0:
        reward += 0.22
    if bool(action.get("crit", False)):
        reward *= 0.75
    return clamp(reward * IMITATION_LEARNING_RATE / max(0.01, LEARNING_RATE), 0.0, IMITATION_MAX_REWARD)


def post_battle_analysis(entry: dict[str, Any]) -> dict[str, Any]:
    decisions = [row for row in entry.get("decisions", []) if isinstance(row, dict)]
    events = [row for row in entry.get("events", []) if isinstance(row, dict)]
    missed_finishes = missed_finish_opportunities(decisions)
    useless_switches = useless_switch_decisions(decisions)
    lost_damage = lost_damage_from_ability_choices(decisions)
    losing_move = suspected_losing_move(decisions, events, str(entry.get("outcome", "")))
    severity = "ok"
    if str(entry.get("outcome", "")) == "defeat":
        severity = "critical" if losing_move else "bad"
    elif missed_finishes["count"] or useless_switches["count"] or lost_damage["total"] >= 30.0:
        severity = "needs_review"
    return {
        "severity": severity,
        "summary": {
            "missed_finishes": missed_finishes["count"],
            "useless_switches": useless_switches["count"],
            "lost_damage": round(lost_damage["total"], 3),
            "has_suspected_losing_move": bool(losing_move),
        },
        "suspected_losing_move": losing_move,
        "missed_finishes": missed_finishes["items"],
        "useless_switches": useless_switches["items"],
        "lost_damage": lost_damage["items"],
    }


def missed_finish_opportunities(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    items = []
    for row in decisions:
        if not decision_applied(row):
            continue
        decision = row.get("decision", {}) if isinstance(row.get("decision"), dict) else {}
        candidates = decision.get("candidates", []) if isinstance(decision.get("candidates"), list) else []
        lethal_candidates = [item for item in candidates if isinstance(item, dict) and bool(item.get("lethal", False))]
        if not lethal_candidates:
            continue
        chose_lethal = str(decision.get("type", "")) == "ability" and bool(decision.get("lethal", False))
        if chose_lethal:
            continue
        best = max(lethal_candidates, key=lambda item: float(item.get("damage", 0.0) or 0.0))
        items.append(
            {
                "turns": row.get("turns", 0),
                "chosen": compact_decision_ref(decision),
                "best_finisher": compact_decision_ref(best),
                "active": row.get("active", {}),
                "foe": row.get("foe", {}),
            }
        )
    return {"count": len(items), "items": items[:12]}


def useless_switch_decisions(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    items = []
    for index, row in enumerate(decisions):
        if not decision_applied(row):
            continue
        decision = row.get("decision", {}) if isinstance(row.get("decision"), dict) else {}
        if str(decision.get("type", "")) != "switch":
            continue
        reason = str(decision.get("reason", ""))
        if reason == "active_dead":
            continue
        gain = optional_float(decision.get("gain"))
        lethal_incoming = bool(decision.get("lethal_incoming", False))
        survives = decision.get("survives", True)
        next_row = next_sent_decision(decisions, index)
        hp_drop = 0.0
        foe_drop = 0.0
        if next_row:
            hp_drop = hp_ratio_delta(row, next_row, "player")
            foe_drop = hp_ratio_delta(row, next_row, "foe")
        useless = lethal_incoming or survives is False or (gain is not None and gain <= 8.0) or (hp_drop >= 0.16 and foe_drop <= 0.04)
        if not useless:
            continue
        items.append(
            {
                "turns": row.get("turns", 0),
                "chosen": compact_decision_ref(decision),
                "reason": "lethal_incoming" if lethal_incoming else "low_gain" if gain is not None and gain <= 8.0 else "bad_trade",
                "gain": gain,
                "hp_drop_after": round(hp_drop, 4),
                "foe_hp_drop_after": round(foe_drop, 4),
                "active_before": row.get("active", {}),
                "foe": row.get("foe", {}),
            }
        )
    return {"count": len(items), "items": items[:12]}


def lost_damage_from_ability_choices(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0.0
    items = []
    for row in decisions:
        if not decision_applied(row):
            continue
        decision = row.get("decision", {}) if isinstance(row.get("decision"), dict) else {}
        if str(decision.get("type", "")) != "ability":
            continue
        candidates = [item for item in decision.get("candidates", []) if isinstance(item, dict)] if isinstance(decision.get("candidates"), list) else []
        if not candidates:
            continue
        chosen_damage = optional_float(decision.get("damage")) or 0.0
        best = max(candidates, key=lambda item: float(item.get("damage", 0.0) or 0.0))
        best_damage = float(best.get("damage", 0.0) or 0.0)
        lost = max(0.0, best_damage - chosen_damage)
        if lost < max(8.0, best_damage * 0.18):
            continue
        total += lost
        items.append(
            {
                "turns": row.get("turns", 0),
                "chosen": compact_decision_ref(decision),
                "best_damage_option": compact_decision_ref(best),
                "lost_damage": round(lost, 3),
                "active": row.get("active", {}),
                "foe": row.get("foe", {}),
            }
        )
    return {"total": total, "items": items[:16]}


def suspected_losing_move(decisions: list[dict[str, Any]], events: list[dict[str, Any]], outcome: str) -> dict[str, Any]:
    if outcome != "defeat":
        return {}
    best_row: dict[str, Any] | None = None
    best_score = 0.0
    for index, row in enumerate(decisions):
        if not decision_applied(row):
            continue
        next_row = next_sent_decision(decisions, index)
        hp_drop = hp_ratio_delta(row, next_row, "player") if next_row else hp_ratio_to_finish_drop(row, events)
        foe_drop = hp_ratio_delta(row, next_row, "foe") if next_row else 0.0
        decision = row.get("decision", {}) if isinstance(row.get("decision"), dict) else {}
        penalty = hp_drop - foe_drop * 0.45
        if str(decision.get("type", "")) == "switch" and bool(decision.get("lethal_incoming", False)):
            penalty += 0.28
        if missed_finish_opportunities([row])["count"]:
            penalty += 0.35
        if penalty > best_score:
            best_score = penalty
            best_row = row
    if not best_row or best_score < 0.18:
        return {}
    decision = best_row.get("decision", {}) if isinstance(best_row.get("decision"), dict) else {}
    return {
        "turns": best_row.get("turns", 0),
        "score": round(best_score, 4),
        "decision": compact_decision_ref(decision),
        "active": best_row.get("active", {}),
        "foe": best_row.get("foe", {}),
        "hp": best_row.get("hp", {}),
    }


def next_sent_decision(decisions: list[dict[str, Any]], index: int) -> dict[str, Any]:
    for row in decisions[index + 1 :]:
        if isinstance(row, dict) and decision_applied(row):
            return row
    return {}


def hp_ratio_delta(before: dict[str, Any], after: dict[str, Any], side: str) -> float:
    before_ratio = nested_hp_ratio(before, side)
    after_ratio = nested_hp_ratio(after, side)
    return max(0.0, before_ratio - after_ratio)


def hp_ratio_to_finish_drop(row: dict[str, Any], events: list[dict[str, Any]]) -> float:
    before_ratio = nested_hp_ratio(row, "player")
    final_ratio = before_ratio
    for event in reversed(events):
        hp = event.get("hp", {}) if isinstance(event.get("hp"), dict) else {}
        player = hp.get("player", {}) if isinstance(hp.get("player"), dict) else {}
        if "hp_ratio" in player:
            final_ratio = float(player.get("hp_ratio", final_ratio) or 0.0)
            break
    return max(0.0, before_ratio - final_ratio)


def nested_hp_ratio(row: dict[str, Any], side: str) -> float:
    hp = row.get("hp", {}) if isinstance(row.get("hp"), dict) else {}
    side_data = hp.get(side, {}) if isinstance(hp.get(side), dict) else {}
    return float(side_data.get("hp_ratio", 0.0) or 0.0)


def compact_decision_ref(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": decision.get("type", ""),
        "id": decision.get("id", 0),
        "name": decision.get("name", ""),
        "reason": decision.get("reason", ""),
        "score": decision.get("score"),
        "damage": decision.get("damage"),
        "lethal": decision.get("lethal"),
    }


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def update_bucket(bucket: dict[str, Any], key: str, reward: float) -> None:
    if not key:
        return
    entry = bucket.get(key, {})
    if not isinstance(entry, dict):
        entry = {}
    count = int(entry.get("count", 0) or 0) + 1
    total = float(entry.get("total", 0.0) or 0.0) + reward
    old_weight = float(entry.get("weight", 0.0) or 0.0)
    target = clamp(total / max(1, count) * 12.0, -MAX_WEIGHT, MAX_WEIGHT)
    entry.update(
        {
            "count": count,
            "total": round(total, 5),
            "avg": round(total / max(1, count), 5),
            "weight": round(clamp(old_weight * (1.0 - LEARNING_RATE) + target * LEARNING_RATE, -MAX_WEIGHT, MAX_WEIGHT), 5),
            "updated_at": time.time(),
        }
    )
    bucket[key] = entry


def update_damage_model(weights: dict[str, Any], samples: Any) -> bool:
    if not isinstance(samples, list) or not samples:
        return False
    model = weights.get("damage_model")
    if not isinstance(model, dict):
        model = default_damage_model()
        weights["damage_model"] = model
    buckets = model.setdefault("buckets", {})
    if not isinstance(buckets, dict):
        buckets = {}
        model["buckets"] = buckets
    changed = False
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        base = float(sample.get("base_damage", 0.0) or 0.0)
        predicted = float(sample.get("predicted_damage", 0.0) or 0.0)
        actual = float(sample.get("actual_damage", 0.0) or 0.0)
        if base <= 0.0 or actual <= 0.0:
            continue
        observed = clamp(actual / base, 0.2, 4.0)
        features = sample.get("features", {})
        if not isinstance(features, dict):
            features = {}
        update_damage_bucket(model, "global", observed)
        for key in damage_feature_keys(features):
            update_damage_bucket(buckets, key, observed)
        count = int(model.get("samples", 0) or 0) + 1
        abs_error = abs(actual - predicted)
        pct_error = abs_error / max(1.0, actual)
        model["samples"] = count
        model["mae"] = round(rolling_average(float(model.get("mae", 0.0) or 0.0), abs_error, count), 5)
        model["mape"] = round(rolling_average(float(model.get("mape", 0.0) or 0.0), pct_error, count), 5)
        changed = True
    return changed


def update_damage_bucket(bucket: dict[str, Any], key: str, observed_scale: float) -> None:
    entry = bucket.get(key, {}) if key != "global" else bucket.get("global", {})
    if not isinstance(entry, dict):
        entry = {}
    count = int(entry.get("count", 0) or 0) + 1
    old_avg = float(entry.get("avg", 1.0) or 1.0)
    old_scale = float(entry.get("scale", old_avg) or old_avg)
    avg = rolling_average(old_avg, observed_scale, count)
    scale = old_scale * (1.0 - DAMAGE_LEARNING_RATE) + observed_scale * DAMAGE_LEARNING_RATE
    entry.update(
        {
            "count": count,
            "avg": round(avg, 5),
            "scale": round(clamp(scale, 0.25, 3.5), 5),
            "updated_at": time.time(),
        }
    )
    if key == "global":
        bucket["global"] = entry
    else:
        bucket[key] = entry


def damage_feature_keys(features: dict[str, Any]) -> list[str]:
    if not isinstance(features, dict):
        return []
    keys = [
        f"kind:{features.get('kind', 'Unknown')}",
        f"element:{features.get('element', 'Physical')}",
        f"ratio:{features.get('ratio_bucket', 'even')}",
        f"defense:{features.get('defense_state', 'normal')}",
    ]
    ability_id = int(features.get("ability_id", 0) or 0)
    if ability_id:
        keys.append(f"ability:{ability_id}")
    if bool(features.get("true_damage", False)):
        keys.append("true_damage")
    if bool(features.get("multi_hit", False)):
        keys.append("multi_hit")
    if bool(features.get("crit", False)):
        keys.append("crit")
    element_multiplier_value = float(features.get("element_multiplier", 1.0) or 1.0)
    if element_multiplier_value > 1.0:
        keys.append("element:strong")
    elif element_multiplier_value < 1.0:
        keys.append("element:resisted")
    attacker_mid = int(features.get("attacker_mid", 0) or 0)
    defender_mid = int(features.get("defender_mid", 0) or 0)
    if attacker_mid and defender_mid:
        keys.append(f"matchup:{attacker_mid}>{defender_mid}")
    return keys


def damage_model_status(model: Any) -> dict[str, Any]:
    if not isinstance(model, dict):
        model = default_damage_model()
    buckets = model.get("buckets", {}) if isinstance(model.get("buckets"), dict) else {}
    return {
        "samples": int(model.get("samples", 0) or 0),
        "mae": model.get("mae", 0.0),
        "mape": model.get("mape", 0.0),
        "global": model.get("global", {}),
        "top_buckets": top_entries(buckets, 16),
    }


def scale_from_bucket(bucket: Any) -> float:
    if not isinstance(bucket, dict):
        return 1.0
    return float(bucket.get("scale", bucket.get("avg", 1.0)) or 1.0)


def rolling_average(old: float, value: float, count: int) -> float:
    if count <= 1:
        return value
    return old + (value - old) / float(count)


def grade_battle(outcome: str, start: dict[str, Any], finish: dict[str, Any], turns_sent: int) -> dict[str, Any]:
    player_start = side_stats(start, "player")
    player_finish = side_stats(finish, "player")
    foe_finish = side_stats(finish, "foe")
    hp_retained = player_finish["hp_ratio"]
    hp_lost = max(0.0, player_start["hp_ratio"] - player_finish["hp_ratio"])
    ally_deaths = player_finish["dead"]
    foe_deaths = foe_finish["dead"]
    death_gap = int(ally_deaths) - int(foe_deaths)
    foe_alive = player_finish["foe_alive"] if "foe_alive" in player_finish else foe_finish["alive"]
    score = 50.0
    if outcome == "victory":
        score = 72.0 + foe_deaths * 4.5 + hp_retained * 12.0 - ally_deaths * 3.0 - max(0, death_gap) * 8.0 - min(10.0, turns_sent * 0.06)
        if ally_deaths == 0 and hp_retained >= 0.94:
            label = "победа в сухую"
        elif ally_deaths == 0 and hp_retained >= 0.70:
            label = "уверенная победа"
        elif hp_retained >= 0.42:
            label = "нелёгкая победа"
        elif hp_retained >= 0.16:
            label = "победа на грани"
        else:
            label = "случайная победа"
    elif outcome == "defeat":
        foe_hp = foe_finish["hp_ratio"]
        score = 32.0 + foe_deaths * 10.0 - foe_hp * 18.0 - ally_deaths * 4.0 - max(0, death_gap) * 12.0
        if ally_deaths >= 4:
            score -= 10.0
        if foe_finish["dead"] == 0 and foe_hp >= 0.78:
            label = "фатальное поражение"
        elif foe_hp >= 0.45:
            label = "тяжёлое поражение"
        elif foe_hp >= 0.18 or foe_finish["alive"] <= 1:
            label = "поражение в борьбе"
        else:
            label = "поражение на волоске"
    else:
        score = 42.0 + hp_retained * 8.0 + foe_deaths * 4.0 - ally_deaths * 4.0 - max(0, death_gap) * 6.0
        label = "неопределённый результат"
    score = clamp(score, 0.0, 100.0)
    learning_reward = (score - 50.0) / 50.0
    if outcome == "victory":
        learning_reward += 0.35
    elif outcome == "defeat":
        learning_reward -= 0.35
    label = readable_grade_label(outcome, hp_retained, ally_deaths, foe_finish)
    return {
        "label": label,
        "score": round(score, 2),
        "learning_reward": round(clamp(learning_reward, -1.4, 1.4), 5),
        "outcome": outcome,
        "turns_sent": turns_sent,
        "player_hp_retained": round(hp_retained, 4),
        "player_hp_lost": round(hp_lost, 4),
        "ally_deaths": ally_deaths,
        "foe_deaths": foe_deaths,
        "death_gap": death_gap,
        "foe_alive": foe_alive,
        "foe_hp_remaining": round(foe_finish["hp_ratio"], 4),
    }


def readable_grade_label(outcome: str, hp_retained: float, ally_deaths: int, foe_finish: dict[str, Any]) -> str:
    if outcome == "victory":
        if ally_deaths == 0 and hp_retained >= 0.94:
            return "победа в сухую"
        if ally_deaths == 0 and hp_retained >= 0.70:
            return "уверенная победа"
        if hp_retained >= 0.42:
            return "нелёгкая победа"
        if hp_retained >= 0.16:
            return "победа на грани"
        return "случайная победа"
    if outcome == "defeat":
        foe_hp = float(foe_finish.get("hp_ratio", 0.0) or 0.0)
        if int(foe_finish.get("dead", 0) or 0) == 0 and foe_hp >= 0.78:
            return "фатальное поражение"
        if foe_hp >= 0.45:
            return "тяжёлое поражение"
        if foe_hp >= 0.18 or int(foe_finish.get("alive", 0) or 0) <= 1:
            return "поражение в борьбе"
        return "поражение на волоске"
    return "неопределённый результат"


def side_stats(snapshot: dict[str, Any], side: str) -> dict[str, Any]:
    team = snapshot.get(side, {}).get("team", []) if isinstance(snapshot.get(side), dict) else []
    if not isinstance(team, list) or not team:
        return {"alive": 0, "dead": 0, "hp_ratio": 0.0}
    total_hp = 0.0
    total_max = 0.0
    dead = 0
    for item in team:
        if not isinstance(item, dict):
            continue
        chp = max(0.0, float(item.get("chp", 0) or 0))
        max_hp = max(1.0, float(item.get("max_hp", item.get("hp", chp or 1)) or 1))
        total_hp += min(chp, max_hp)
        total_max += max_hp
        if chp <= 0:
            dead += 1
    alive = max(0, len(team) - dead)
    return {"alive": alive, "dead": dead, "hp_ratio": total_hp / max(1.0, total_max)}


def compact_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "battle_type": snapshot.get("battle_type", ""),
        "turns": snapshot.get("turns", 0),
        "winner": snapshot.get("winner", ""),
        "player": compact_side(snapshot.get("player", {})),
        "foe": compact_side(snapshot.get("foe", {})),
    }


def compact_side(side: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(side, dict):
        return {"user_id": "", "username": "", "team": []}
    team = side.get("team", [])
    return {
        "user_id": side.get("user_id", ""),
        "username": side.get("username", ""),
        "team": [compact_miscrit(item) for item in team if isinstance(item, dict)] if isinstance(team, list) else [],
    }


def compact_miscrit(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id", 0),
        "mid": item.get("mid", 0),
        "name": item.get("name", ""),
        "element": item.get("element", ""),
        "level": item.get("level", ""),
        "chp": item.get("chp", 0),
        "max_hp": item.get("max_hp", item.get("hp", 0)),
        "active": item.get("active", False),
        "dead": item.get("dead", False),
    }


def clean_decision(decision: dict[str, Any]) -> dict[str, Any]:
    debug = decision.get("debug", {}) if isinstance(decision.get("debug"), dict) else {}
    return {
        "type": decision.get("type", "none"),
        "id": decision.get("id", 0),
        "name": debug.get("name", ""),
        "reason": decision.get("reason", ""),
        "score": debug.get("score"),
        "damage": debug.get("damage"),
        "utility": debug.get("utility"),
        "lethal": debug.get("lethal"),
        "near_lethal": debug.get("near_lethal"),
        "redundant": debug.get("redundant"),
        "immune_blocked": debug.get("immune_blocked"),
        "gain": debug.get("gain"),
        "active_score": debug.get("active_score"),
        "incoming": debug.get("incoming"),
        "incoming_ratio": debug.get("incoming_ratio"),
        "survives": debug.get("survives"),
        "lethal_incoming": debug.get("lethal_incoming"),
        "win_plan": debug.get("win_plan") or (debug.get("plan", {}) if isinstance(debug.get("plan"), dict) else {}).get("mode"),
        "win_adjustment": debug.get("win_adjustment"),
        "lookahead_adjustment": debug.get("lookahead_adjustment"),
        "lookahead": debug.get("lookahead", {}),
        "candidates": debug.get("candidates", [])[:6] if isinstance(debug.get("candidates", []), list) else [],
    }


def decision_type_from_opcode(opcode: int) -> str:
    if int(opcode) == 1:
        return "switch"
    if int(opcode) in {2, 6, 3}:
        return "ability"
    return ""


def decision_applied(row: dict[str, Any]) -> bool:
    if "executed" in row:
        return bool(row.get("executed", False))
    return bool(row.get("sent", False))


def hp_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "player": side_stats(snapshot, "player"),
        "foe": side_stats(snapshot, "foe"),
    }


def active_miscrit(snapshot: dict[str, Any], side: str) -> dict[str, Any]:
    team = snapshot.get(side, {}).get("team", []) if isinstance(snapshot.get(side), dict) else []
    if not isinstance(team, list):
        return {}
    for item in team:
        if isinstance(item, dict) and bool(item.get("active", False)):
            return item
    return team[0] if team and isinstance(team[0], dict) else {}


def summarize_event_data(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    keep = {}
    for key in ("type", "user_id", "next_turn", "turns", "winner", "loser", "id", "pending", "captured", "capture_chance"):
        if key in data:
            keep[key] = data[key]
    actions = data.get("actions", [])
    if isinstance(actions, list):
        keep["actions"] = [summarize_action(item) for item in actions if isinstance(item, dict)]
    return keep


def summarize_action(action: dict[str, Any]) -> dict[str, Any]:
    return {
        key: action[key]
        for key in ("type", "target", "id", "hp", "chp", "damage", "ap", "dead", "crit", "miss", "winner")
        if key in action
    }


def action_key(kind: str, action_id: int) -> str:
    return f"{kind}:{int(action_id or 0)}"


def matchup_key(attacker: dict[str, Any], defender: dict[str, Any], kind: str, action_id: int) -> str:
    return f"{int(attacker.get('mid', 0) or 0)}>{int(defender.get('mid', 0) or 0)}:{kind}:{int(action_id or 0)}"


def opponent_action_entry(bucket: Any, kind: str, action_id: int) -> dict[str, Any]:
    if not isinstance(bucket, dict):
        return {}
    canonical_key = action_key(kind, action_id)
    canonical = bucket.get(canonical_key)
    if canonical_key in bucket and isinstance(canonical, dict):
        return canonical
    suffix = f":{int(action_id or 0)}"
    for key, value in bucket.items():
        if str(key).endswith(suffix) and isinstance(value, dict):
            return value
    return {}


def opponent_matchup_entry(
    bucket: Any,
    attacker: dict[str, Any],
    defender: dict[str, Any],
    kind: str,
    action_id: int,
) -> dict[str, Any]:
    if not isinstance(bucket, dict):
        return {}
    canonical_key = matchup_key(attacker, defender, kind, action_id)
    canonical = bucket.get(canonical_key)
    if canonical_key in bucket and isinstance(canonical, dict):
        return canonical
    prefix = f"{int(attacker.get('mid', 0) or 0)}>{int(defender.get('mid', 0) or 0)}:"
    suffix = f":{int(action_id or 0)}"
    for key, value in bucket.items():
        text = str(key)
        if text.startswith(prefix) and text.endswith(suffix) and isinstance(value, dict):
            return value
    return {}


def pair_matchup_key(attacker: dict[str, Any], defender: dict[str, Any]) -> str:
    return f"{int(attacker.get('mid', 0) or 0)}>{int(defender.get('mid', 0) or 0)}"


def top_entries(bucket: Any, limit: int) -> list[dict[str, Any]]:
    if not isinstance(bucket, dict):
        return []
    rows = []
    for key, value in bucket.items():
        if isinstance(value, dict):
            rows.append({"key": key, **value})
    rows.sort(
        key=lambda item: max(
            abs(float(item.get("weight", 0.0) or 0.0)),
            abs(float(item.get("scale", 1.0) or 1.0) - 1.0),
        ),
        reverse=True,
    )
    return rows[:limit]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
