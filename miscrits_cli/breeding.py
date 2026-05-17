from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from .data_cache import DataCache
from .nakama import MiscritsClient, RpcResult
from .storage import DATA_DIR, load_json, save_json


STAT_KEYS = ("hp", "spd", "ea", "pa", "ed", "pd")
RATINGS = ["S+", "S", "A+", "A", "B+", "B", "C+", "C", "D+", "D", "F+", "F", "F-"]
MAX_POOL_SIZE = 36
MIN_TARGET_SUM = 15
GREEN_STAT_WEIGHT = 9.0
NEUTRAL_STAT_WEIGHT = 4.0
RED_STAT_PENALTY = 3.0
MAX_SUM_WEIGHT = 12.0
EXPECTED_SUM_WEIGHT = 8.0
SPECIES_WEIGHT = 4.0


@dataclass(frozen=True)
class BreedConfig:
    max_breeds: int = 1
    min_max_sum: int = MIN_TARGET_SUM
    allow_splus_parents: int = 0
    dry_run: bool = False
    delay_seconds: float = 0.4
    target_mid: int | None = None
    target_element: str | None = None


def rating_from_sum(value: int) -> str:
    if value <= 0:
        return "?"
    clamped = max(6, min(18, value))
    return RATINGS[18 - clamped]


def evo_from_level(level: int) -> int:
    return max(0, min(3, int(level or 1) // 10))


def normalize_evo(raw: dict[str, Any], level: int) -> int:
    for key in ("evo", "evo_id", "evolution"):
        value = raw.get(key)
        if value in (None, ""):
            continue
        try:
            return max(0, min(3, int(value)))
        except (TypeError, ValueError):
            continue
    return evo_from_level(level)


def normalize_miscrit(raw: dict[str, Any]) -> dict[str, Any]:
    version = int(raw.get("v", 1) or 1)
    if version == 2:
        out = {
            "id": int(raw.get("id", 0) or 0),
            "mid": int(raw.get("m", 0) or 0),
            "level": int(raw.get("l", 1) or 1),
            "chp": int(raw.get("c", raw.get("chp", 1)) or 0),
            "fav": bool(raw.get("f", False)),
            "nick": str(raw.get("n", "") or ""),
            "hp": int(raw.get("h", 1) or 1),
            "spd": int(raw.get("s", 1) or 1),
            "ea": int(raw.get("e", 1) or 1),
            "pa": int(raw.get("p", 1) or 1),
            "ed": int(raw.get("d", 1) or 1),
            "pd": int(raw.get("pd", 1) or 1),
            "enchants": raw.get("enchants", raw.get("en", [])),
            "active_relics": active_relics_from_raw(raw),
        }
    else:
        out = {
            "id": int(raw.get("id", 0) or 0),
            "mid": int(raw.get("mId", raw.get("mid", 0)) or 0),
            "level": int(raw.get("level", 1) or 1),
            "chp": int(raw.get("chp", raw.get("c", 1)) or 0),
            "fav": bool(raw.get("fav", False)),
            "nick": str(raw.get("nick", "") or ""),
            "hp": int(raw.get("hp", 1) or 1),
            "spd": int(raw.get("spd", 1) or 1),
            "ea": int(raw.get("ea", 1) or 1),
            "pa": int(raw.get("pa", 1) or 1),
            "ed": int(raw.get("ed", 1) or 1),
            "pd": int(raw.get("pd", 1) or 1),
            "enchants": raw.get("enchants", raw.get("en", [])),
            "active_relics": active_relics_from_raw(raw),
        }
    out["evo"] = normalize_evo(raw, int(out.get("level", 1) or 1))
    out["rating_sum"] = sum(int(out[stat]) for stat in STAT_KEYS)
    out["rating"] = rating_from_sum(int(out["rating_sum"]))
    return out


def active_relics_from_raw(raw: dict[str, Any]) -> list[Any]:
    relics = raw.get("relics")
    if isinstance(relics, dict):
        return [item for item in relics.values() if item is not None]
    if isinstance(relics, list):
        return [item for item in relics if item is not None]
    return [raw.get(key) for key in ("r1", "r2", "r3", "r4") if raw.get(key) is not None]


def metadata_by_mid(items: Any) -> dict[int, dict[str, Any]]:
    if not isinstance(items, list):
        return {}
    out: dict[int, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        mid = int(item.get("id", item.get("mId", item.get("mid", 0))) or 0)
        if mid <= 0:
            continue
        names = item.get("names", [])
        if isinstance(names, list) and names:
            name = str(names[0])
        else:
            name = str(item.get("name", item.get("display_name", f"#{mid}")))
        out[mid] = {
            "mid": mid,
            "name": name,
            "names": [str(name) for name in names] if isinstance(names, list) else [],
            "element": str(item.get("element", "") or ""),
            "rarity": str(item.get("rarity", "") or ""),
            "abilities": item.get("abilities", []),
            "hp": str(item.get("hp", "") or ""),
            "spd": str(item.get("spd", "") or ""),
            "ea": str(item.get("ea", "") or ""),
            "pa": str(item.get("pa", "") or ""),
            "ed": str(item.get("ed", "") or ""),
            "pd": str(item.get("pd", "") or ""),
        }
    return out


def enrich_player(player: dict[str, Any], metadata: dict[int, dict[str, Any]]) -> dict[str, Any]:
    for miscrit in player["miscrits"]:
        meta = metadata.get(int(miscrit["mid"]), {})
        names = meta.get("names", [])
        evo = int(miscrit.get("evo", evo_from_level(int(miscrit.get("level", 1) or 1))) or 0)
        if isinstance(names, list) and names:
            miscrit["name"] = str(names[min(evo, len(names) - 1)] or names[0])
        else:
            miscrit["name"] = meta.get("name", f"#{miscrit['mid']}")
        miscrit["element"] = meta.get("element", "")
        miscrit["rarity"] = meta.get("rarity", "")
    return player


def normalize_player(data: dict[str, Any]) -> dict[str, Any]:
    miscrits = [normalize_miscrit(item) for item in data.get("miscrits", [])]
    return {
        "gold": int(data.get("gold", 0) or 0),
        "team": [int(item) for item in data.get("team_order", data.get("team", []))],
        "right_traits": [int(item) for item in data.get("right_traits", [])],
        "miscrits": miscrits,
    }


def breed_cost(player: dict[str, Any]) -> int:
    return 1500 if 400 in set(player.get("right_traits", [])) else 3000


def is_splus(miscrit: dict[str, Any]) -> bool:
    return int(miscrit.get("rating_sum", 0)) >= 18 or str(miscrit.get("rating", "")) == "S+"


def count_species_splus(miscrits: list[dict[str, Any]], mid: int) -> int:
    return sum(1 for item in miscrits if int(item["mid"]) == mid and is_splus(item))


def is_candidate(miscrit: dict[str, Any], player: dict[str, Any]) -> bool:
    if int(miscrit["level"]) != 1:
        return False
    if int(miscrit["id"]) in set(player.get("team", [])):
        return False
    if bool(miscrit.get("fav", False)):
        return False
    if is_splus(miscrit):
        return count_species_splus(player["miscrits"], int(miscrit["mid"])) > 1
    return True


def candidate_pool(player: dict[str, Any], max_pool_size: int = MAX_POOL_SIZE) -> list[dict[str, Any]]:
    items = [item for item in player["miscrits"] if is_candidate(item, player)]
    items.sort(key=lambda item: (-int(item["rating_sum"]), -count_green_stats(item), -int(item["spd"]), int(item["id"])))
    return items[:max_pool_size]


def count_green_stats(miscrit: dict[str, Any]) -> int:
    return sum(1 for stat in STAT_KEYS if int(miscrit[stat]) >= 3)


def score_triple(
    parents: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    player: dict[str, Any],
    min_max_sum: int,
    target_mid: int | None = None,
    target_element: str | None = None,
) -> dict[str, Any]:
    if target_mid is not None and all(int(parent["mid"]) != target_mid for parent in parents):
        return {}
    if target_element:
        target_normalized = target_element.casefold()
        if all(str(parent.get("element", "")).casefold() != target_normalized for parent in parents):
            return {}
    species_counts: dict[int, int] = {}
    expected_sum = 0.0
    max_sum = 0
    stat_score = 0.0
    for parent in parents:
        species_counts[int(parent["mid"])] = species_counts.get(int(parent["mid"]), 0) + 1
        expected_sum += float(parent["rating_sum"])
    expected_sum /= 3.0

    for stat in STAT_KEYS:
        counts = {1: 0, 2: 0, 3: 0}
        best = 1
        for parent in parents:
            color = max(1, min(3, int(parent[stat])))
            counts[color] += 1
            best = max(best, color)
        max_sum += best
        green_chance = counts[3] / 3.0
        neutral_chance = counts[2] / 3.0
        red_chance = counts[1] / 3.0
        stat_weight = 1.25 if stat == "spd" else 1.0
        stat_score += stat_weight * (
            green_chance * GREEN_STAT_WEIGHT
            + neutral_chance * NEUTRAL_STAT_WEIGHT
            - red_chance * RED_STAT_PENALTY
        )
        if counts[3] >= 2:
            stat_score += stat_weight * 2.5
        if counts[3] == 3:
            stat_score += stat_weight * 4.0

    if max_sum < min_max_sum:
        return {}
    if would_only_create_closed_splus_species(species_counts, max_sum, parents, player):
        return {}
    if single_species_closed_line(species_counts, player):
        return {}
    if dominant_species_closed_for_splus(species_counts, max_sum, player):
        return {}

    dominant_mid, dominant_count = max(species_counts.items(), key=lambda item: item[1])
    species_score = 0.0
    for mid, count in species_counts.items():
        probability = count / 3.0
        best_owned_sum = max((int(item["rating_sum"]) for item in player["miscrits"] if int(item["mid"]) == mid), default=0)
        has_splus = count_species_splus(player["miscrits"], mid) > 0
        missing_power = max(0, 18 - best_owned_sum)
        need_bonus = 12.0 if not has_splus else 2.0
        species_score += probability * (need_bonus + missing_power * SPECIES_WEIGHT)

    score = stat_score + max_sum * MAX_SUM_WEIGHT + expected_sum * EXPECTED_SUM_WEIGHT + species_score
    if dominant_count == 3:
        score += 8.0
    elif dominant_count == 2:
        score += 4.0
    else:
        score -= 6.0
    if target_mid is not None:
        target_count = sum(1 for parent in parents if int(parent["mid"]) == target_mid)
        score += target_count * 18.0
    if target_element:
        target_normalized = target_element.casefold()
        element_count = sum(1 for parent in parents if str(parent.get("element", "")).casefold() == target_normalized)
        score += element_count * 8.0

    ids = [int(parent["id"]) for parent in parents]
    return {
        "ids": ids,
        "score": round(score, 3),
        "max_sum": max_sum,
        "max_rating": rating_from_sum(max_sum),
        "expected_sum": round(expected_sum, 3),
        "dominant_mid": dominant_mid,
        "dominant_chance": round(dominant_count / 3.0, 3),
        "species_counts": {str(mid): count for mid, count in species_counts.items()},
        "parents": [snapshot_parent(parent) for parent in parents],
        "splus_parent_count": sum(1 for parent in parents if is_splus(parent)),
    }


def would_only_create_closed_splus_species(
    species_counts: dict[int, int],
    max_sum: int,
    parents: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    player: dict[str, Any],
) -> bool:
    if max_sum < 18:
        return False
    if all(is_splus(parent) for parent in parents):
        return True
    return all(count_species_splus(player["miscrits"], mid) > 0 for mid in species_counts)


def single_species_closed_line(species_counts: dict[int, int], player: dict[str, Any]) -> bool:
    return len(species_counts) == 1 and count_species_splus(player["miscrits"], next(iter(species_counts))) > 0


def dominant_species_closed_for_splus(species_counts: dict[int, int], max_sum: int, player: dict[str, Any]) -> bool:
    if max_sum < 18:
        return False
    dominant_mid, dominant_count = max(species_counts.items(), key=lambda item: item[1])
    return dominant_count >= 2 and count_species_splus(player["miscrits"], dominant_mid) > 0


def find_best_plan(
    player: dict[str, Any],
    min_max_sum: int = MIN_TARGET_SUM,
    allow_splus_parents: int = 0,
    target_mid: int | None = None,
    target_element: str | None = None,
) -> dict[str, Any]:
    pool = candidate_pool(player)
    best: dict[str, Any] = {}
    for parents in combinations(pool, 3):
        if sum(1 for parent in parents if is_splus(parent)) > allow_splus_parents:
            continue
        plan = score_triple(parents, player, min_max_sum, target_mid, target_element)
        if not plan:
            continue
        if not best or float(plan["score"]) > float(best["score"]):
            best = plan
    if best:
        best["candidate_count"] = len(pool)
        best["cost"] = breed_cost(player)
        best["target_mid"] = target_mid
        best["target_element"] = target_element or ""
    return best


def snapshot_parent(miscrit: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(miscrit["id"]),
        "mid": int(miscrit["mid"]),
        "level": int(miscrit["level"]),
        "rating_sum": int(miscrit["rating_sum"]),
        "rating": str(miscrit["rating"]),
        "name": str(miscrit.get("name", f"#{miscrit['mid']}")),
        "element": str(miscrit.get("element", "")),
        "rarity": str(miscrit.get("rarity", "")),
        "stats": {stat: int(miscrit[stat]) for stat in STAT_KEYS},
    }


class BreedRunner:
    def __init__(self, client: MiscritsClient) -> None:
        self.client = client

    def load_player(self) -> dict[str, Any]:
        result = self.client.get_player()
        if not result.success:
            raise RuntimeError(f"get_player failed: {result.raw}")
        return enrich_player(normalize_player(result.data), self.load_metadata())

    def load_metadata(self) -> dict[int, dict[str, Any]]:
        try:
            items = DataCache(self.client).get_json("miscrits.json")
        except Exception:
            return {}
        return metadata_by_mid(items)

    def plan(
        self,
        min_max_sum: int = MIN_TARGET_SUM,
        allow_splus_parents: int = 0,
        target_mid: int | None = None,
        target_element: str | None = None,
    ) -> dict[str, Any]:
        player = self.load_player()
        plan = find_best_plan(player, min_max_sum, allow_splus_parents, target_mid, target_element)
        return {"ok": bool(plan), "gold": player["gold"], "plan": plan}

    def options(self) -> dict[str, Any]:
        player = self.load_player()
        candidates = candidate_pool(player, 9999)
        by_mid: dict[int, dict[str, Any]] = {}
        elements: set[str] = set()
        for item in candidates:
            mid = int(item["mid"])
            elements.add(str(item.get("element", "")))
            entry = by_mid.setdefault(
                mid,
                {
                    "mid": mid,
                    "name": str(item.get("name", f"#{mid}")),
                    "element": str(item.get("element", "")),
                    "rarity": str(item.get("rarity", "")),
                    "count": 0,
                    "best_rating_sum": 0,
                    "best_rating": "?",
                },
            )
            entry["count"] += 1
            if int(item["rating_sum"]) > int(entry["best_rating_sum"]):
                entry["best_rating_sum"] = int(item["rating_sum"])
                entry["best_rating"] = str(item["rating"])
        targets = sorted(by_mid.values(), key=lambda item: (str(item["name"]).casefold(), int(item["mid"])))
        return {
            "ok": True,
            "gold": player["gold"],
            "cost": breed_cost(player),
            "candidate_count": len(candidates),
            "targets": targets,
            "elements": sorted(item for item in elements if item),
        }

    def breed_once(self, config: BreedConfig) -> dict[str, Any]:
        player = self.load_player()
        cost = breed_cost(player)
        if player["gold"] < cost:
            return {"ok": False, "reason": "not_enough_gold", "gold": player["gold"], "cost": cost}
        plan = find_best_plan(
            player,
            config.min_max_sum,
            config.allow_splus_parents,
            config.target_mid,
            config.target_element,
        )
        if not plan:
            return {"ok": False, "reason": "no_valid_plan", "gold": player["gold"]}
        if config.dry_run:
            return {"ok": True, "dry_run": True, "plan": plan}
        result = self.client.breed(plan["ids"])
        entry = build_log_entry(result, plan, cost)
        append_breed_log(entry)
        return {"ok": result.success, "result": result.__dict__, "plan": plan, "log": entry}

    def auto_breed(self, config: BreedConfig) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        stop_reason = "max_breeds_reached"
        for _ in range(max(1, config.max_breeds)):
            outcome = self.breed_once(config)
            if not outcome.get("ok"):
                stop_reason = str(outcome.get("reason", "breed_failed"))
                return {"ok": bool(entries), "bred": len(entries), "stop_reason": stop_reason, "entries": entries, "last": outcome}
            if config.dry_run:
                return {"ok": True, "dry_run": True, "plan": outcome.get("plan")}
            entries.append(outcome)
            time.sleep(max(0.0, config.delay_seconds))
        return {"ok": True, "bred": len(entries), "stop_reason": stop_reason, "entries": entries}


def build_log_entry(result: RpcResult, plan: dict[str, Any], cost: int) -> dict[str, Any]:
    data = result.data if isinstance(result.data, dict) else {}
    child = normalize_miscrit(data) if data else {}
    if child:
        for parent in plan.get("parents", []):
            if int(parent.get("mid", 0)) != int(child.get("mid", 0)):
                continue
            child.setdefault("name", parent.get("name", f"#{child['mid']}"))
            child.setdefault("element", parent.get("element", ""))
            child.setdefault("rarity", parent.get("rarity", ""))
            break
    return {
        "timestamp_unix": int(time.time()),
        "success": bool(result.success),
        "cost": cost,
        "parent_ids": list(plan.get("ids", [])),
        "parents": plan.get("parents", []),
        "plan_score": plan.get("score"),
        "plan_max_sum": plan.get("max_sum"),
        "plan_max_rating": plan.get("max_rating"),
        "child": child,
        "raw": result.raw,
    }


def append_breed_log(entry: dict[str, Any]) -> None:
    path = DATA_DIR / "breed_log.json"
    items = load_json(path, [])
    if not isinstance(items, list):
        items = []
    items.insert(0, entry)
    save_json(path, items[:500])


def load_breed_logs(limit: int = 100) -> list[dict[str, Any]]:
    path = DATA_DIR / "breed_log.json"
    items = load_json(path, [])
    if not isinstance(items, list):
        return []
    return items[:limit]
