from __future__ import annotations

import time
from typing import Any

from .config import DATA_DIR
from .storage import load_json, save_json


PLAYER_SNAPSHOT_FILE = DATA_DIR / "player_snapshot.json"
OWNED_MISCRITS_FILE = DATA_DIR / "owned_miscrits.json"


def save_player_snapshot(player_data: dict[str, Any]) -> dict[str, Any]:
    saved_at = int(time.time())
    miscrits = player_data.get("miscrits", [])
    if not isinstance(miscrits, list):
        miscrits = []
    snapshot = {
        "saved_at": saved_at,
        "user_id": player_data.get("userId") or player_data.get("user_id"),
        "username": player_data.get("username"),
        "display_name": player_data.get("displayName") or player_data.get("display_name"),
        "gold": player_data.get("gold"),
        "gems": player_data.get("gems"),
        "platinum": player_data.get("platinum"),
        "virtue": player_data.get("virtue"),
        "team": player_data.get("team_order", player_data.get("team", [])),
        "miscrit_count": len(miscrits),
        "raw": player_data,
    }
    owned = {
        "saved_at": saved_at,
        "user_id": snapshot["user_id"],
        "username": snapshot["username"],
        "team": snapshot["team"],
        "count": len(miscrits),
        "miscrits": miscrits,
    }
    save_json(PLAYER_SNAPSHOT_FILE, snapshot)
    save_json(OWNED_MISCRITS_FILE, owned)
    return {"saved_at": saved_at, "miscrit_count": len(miscrits)}


def load_player_snapshot() -> dict[str, Any]:
    data = load_json(PLAYER_SNAPSHOT_FILE, {})
    return data if isinstance(data, dict) else {}


def load_owned_miscrits() -> dict[str, Any]:
    data = load_json(OWNED_MISCRITS_FILE, {})
    return data if isinstance(data, dict) else {}


def saved_data_status() -> dict[str, Any]:
    player = load_player_snapshot()
    owned = load_owned_miscrits()
    return {
        "ok": True,
        "player_cached": bool(player),
        "miscrits_cached": bool(owned),
        "player_saved_at": player.get("saved_at"),
        "miscrits_saved_at": owned.get("saved_at"),
        "miscrit_count": owned.get("count", player.get("miscrit_count", 0)),
        "player_path": str(PLAYER_SNAPSHOT_FILE),
        "miscrits_path": str(OWNED_MISCRITS_FILE),
    }
