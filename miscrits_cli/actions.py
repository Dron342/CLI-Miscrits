from __future__ import annotations

from typing import Any

from .nakama import MiscritsClient, RpcResult


def player_summary(player_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": player_data.get("userId") or player_data.get("user_id"),
        "username": player_data.get("username"),
        "display_name": player_data.get("displayName") or player_data.get("display_name"),
        "level": player_data.get("level"),
        "gold": player_data.get("gold"),
        "gems": player_data.get("gems"),
        "platinum": player_data.get("platinum"),
        "virtue": player_data.get("virtue"),
        "location": player_data.get("location"),
        "team": player_data.get("team"),
        "active_battle": player_data.get("activeBattle") or player_data.get("active_battle"),
    }


def run_named_action(client: MiscritsClient, action: str) -> RpcResult:
    if action == "player":
        return client.get_player()
    if action == "heal":
        return client.heal_team()
    if action == "wish_sk":
        return client.wish("sk")
    if action == "wish_vi":
        return client.wish("vi")
    if action == "wish_xmas":
        return client.wish("xmas")
    raise ValueError(f"Unknown action: {action}")
