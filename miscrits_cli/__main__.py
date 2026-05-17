from __future__ import annotations

import argparse
import json
import sys
import time

from .actions import player_summary
from .arena import ArenaRunConfig, ArenaRunner
from .asset_cache import AssetCache
from .battle_learning import learning_status, load_battle_history, load_battle_log, load_battle_log_index
from .breeding import BreedConfig, BreedRunner, evo_from_level
from .config import DEFAULT_CONFIG, SESSION_FILE
from .credentials import clear_credentials, credentials_status, save_credentials
from .data_cache import DataCache
from .event_log import load_events, log_event
from .nakama import MiscritsClient, MiscritsError
from .player_store import load_owned_miscrits, load_player_snapshot, saved_data_status
from .realtime import diagnose_socket_endpoints
from .web.server import run_server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="miscrits-cli")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Show local configuration.")
    sub.add_parser("request-info", help="Show sanitized HTTP request settings.")
    sub.add_parser("cache-list", help="Show local reference cache state.")

    cache_sync_parser = sub.add_parser("cache-sync", help="Download or update CDN reference data.")
    cache_sync_parser.add_argument("names", nargs="*", help="Reference JSON names, e.g. miscrits.json.")
    cache_sync_parser.add_argument("--all", action="store_true", help="Try every JSON file listed in cache.json.")
    cache_sync_parser.add_argument("--force", action="store_true", help="Download even when local version matches.")

    avatar_sync_parser = sub.add_parser("avatar-sync", help="Download miscrit avatar PNGs from the CDN.")
    avatar_sync_parser.add_argument("mids", nargs="*", type=int, help="Miscrit mids to download.")
    avatar_sync_parser.add_argument("--all", action="store_true", help="Download avatars for every known miscrit.")
    avatar_sync_parser.add_argument("--evo", type=int, default=0, help="Evolution image index, 0-3.")
    avatar_sync_parser.add_argument("--force", action="store_true", help="Redownload existing PNG files.")
    avatar_sync_parser.add_argument("--limit", type=int, default=0, help="Maximum number of avatars to download.")

    login_parser = sub.add_parser("login", help="Authenticate and save a local session.")
    login_parser.add_argument("login")
    login_parser.add_argument("password")
    login_parser.add_argument("--remember", action="store_true", help="Save encrypted credentials for automatic relogin.")

    sub.add_parser("logout", help="Delete the saved local session.")
    sub.add_parser("login-saved", help="Authenticate with locally saved credentials.")
    sub.add_parser("credentials-status", help="Show saved credentials state.")
    sub.add_parser("credentials-clear", help="Delete saved credentials.")
    sub.add_parser("socket-doctor", help="Check realtime WebSocket endpoints.")
    sub.add_parser("player", help="Fetch and print player summary.")
    sub.add_parser("saved-player", help="Print the locally saved player snapshot.")
    sub.add_parser("saved-miscrits", help="Print the locally saved owned miscrits list.")
    sub.add_parser("saved-status", help="Show local saved player/miscrits cache state.")
    sub.add_parser("heal", help="Call heal_team RPC.")
    sub.add_parser("ai-learning", help="Show learned battle AI weights and recent battle grades.")
    battle_history_parser = sub.add_parser("battle-history", help="Show saved battle logs and grades.")
    battle_history_parser.add_argument("--limit", type=int, default=20)
    battle_logs_parser = sub.add_parser("battle-logs", help="List saved per-battle logs.")
    battle_logs_parser.add_argument("--limit", type=int, default=100)
    battle_logs_parser.add_argument("--mode", default="")
    battle_logs_parser.add_argument("--outcome", default="")
    battle_logs_parser.add_argument("--text", default="")
    battle_log_parser = sub.add_parser("battle-log", help="Show one saved battle log by id.")
    battle_log_parser.add_argument("id")

    logs_parser = sub.add_parser("logs", help="Show persistent CLI/web event logs.")
    logs_parser.add_argument("--limit", type=int, default=100)
    logs_parser.add_argument("--category", default="")
    logs_parser.add_argument("--level", default="")
    logs_parser.add_argument("--event", default="")
    logs_parser.add_argument("--reason", default="")
    logs_parser.add_argument("--text", default="")

    arena_status_parser = sub.add_parser("arena-status", help="Validate arena rules and show current team.")
    arena_status_parser.add_argument("mode", choices=["battle", "random", "platinum", "daily"], nargs="?", default="battle")

    arena_run_parser = sub.add_parser("arena-run", help="Queue and auto-play an arena battle.")
    arena_run_parser.add_argument("mode", choices=["battle", "random", "platinum", "daily"])
    arena_run_parser.add_argument("--timeout", type=float, default=300.0)
    arena_run_parser.add_argument("--max-turns", type=int, default=150)
    arena_run_parser.add_argument("--dry-run", action="store_true")
    arena_run_parser.add_argument("--no-prepare", action="store_true", help="Skip arena location/team/heal preparation.")
    arena_run_parser.add_argument("--location-id", type=int, default=1, help="Target arena location id for preparation.")
    arena_run_parser.add_argument("--area-id", type=int, default=2, help="Target arena area id for preparation.")
    arena_run_parser.add_argument("--repeat", type=int, default=1, help="Number of battles to run. Use 0 for endless loop.")
    arena_run_parser.add_argument("--repeat-delay", type=float, default=3.0, help="Delay between battles in loop mode.")
    arena_run_parser.add_argument("--continue-on-error", action="store_true", help="Keep looping even if one battle fails.")
    arena_run_parser.add_argument("--stop-on-error", action="store_true", help="Stop an endless loop after the first failed battle.")

    wish_parser = sub.add_parser("wish", help="Call a wishing-well RPC.")
    wish_parser.add_argument("kind", choices=["sk", "vi", "xmas"])

    rpc_parser = sub.add_parser("rpc", help="Call any Nakama RPC method.")
    rpc_parser.add_argument("method")
    rpc_parser.add_argument("--payload", default="{}", help="JSON object payload.")

    battle_parser = sub.add_parser("battle", help="Create a battle through create_battle RPC.")
    battle_parser.add_argument("type", choices=["Wild", "Boss", "Campaign", "GlobalBoss"])
    battle_parser.add_argument("--payload", default="{}", help="JSON object payload.")

    breed_plan_parser = sub.add_parser("breed-plan", help="Find the best S+ breeding triple without spending gold.")
    breed_plan_parser.add_argument("--min-max-sum", type=int, default=15)
    breed_plan_parser.add_argument("--allow-splus-parents", type=int, default=0)
    breed_plan_parser.add_argument("--target-mid", type=int)
    breed_plan_parser.add_argument("--target-element")

    breed_once_parser = sub.add_parser("breed-once", help="Run one planned breed RPC.")
    breed_once_parser.add_argument("--min-max-sum", type=int, default=15)
    breed_once_parser.add_argument("--allow-splus-parents", type=int, default=0)
    breed_once_parser.add_argument("--target-mid", type=int)
    breed_once_parser.add_argument("--target-element")
    breed_once_parser.add_argument("--dry-run", action="store_true")

    auto_breed_parser = sub.add_parser("auto-breed", help="Run repeated planned breed RPCs.")
    auto_breed_parser.add_argument("--max-breeds", type=int, default=1)
    auto_breed_parser.add_argument("--min-max-sum", type=int, default=15)
    auto_breed_parser.add_argument("--allow-splus-parents", type=int, default=0)
    auto_breed_parser.add_argument("--target-mid", type=int)
    auto_breed_parser.add_argument("--target-element")
    auto_breed_parser.add_argument("--delay", type=float, default=0.4)
    auto_breed_parser.add_argument("--dry-run", action="store_true")

    serve_parser = sub.add_parser("serve", help="Start local web UI.")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8765)

    args = parser.parse_args(argv)
    client = MiscritsClient()

    try:
        if args.command == "doctor":
            print_json(
                {
                    "api_url": DEFAULT_CONFIG.api_url,
                    "cdn_url": DEFAULT_CONFIG.cdn_url,
                    "session_file": str(SESSION_FILE),
                    "standalone": True,
                    "user_agent": DEFAULT_CONFIG.user_agent,
                    "logged_in": client.is_logged_in(),
                    "client_version": DEFAULT_CONFIG.client_version,
                }
            )
            return 0
        if args.command == "request-info":
            print_json(client.request_info())
            return 0
        if args.command == "cache-list":
            print_json({"ok": True, "references": DataCache(client).list_known_references(refresh_index=True)})
            return 0
        if args.command == "cache-sync":
            print_json(
                DataCache(client).sync_references(
                    names=args.names or None,
                    force=args.force,
                    refresh_index=True,
                    include_all_index_json=args.all,
                )
            )
            return 0
        if args.command == "avatar-sync":
            mids = args.mids or None
            specs = None
            if not mids and not args.all:
                owned = load_owned_miscrits()
                raw_miscrits = owned.get("miscrits", []) if isinstance(owned, dict) else owned
                if isinstance(raw_miscrits, list):
                    specs = []
                    for item in raw_miscrits:
                        if not isinstance(item, dict):
                            continue
                        mid = int(item.get("mid", item.get("mId", item.get("m", 0))) or 0)
                        level = int(item.get("level", item.get("l", 1)) or 1)
                        if mid > 0:
                            specs.append({"mid": mid, "evo": evo_from_level(level)})
            if specs is not None:
                print_json(AssetCache(client).sync_miscrit_asset_specs(specs, asset_type="avatars", force=args.force, limit=args.limit))
                return 0
            print_json(
                AssetCache(client).sync_miscrit_assets(
                    mids=mids,
                    asset_type="avatars",
                    evo=args.evo,
                    force=args.force,
                    include_all=args.all,
                    limit=args.limit,
                )
            )
            return 0
        if args.command == "login":
            session = client.login(args.login, args.password)
            remembered = False
            if args.remember:
                save_credentials(args.login, args.password)
                remembered = True
            print_json({"ok": True, "login": session["login"], "remembered": remembered, "session_file": str(SESSION_FILE)})
            return 0
        if args.command == "logout":
            client.logout()
            print_json({"ok": True})
            return 0
        if args.command == "login-saved":
            session = client.login_saved()
            print_json({"ok": True, "login": session["login"], "session_file": str(SESSION_FILE)})
            return 0
        if args.command == "credentials-status":
            print_json(credentials_status())
            return 0
        if args.command == "credentials-clear":
            clear_credentials()
            print_json({"ok": True})
            return 0
        if args.command == "socket-doctor":
            client.ensure_realtime_session()
            print_json({"ok": True, "endpoints": diagnose_socket_endpoints(client.session["token"], client.config)})
            return 0
        if args.command == "player":
            result = client.get_player()
            print_json(player_summary(result.data) if result.success else result.raw)
            return 0 if result.success else 2
        if args.command == "saved-player":
            print_json(load_player_snapshot())
            return 0
        if args.command == "saved-miscrits":
            print_json(load_owned_miscrits())
            return 0
        if args.command == "saved-status":
            print_json(saved_data_status())
            return 0
        if args.command == "heal":
            print_json(client.heal_team().__dict__)
            return 0
        if args.command == "ai-learning":
            print_json(learning_status())
            return 0
        if args.command == "battle-history":
            print_json({"ok": True, "history": list(reversed(load_battle_history(args.limit)))})
            return 0
        if args.command == "battle-logs":
            print_json({"ok": True, "logs": load_battle_log_index(args.limit, args.mode, args.outcome, args.text)})
            return 0
        if args.command == "battle-log":
            battle = load_battle_log(args.id)
            print_json({"ok": bool(battle), "battle": battle} if battle else {"ok": False, "error": "Battle log not found"})
            return 0
        if args.command == "logs":
            print_json(
                {
                    "ok": True,
                    "logs": load_events(
                        limit=args.limit,
                        category=args.category,
                        level=args.level,
                        event=args.event,
                        reason=args.reason,
                        text=args.text,
                    ),
                }
            )
            return 0
        if args.command == "arena-status":
            print_json(ArenaRunner(client).status(args.mode))
            return 0
        if args.command == "arena-run":
            config = ArenaRunConfig(
                mode=args.mode,
                timeout_seconds=args.timeout,
                max_turns=args.max_turns,
                dry_run=args.dry_run,
                prepare=not args.no_prepare,
                target_location_id=args.location_id,
                target_area_id=args.area_id,
                repeat_count=args.repeat,
                repeat_delay_seconds=args.repeat_delay,
                stop_on_error=args.stop_on_error if args.repeat <= 0 else not args.continue_on_error,
            )
            job_id = f"cli-{int(time.time())}"

            def progress(update: dict[str, object]) -> None:
                event_name = str(update.get("event", "arena_event"))
                phase = str(update.get("phase", ""))
                level = "warning" if phase == "recovering" or event_name.startswith("recoverable_") else ("error" if phase == "error" or update.get("error") else "info")
                log_event(
                    event_name,
                    category="arena",
                    level=level,
                    source="cli_command",
                    initiator="cli",
                    job_id=job_id,
                    mode=str(update.get("mode", args.mode)),
                    phase=phase,
                    reason=str(update.get("reason") or update.get("error") or ""),
                    payload={key: value for key, value in update.items() if key != "timestamp"},
                )

            log_event(
                "arena_cli_started",
                category="arena",
                source="cli_command",
                initiator="cli",
                job_id=job_id,
                mode=args.mode,
                payload={
                    "repeat": args.repeat,
                    "repeat_delay": args.repeat_delay,
                    "stop_on_error": config.stop_on_error,
                    "dry_run": args.dry_run,
                },
            )
            runner = ArenaRunner(client, progress=progress)
            print_json(
                runner.run_loop(config) if args.repeat != 1 and not args.dry_run else runner.run(config)
            )
            return 0
        if args.command == "wish":
            print_json(client.wish(args.kind).__dict__)
            return 0
        if args.command == "rpc":
            print_json(client.rpc(args.method, parse_payload(args.payload)).__dict__)
            return 0
        if args.command == "battle":
            print_json(client.create_battle(args.type, parse_payload(args.payload)).__dict__)
            return 0
        if args.command == "breed-plan":
            runner = BreedRunner(client)
            print_json(
                runner.plan(
                    args.min_max_sum,
                    args.allow_splus_parents,
                    args.target_mid,
                    args.target_element,
                )
            )
            return 0
        if args.command == "breed-once":
            runner = BreedRunner(client)
            print_json(
                runner.breed_once(
                    BreedConfig(
                        max_breeds=1,
                        min_max_sum=args.min_max_sum,
                        allow_splus_parents=args.allow_splus_parents,
                        dry_run=args.dry_run,
                        target_mid=args.target_mid,
                        target_element=args.target_element,
                    )
                )
            )
            return 0
        if args.command == "auto-breed":
            runner = BreedRunner(client)
            print_json(
                runner.auto_breed(
                    BreedConfig(
                        max_breeds=args.max_breeds,
                        min_max_sum=args.min_max_sum,
                        allow_splus_parents=args.allow_splus_parents,
                        dry_run=args.dry_run,
                        delay_seconds=args.delay,
                        target_mid=args.target_mid,
                        target_element=args.target_element,
                    )
                )
            )
            return 0
        if args.command == "serve":
            run_server(args.host, args.port)
            return 0
    except (MiscritsError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 1


def parse_payload(raw: str) -> dict:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("--payload must be a JSON object")
    return payload


def print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
