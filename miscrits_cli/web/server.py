from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..actions import player_summary, run_named_action
from ..account_plan import AccountPlanRunner, default_plan_config, get_account_plan, save_account_plan
from ..accounts import credentials_path, get_account, list_accounts, remove_account, save_account, session_path
from ..arena import ArenaRunConfig, ArenaRunner
from ..asset_cache import AssetCache
from ..battle_learning import ai_dashboard, learning_status, load_battle_history, load_battle_log, load_battle_log_index, set_ai_weight
from ..breeding import BreedConfig, BreedRunner, load_breed_logs
from ..config import DATA_DIR
from ..credentials import clear_credentials, credentials_status, save_credentials
from ..data_cache import DataCache
from ..event_log import event_log_status, load_events, log_event
from ..nakama import MiscritsClient, MiscritsError
from ..player_store import load_owned_miscrits, load_player_snapshot, saved_data_status
from ..realtime import diagnose_socket_endpoints


ARENA_JOBS: dict[str, dict[str, Any]] = {}
ARENA_JOBS_LOCK = threading.RLock()
MAX_ARENA_JOBS = 12
PLAN_JOBS: dict[str, dict[str, Any]] = {}
PLAN_JOBS_LOCK = threading.RLock()


class MiscritsWebHandler(BaseHTTPRequestHandler):
    server_version = "CLI_Miscrits/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(INDEX_HTML)
            return
        if parsed.path == "/api/status":
            client = MiscritsClient()
            self._send_json({"logged_in": client.is_logged_in(), "credentials": credentials_status()})
            return
        if parsed.path == "/api/auth-bootstrap":
            self._handle_auth_bootstrap()
            return
        if parsed.path == "/api/accounts":
            self._send_json({"ok": True, "accounts": list_accounts()})
            return
        if parsed.path == "/api/player":
            query = parse_qs(parsed.query)
            self._handle_player((query.get("account_id") or [""])[0])
            return
        if parsed.path == "/api/saved-data":
            self._handle_saved_data()
            return
        if parsed.path == "/api/socket-doctor":
            self._handle_socket_doctor()
            return
        if parsed.path == "/api/breed-options":
            query = parse_qs(parsed.query)
            self._handle_breed_options((query.get("account_id") or [""])[0])
            return
        if parsed.path == "/api/breed-logs":
            self._send_json({"ok": True, "logs": load_breed_logs(100)})
            return
        if parsed.path == "/api/cache":
            self._handle_cache_list()
            return
        if parsed.path == "/api/miscrit-asset":
            self._handle_miscrit_asset(parse_qs(parsed.query))
            return
        if parsed.path == "/api/element-asset":
            self._handle_element_asset(parse_qs(parsed.query))
            return
        if parsed.path == "/api/arena-status":
            query = parse_qs(parsed.query)
            self._handle_arena_status((query.get("mode") or ["battle"])[0], (query.get("account_id") or [""])[0])
            return
        if parsed.path == "/api/arena-current":
            query = parse_qs(parsed.query)
            self._handle_arena_current((query.get("account_id") or [""])[0])
            return
        if parsed.path == "/api/arena-live":
            query = parse_qs(parsed.query)
            self._handle_arena_live((query.get("id") or [""])[0])
            return
        if parsed.path == "/api/plan":
            query = parse_qs(parsed.query)
            self._handle_plan_get((query.get("account_id") or [""])[0])
            return
        if parsed.path == "/api/plan-live":
            query = parse_qs(parsed.query)
            self._handle_plan_live((query.get("account_id") or [""])[0])
            return
        if parsed.path == "/api/battle-learning":
            self._send_json(learning_status())
            return
        if parsed.path == "/api/ai-dashboard":
            self._send_json(ai_dashboard())
            return
        if parsed.path == "/api/battle-history":
            query = parse_qs(parsed.query)
            raw_limit = (query.get("limit") or ["50"])[0]
            limit = None if str(raw_limit).strip().lower() in {"all", "0", "-1"} else int(raw_limit or 50)
            self._send_json({"ok": True, "history": list(reversed(load_battle_history(limit)))})
            return
        if parsed.path == "/api/battle-logs":
            query = parse_qs(parsed.query)
            raw_limit = (query.get("limit") or ["100"])[0]
            limit = None if str(raw_limit).strip().lower() in {"all", "0", "-1"} else int(raw_limit or 100)
            self._send_json(
                {
                    "ok": True,
                    "logs": load_battle_log_index(
                        limit=limit,
                        mode=(query.get("mode") or [""])[0],
                        outcome=(query.get("outcome") or [""])[0],
                        text=(query.get("text") or [""])[0],
                    ),
                }
            )
            return
        if parsed.path == "/api/battle-log":
            query = parse_qs(parsed.query)
            battle_id = (query.get("id") or [""])[0]
            log = load_battle_log(battle_id)
            self._send_json({"ok": bool(log), "battle": log} if log else {"ok": False, "error": "Battle log not found"}, HTTPStatus.OK if log else HTTPStatus.NOT_FOUND)
            return
        if parsed.path == "/api/logs":
            query = parse_qs(parsed.query)
            self._handle_logs(query)
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/login":
            body = self._read_json()
            client = MiscritsClient()
            try:
                session = client.login(str(body.get("login", "")), str(body.get("password", "")))
                remembered = False
                if bool(body.get("remember", False)):
                    save_credentials(str(body.get("login", "")), str(body.get("password", "")))
                    remembered = True
                self._send_json(
                    {
                        "ok": True,
                        "logged_in": client.is_logged_in(),
                        "login": session.get("login"),
                        "remembered": remembered,
                        "credentials": credentials_status(),
                        "method": "password",
                    }
                )
            except MiscritsError as exc:
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/login-saved":
            try:
                client = MiscritsClient()
                session = client.login_saved()
                self._send_json(
                    {
                        "ok": True,
                        "login": session.get("login"),
                        "logged_in": client.is_logged_in(),
                        "credentials": credentials_status(),
                        "method": "saved_credentials",
                    }
                )
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/logout":
            MiscritsClient().logout()
            self._send_json({"ok": True, "logged_in": False, "credentials": credentials_status()})
            return
        if parsed.path == "/api/account-save":
            body = self._read_json()
            self._handle_account_save(body)
            return
        if parsed.path == "/api/account-remove":
            body = self._read_json()
            self._handle_account_remove(body)
            return
        if parsed.path == "/api/credentials-clear":
            clear_credentials()
            self._send_json({"ok": True, "credentials": credentials_status()})
            return
        if parsed.path == "/api/action":
            query = parse_qs(parsed.query)
            action = (query.get("name") or [""])[0]
            self._handle_action(action, (query.get("account_id") or [""])[0])
            return
        if parsed.path == "/api/rpc":
            body = self._read_json()
            self._handle_rpc(str(body.get("method", "")), body.get("payload", {}))
            return
        if parsed.path == "/api/breed-plan":
            body = self._read_json()
            self._handle_breed_plan(body)
            return
        if parsed.path == "/api/breed-once":
            body = self._read_json()
            self._handle_breed_once(body)
            return
        if parsed.path == "/api/auto-breed":
            body = self._read_json()
            self._handle_auto_breed(body)
            return
        if parsed.path == "/api/cache-sync":
            body = self._read_json()
            self._handle_cache_sync(body)
            return
        if parsed.path == "/api/avatar-sync":
            body = self._read_json()
            self._handle_avatar_sync(body)
            return
        if parsed.path == "/api/arena-run":
            body = self._read_json()
            self._handle_arena_run(body)
            return
        if parsed.path == "/api/arena-start":
            body = self._read_json()
            self._handle_arena_start(body)
            return
        if parsed.path == "/api/arena-stop":
            body = self._read_json()
            self._handle_arena_stop(body)
            return
        if parsed.path == "/api/plan-save":
            body = self._read_json()
            self._handle_plan_save(body)
            return
        if parsed.path == "/api/plan-start":
            body = self._read_json()
            self._handle_plan_start(body)
            return
        if parsed.path == "/api/plan-stop":
            body = self._read_json()
            self._handle_plan_stop(body)
            return
        if parsed.path == "/api/ai-weight":
            body = self._read_json()
            try:
                entry = set_ai_weight(str(body.get("category", "")), str(body.get("key", "")), float(body.get("weight", 0.0) or 0.0))
                self._send_json({"ok": True, "entry": entry, "dashboard": ai_dashboard()})
            except (TypeError, ValueError) as exc:
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        print("[%s] %s" % (self.log_date_time_string(), format % args))

    def _handle_player(self, account_id: str = "") -> None:
        client = client_for_account(account_id)
        try:
            result = client.get_player()
            payload = player_summary(result.data) if result.success else result.raw
            self._send_json({"ok": result.success, "player": payload})
        except MiscritsError as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _handle_auth_bootstrap(self) -> None:
        client = MiscritsClient()
        credentials = credentials_status()
        method = "none"
        error = ""
        try:
            if client.is_logged_in():
                method = "session"
            client.ensure_session()
            if method == "none":
                method = "saved_credentials"
            elif int(client.session.get("saved_at", 0) or 0):
                method = "session_or_refresh"
        except Exception as exc:
            error = str(exc)
        self._send_json(
            {
                "ok": client.is_logged_in(),
                "logged_in": client.is_logged_in(),
                "login": client.session.get("login"),
                "method": method if client.is_logged_in() else "none",
                "credentials": credentials,
                "error": error,
            }
        )

    def _handle_logs(self, query: dict[str, list[str]]) -> None:
        def item(name: str, default: str = "") -> str:
            return (query.get(name) or [default])[0] or default

        self._send_json(
            {
                "ok": True,
                "status": event_log_status(),
                "logs": load_events(
                    limit=optional_int_default(item("limit"), 250) or 250,
                    level=item("level"),
                    category=item("category"),
                    event=item("event"),
                    account_id=item("account_id"),
                    job_id=item("job_id"),
                    source=item("source"),
                    initiator=item("initiator"),
                    reason=item("reason"),
                    text=item("text"),
                    since=optional_float(item("since")),
                    until=optional_float(item("until")),
                ),
            }
        )

    def _handle_account_save(self, body: dict[str, Any]) -> None:
        try:
            account = save_account(
                str(body.get("login", "")).strip(),
                str(body.get("password", "")),
                str(body.get("label", "")).strip(),
                str(body.get("id", "")).strip(),
            )
            log_event(
                "account_saved",
                category="account",
                source="web",
                initiator="user",
                account_id=str(account.get("id", "")),
                message=str(account.get("label") or account.get("login") or "account"),
                payload={"label": account.get("label"), "login": account.get("login")},
            )
            self._send_json({"ok": True, "account": account, "accounts": list_accounts()})
        except Exception as exc:
            log_event("account_save_failed", category="account", level="error", source="web", initiator="user", reason=str(exc))
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _handle_account_remove(self, body: dict[str, Any]) -> None:
        account_id = str(body.get("id", "")).strip()
        removed = remove_account(account_id)
        log_event(
            "account_removed" if removed else "account_remove_failed",
            category="account",
            level="info" if removed else "warning",
            source="web",
            initiator="user",
            account_id=account_id,
            reason="" if removed else "account_not_found",
        )
        self._send_json({"ok": removed, "accounts": list_accounts()})

    def _handle_saved_data(self) -> None:
        self._send_json(
            {
                **saved_data_status(),
                "player": load_player_snapshot(),
                "owned_miscrits": load_owned_miscrits(),
            }
        )

    def _handle_socket_doctor(self) -> None:
        try:
            client = MiscritsClient()
            client.ensure_realtime_session()
            self._send_json({"ok": True, "endpoints": diagnose_socket_endpoints(client.session["token"], client.config)})
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _handle_action(self, action: str, account_id: str = "") -> None:
        client = client_for_account(account_id)
        try:
            result = run_named_action(client, action)
            self._send_json({"ok": result.success, "result": result.__dict__})
        except (MiscritsError, ValueError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _handle_rpc(self, method: str, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            self._send_json({"ok": False, "error": "payload must be an object"}, HTTPStatus.BAD_REQUEST)
            return
        try:
            account_id = str(payload.pop("_account_id", "") or "")
            result = client_for_account(account_id).rpc(method, payload)
            self._send_json({"ok": result.success, "result": result.__dict__})
        except MiscritsError as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _handle_breed_options(self, account_id: str = "") -> None:
        try:
            self._send_json(BreedRunner(client_for_account(account_id)).options())
        except (MiscritsError, RuntimeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _handle_breed_plan(self, body: dict[str, Any]) -> None:
        try:
            result = BreedRunner(client_for_account(str(body.get("account_id", "") or ""))).plan(
                int(body.get("min_max_sum", 15) or 15),
                int(body.get("allow_splus_parents", 0) or 0),
                optional_int(body.get("target_mid")),
                optional_text(body.get("target_element")),
            )
            self._send_json(result)
        except (MiscritsError, RuntimeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _handle_breed_once(self, body: dict[str, Any]) -> None:
        try:
            result = BreedRunner(client_for_account(str(body.get("account_id", "") or ""))).breed_once(
                breed_config_from_body(body)
            )
            self._send_json(result)
        except (MiscritsError, RuntimeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _handle_auto_breed(self, body: dict[str, Any]) -> None:
        try:
            result = BreedRunner(client_for_account(str(body.get("account_id", "") or ""))).auto_breed(breed_config_from_body(body))
            self._send_json(result)
        except (MiscritsError, RuntimeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _handle_cache_list(self) -> None:
        try:
            self._send_json({"ok": True, "references": DataCache(MiscritsClient()).list_known_references(True)})
        except (MiscritsError, RuntimeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _handle_cache_sync(self, body: dict[str, Any]) -> None:
        try:
            names = body.get("names")
            if not isinstance(names, list):
                names = None
            result = DataCache(MiscritsClient()).sync_references(
                names=[str(item) for item in names] if names else None,
                force=bool(body.get("force", False)),
                refresh_index=True,
                include_all_index_json=bool(body.get("all", False)),
            )
            self._send_json(result)
        except (MiscritsError, RuntimeError, ValueError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _handle_miscrit_asset(self, query: dict[str, list[str]]) -> None:
        try:
            mid = int((query.get("mid") or ["0"])[0] or 0)
            evo = int((query.get("evo") or ["0"])[0] or 0)
            asset_type = (query.get("type") or ["avatars"])[0]
            path = AssetCache(MiscritsClient()).ensure_miscrit_asset(mid, evo, asset_type)
            self._send_bytes(path.read_bytes(), "image/png")
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.NOT_FOUND)

    def _handle_element_asset(self, query: dict[str, list[str]]) -> None:
        try:
            element = str((query.get("element") or [""])[0] or "")
            path = element_asset_path(element)
            self._send_bytes(path.read_bytes(), "image/png")
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.NOT_FOUND)

    def _handle_avatar_sync(self, body: dict[str, Any]) -> None:
        try:
            account_id = str(body.get("account_id", "") or "")
            mids = body.get("mids")
            selected: list[int] | None = None
            if isinstance(mids, list) and mids:
                selected = [int(item) for item in mids if int(item) > 0]
            elif not bool(body.get("all", False)):
                player = BreedRunner(client_for_account(account_id)).load_player()
                specs = [
                    {"mid": int(item.get("mid", 0) or 0), "evo": int(item.get("evo", 0) or 0)}
                    for item in player.get("miscrits", [])
                    if isinstance(item, dict) and int(item.get("mid", 0) or 0) > 0
                ]
                result = AssetCache(MiscritsClient()).sync_miscrit_asset_specs(
                    specs,
                    asset_type="avatars",
                    force=bool(body.get("force", False)),
                    limit=int(body.get("limit", 0) or 0),
                )
            else:
                result = AssetCache(MiscritsClient()).sync_miscrit_assets(
                    mids=selected,
                    asset_type="avatars",
                    evo=int(body.get("evo", 0) or 0),
                    force=bool(body.get("force", False)),
                    include_all=bool(body.get("all", False)),
                    limit=int(body.get("limit", 0) or 0),
                )
            self._send_json(result)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _handle_arena_status(self, mode: str, account_id: str = "") -> None:
        try:
            self._send_json(ArenaRunner(client_for_account(account_id)).status(mode))
        except (MiscritsError, RuntimeError, ValueError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _handle_arena_live(self, job_id: str) -> None:
        with ARENA_JOBS_LOCK:
            job = ARENA_JOBS.get(job_id)
            payload = json.loads(json.dumps(job, ensure_ascii=False)) if job else None
        if not payload:
            payload = plan_arena_job_by_id(job_id)
        if not payload:
            self._send_json({"ok": False, "error": "arena job not found"}, HTTPStatus.NOT_FOUND)
            return
        self._send_json({"ok": True, "job": payload})

    def _handle_arena_current(self, account_id: str = "") -> None:
        with ARENA_JOBS_LOCK:
            job = current_arena_job(account_id)
            payload = json.loads(json.dumps(job, ensure_ascii=False)) if job else None
        if not payload:
            payload = current_plan_arena_job(account_id)
        self._send_json({"ok": True, "job": payload})

    def _handle_arena_start(self, body: dict[str, Any]) -> None:
        try:
            config = arena_config_from_body({**body, "dry_run": False})
            job = create_arena_job(config, str(body.get("account_id", "") or ""))
            log_event(
                "arena_start_requested",
                category="arena",
                source="web",
                initiator="user",
                account_id=str(body.get("account_id", "") or ""),
                job_id=str(job["id"]),
                mode=config.mode,
                payload={
                    "repeat_count": config.repeat_count,
                    "repeat_delay_seconds": config.repeat_delay_seconds,
                    "stop_on_error": config.stop_on_error,
                },
            )
            self._send_json({"ok": True, "job_id": job["id"], "job": job})
        except (MiscritsError, RuntimeError, TimeoutError, ValueError) as exc:
            log_event("arena_start_failed", category="arena", level="error", source="web", initiator="user", reason=str(exc))
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _handle_arena_stop(self, body: dict[str, Any]) -> None:
        job_id = str(body.get("id", ""))
        plan_arena = plan_arena_job_by_id(job_id)
        if plan_arena:
            account_id = str(plan_arena.get("account_id", "") or "")
            with PLAN_JOBS_LOCK:
                plan_job = PLAN_JOBS.get(account_id)
                if plan_job:
                    plan_job["stop_requested"] = True
                    plan_job["updated_at"] = time.time()
                    active = plan_job.get("active_arena")
                    if isinstance(active, dict):
                        active["stop_requested"] = True
                        active["updated_at"] = time.time()
                    payload = json.loads(json.dumps(active, ensure_ascii=False)) if isinstance(active, dict) else plan_arena
                    self._send_json({"ok": True, "job": payload})
                    return
        with ARENA_JOBS_LOCK:
            job = ARENA_JOBS.get(job_id)
            if not job:
                self._send_json({"ok": False, "error": "arena job not found"}, HTTPStatus.NOT_FOUND)
                return
            job["stop_requested"] = True
            job["updated_at"] = time.time()
            payload = json.loads(json.dumps(job, ensure_ascii=False))
        log_event(
            "arena_stop_requested",
            category="arena",
            source="web",
            initiator="user",
            account_id=str(payload.get("account_id", "")),
            job_id=job_id,
            mode=str(payload.get("mode", "")),
            phase=str(payload.get("phase", "")),
        )
        self._send_json({"ok": True, "job": payload})

    def _handle_arena_run(self, body: dict[str, Any]) -> None:
        try:
            result = ArenaRunner(client_for_account(str(body.get("account_id", "") or ""))).run(arena_config_from_body(body))
            self._send_json(result)
        except (MiscritsError, RuntimeError, TimeoutError, ValueError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _handle_plan_get(self, account_id: str) -> None:
        if not account_id:
            self._send_json({"ok": False, "error": "Сначала выбери аккаунт Miscrits."}, HTTPStatus.BAD_REQUEST)
            return
        payload = get_account_plan(account_id)
        job = current_plan_job(account_id)
        if not job or job.get("status") != "running":
            try:
                payload["state"] = AccountPlanRunner(account_id, client_for_account(account_id)).refresh_arena_states()
            except Exception as exc:
                payload["arena_refresh_error"] = str(exc)
        payload["job"] = job
        payload["default_config"] = default_plan_config()
        self._send_json(payload)

    def _handle_plan_live(self, account_id: str) -> None:
        self._send_json({"ok": True, "job": current_plan_job(account_id), **get_account_plan(account_id)})

    def _handle_plan_save(self, body: dict[str, Any]) -> None:
        account_id = str(body.get("account_id", "") or "")
        if not account_id:
            self._send_json({"ok": False, "error": "Сначала выбери аккаунт Miscrits."}, HTTPStatus.BAD_REQUEST)
            return
        config = body.get("config", {})
        self._send_json(save_account_plan(account_id, config if isinstance(config, dict) else {}))

    def _handle_plan_start(self, body: dict[str, Any]) -> None:
        account_id = str(body.get("account_id", "") or "")
        if not account_id:
            self._send_json({"ok": False, "error": "Сначала выбери аккаунт Miscrits."}, HTTPStatus.BAD_REQUEST)
            return
        try:
            job = create_plan_job(account_id, run_forever=bool(body.get("run_forever", True)))
            self._send_json({"ok": True, "job": job})
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _handle_plan_stop(self, body: dict[str, Any]) -> None:
        account_id = str(body.get("account_id", "") or "")
        with PLAN_JOBS_LOCK:
            job = current_plan_job_unlocked(account_id)
            if not job:
                self._send_json({"ok": False, "error": "plan job not found"}, HTTPStatus.NOT_FOUND)
                return
            job["stop_requested"] = True
            job["updated_at"] = time.time()
            payload = json.loads(json.dumps(job, ensure_ascii=False))
        log_event("plan_stop_requested", category="plan", source="web", initiator="user", account_id=account_id)
        self._send_json({"ok": True, "job": payload})

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_bytes(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(body)


def run_server(host: str, port: int) -> None:
    httpd = ThreadingHTTPServer((host, port), MiscritsWebHandler)
    print(f"CLI Miscrits web UI: http://{host}:{port}")
    for url in local_network_urls(host, port):
        print(f"Phone/LAN URL: {url}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def local_network_urls(host: str, port: int) -> list[str]:
    if host not in {"0.0.0.0", "::", ""}:
        return []
    addresses: set[str] = set()
    try:
        hostname = socket.gethostname()
        for value in socket.gethostbyname_ex(hostname)[2]:
            if value and not value.startswith("127."):
                addresses.add(value)
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            value = sock.getsockname()[0]
            if value and not value.startswith("127."):
                addresses.add(value)
    except OSError:
        pass
    return [f"http://{address}:{port}" for address in sorted(addresses)]


def optional_int(value: Any) -> int | None:
    if value in (None, "", "0", 0):
        return None
    return int(value)


def optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def element_asset_path(value: str) -> Path:
    filename = element_asset_filename(value)
    if not filename:
        raise ValueError(f"Unknown element asset: {value}")
    path = DATA_DIR / "img" / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def element_asset_filename(value: str) -> str:
    compact = "".join(ch for ch in str(value or "") if ch.isalpha())
    if not compact:
        return ""
    parts: list[str] = []
    cursor = 0
    known = ("Lightning", "Nature", "Water", "Earth", "Fire", "Wind")
    while cursor < len(compact):
        match = next((item for item in known if compact[cursor:].casefold().startswith(item.casefold())), "")
        if not match:
            return ""
        parts.append(match.casefold())
        cursor += len(match)
    return "-".join(parts) + ".png"


def breed_config_from_body(body: dict[str, Any]) -> BreedConfig:
    return BreedConfig(
        max_breeds=max(1, int(body.get("max_breeds", 1) or 1)),
        min_max_sum=int(body.get("min_max_sum", 15) or 15),
        allow_splus_parents=int(body.get("allow_splus_parents", 0) or 0),
        dry_run=bool(body.get("dry_run", True)),
        delay_seconds=float(body.get("delay_seconds", 0.4) or 0.4),
        target_mid=optional_int(body.get("target_mid")),
        target_element=optional_text(body.get("target_element")),
    )


def arena_config_from_body(body: dict[str, Any]) -> ArenaRunConfig:
    return ArenaRunConfig(
        mode=str(body.get("mode", "battle")),
        timeout_seconds=float(body.get("timeout_seconds", 300) or 300),
        max_turns=int(body.get("max_turns", 150) or 150),
        dry_run=bool(body.get("dry_run", False)),
        prepare=bool(body.get("prepare", True)),
        target_location_id=int(body.get("location_id", 1) or 1),
        target_area_id=int(body.get("area_id", 2) or 2),
        repeat_count=optional_int_default(body.get("repeat_count"), 1),
        repeat_delay_seconds=float(body.get("repeat_delay_seconds", 3) or 3),
        stop_on_error=bool(body.get("stop_on_error", True)),
    )


def optional_int_default(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    return int(value)


def optional_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def client_for_account(account_id: str = "") -> MiscritsClient:
    account_id = str(account_id or "").strip()
    if not account_id:
        accounts = list_accounts()
        if accounts:
            account_id = str(accounts[0].get("id", "") or "")
    if not account_id:
        return MiscritsClient()
    if not get_account(account_id):
        raise MiscritsError(f"Unknown account: {account_id}")
    return MiscritsClient(session_file=session_path(account_id), credentials_file=credentials_path(account_id))


def create_arena_job(config: ArenaRunConfig, account_id: str = "") -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    now = time.time()
    account = get_account(account_id) if account_id else None
    job: dict[str, Any] = {
        "id": job_id,
        "ok": True,
        "status": "running",
        "phase": "starting",
        "mode": config.mode,
        "account_id": account_id,
        "account": {"id": account_id, "label": account.get("label"), "login": account.get("login")} if account else {},
        "created_at": now,
        "updated_at": now,
        "progress": {},
        "events": [],
        "result": None,
        "error": "",
        "stop_requested": False,
        "repeat": {
            "count": config.repeat_count,
            "delay_seconds": config.repeat_delay_seconds,
            "stop_on_error": config.stop_on_error,
        },
    }
    with ARENA_JOBS_LOCK:
        ARENA_JOBS[job_id] = job
        trim_arena_jobs()
    log_event(
        "arena_job_created",
        category="arena",
        source="web",
        initiator="user",
        account_id=account_id,
        job_id=job_id,
        mode=config.mode,
        phase="starting",
        payload={
            "repeat_count": config.repeat_count,
            "repeat_delay_seconds": config.repeat_delay_seconds,
            "stop_on_error": config.stop_on_error,
            "dry_run": config.dry_run,
        },
    )

    def progress(update: dict[str, Any]) -> None:
        raw = json.loads(json.dumps(update, ensure_ascii=False))
        clean = compact_live_update(raw)
        with ARENA_JOBS_LOCK:
            stored = ARENA_JOBS.get(job_id)
            if not stored:
                return
            stored["updated_at"] = time.time()
            stored["phase"] = clean.get("phase", stored.get("phase", "running"))
            merge_arena_progress(stored["progress"], clean)
            stored["events"].append(clean)
            stored["events"] = stored["events"][-80:]
        log_event(
            str(raw.get("event", "arena_event")),
            category="arena",
            level=arena_log_level(raw),
            source="arena_runner",
            initiator="cli",
            account_id=account_id,
            job_id=job_id,
            mode=str(raw.get("mode", config.mode) or config.mode),
            phase=str(raw.get("phase", "")),
            reason=event_reason(raw),
            message=event_message(raw),
            payload=compact_event_payload(raw),
        )

    def run_job() -> None:
        try:
            def should_stop() -> bool:
                with ARENA_JOBS_LOCK:
                    stored = ARENA_JOBS.get(job_id)
                    return bool(stored and stored.get("stop_requested"))

            runner = ArenaRunner(client_for_account(account_id), progress=progress)
            result = runner.run_loop(config, should_stop=should_stop) if config.repeat_count != 1 else runner.run(config)
            with ARENA_JOBS_LOCK:
                stored = ARENA_JOBS.get(job_id)
                if stored:
                    stored["status"] = "done" if result.get("ok") else "error"
                    if result.get("stopped"):
                        stored["phase"] = "stopped"
                    elif result.get("ok"):
                        stored["phase"] = stored.get("phase") if stored.get("phase") == "finished" else "finished"
                    else:
                        stored["phase"] = "error"
                    stored["result"] = compact_result_payload(result)
                    stored["error"] = "" if result.get("ok") else str(result.get("error", result.get("prepare", "")))
                    stored["updated_at"] = time.time()
            log_event(
                "arena_job_finished" if result.get("ok") else "arena_job_failed",
                category="arena",
                level="info" if result.get("ok") else "error",
                source="arena_runner",
                initiator="cli",
                account_id=account_id,
                job_id=job_id,
                mode=config.mode,
                phase="finished" if result.get("ok") else "error",
                reason="" if result.get("ok") else str(result.get("error", result.get("prepare", "")))[:300],
                payload=compact_result_payload(result),
            )
        except Exception as exc:
            with ARENA_JOBS_LOCK:
                stored = ARENA_JOBS.get(job_id)
                if stored:
                    stored["status"] = "error"
                    stored["phase"] = "error"
                    stored["error"] = str(exc)
                    stored["updated_at"] = time.time()
            log_event(
                "arena_job_exception",
                category="arena",
                level="error",
                source="arena_runner",
                initiator="cli",
                account_id=account_id,
                job_id=job_id,
                mode=config.mode,
                phase="error",
                reason=str(exc),
            )

    threading.Thread(target=run_job, name=f"arena-job-{job_id[:8]}", daemon=True).start()
    return json.loads(json.dumps(job, ensure_ascii=False))


def arena_log_level(event: dict[str, Any]) -> str:
    name = str(event.get("event", "")).lower()
    phase = str(event.get("phase", "")).lower()
    if event.get("recoverable") or name.startswith("recoverable_") or phase == "recovering":
        return "warning"
    if phase == "error" or "failed" in name or name == "error" or event.get("error"):
        return "error"
    if "stop" in name or event.get("stopped"):
        return "warning"
    return "info"


def merge_arena_progress(progress: dict[str, Any], update: dict[str, Any]) -> None:
    phase = str(update.get("phase", "") or "").lower()
    event = str(update.get("event", "") or "").lower()
    if phase in {"preparing", "connecting", "searching", "cooldown", "ready"} or event in {
        "start",
        "loop_start",
        "socket_connecting",
        "socket_connected",
        "matchmaker_add_send",
        "queue_started",
        "matchmaker_wait",
    }:
        for key in ("battle", "draft", "outcome", "grade"):
            progress.pop(key, None)
    elif phase == "draft":
        for key in ("battle", "outcome", "grade"):
            progress.pop(key, None)
    elif phase == "battle":
        for key in ("outcome", "grade"):
            progress.pop(key, None)
        if event in {"matched", "match_join_send", "match_joined", "random_battle_match", "active_battle_found", "active_battle_joined", "battle_rejoined"}:
            progress.pop("battle", None)
            progress.pop("draft", None)
    for key, value in update.items():
        if key == "timestamp":
            continue
        if phase == "battle" and key in {"outcome", "grade"}:
            continue
        else:
            progress[key] = value


def merge_plan_arena_job(plan_job: dict[str, Any], update: dict[str, Any]) -> None:
    event = str(update.get("event", "") or "")
    now = time.time()
    if event == "plan_arena_start":
        mode = canonical_web_arena_mode(str(update.get("mode", "") or ""))
        plan_job["active_arena"] = {
            "id": plan_arena_job_id(str(plan_job.get("id", ""))),
            "ok": True,
            "status": "running",
            "phase": "starting",
            "mode": mode,
            "account_id": plan_job.get("account_id", ""),
            "account": {},
            "created_at": now,
            "updated_at": now,
            "progress": {},
            "events": [],
            "result": None,
            "error": "",
            "stop_requested": bool(plan_job.get("stop_requested", False)),
            "repeat": {"count": 1, "delay_seconds": 0, "stop_on_error": False},
            "source": "plan",
            "plan_job_id": plan_job.get("id", ""),
        }
        return

    active = plan_job.get("active_arena")
    if not isinstance(active, dict):
        return
    if event == "plan_arena_event" and isinstance(update.get("arena_event"), dict):
        arena_update = compact_live_update(update["arena_event"])
        active["updated_at"] = now
        active["phase"] = arena_update.get("phase", active.get("phase", "running"))
        merge_arena_progress(active["progress"], arena_update)
        active["events"].append(arena_update)
        active["events"] = active["events"][-80:]
        if active["phase"] == "error":
            active["status"] = "error"
        return
    if event == "plan_arena_complete":
        result = update.get("result", {})
        active["updated_at"] = now
        active["status"] = "done" if update.get("ok") else "error"
        active["phase"] = "finished" if update.get("ok") else "error"
        active["result"] = compact_result_payload(result) if isinstance(result, dict) else None
        active["error"] = "" if update.get("ok") else str(update.get("error", ""))
        active["events"].append(compact_live_update(update))
        active["events"] = active["events"][-80:]


def canonical_web_arena_mode(mode: str) -> str:
    return {"battle": "Default", "default": "Default", "random": "Random", "platinum": "Platinum", "daily": "Daily"}.get(str(mode or "").lower(), mode)


def event_reason(event: dict[str, Any]) -> str:
    for key in ("reason", "error"):
        value = event.get(key)
        if value:
            return str(value)[:500]
    for key in ("validation", "prepare", "result"):
        value = event.get(key)
        if isinstance(value, dict):
            nested = value.get("reason") or value.get("error")
            if nested:
                return str(nested)[:500]
            prepare = value.get("prepare")
            if isinstance(prepare, dict) and (prepare.get("reason") or prepare.get("error")):
                return str(prepare.get("reason") or prepare.get("error"))[:500]
    return ""


def event_message(event: dict[str, Any]) -> str:
    name = str(event.get("event", "event"))
    loop = event.get("loop")
    if isinstance(loop, dict):
        current = loop.get("current")
        target = loop.get("target", loop.get("requested", ""))
        if target == 0 or loop.get("infinite"):
            return f"{name}: battle {current or loop.get('completed', '')} / inf"
        if current:
            return f"{name}: battle {current} / {target or '?'}"
    battle = event.get("battle")
    if isinstance(battle, dict):
        outcome = battle.get("outcome")
        if outcome:
            return f"{name}: {outcome}"
    result = event.get("result")
    if isinstance(result, dict):
        nested_battle = result.get("battle")
        if isinstance(nested_battle, dict) and nested_battle.get("outcome"):
            return f"{name}: {nested_battle.get('outcome')}"
    return name


def compact_event_payload(event: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in event.items():
        if key == "timestamp":
            continue
        if key == "result" and isinstance(value, dict):
            payload[key] = compact_result_payload(value)
        elif key == "status" and isinstance(value, dict):
            payload[key] = {
                "ok": value.get("ok"),
                "mode": value.get("mode"),
                "validation": value.get("validation"),
                "location_valid": value.get("location_valid"),
                "team_rating": value.get("team_rating"),
                "reward_progress": value.get("reward_progress"),
            }
        elif key == "prepare" and isinstance(value, dict):
            payload[key] = compact_prepare_payload(value)
        elif key == "battle" and isinstance(value, dict):
            payload[key] = compact_battle_snapshot_payload(value)
        elif key == "data" and isinstance(value, dict):
            payload[key] = {item: value.get(item) for item in ("turns", "next_turn", "winner", "state") if item in value}
        elif key == "log" and isinstance(value, list):
            payload["log_count"] = len(value)
        elif key == "battles" and isinstance(value, list):
            payload["battle_count"] = len(value)
            payload["recent_battles"] = [compact_result_payload(item) for item in value[-3:] if isinstance(item, dict)]
        elif key == "arena_event" and isinstance(value, dict):
            payload[key] = compact_live_update(value)
        else:
            payload[key] = value
    return payload


def compact_live_update(event: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in event.items():
        if key == "timestamp":
            payload[key] = value
        elif key == "result" and isinstance(value, dict):
            payload[key] = compact_result_payload(value)
        elif key == "status" and isinstance(value, dict):
            payload[key] = {
                "ok": value.get("ok"),
                "mode": value.get("mode"),
                "validation": value.get("validation"),
                "location_valid": value.get("location_valid"),
                "team_rating": value.get("team_rating"),
                "reward_progress": value.get("reward_progress"),
            }
        elif key == "prepare" and isinstance(value, dict):
            payload[key] = compact_prepare_payload(value)
        elif key == "log" and isinstance(value, list):
            payload["log_count"] = len(value)
        elif key == "battles" and isinstance(value, list):
            payload["battle_count"] = len(value)
            payload["recent_battles"] = [compact_result_payload(item) for item in value[-3:] if isinstance(item, dict)]
        elif key == "arena_event" and isinstance(value, dict):
            payload[key] = compact_live_update(value)
        else:
            payload[key] = value
    if "timestamp" not in payload:
        payload["timestamp"] = time.time()
    return payload


def compact_result_payload(result: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": result.get("ok"),
        "mode": result.get("mode"),
    }
    for key in ("dry_run", "stopped", "failed", "recoverable", "reason", "error", "rules", "loop"):
        if key in result:
            payload[key] = result.get(key)
    if isinstance(result.get("prepare"), dict):
        payload["prepare"] = compact_prepare_payload(result["prepare"])
    if isinstance(result.get("status"), dict):
        payload["status"] = {
            "ok": result["status"].get("ok"),
            "validation": result["status"].get("validation"),
            "team_rating": result["status"].get("team_rating"),
            "reward_progress": result["status"].get("reward_progress"),
        }
    if isinstance(result.get("battle"), dict):
        payload["battle"] = compact_battle_result(result["battle"])
    battles = result.get("battles")
    if isinstance(battles, list):
        payload["battle_count"] = len(battles)
        payload["recent_battles"] = [compact_result_payload(item) for item in battles[-3:] if isinstance(item, dict)]
    return payload


def compact_prepare_payload(prepare: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in prepare.items()
        if key in {"ok", "reason", "error", "mode", "location_valid", "team_size", "healed", "moved", "team_changed"}
    }


def compact_battle_result(battle: dict[str, Any]) -> dict[str, Any]:
    return {
        key: battle.get(key)
        for key in ("ended", "terminated", "winner", "outcome", "turns_sent", "reason", "recoverable", "grade", "battle_log_id")
        if key in battle
    }


def compact_battle_snapshot_payload(battle: dict[str, Any]) -> dict[str, Any]:
    payload = {
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
        payload[side_key] = {
            "username": side.get("username"),
            "team_count": len(team),
            "deaths": sum(1 for item in team if isinstance(item, dict) and item.get("dead")),
            "active": {
                key: active.get(key)
                for key in ("id", "mid", "evo", "name", "chp", "max_hp", "element", "level")
                if key in active
            },
        }
    return payload


def current_arena_job(account_id: str = "") -> dict[str, Any] | None:
    running = [job for job in ARENA_JOBS.values() if job.get("status") == "running"]
    if account_id:
        running = [job for job in running if str(job.get("account_id", "")) == str(account_id)]
    if not running:
        return None
    return max(running, key=lambda job: float(job.get("updated_at", job.get("created_at", 0)) or 0))


def plan_arena_job_id(plan_job_id: str) -> str:
    return f"plan-arena:{plan_job_id}"


def plan_arena_job_by_id(job_id: str) -> dict[str, Any] | None:
    if not str(job_id or "").startswith("plan-arena:"):
        return None
    with PLAN_JOBS_LOCK:
        for plan_job in PLAN_JOBS.values():
            active = plan_job.get("active_arena")
            if isinstance(active, dict) and str(active.get("id", "")) == str(job_id):
                return json.loads(json.dumps(active, ensure_ascii=False))
    return None


def current_plan_arena_job(account_id: str = "") -> dict[str, Any] | None:
    with PLAN_JOBS_LOCK:
        jobs = [job for job in PLAN_JOBS.values() if job.get("status") == "running"]
        if account_id:
            jobs = [job for job in jobs if str(job.get("account_id", "")) == str(account_id)]
        active = [
            job.get("active_arena")
            for job in jobs
            if isinstance(job.get("active_arena"), dict) and job.get("active_arena", {}).get("status") == "running"
        ]
        if not active:
            return None
        payload = max(active, key=lambda job: float(job.get("updated_at", job.get("created_at", 0)) or 0))
        return json.loads(json.dumps(payload, ensure_ascii=False))


def trim_arena_jobs() -> None:
    if len(ARENA_JOBS) <= MAX_ARENA_JOBS:
        return
    old_ids = sorted(ARENA_JOBS, key=lambda key: float(ARENA_JOBS[key].get("created_at", 0)))
    for job_id in old_ids[: max(0, len(ARENA_JOBS) - MAX_ARENA_JOBS)]:
        if ARENA_JOBS[job_id].get("status") != "running":
            ARENA_JOBS.pop(job_id, None)


def create_plan_job(account_id: str, run_forever: bool = True) -> dict[str, Any]:
    with PLAN_JOBS_LOCK:
        existing = current_plan_job_unlocked(account_id)
        if existing and existing.get("status") == "running":
            return json.loads(json.dumps(existing, ensure_ascii=False))
        job = {
            "id": uuid.uuid4().hex,
            "ok": True,
            "status": "running",
            "phase": "starting",
            "account_id": account_id,
            "created_at": time.time(),
            "updated_at": time.time(),
            "events": [],
            "progress": {},
            "state": {},
            "error": "",
            "stop_requested": False,
            "run_forever": run_forever,
        }
        PLAN_JOBS[account_id] = job

    log_event("plan_job_created", category="plan", source="web", initiator="user", account_id=account_id, job_id=job["id"])

    def progress(update: dict[str, Any]) -> None:
        raw = json.loads(json.dumps(update, ensure_ascii=False))
        clean = compact_live_update(raw)
        with PLAN_JOBS_LOCK:
            stored = PLAN_JOBS.get(account_id)
            if not stored:
                return
            stored["phase"] = str(clean.get("phase", stored.get("phase", "running")))
            stored["updated_at"] = time.time()
            stored["progress"].update({key: value for key, value in clean.items() if key != "timestamp"})
            stored["events"].append(clean)
            stored["events"] = stored["events"][-100:]
            merge_plan_arena_job(stored, raw)

    def should_stop() -> bool:
        with PLAN_JOBS_LOCK:
            stored = PLAN_JOBS.get(account_id)
            return bool(stored and stored.get("stop_requested"))

    def run_job() -> None:
        try:
            runner = AccountPlanRunner(account_id, client_for_account(account_id), progress=progress)
            result = runner.run_forever(should_stop=should_stop) if run_forever else runner.run_once(should_stop=should_stop)
            with PLAN_JOBS_LOCK:
                stored = PLAN_JOBS.get(account_id)
                if stored:
                    stored["status"] = "done" if result.ok else "error"
                    stored["phase"] = result.status
                    stored["state"] = result.state
                    stored["error"] = result.reason
                    stored["updated_at"] = time.time()
            log_event(
                "plan_job_finished" if result.ok else "plan_job_failed",
                category="plan",
                level="info" if result.ok else "error",
                source="plan_runner",
                initiator="cli",
                account_id=account_id,
                job_id=job["id"],
                phase=result.status,
                reason=result.reason,
                payload=result.state,
            )
        except Exception as exc:
            with PLAN_JOBS_LOCK:
                stored = PLAN_JOBS.get(account_id)
                if stored:
                    stored["status"] = "error"
                    stored["phase"] = "error"
                    stored["error"] = str(exc)
                    stored["updated_at"] = time.time()
            log_event(
                "plan_job_exception",
                category="plan",
                level="error",
                source="plan_runner",
                initiator="cli",
                account_id=account_id,
                job_id=job["id"],
                phase="error",
                reason=str(exc),
            )

    threading.Thread(target=run_job, name=f"plan-job-{job['id'][:8]}", daemon=True).start()
    return json.loads(json.dumps(job, ensure_ascii=False))


def current_plan_job(account_id: str = "") -> dict[str, Any] | None:
    with PLAN_JOBS_LOCK:
        job = current_plan_job_unlocked(account_id)
        return json.loads(json.dumps(job, ensure_ascii=False)) if job else None


def current_plan_job_unlocked(account_id: str = "") -> dict[str, Any] | None:
    if account_id:
        return PLAN_JOBS.get(account_id)
    running = [job for job in PLAN_JOBS.values() if job.get("status") == "running"]
    return max(running, key=lambda item: float(item.get("updated_at", 0) or 0)) if running else None


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CLI Miscrits</title>
  <style>
    :root {
      font-family: Inter, Segoe UI, system-ui, sans-serif;
      color: #182026;
      background: #eef1f3;
      --panel: #ffffff;
      --line: #d8dde2;
      --muted: #66727c;
      --ink: #182026;
      --brand: #227c75;
      --brand-strong: #14635e;
      --accent: #b75d36;
      --soft: #f7f9fa;
      --good: #0f8f5f;
      --warn: #b26022;
      --bad: #b23b48;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: #eef1f3; color: var(--ink); }
    button, input, select, textarea { font: inherit; }
    button {
      border: 0;
      border-radius: 7px;
      background: var(--brand);
      color: white;
      padding: 9px 12px;
      cursor: pointer;
      min-height: 38px;
    }
    button:hover { background: var(--brand-strong); }
    button.secondary { background: #52616b; }
    button.warn { background: var(--accent); }
    button.ghost { background: #e7ecef; color: #263238; }
    input, select, textarea {
      border: 1px solid #cbd3d9;
      border-radius: 7px;
      background: white;
      color: var(--ink);
      padding: 9px 10px;
      min-height: 38px;
    }
    textarea { width: 100%; min-height: 120px; font-family: Consolas, monospace; resize: vertical; }
    .auth-shell {
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
      background: linear-gradient(135deg, #eef1f3 0%, #dfe8ea 100%);
    }
    .auth-card {
      width: min(460px, 100%);
      background: white;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 24px;
      box-shadow: 0 18px 50px rgba(24, 32, 38, .12);
      display: grid;
      gap: 14px;
    }
    .auth-card h1 { margin: 0; font-size: 24px; }
    .auth-card p { margin: 0; color: var(--muted); line-height: 1.45; }
    .auth-card .field input { width: 100%; }
    .auth-actions { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: center; }
    .auth-error { color: var(--bad); min-height: 18px; font-size: 13px; }
    .app { min-height: 100vh; display: grid; grid-template-columns: 246px minmax(0, 1fr); }
    aside {
      background: #202c33;
      color: #eef6f6;
      padding: 20px 16px;
      position: sticky;
      top: 0;
      height: 100vh;
    }
    .brand { display: flex; align-items: center; gap: 10px; margin-bottom: 22px; }
    .brand-mark { width: 34px; height: 34px; border-radius: 8px; background: #33a399; display: grid; place-items: center; font-weight: 800; }
    .brand h1 { margin: 0; font-size: 18px; }
    .status-pill { margin-left: auto; border-radius: 999px; padding: 4px 8px; background: #30424b; font-size: 12px; color: #d8efef; }
    nav { display: grid; gap: 7px; }
    nav button {
      width: 100%;
      background: transparent;
      color: #dce7eb;
      text-align: left;
      padding: 10px 12px;
    }
    nav button.active, nav button:hover { background: #30424b; }
    main { padding: 24px; display: grid; gap: 18px; align-content: start; }
    .topbar { display: flex; justify-content: space-between; gap: 12px; align-items: center; }
    .topbar h2 { margin: 0; font-size: 24px; }
    .cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
    .metric { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
    .metric span { display: block; color: var(--muted); font-size: 12px; margin-bottom: 5px; }
    .metric strong { font-size: 22px; }
    section.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }
    .panel h3 { margin: 0 0 14px; font-size: 16px; }
    .view { display: none; gap: 16px; }
    .view.active { display: grid; }
    .grid-2 { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; }
    .grid-3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
    .row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
    .field { display: grid; gap: 6px; }
    .field label { color: var(--muted); font-size: 12px; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: end; }
    .plan-card { display: grid; gap: 12px; background: var(--soft); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
    .plan-builder-toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: end; margin-top: 14px; }
    .plan-blocks { display: grid; gap: 10px; margin-top: 14px; }
    .plan-block {
      display: grid;
      grid-template-columns: minmax(180px, 1fr) minmax(280px, 360px) auto;
      gap: 12px;
      align-items: center;
      padding: 12px;
      border: 1px solid rgba(176, 132, 255, .22);
      border-radius: 8px;
      background: rgba(255,255,255,.03);
    }
    .plan-block-main { display: flex; gap: 10px; align-items: center; min-width: 0; }
    .plan-block-index {
      display: grid;
      place-items: center;
      width: 28px;
      height: 28px;
      border-radius: 999px;
      background: rgba(176, 132, 255, .16);
      color: #eadcff;
      font-weight: 700;
      flex: 0 0 auto;
    }
    .plan-block-title { display: grid; gap: 2px; min-width: 0; }
    .plan-block-title strong { font-size: 14px; }
    .plan-block-title span { color: var(--muted); font-size: 12px; }
    .plan-block-goal { display: grid; grid-template-columns: minmax(140px, 1fr) minmax(120px, 1fr); gap: 10px; }
    .plan-block-actions { display: flex; gap: 8px; justify-content: flex-end; flex-wrap: wrap; }
    .parents { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
    .parent, .log-card { background: white; border: 1px solid var(--line); border-radius: 8px; padding: 12px; }
    .log-card.splus-result {
      border-color: rgba(46,242,166,.72);
      box-shadow: 0 0 0 1px rgba(46,242,166,.2), 0 0 30px rgba(46,242,166,.18);
    }
    .splus-banner {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 10px;
      border: 1px solid rgba(46,242,166,.42);
      border-radius: 999px;
      background: rgba(46,242,166,.14);
      color: #0d6d4a;
      font-weight: 800;
      letter-spacing: 0;
    }
    .miscrit-card {
      --rarity-base: #f4edcf;
      --rarity-mid: #efe5bd;
      --rarity-line: #d3c898;
      --rarity-ink: #554f39;
      position: relative;
      display: grid;
      grid-template-columns: 102px minmax(0, 1fr) 40px;
      grid-template-rows: auto minmax(0, 1fr) auto;
      gap: 8px 12px;
      min-height: 188px;
      padding: 12px;
      border: 1px solid color-mix(in srgb, var(--rarity-line), #6e6a58 26%);
      border-radius: 8px;
      background:
        linear-gradient(145deg, color-mix(in srgb, var(--rarity-base), #fff 58%), var(--rarity-base) 58%, var(--rarity-mid)),
        radial-gradient(circle at 78% 18%, rgba(255,255,255,.42), transparent 34%);
      color: var(--rarity-ink);
      box-shadow: inset 0 1px 0 rgba(255,255,255,.68), 0 1px 2px rgba(28, 34, 38, .06);
      overflow: hidden;
    }
    .miscrit-card::after {
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      background: linear-gradient(180deg, rgba(255,255,255,.34), transparent 36%);
    }
    .miscrit-card.rarity-common { --rarity-base: #f5edcf; --rarity-mid: #eadfac; --rarity-line: #cfc28f; --rarity-ink: #57513b; }
    .miscrit-card.rarity-rare { --rarity-base: #dceffc; --rarity-mid: #bcdcf2; --rarity-line: #85b8d7; --rarity-ink: #25485f; }
    .miscrit-card.rarity-epic { --rarity-base: #dcf2df; --rarity-mid: #b8e0c0; --rarity-line: #7dbb8a; --rarity-ink: #285536; }
    .miscrit-card.rarity-exotic { --rarity-base: #ffe3c1; --rarity-mid: #ffc47d; --rarity-line: #d88b35; --rarity-ink: #704012; }
    .miscrit-card.rarity-legend, .miscrit-card.rarity-legendary { --rarity-base: #ffefad; --rarity-mid: #f2c94c; --rarity-line: #b88a12; --rarity-ink: #67500d; }
    .miscrit-card.splus-card {
      border-color: rgba(46,242,166,.74);
      box-shadow:
        inset 0 1px 0 rgba(255,255,255,.68),
        0 0 0 1px rgba(46,242,166,.22),
        0 0 28px rgba(46,242,166,.24);
      animation: splusGlow 2.1s ease-in-out infinite;
    }
    .miscrit-avatar {
      position: relative;
      z-index: 1;
      grid-row: 1 / 3;
      width: 102px;
      aspect-ratio: 1;
      border: 1px solid color-mix(in srgb, var(--rarity-line), #fff 30%);
      border-radius: 6px;
      background: rgba(255,255,255,.72);
      display: grid;
      place-items: center;
      overflow: hidden;
      font-weight: 800;
      font-size: 28px;
      color: color-mix(in srgb, var(--rarity-ink), #fff 18%);
    }
    .miscrit-avatar img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .rating-mark {
      position: absolute;
      right: 6px;
      bottom: 6px;
      min-width: 28px;
      height: 28px;
      padding: 0 5px;
      border-radius: 6px;
      display: grid;
      place-items: center;
      background: rgba(255,255,255,.9);
      border: 1px solid color-mix(in srgb, var(--rarity-line), #6e6a58 22%);
      font-weight: 800;
      font-size: 13px;
      color: var(--rarity-ink);
      box-shadow: 0 1px 2px rgba(28,34,38,.12);
    }
    .miscrit-main { position: relative; z-index: 1; min-width: 0; }
    .miscrit-name {
      font-size: 18px;
      font-weight: 800;
      line-height: 1.1;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      border-bottom: 3px solid color-mix(in srgb, var(--rarity-line), #6e6a58 28%);
      padding-bottom: 5px;
      margin-right: 4px;
    }
    .miscrit-mid {
      margin-top: 8px;
      font-size: 28px;
      font-weight: 800;
      line-height: 1;
      border-bottom: 3px solid color-mix(in srgb, var(--rarity-line), #6e6a58 28%);
      padding-bottom: 7px;
    }
    .miscrit-subline { margin-top: 6px; font-size: 12px; color: color-mix(in srgb, var(--rarity-ink), #5b6470 38%); }
    .element-orb {
      position: relative;
      z-index: 1;
      align-self: start;
      justify-self: end;
      width: 38px;
      height: 38px;
      border-radius: 50%;
      border: 2px solid color-mix(in srgb, var(--rarity-line), #6e6a58 25%);
      background: rgba(255,255,255,.48);
      display: grid;
      place-items: center;
      font-size: 11px;
      font-weight: 800;
      color: var(--rarity-ink);
      text-transform: uppercase;
    }
    .element-orb img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }
    .element-badge {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      min-height: 20px;
      color: inherit;
      white-space: nowrap;
    }
    .element-badge img {
      width: 20px;
      height: 20px;
      object-fit: contain;
      flex: 0 0 auto;
    }
    .miscrit-stat-grid {
      position: relative;
      z-index: 1;
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 7px;
      align-self: end;
    }
    .miscrit-stat {
      min-height: 36px;
      border: 1px solid color-mix(in srgb, var(--rarity-line), #6e6a58 22%);
      border-radius: 4px;
      background: rgba(255,255,255,.38);
      display: grid;
      place-items: center;
      font-weight: 800;
      font-size: 12px;
      text-align: center;
    }
    .miscrit-stat.v3 { background: rgba(176, 232, 197, .72); color: #17653f; }
    .miscrit-stat.v2 { background: rgba(255,255,255,.48); color: color-mix(in srgb, var(--rarity-ink), #344 28%); }
    .miscrit-stat.v1 { background: rgba(255, 212, 217, .82); color: #8c2832; }
    .chips { display: flex; gap: 6px; flex-wrap: wrap; }
    .chip { border-radius: 999px; background: #e8eff1; color: #2d3b43; padding: 3px 8px; font-size: 12px; }
    .chip.good { background: #dff4ea; color: #0d6d4a; }
    .chip.warn { background: #fde9d8; color: #8a4518; }
    .stat-line { display: grid; grid-template-columns: repeat(6, 1fr); gap: 5px; margin-top: 8px; }
    .stat { border-radius: 5px; padding: 5px 4px; text-align: center; font-size: 12px; background: #edf1f3; }
    .stat.v3 { background: #dff4ea; color: #0d6d4a; }
    .stat.v2 { background: #eef1f3; color: #394750; }
    .stat.v1 { background: #ffe2e4; color: #8c2832; }
    .logs { display: grid; gap: 10px; max-height: 520px; overflow: auto; padding-right: 4px; }
    .log-head { display: flex; justify-content: space-between; gap: 10px; align-items: start; }
    .stage-track { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }
    .stage { border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #edf1f3; color: #52616b; text-align: center; font-size: 13px; }
    .stage.active { background: #dff4ea; color: #0d6d4a; border-color: #9ed9bf; font-weight: 700; }
    .stage.done { background: #eef8f5; color: #227c75; }
    .battle-board { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; }
    .battle-side { border: 1px solid var(--line); border-radius: 8px; background: var(--soft); padding: 12px; display: grid; gap: 8px; }
    .battle-team { display: grid; gap: 8px; }
    .miscrit-pill { display: grid; grid-template-columns: 42px minmax(0, 1fr); gap: 9px; align-items: center; border: 1px solid #d8dde2; border-radius: 8px; background: white; padding: 8px; opacity: .78; }
    .miscrit-pill.active { border-color: #227c75; box-shadow: 0 0 0 2px rgba(34,124,117,.15); opacity: 1; }
    .miscrit-icon { width: 42px; height: 42px; border-radius: 50%; display: grid; place-items: center; background: #dfe9ec; color: #203039; font-weight: 800; }
    .miscrit-pill.dead .miscrit-icon { background: #f2d9dc; color: #8c2832; }
    .hpbar { height: 7px; background: #dce3e7; border-radius: 999px; overflow: hidden; margin-top: 5px; }
    .hpfill { height: 100%; background: var(--good); border-radius: inherit; }
    .hpfill.low { background: var(--bad); }
    .draft-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
    .arena-rewards {
      display: grid;
      gap: 12px;
      margin-top: 14px;
    }
    .arena-reward-summary {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 8px;
    }
    .arena-reward-total {
      border: 1px solid rgba(184,132,255,.24);
      border-radius: 8px;
      padding: 10px;
      background: rgba(255,255,255,.05);
      display: grid;
      gap: 4px;
    }
    .arena-reward-total strong { font-size: 18px; }
    .arena-reward-total.platinum strong { color: #e4eeff; }
    .arena-reward-total.gems strong { color: #7dffca; }
    .arena-reward-total.miscrit strong { color: #ffc9f3; }
    .arena-reward-list {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 8px;
    }
    .arena-reward-item {
      display: grid;
      grid-template-columns: 44px minmax(0, 1fr);
      gap: 10px;
      align-items: center;
      min-height: 70px;
      border: 1px solid rgba(184,132,255,.24);
      border-radius: 8px;
      padding: 10px;
      background: rgba(255,255,255,.05);
    }
    .arena-reward-item.claimed {
      border-color: rgba(46,242,166,.38);
      background: rgba(46,242,166,.09);
    }
    .arena-reward-icon {
      width: 44px;
      height: 44px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      overflow: hidden;
      background: rgba(255,255,255,.1);
      font-size: 20px;
      font-weight: 800;
    }
    .arena-reward-icon img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    .arena-reward-item .muted { font-size: 12px; }
    .ai-metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
    }
    .ai-dashboard {
      display: grid;
      gap: 10px;
    }
    .ai-metric {
      border: 1px solid rgba(184,132,255,.24);
      border-radius: 8px;
      padding: 12px;
      background: rgba(255,255,255,.05);
      display: grid;
      gap: 5px;
    }
    .ai-metric strong { font-size: 24px; }
    .ai-chart {
      display: grid;
      gap: 8px;
    }
    .ai-summary-metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 8px;
    }
    .ai-summary-metric {
      min-height: 58px;
      border: 1px solid rgba(184,132,255,.24);
      border-radius: 8px;
      padding: 9px 11px;
      background: rgba(255,255,255,.04);
      display: grid;
      gap: 3px;
      align-content: center;
    }
    .ai-summary-metric strong { font-size: 18px; }
    .ai-bars { display: grid; gap: 7px; }
    .ai-bar {
      display: grid;
      grid-template-columns: minmax(120px, 1fr) minmax(120px, 2fr) 56px;
      gap: 8px;
      align-items: center;
    }
    .ai-bar-track {
      height: 10px;
      border-radius: 999px;
      background: rgba(255,255,255,.08);
      overflow: hidden;
    }
    .ai-bar-fill {
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, #ff4fd8, #8b5cf6);
    }
    .ai-bar-fill.good { background: linear-gradient(90deg, #2ef2a6, #00e5ff); }
    .ai-bar-fill.warn { background: linear-gradient(90deg, #ffb86b, #ff4e7a); }
    .ai-sparkline {
      display: flex;
      align-items: end;
      gap: 3px;
      min-height: 120px;
      padding: 8px;
      border: 1px solid rgba(184,132,255,.24);
      border-radius: 8px;
      background: rgba(255,255,255,.04);
    }
    .ai-spark {
      flex: 1 1 0;
      min-width: 4px;
      border-radius: 999px 999px 0 0;
      background: rgba(184,132,255,.72);
    }
    .ai-spark.victory { background: linear-gradient(180deg, #2ef2a6, #00e5ff); }
    .ai-spark.defeat { background: linear-gradient(180deg, #ffb86b, #ff4e7a); }
    .ai-weight-grid { display: grid; gap: 14px; }
    .ai-weight-table { display: grid; gap: 6px; }
    .ai-weight-row {
      display: grid;
      grid-template-columns: minmax(180px, 1fr) 88px 72px 110px;
      gap: 8px;
      align-items: center;
    }
    .ai-weight-row input { width: 100%; }
    .ai-weight-key {
      overflow-wrap: anywhere;
      font-size: 13px;
    }
    .muted { color: var(--muted); }
    .hidden { display: none !important; }
    .boot-fallback {
      position: fixed;
      inset: 0;
      z-index: 9999;
      display: grid;
      place-items: center;
      padding: 18px;
      background:
        radial-gradient(circle at 18% -12%, rgba(174, 97, 255, .26), transparent 38%),
        linear-gradient(135deg, #120a26 0%, #1a0d38 42%, #0d1028 100%);
      color: #f4edff;
    }
    .boot-fallback-card {
      width: min(520px, 100%);
      border: 1px solid rgba(184,132,255,.3);
      border-radius: 8px;
      padding: 18px;
      background: linear-gradient(145deg, rgba(34, 22, 72, .94), rgba(16, 14, 42, .92));
      box-shadow: 0 18px 42px rgba(3,0,20,.34);
    }
    .boot-fallback-card strong { display: block; margin-bottom: 8px; font-size: 18px; }
    .boot-ok .boot-fallback { display: none; }
    body {
      background:
        radial-gradient(circle at 18% -12%, rgba(174, 97, 255, .26), transparent 38%),
        radial-gradient(circle at 92% 8%, rgba(0, 229, 255, .16), transparent 30%),
        linear-gradient(135deg, #120a26 0%, #1a0d38 42%, #0d1028 100%);
      color: #f4edff;
    }
    :root {
      --panel: rgba(26, 18, 55, .84);
      --line: rgba(187, 136, 255, .26);
      --muted: #b9a9d8;
      --ink: #f4edff;
      --brand: #8b5cf6;
      --brand-strong: #a855f7;
      --accent: #ff4fd8;
      --soft: rgba(255,255,255,.06);
      --good: #2ef2a6;
      --warn: #ffb86b;
      --bad: #ff4e7a;
    }
    .auth-shell {
      background:
        radial-gradient(circle at 22% 18%, rgba(255, 79, 216, .22), transparent 28%),
        linear-gradient(135deg, #120a26, #24104d 58%, #0b1028);
    }
    .auth-card, section.panel, .metric, .parent, .log-card, .battle-side, .plan-card {
      background: linear-gradient(145deg, rgba(34, 22, 72, .92), rgba(16, 14, 42, .88));
      border-color: rgba(184, 132, 255, .28);
      box-shadow: 0 18px 42px rgba(3, 0, 20, .34), inset 0 1px 0 rgba(255,255,255,.08);
      color: var(--ink);
      backdrop-filter: blur(16px);
    }
    aside {
      background: linear-gradient(180deg, rgba(27, 13, 58, .96), rgba(12, 12, 32, .98));
      border-right: 1px solid rgba(184, 132, 255, .24);
      box-shadow: 8px 0 32px rgba(3, 0, 20, .24);
    }
    .brand-mark {
      background: linear-gradient(135deg, #ff4fd8, #7c3aed 52%, #00e5ff);
      box-shadow: 0 0 24px rgba(168, 85, 247, .55);
    }
    .status-pill, nav button.active, nav button:hover {
      background: rgba(139, 92, 246, .2);
      border: 1px solid rgba(184, 132, 255, .26);
      color: #f4edff;
    }
    button {
      background: linear-gradient(135deg, #7c3aed, #a855f7);
      box-shadow: 0 0 18px rgba(139, 92, 246, .24);
    }
    button:hover { background: linear-gradient(135deg, #8b5cf6, #c026d3); }
    button.secondary { background: linear-gradient(135deg, #334155, #5b4b8a); }
    button.warn { background: linear-gradient(135deg, #db2777, #f97316); }
    button.ghost {
      background: rgba(255,255,255,.08);
      color: #eadcff;
      border: 1px solid rgba(184, 132, 255, .24);
    }
    button.danger { color: #ffd6df; border-color: rgba(255, 78, 122, .42); }
    input, select, textarea {
      background: rgba(13, 10, 31, .72);
      border-color: rgba(184, 132, 255, .28);
      color: #f4edff;
    }
    input::placeholder { color: #8e7aad; }
    .metric {
      position: relative;
      min-height: 92px;
      overflow: hidden;
    }
    .metric::before {
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(120deg, rgba(255,255,255,.08), transparent 38%, rgba(0,229,255,.08));
      pointer-events: none;
    }
    .metric span { color: #cbbdf2; text-transform: uppercase; letter-spacing: 0; }
    .metric strong {
      position: relative;
      font-size: 28px;
      text-shadow: 0 0 18px rgba(168, 85, 247, .55);
    }
    .metric.resource-gold {
      border-color: rgba(255, 215, 96, .48);
      background:
        radial-gradient(circle at 82% 18%, rgba(255, 215, 96, .22), transparent 34%),
        linear-gradient(145deg, rgba(54, 38, 8, .92), rgba(26, 18, 55, .84));
    }
    .metric.resource-gold strong {
      color: #ffe680;
      text-shadow: 0 0 20px rgba(255, 215, 96, .58);
    }
    .metric.resource-gems {
      border-color: rgba(46, 242, 166, .46);
      background:
        radial-gradient(circle at 82% 18%, rgba(46, 242, 166, .2), transparent 34%),
        linear-gradient(145deg, rgba(7, 48, 38, .9), rgba(26, 18, 55, .84));
    }
    .metric.resource-gems strong {
      color: #7dffca;
      text-shadow: 0 0 20px rgba(46, 242, 166, .52);
    }
    .metric.resource-platinum {
      border-color: rgba(218, 232, 255, .48);
      background:
        radial-gradient(circle at 82% 18%, rgba(218, 232, 255, .2), transparent 34%),
        linear-gradient(145deg, rgba(58, 66, 86, .9), rgba(26, 18, 55, .84));
    }
    .metric.resource-platinum strong {
      color: #eaf2ff;
      text-shadow: 0 0 20px rgba(218, 232, 255, .46);
    }
    .chip {
      background: rgba(255,255,255,.08);
      color: #eadcff;
      border: 1px solid rgba(184, 132, 255, .18);
    }
    .chip.good { background: rgba(46,242,166,.12); color: #a8ffd8; border-color: rgba(46,242,166,.28); }
    .chip.warn { background: rgba(255,78,122,.12); color: #ffd1dc; border-color: rgba(255,78,122,.28); }
    .miscrit-card {
      --rarity-base: rgba(139, 92, 246, .22);
      --rarity-mid: rgba(255, 79, 216, .12);
      --rarity-line: rgba(184, 132, 255, .48);
      --rarity-ink: #f4edff;
      background:
        radial-gradient(circle at 18% 10%, color-mix(in srgb, var(--rarity-line), transparent 42%), transparent 36%),
        linear-gradient(145deg, rgba(32, 20, 70, .92), rgba(13, 14, 38, .94) 58%, var(--rarity-base));
      color: #f4edff;
      border-color: color-mix(in srgb, var(--rarity-line), #ffffff 8%);
      box-shadow:
        0 18px 42px rgba(3, 0, 20, .28),
        0 0 26px color-mix(in srgb, var(--rarity-line), transparent 68%),
        inset 0 1px 0 rgba(255,255,255,.08);
    }
    .miscrit-card::after {
      background:
        linear-gradient(180deg, rgba(255,255,255,.08), transparent 34%),
        linear-gradient(90deg, color-mix(in srgb, var(--rarity-line), transparent 30%), transparent 3px);
      opacity: .9;
    }
    .miscrit-card.rarity-common { --rarity-base: rgba(210, 200, 255, .12); --rarity-mid: rgba(255,255,255,.055); --rarity-line: rgba(196, 181, 253, .42); --rarity-ink: #f4edff; }
    .miscrit-card.rarity-rare { --rarity-base: rgba(0, 229, 255, .14); --rarity-mid: rgba(139, 92, 246, .12); --rarity-line: rgba(0, 229, 255, .58); --rarity-ink: #effcff; }
    .miscrit-card.rarity-epic { --rarity-base: rgba(46, 242, 166, .13); --rarity-mid: rgba(139, 92, 246, .12); --rarity-line: rgba(46, 242, 166, .58); --rarity-ink: #effff8; }
    .miscrit-card.rarity-exotic { --rarity-base: rgba(255, 184, 107, .16); --rarity-mid: rgba(255, 79, 216, .12); --rarity-line: rgba(255, 184, 107, .62); --rarity-ink: #fff6e9; }
    .miscrit-card.rarity-legend, .miscrit-card.rarity-legendary { --rarity-base: rgba(255, 215, 96, .18); --rarity-mid: rgba(255, 79, 216, .14); --rarity-line: rgba(255, 215, 96, .68); --rarity-ink: #fff8dc; }
    .log-card.splus-result {
      background:
        radial-gradient(circle at 86% 18%, rgba(46,242,166,.16), transparent 28%),
        linear-gradient(145deg, rgba(255,255,255,.07), rgba(255,255,255,.025));
    }
    .splus-banner {
      color: #a8ffd8;
      box-shadow: 0 0 18px rgba(46,242,166,.18);
    }
    .miscrit-avatar {
      background:
        radial-gradient(circle at 30% 22%, rgba(255,255,255,.28), transparent 30%),
        linear-gradient(145deg, rgba(255,255,255,.12), rgba(255,255,255,.035));
      border-color: color-mix(in srgb, var(--rarity-line), #ffffff 12%);
      color: var(--rarity-ink);
      box-shadow: inset 0 0 20px rgba(255,255,255,.055), 0 0 18px color-mix(in srgb, var(--rarity-line), transparent 68%);
    }
    .rating-mark {
      background: linear-gradient(135deg, rgba(255,255,255,.95), color-mix(in srgb, var(--rarity-line), #ffffff 62%));
      border-color: color-mix(in srgb, var(--rarity-line), #ffffff 18%);
      color: #160b2e;
      box-shadow: 0 0 18px color-mix(in srgb, var(--rarity-line), transparent 58%);
    }
    .miscrit-name, .miscrit-mid {
      border-bottom-color: color-mix(in srgb, var(--rarity-line), #ffffff 8%);
      color: var(--rarity-ink);
      text-shadow: 0 0 14px color-mix(in srgb, var(--rarity-line), transparent 60%);
    }
    .miscrit-subline {
      color: #cbbdf2;
    }
    .element-orb {
      border-color: color-mix(in srgb, var(--rarity-line), #ffffff 12%);
      background: radial-gradient(circle at 35% 30%, rgba(255,255,255,.28), rgba(255,255,255,.06));
      color: var(--rarity-ink);
      box-shadow: 0 0 18px color-mix(in srgb, var(--rarity-line), transparent 66%);
    }
    .element-badge img {
      filter: drop-shadow(0 0 8px rgba(255,255,255,.22));
    }
    .miscrit-stat {
      background: rgba(255,255,255,.055);
      border-color: rgba(255,255,255,.1);
      color: #eadcff;
    }
    .miscrit-stat.v3 {
      background: rgba(46, 242, 166, .16);
      color: #a8ffd8;
      border-color: rgba(46, 242, 166, .28);
    }
    .miscrit-stat.v2 {
      background: rgba(255,255,255,.065);
      color: #d7c8f7;
      border-color: rgba(184,132,255,.2);
    }
    .miscrit-stat.v1 {
      background: rgba(255, 78, 122, .16);
      color: #ffd1dc;
      border-color: rgba(255, 78, 122, .28);
    }
    .dashboard-stack { display: grid; gap: 16px; }
    .account-card-grid, .gauge-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
    .profile-card {
      border: 1px solid rgba(184,132,255,.26);
      border-radius: 8px;
      padding: 14px;
      background: linear-gradient(145deg, rgba(255,255,255,.07), rgba(255,255,255,.025));
      display: grid;
      gap: 12px;
      min-height: 142px;
    }
    .profile-card.active {
      border-color: rgba(46,242,166,.48);
      box-shadow: 0 0 0 1px rgba(46,242,166,.18), 0 0 28px rgba(46,242,166,.12);
    }
    .profile-top { display: flex; justify-content: space-between; gap: 12px; align-items: start; }
    .profile-avatar {
      width: 46px;
      height: 46px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      background: linear-gradient(135deg, #ff4fd8, #7c3aed);
      font-weight: 900;
      color: white;
      box-shadow: 0 0 20px rgba(255,79,216,.34);
    }
    .player-card {
      display: grid;
      grid-template-columns: 72px minmax(0, 1fr);
      gap: 14px;
      align-items: center;
      border: 1px solid rgba(184,132,255,.26);
      border-radius: 8px;
      padding: 14px;
      background: linear-gradient(145deg, rgba(124,58,237,.18), rgba(0,229,255,.06));
    }
    .player-avatar {
      width: 72px;
      height: 72px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      background: linear-gradient(135deg, #00e5ff, #8b5cf6);
      font-size: 30px;
      font-weight: 900;
      color: #0b1028;
    }
    .gauge-card {
      border: 1px solid rgba(184,132,255,.26);
      border-radius: 8px;
      padding: 14px;
      display: grid;
      justify-items: center;
      gap: 8px;
      background: linear-gradient(145deg, rgba(255,255,255,.07), rgba(255,255,255,.025));
    }
    .gauge-arc {
      --win: 0;
      --loss: 100;
      width: min(210px, 100%);
      aspect-ratio: 2 / 1;
      position: relative;
      overflow: hidden;
      border-radius: 210px 210px 0 0;
      background:
        conic-gradient(from 270deg at 50% 100%,
          #ff4e7a 0 var(--loss-deg),
          #2ef2a6 var(--loss-deg) var(--total-deg),
          rgba(255,255,255,.12) var(--total-deg) 180deg,
          transparent 180deg 360deg);
      box-shadow: 0 0 24px rgba(139,92,246,.22);
    }
    .gauge-arc::after {
      content: "";
      position: absolute;
      left: 14%;
      right: 14%;
      bottom: 0;
      height: 72%;
      border-radius: 180px 180px 0 0;
      background: #171034;
      box-shadow: inset 0 0 18px rgba(255,255,255,.05);
    }
    .gauge-value {
      position: absolute;
      z-index: 1;
      left: 0;
      right: 0;
      bottom: 14px;
      text-align: center;
      font-size: 24px;
      font-weight: 900;
      color: #f4edff;
    }
    .arena-console {
      display: grid;
      gap: 14px;
      min-height: 520px;
    }
    .arena-header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 14px;
      border-radius: 8px;
      border: 1px solid rgba(184,132,255,.24);
      background: linear-gradient(145deg, rgba(124,58,237,.16), rgba(15,23,42,.36));
    }
    .stage-track {
      gap: 10px;
      background: rgba(255,255,255,.04);
      border: 1px solid rgba(184,132,255,.18);
      border-radius: 8px;
      padding: 8px;
    }
    .stage {
      border: 1px solid rgba(184,132,255,.2);
      background: rgba(255,255,255,.045);
      color: #a897cf;
      text-transform: uppercase;
      font-weight: 800;
    }
    .stage.active {
      background: linear-gradient(135deg, rgba(168,85,247,.36), rgba(0,229,255,.16));
      color: #fff;
      border-color: rgba(0,229,255,.42);
      box-shadow: 0 0 22px rgba(0,229,255,.14);
    }
    .stage.done {
      background: rgba(46,242,166,.1);
      color: #a8ffd8;
      border-color: rgba(46,242,166,.24);
    }
    .arena-scene {
      min-height: 370px;
      border: 1px solid rgba(184,132,255,.24);
      border-radius: 8px;
      padding: 18px;
      background:
        linear-gradient(145deg, rgba(20,12,49,.92), rgba(11,16,40,.94));
      overflow: hidden;
    }
    .search-scene {
      min-height: 330px;
      display: grid;
      place-items: center;
      text-align: center;
    }
    .search-core {
      width: 172px;
      height: 172px;
      border-radius: 50%;
      border: 2px solid rgba(0,229,255,.42);
      position: relative;
      display: grid;
      place-items: center;
      color: #fff;
      font-weight: 900;
      background: radial-gradient(circle, rgba(168,85,247,.34), rgba(0,229,255,.08) 58%, transparent 60%);
      box-shadow: 0 0 42px rgba(0,229,255,.18), inset 0 0 28px rgba(168,85,247,.22);
      animation: searchPulse 1.7s ease-in-out infinite;
    }
    .search-core::before, .search-core::after {
      content: "";
      position: absolute;
      inset: -18px;
      border-radius: inherit;
      border: 1px solid rgba(255,79,216,.36);
      animation: searchRing 2.2s linear infinite;
    }
    .search-core::after {
      inset: -36px;
      border-color: rgba(46,242,166,.28);
      animation-duration: 3.1s;
      animation-direction: reverse;
    }
    .draft-board { display: grid; gap: 16px; }
    .draft-card-grid { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; }
    .draft-card {
      --rarity-base: rgba(139, 92, 246, .12);
      --rarity-line: rgba(184, 132, 255, .34);
      --rarity-ink: #f4edff;
      position: relative;
      min-height: 112px;
      border: 1px solid color-mix(in srgb, var(--rarity-line), #ffffff 8%);
      border-radius: 8px;
      padding: 10px;
      background:
        radial-gradient(circle at 18% 12%, color-mix(in srgb, var(--rarity-line), transparent 55%), transparent 38%),
        linear-gradient(145deg, rgba(255,255,255,.065), var(--rarity-base));
      display: grid;
      gap: 8px;
      align-content: start;
      transition: transform .18s ease, border-color .18s ease, opacity .18s ease;
      box-shadow: 0 0 18px color-mix(in srgb, var(--rarity-line), transparent 78%);
    }
    .draft-card.rarity-common, .battle-card.rarity-common { --rarity-base: rgba(210, 200, 255, .10); --rarity-line: rgba(196, 181, 253, .38); --rarity-ink: #f4edff; }
    .draft-card.rarity-rare, .battle-card.rarity-rare { --rarity-base: rgba(0, 229, 255, .13); --rarity-line: rgba(0, 229, 255, .56); --rarity-ink: #effcff; }
    .draft-card.rarity-epic, .battle-card.rarity-epic { --rarity-base: rgba(46, 242, 166, .13); --rarity-line: rgba(46, 242, 166, .56); --rarity-ink: #effff8; }
    .draft-card.rarity-exotic, .battle-card.rarity-exotic { --rarity-base: rgba(255, 184, 107, .15); --rarity-line: rgba(255, 184, 107, .6); --rarity-ink: #fff6e9; }
    .draft-card.rarity-legend, .draft-card.rarity-legendary, .battle-card.rarity-legend, .battle-card.rarity-legendary { --rarity-base: rgba(255, 215, 96, .18); --rarity-line: rgba(255, 215, 96, .68); --rarity-ink: #fff8dc; }
    .draft-card.picked {
      border-color: rgba(46,242,166,.62);
      box-shadow: 0 0 24px rgba(46,242,166,.16);
      transform: translateY(-3px);
    }
    .draft-card.banned { opacity: .48; }
    .draft-card.banned::after {
      content: "";
      position: absolute;
      left: 8px;
      right: 8px;
      top: 50%;
      height: 2px;
      background: #ff4e7a;
      box-shadow: 0 0 12px rgba(255,78,122,.8);
      transform: rotate(-12deg);
    }
    .draft-icon, .battle-avatar {
      width: 46px;
      height: 46px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      background:
        radial-gradient(circle at 30% 22%, rgba(255,255,255,.32), transparent 30%),
        linear-gradient(135deg, color-mix(in srgb, var(--rarity-line), #ffffff 10%), rgba(168,85,247,.72));
      color: #0b1028;
      font-weight: 900;
      overflow: hidden;
      box-shadow: 0 0 18px color-mix(in srgb, var(--rarity-line), transparent 62%);
    }
    .draft-icon img, .battle-avatar img { width: 100%; height: 100%; object-fit: cover; }
    .battle-board {
      align-items: start;
      gap: clamp(24px, 6vw, 84px);
    }
    .battle-side {
      background: rgba(255,255,255,.045);
    }
    .battle-side.turn {
      border-color: rgba(0, 229, 255, .52);
      background:
        radial-gradient(circle at 50% 0%, rgba(0, 229, 255, .13), transparent 36%),
        rgba(255,255,255,.055);
      box-shadow: 0 0 28px rgba(0, 229, 255, .12), inset 0 1px 0 rgba(255,255,255,.08);
    }
    .battle-team {
      grid-template-columns: 1fr;
      align-items: start;
      gap: 12px;
    }
    .battle-side.player .battle-team { justify-items: start; }
    .battle-side.foe .battle-team { justify-items: end; }
    .battle-card {
      --rarity-base: rgba(139, 92, 246, .12);
      --rarity-line: rgba(184, 132, 255, .34);
      --rarity-ink: #f4edff;
      position: relative;
      display: grid;
      grid-template-columns: 62px minmax(0, 1fr);
      gap: 10px;
      align-items: center;
      border: 1px solid color-mix(in srgb, var(--rarity-line), #ffffff 8%);
      border-radius: 8px;
      padding: 10px;
      background:
        radial-gradient(circle at 14% 16%, color-mix(in srgb, var(--rarity-line), transparent 58%), transparent 40%),
        linear-gradient(145deg, rgba(255,255,255,.075), rgba(255,255,255,.025) 56%, var(--rarity-base));
      opacity: .74;
      transition: transform .2s ease, opacity .2s ease, box-shadow .2s ease;
      color: var(--rarity-ink);
      box-shadow: 0 0 18px color-mix(in srgb, var(--rarity-line), transparent 78%);
      width: min(360px, 100%);
      min-height: 86px;
    }
    .battle-card.active {
      opacity: 1;
      z-index: 2;
      border-color: color-mix(in srgb, var(--rarity-line), #ffffff 18%);
      box-shadow: 0 20px 42px rgba(0,0,0,.22), 0 0 30px color-mix(in srgb, var(--rarity-line), transparent 58%);
      width: min(430px, 108%);
    }
    .battle-side.player .battle-card.active {
      transform: translateX(34px) scale(1.045);
    }
    .battle-side.foe .battle-card.active {
      transform: translateX(-34px) scale(1.045);
    }
    .battle-card.dead {
      opacity: .72;
      filter: grayscale(.35);
      border-color: rgba(255, 78, 122, .72);
      background:
        linear-gradient(145deg, rgba(255, 78, 122, .22), rgba(62, 10, 28, .72)),
        radial-gradient(circle at 14% 16%, rgba(255,78,122,.22), transparent 42%);
      box-shadow: 0 0 24px rgba(255,78,122,.18);
    }
    .battle-card.damaged {
      animation: hitPulse .62s ease;
    }
    .battle-avatar { width: 62px; height: 62px; }
    .hpbar { background: rgba(255,255,255,.12); }
    .hpfill {
      background: linear-gradient(90deg, #2ef2a6, #00e5ff);
      box-shadow: 0 0 12px rgba(46,242,166,.34);
      transition: width .55s cubic-bezier(.2,.8,.2,1), background .25s ease;
    }
    .hpfill.low { background: linear-gradient(90deg, #ff4e7a, #ffb86b); }
    .status-row {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
      margin-top: 6px;
    }
    .status-badge {
      border: 1px solid rgba(184,132,255,.24);
      border-radius: 999px;
      padding: 2px 6px;
      font-size: 10px;
      line-height: 1.2;
      color: #eadcff;
      background: rgba(255,255,255,.07);
      text-transform: uppercase;
      max-width: 112px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .status-badge.bad { border-color: rgba(255,78,122,.36); background: rgba(255,78,122,.14); color: #ffd1dc; }
    .status-badge.good { border-color: rgba(46,242,166,.32); background: rgba(46,242,166,.12); color: #a8ffd8; }
    .status-badge.control { border-color: rgba(0,229,255,.32); background: rgba(0,229,255,.12); color: #c9fbff; }
    .result-scene {
      min-height: 330px;
      display: grid;
      place-items: center;
      text-align: center;
    }
    .result-title {
      font-size: clamp(34px, 6vw, 82px);
      line-height: 1;
      font-weight: 950;
      text-transform: uppercase;
      text-shadow: 0 0 32px rgba(168,85,247,.56);
    }
    .result-title.victory { color: #a8ffd8; text-shadow: 0 0 34px rgba(46,242,166,.48); }
    .result-title.defeat { color: #ffd1dc; text-shadow: 0 0 34px rgba(255,78,122,.48); }
    .collapsible {
      border: 1px solid rgba(184,132,255,.22);
      border-radius: 8px;
      background: rgba(255,255,255,.04);
      padding: 0;
    }
    .collapsible > summary {
      cursor: pointer;
      list-style: none;
      padding: 12px 14px;
      font-weight: 800;
      color: #eadcff;
    }
    .collapsible > summary::-webkit-details-marker { display: none; }
    .collapsible-body { padding: 0 14px 14px; }
    @keyframes searchRing {
      from { transform: rotate(0deg) scale(.94); opacity: .9; }
      to { transform: rotate(360deg) scale(1.05); opacity: .42; }
    }
    @keyframes searchPulse {
      0%, 100% { transform: scale(1); }
      50% { transform: scale(1.035); }
    }
    @keyframes hitPulse {
      0% { transform: translateX(0) scale(1); box-shadow: 0 0 0 rgba(255,78,122,0); }
      22% { transform: translateX(-4px) scale(1.035); box-shadow: 0 0 38px rgba(255,78,122,.42); }
      46% { transform: translateX(4px) scale(1.02); }
      100% { transform: translateX(0) scale(1); }
    }
    @keyframes splusGlow {
      0%, 100% { box-shadow: 0 0 0 1px rgba(46,242,166,.18), 0 0 20px rgba(46,242,166,.16); }
      50% { box-shadow: 0 0 0 1px rgba(46,242,166,.34), 0 0 36px rgba(46,242,166,.34); }
    }
    @media (prefers-reduced-motion: reduce) {
      .search-core, .search-core::before, .search-core::after, .battle-card.damaged, .miscrit-card.splus-card { animation: none; }
      .battle-card, .draft-card { transition: none; }
    }
    .output-toggle {
      position: fixed;
      right: 18px;
      bottom: 18px;
      z-index: 10;
      box-shadow: 0 8px 28px rgba(20, 35, 45, 0.24);
    }
    .output-drawer {
      position: fixed;
      left: 246px;
      right: 0;
      bottom: 0;
      max-height: 45vh;
      background: #141d22;
      color: #d7f2df;
      border-top: 1px solid #2e414a;
      transform: translateY(calc(100% - 0px));
      transition: transform .18s ease;
      z-index: 9;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
    }
    .output-drawer.open { transform: translateY(0); }
    .output-head { display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: #1d2a31; }
    pre { margin: 0; padding: 14px; overflow: auto; white-space: pre-wrap; word-break: break-word; font-family: Consolas, monospace; }
    @media (max-width: 900px) {
      .app { grid-template-columns: 1fr; }
      aside { position: static; height: auto; }
      nav { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .cards, .grid-2, .grid-3, .parents, .battle-board, .battle-team, .draft-grid, .draft-card-grid, .stage-track, .account-card-grid, .gauge-grid, .plan-block { grid-template-columns: 1fr; }
      .output-drawer { left: 0; }
    }
    @media (max-width: 640px) {
      .auth-shell { padding: 14px; }
      .auth-card { padding: 18px; }
      .app { min-width: 0; }
      aside { padding: 14px; }
      .brand { margin-bottom: 14px; }
      nav { grid-template-columns: 1fr 1fr; gap: 6px; }
      nav button { min-height: 36px; padding: 8px 9px; font-size: 13px; }
      main { padding: 14px; gap: 14px; }
      .topbar { align-items: stretch; flex-direction: column; }
      .topbar h2 { font-size: 21px; }
      .topbar .row, .toolbar, .row { align-items: stretch; }
      .topbar select, .topbar button, .toolbar .field, .toolbar button, .row button, .field input, .field select { width: 100%; }
      .cards { gap: 10px; }
      .metric { min-height: 78px; padding: 12px; }
      .metric strong { font-size: 24px; }
      section.panel { padding: 12px; }
      .panel h3 { font-size: 15px; }
      .logs { max-height: 420px; padding-right: 0; }
      .miscrit-card {
        grid-template-columns: 76px minmax(0, 1fr) 34px;
        min-height: 150px;
        padding: 10px;
        gap: 8px 10px;
      }
      .miscrit-avatar { width: 76px; font-size: 22px; }
      .rating-mark { min-width: 24px; height: 24px; font-size: 12px; }
      .miscrit-name { font-size: 15px; }
      .miscrit-mid { font-size: 20px; padding-bottom: 5px; }
      .miscrit-subline { font-size: 11px; }
      .element-orb { width: 32px; height: 32px; font-size: 10px; }
      .miscrit-stat-grid {
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 6px;
      }
      .miscrit-stat {
        min-height: 30px;
        font-size: 11px;
        padding: 3px;
      }
      .player-card { grid-template-columns: 54px minmax(0, 1fr); padding: 12px; }
      .player-avatar { width: 54px; height: 54px; font-size: 22px; }
      .profile-card { min-height: 126px; }
      .gauge-arc { width: min(190px, 100%); }
      .arena-header { flex-direction: column; align-items: stretch; }
      .arena-scene { min-height: 300px; padding: 12px; }
      .search-core { width: 132px; height: 132px; }
      .battle-card {
        grid-template-columns: 52px minmax(0, 1fr);
        transform: none;
      }
      .battle-card.active { transform: translateY(-4px) scale(1.015); }
      .battle-avatar { width: 52px; height: 52px; }
      .draft-card-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .draft-card { min-height: 106px; padding: 9px; }
      .ai-bar, .ai-weight-row { grid-template-columns: 1fr; }
      .output-toggle { right: 12px; bottom: 12px; }
    }
    @media (max-width: 420px) {
      nav { grid-template-columns: 1fr; }
      .draft-card-grid { grid-template-columns: 1fr; }
      .miscrit-card {
        grid-template-columns: 68px minmax(0, 1fr);
      }
      .element-orb {
        position: absolute;
        right: 10px;
        top: 10px;
      }
      .miscrit-main { padding-right: 36px; }
    }
  </style>
</head>
<body>
  <div id="bootFallback" class="boot-fallback">
    <div class="boot-fallback-card">
      <strong>CLI Miscrits загружается...</strong>
      <div class="muted">Если это окно не исчезает на телефоне, браузер не запустил скрипт панели. Попробуй Chrome или Firefox и обнови страницу.</div>
    </div>
  </div>
  <div id="authShell" class="auth-shell">
    <section class="auth-card">
      <div class="brand" style="margin-bottom:0">
        <div class="brand-mark">M</div>
        <div>
          <h1>CLI Miscrits</h1>
          <p id="authStatus">Проверяю сохранённую сессию...</p>
        </div>
      </div>
      <div class="field"><label>Логин</label><input id="login" autocomplete="username"></div>
      <div class="field"><label>Пароль</label><input id="password" type="password" autocomplete="current-password"></div>
      <label class="chip" style="width:max-content"><input id="rememberLogin" type="checkbox" checked> Запомнить на этом ПК</label>
      <div class="auth-actions">
        <button onclick="login()">Войти</button>
        <button class="secondary" onclick="loginSaved()">Сохранённый вход</button>
      </div>
      <div class="auth-error" id="authError"></div>
    </section>
  </div>

  <div id="appShell" class="app hidden">
    <aside>
      <div class="brand">
        <div class="brand-mark">M</div>
        <h1>CLI Miscrits</h1>
        <span class="status-pill" id="status">...</span>
      </div>
      <nav>
        <button class="active" data-view="dashboard" onclick="switchView('dashboard')">Дашборд</button>
        <button data-view="breeding" onclick="switchView('breeding')">Скрещивание</button>
        <button data-view="arena" onclick="switchView('arena')">Арена</button>
        <button data-view="plan" onclick="switchView('plan')">План</button>
        <button data-view="ai" onclick="switchView('ai')">AI</button>
        <button data-view="battlelogs" onclick="switchView('battlelogs')">Логи боёв</button>
        <button data-view="logs" onclick="switchView('logs')">События</button>
        <button data-view="actions" onclick="switchView('actions')">Действия</button>
        <button data-view="cache" onclick="switchView('cache')">Справочники</button>
        <button data-view="tools" onclick="switchView('tools')">RPC-инструменты</button>
      </nav>
    </aside>
    <main>
      <div class="topbar">
        <h2 id="viewTitle">Дашборд</h2>
        <div class="row">
          <select id="accountSelect" onchange="selectAccount(this.value)"><option value="">Нет аккаунта Miscrits</option></select>
          <button class="ghost" onclick="loadPlayer()">Обновить</button>
          <button class="ghost" onclick="loadBreedOptions()">Обновить скрещивание</button>
        </div>
      </div>

      <div class="cards">
        <div class="metric resource-gold"><span>Золото</span><strong id="goldMetric">-</strong></div>
        <div class="metric resource-gems"><span>Гемы</span><strong id="gemsMetric">-</strong></div>
        <div class="metric resource-platinum"><span>Платина</span><strong id="platMetric">-</strong></div>
        <div class="metric"><span>Пул скрещивания</span><strong id="poolMetric">-</strong></div>
      </div>

      <div id="dashboard" class="view active">
        <div class="dashboard-stack">
          <section class="panel">
            <h3>Винрейт арен</h3>
            <div id="arenaGaugeGrid" class="gauge-grid">
              <div class="gauge-card muted">Данные о боях ещё не загружены</div>
            </div>
          </section>
          <section class="panel">
            <h3>Аккаунты Miscrits</h3>
            <div id="accountSummary" class="chips"><span class="chip">Аккаунт не выбран</span></div>
            <div id="accountCards" class="account-card-grid" style="margin-top:12px"></div>
            <div class="toolbar" style="margin-top:12px">
              <div class="field"><label>Название</label><input id="accountLabel" placeholder="Основной"></div>
              <div class="field"><label>Логин</label><input id="accountLogin" autocomplete="username"></div>
              <div class="field"><label>Пароль</label><input id="accountPassword" type="password" autocomplete="current-password"></div>
              <button onclick="saveMiscritsAccount()">Сохранить аккаунт</button>
              <button class="ghost" onclick="removeMiscritsAccount()">Удалить выбранный</button>
            </div>
          </section>
          <section class="panel">
            <h3>Игрок</h3>
            <div id="playerSummary" class="muted">Данные ещё не загружены</div>
          </section>
          <section class="panel">
            <h3>Сохранённые данные</h3>
            <div class="row" style="margin-bottom:10px"><button class="ghost" onclick="loadSavedData()">Обновить сохранённое</button></div>
            <div id="savedData" class="muted">Сохранённые данные ещё не загружены</div>
          </section>
        </div>
      </div>

      <div id="breeding" class="view">
        <section class="panel">
          <h3>Настройки скрещивания</h3>
          <div class="grid-3">
            <div class="field"><label>Целевой мискрит</label><select id="targetMid"><option value="">Любой</option></select></div>
            <div class="field"><label>Стихия</label><select id="targetElement"><option value="">Любая</option></select></div>
            <div class="field"><label>Минимально возможный ранг</label><select id="minMaxSum">
              <option value="15">A или лучше</option>
              <option value="16">A+ или лучше</option>
              <option value="17">S или лучше</option>
              <option value="18">Только S+</option>
            </select></div>
            <div class="field"><label>Разрешено родителей S+</label><input id="allowSplus" type="number" min="0" max="3" value="0"></div>
            <div class="field"><label>Максимум автоскрещиваний</label><input id="maxBreeds" type="number" min="1" value="10"></div>
            <div class="field"><label>Задержка</label><input id="breedDelay" type="number" min="0" step="0.1" value="0.4"></div>
          </div>
          <div class="row" style="margin-top:12px">
            <button onclick="breedPlan()">Построить план</button>
            <button class="secondary" onclick="breedOnce(true)">Проверить без запуска</button>
            <button class="warn" onclick="breedOnce(false)">Скрестить один раз</button>
            <button class="secondary" onclick="autoBreed(true)">Проверить авто</button>
            <button class="warn" onclick="autoBreed(false)">Запустить авто</button>
          </div>
        </section>

        <div class="grid-2">
          <section class="panel">
            <h3>Текущий план</h3>
            <div id="planRoot" class="plan-card muted">План ещё не загружен</div>
          </section>
          <section class="panel">
            <h3>Доступные цели</h3>
            <div id="targetList" class="logs muted">Цели ещё не загружены</div>
          </section>
        </div>

        <section class="panel">
          <h3>Логи скрещивания</h3>
          <div class="row" style="margin-bottom:10px"><button class="ghost" onclick="loadBreedLogs()">Обновить логи</button></div>
          <div id="breedLogs" class="logs muted">Логи ещё не загружены</div>
        </section>
      </div>

      <div id="actions" class="view">
        <section class="panel">
          <h3>Быстрые действия</h3>
          <div class="row">
            <button onclick="action('heal')">Вылечить команду</button>
            <button onclick="action('wish_sk')">Wish SK</button>
            <button onclick="action('wish_vi')">Wish VI</button>
            <button onclick="action('wish_xmas')">Wish Xmas</button>
          </div>
        </section>
      </div>

      <div id="arena" class="view">
        <section class="panel">
          <h3>Автобой на арене</h3>
          <div class="grid-3">
            <div class="field"><label>Режим</label><select id="arenaMode">
              <option value="battle">Обычная арена</option>
              <option value="random">Случайная арена</option>
              <option value="platinum">Платиновая арена</option>
              <option value="daily">Ежедневная арена</option>
            </select></div>
            <div class="field"><label>Таймаут, секунд</label><input id="arenaTimeout" type="number" min="30" value="300"></div>
            <div class="field"><label>Максимум ходов</label><input id="arenaMaxTurns" type="number" min="1" value="150"></div>
            <div class="field"><label>Автоподготовка</label><select id="arenaPrepare">
              <option value="1">Перейти, собрать команду, вылечить</option>
              <option value="0">Пропустить подготовку</option>
            </select></div>
            <div class="field"><label>Количество боёв</label><input id="arenaRepeatCount" type="number" min="0" value="1"></div>
            <div class="field"><label>Пауза между боями</label><input id="arenaRepeatDelay" type="number" min="0" step="0.5" value="3"></div>
            <div class="field"><label>При ошибке</label><select id="arenaStopOnError">
              <option value="0">Продолжать цикл</option>
              <option value="1">Остановить цикл</option>
            </select></div>
          </div>
          <div class="row" style="margin-top:12px">
            <button onclick="arenaStatus()">Проверить условия</button>
            <button class="ghost" onclick="socketDoctor()">Проверить сокет</button>
            <button class="secondary" onclick="arenaRun(true)">Проверить без запуска</button>
            <button class="warn" onclick="arenaRun(false)">Искать бой</button>
            <button class="ghost" onclick="arenaStop()">Остановить после текущего боя</button>
          </div>
        </section>
        <section class="panel">
          <h3>Состояние арены</h3>
          <div id="arenaState" class="logs muted">Состояние арены ещё не загружено</div>
        </section>
        <section class="panel">
          <details class="collapsible">
            <summary>Обучение AI</summary>
            <div class="collapsible-body">
              <div class="row" style="margin-bottom:10px"><button class="ghost" onclick="loadBattleLearning()">Обновить обучение</button></div>
              <div id="battleLearning" class="logs muted">Данные обучения ещё не загружены</div>
            </div>
          </details>
        </section>
      </div>

      <div id="plan" class="view">
        <section class="panel">
          <h3>Ежедневный план аккаунта</h3>
          <div class="grid-3">
            <label class="chip"><input id="planEnabled" type="checkbox" checked> Включён</label>
            <div class="field"><label>Пауза проверки, секунд</label><input id="planTickSeconds" type="number" min="5" value="20"></div>
            <div class="field"><label>Пауза между боями</label><input id="planBattleDelay" type="number" min="0" step="0.5" value="3"></div>
            <label class="chip"><input id="planFallbackRandom" type="checkbox" checked> Случайная арена, если лечение недоступно</label>
          </div>
          <div class="plan-builder-toolbar">
            <div class="field">
              <label>Добавить блок</label>
              <select id="planBlockType">
                <option value="wish_sk">Wish SK</option>
                <option value="wish_vi">Wish VI</option>
                <option value="daily_arena">Ежедневная арена</option>
                <option value="battle_arena">Обычная арена</option>
                <option value="platinum_arena">Платиновая арена</option>
                <option value="random_arena">Случайная арена</option>
              </select>
            </div>
            <button onclick="addPlanBlock()">Добавить блок</button>
          </div>
          <div id="planBlocks" class="plan-blocks"></div>
          <div class="row" style="margin-top:12px">
            <button onclick="savePlan()">Сохранить план</button>
            <button class="warn" onclick="startPlan()">Запустить цикл плана</button>
            <button class="secondary" onclick="runPlanOnce()">Выполнить один шаг</button>
            <button class="ghost" onclick="stopPlan()">Остановить план</button>
            <button class="ghost" onclick="loadPlan()">Обновить план</button>
          </div>
        </section>
        <section class="panel">
          <h3>Состояние плана</h3>
          <div id="planState" class="logs muted">План ещё не загружен</div>
        </section>
      </div>

      <div id="logs" class="view">
        <section class="panel">
          <h3>Логи событий</h3>
          <div class="grid-3">
            <div class="field"><label>Аккаунт</label><select id="logAccountScope">
              <option value="">Все аккаунты</option>
              <option value="selected">Выбранный аккаунт</option>
            </select></div>
            <div class="field"><label>Уровень</label><select id="logLevel">
              <option value="">Любой</option>
              <option value="info">Инфо</option>
              <option value="warning">Предупреждение</option>
              <option value="error">Ошибка</option>
            </select></div>
            <div class="field"><label>Категория</label><select id="logCategory">
              <option value="">Любая</option>
              <option value="arena">Арена</option>
              <option value="plan">План</option>
              <option value="account">Аккаунт</option>
              <option value="system">Система</option>
            </select></div>
            <div class="field"><label>Инициатор</label><select id="logInitiator">
              <option value="">Любой</option>
              <option value="cli">CLI</option>
              <option value="user">Пользователь</option>
            </select></div>
            <div class="field"><label>С</label><input id="logSince" type="datetime-local"></div>
            <div class="field"><label>По</label><input id="logUntil" type="datetime-local"></div>
            <div class="field"><label>Лимит</label><input id="logLimit" type="number" min="1" max="1000" value="250"></div>
            <div class="field"><label>Событие</label><input id="logEvent" placeholder="loop_complete"></div>
            <div class="field"><label>Причина</label><input id="logReason" placeholder="socket, validation, error"></div>
            <div class="field"><label>Поиск по тексту</label><input id="logText" placeholder="аккаунт, задача, payload"></div>
          </div>
          <div class="row" style="margin-top:12px">
            <button onclick="loadEventLogs()">Обновить логи</button>
            <button class="ghost" onclick="fillLogLastHours(2)">Последние 2 часа</button>
            <button class="ghost" onclick="fillLogLastHours(24)">Последние 24 часа</button>
            <button class="ghost" onclick="clearLogFilters()">Сбросить фильтры</button>
          </div>
        </section>
        <section class="panel">
          <div class="log-head">
            <h3 style="margin:0">События</h3>
            <div id="logStatus" class="chips"><span class="chip">не загружено</span></div>
          </div>
          <div id="systemLogs" class="logs muted" style="margin-top:12px">Логи ещё не загружены</div>
        </section>
      </div>

      <div id="ai" class="view">
        <section class="panel">
          <div class="log-head">
            <h3 style="margin:0">Панель AI</h3>
            <button class="ghost" onclick="loadAiDashboard()">Обновить AI</button>
          </div>
          <div id="aiDashboard" class="ai-dashboard muted" style="margin-top:12px">Данные AI ещё не загружены</div>
        </section>
      </div>

      <div id="battlelogs" class="view">
        <section class="panel">
          <h3>Логи боёв</h3>
          <div class="grid-3">
            <div class="field"><label>Режим</label><select id="battleLogMode">
              <option value="">Любой</option>
              <option value="battle">Обычная</option>
              <option value="daily">Ежедневная</option>
              <option value="platinum">Платиновая</option>
              <option value="random">Случайная</option>
            </select></div>
            <div class="field"><label>Исход</label><select id="battleLogOutcome">
              <option value="">Любой</option>
              <option value="victory">Победа</option>
              <option value="defeat">Поражение</option>
              <option value="unknown">Неизвестно</option>
            </select></div>
            <div class="field"><label>Лимит</label><input id="battleLogLimit" type="number" min="1" max="1000" value="100"></div>
            <div class="field"><label>Поиск</label><input id="battleLogText" placeholder="id боя, режим, результат"></div>
          </div>
          <div class="row" style="margin-top:12px">
            <button onclick="loadBattleLogs()">Обновить логи боёв</button>
            <button class="ghost" onclick="clearBattleLogFilters()">Сбросить фильтры</button>
          </div>
        </section>
        <div class="grid-2">
          <section class="panel">
            <h3>Сохранённые бои</h3>
            <div id="battleLogList" class="logs muted">Логи боёв ещё не загружены</div>
          </section>
          <section class="panel">
            <h3>Детали боя</h3>
            <div id="battleLogDetail" class="logs muted">Выбери бой</div>
          </section>
        </div>
      </div>

      <div id="cache" class="view">
        <section class="panel">
          <h3>Справочные данные</h3>
          <div class="row">
            <button onclick="syncCache(false, false)">Синхронизировать нужное</button>
            <button class="secondary" onclick="syncCache(true, false)">Принудительная синхронизация</button>
            <button class="secondary" onclick="syncCache(false, true)">Синхронизировать весь JSON</button>
            <button class="secondary" onclick="syncAvatars(false)">Синхронизировать аватары аккаунта</button>
            <button class="ghost" onclick="syncAvatars(true)">Синхронизировать все аватары</button>
            <button class="ghost" onclick="loadCacheState()">Обновить состояние</button>
          </div>
        </section>
        <section class="panel">
          <h3>Локальный кэш</h3>
          <div id="cacheList" class="logs muted">Состояние кэша ещё не загружено</div>
        </section>
      </div>

      <div id="tools" class="view">
        <section class="panel">
          <h3>Сырой RPC</h3>
          <div class="toolbar">
            <div class="field"><label>Метод</label><input id="rpcMethod" value="get_player"></div>
            <button class="warn" onclick="rpc()">Выполнить</button>
          </div>
          <div class="field" style="margin-top:10px"><label>Тело запроса</label><textarea id="rpcPayload">{}</textarea></div>
        </section>
      </div>
    </main>
  </div>

  <button id="outputToggle" class="output-toggle hidden" onclick="toggleOutput()">Вывод</button>
  <div id="outputDrawer" class="output-drawer hidden">
    <div class="output-head"><strong>Вывод</strong><button class="ghost" onclick="toggleOutput()">Скрыть</button></div>
    <pre id="out">{}</pre>
  </div>

  <script>
    document.documentElement.className += ' boot-ok';
    const out = document.getElementById('out');
    const statusEl = document.getElementById('status');
    const authShell = document.getElementById('authShell');
    const appShell = document.getElementById('appShell');
    const authStatus = document.getElementById('authStatus');
    const authError = document.getElementById('authError');
    const outputToggle = document.getElementById('outputToggle');
    const outputDrawer = document.getElementById('outputDrawer');
    let currentPlayer = null;
    let arenaJobId = null;
    let arenaPollTimer = null;
    let logPollTimer = null;
    let planPollTimer = null;
    let planBlocks = [];
    let statusPollTimer = null;
    let lastPlanBattleRefreshKey = '';
    let battleHpMemory = new Map();
    let battleSlotMemory = new Map();
    let authenticated = false;
    let accounts = [];
    let selectedAccountId = localStorage.getItem('miscritsAccountId') || '';

    function pick(value, fallback) {
      return value === undefined || value === null ? fallback : value;
    }

    function modeLabel(value) {
      return ({
        default: 'Обычная арена',
        battle: 'Обычная арена',
        daily: 'Ежедневная арена',
        platinum: 'Платиновая арена',
        random: 'Случайная арена'
      })[String(value || '').toLowerCase()] || String(value || '');
    }

    function outcomeLabel(value) {
      return ({
        victory: 'Победа',
        defeat: 'Поражение',
        unknown: 'Неизвестно'
      })[String(value || '').toLowerCase()] || String(value || '');
    }

    function phaseLabel(value) {
      return ({
        starting: 'Запуск',
        preparing: 'Подготовка',
        connecting: 'Подключение',
        ready: 'Готово',
        searching: 'Поиск',
        draft: 'Драфт',
        battle: 'Бой',
        cooldown: 'Пауза',
        finished: 'Завершено',
        stopped: 'Остановлено',
        error: 'Ошибка'
      })[String(value || '').toLowerCase()] || String(value || '');
    }

    const AI_REASON_LABELS = {
      highest_expected_damage_lookahead_safe: 'Максимальный ожидаемый урон, безопасный прогноз',
      highest_utility_lookahead_safe: 'Максимальная полезность, безопасный прогноз',
      highest_expected_damage: 'Максимальный ожидаемый урон',
      highest_utility: 'Максимальная полезность',
      recovery_value_lookahead_safe: 'Восстановление с безопасным прогнозом',
      lethal_finish: 'Добивание',
      incoming_lethal_lookahead_enemy_kills: 'Защита от входящего добивания',
      damage_pressure_lookahead_safe_lookahead_next_finish: 'Давление уроном с шансом добить следующим ходом'
    };

    function aiReasonLabel(value) {
      const key = String(value || '');
      if (AI_REASON_LABELS[key]) return AI_REASON_LABELS[key];
      return key
        .replaceAll('highest_expected_damage', 'максимальный ожидаемый урон')
        .replaceAll('highest_utility', 'максимальная полезность')
        .replaceAll('recovery_value', 'восстановление')
        .replaceAll('lookahead_next_finish', 'добивание следующим ходом')
        .replaceAll('lookahead_last_ally_risk', 'риск для последнего союзника')
        .replaceAll('lookahead_last_ally_damage_now', 'урон сейчас последним союзником')
        .replaceAll('lookahead_enemy_kills', 'враг добивает')
        .replaceAll('lookahead_safe', 'безопасный прогноз')
        .replaceAll('damage_pressure', 'давление уроном')
        .replaceAll('lethal_finish', 'добивание')
        .replaceAll('_', ' ');
    }

    function aiActionLabel(value) {
      const raw = String(value || '');
      if (raw.startsWith('ability:')) return 'Способность ' + raw.slice('ability:'.length);
      if (raw.startsWith('switch:')) return 'Смена ' + raw.slice('switch:'.length);
      return raw;
    }

    function aiWeightKeyLabel(category, value) {
      if (category === 'reasons') return aiReasonLabel(value);
      if (category === 'actions' || category === 'opponent_actions') return aiActionLabel(value);
      return value;
    }

    function show(data, open = false) {
      const drawer = document.getElementById('outputDrawer');
      if (drawer.classList.contains('open')) {
        const text = JSON.stringify(data, null, 2);
        out.textContent = text.length > 24000 ? text.slice(0, 24000) + '\\n... truncated ...' : text;
      }
      if (authenticated) scheduleStatusCheck();
    }

    async function api(path, options = {}, open = false) {
      const res = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...options });
      const data = await res.json();
      show(data, open);
      return data;
    }

    function scheduleStatusCheck() {
      if (statusPollTimer) clearTimeout(statusPollTimer);
      statusPollTimer = setTimeout(checkStatus, 1200);
    }

    function switchView(name) {
      document.querySelectorAll('.view').forEach(el => el.classList.toggle('active', el.id === name));
      document.querySelectorAll('nav button').forEach(el => el.classList.toggle('active', el.dataset.view === name));
      document.getElementById('viewTitle').textContent = ({dashboard:'Дашборд', breeding:'Скрещивание', arena:'Арена', plan:'План', ai:'AI', battlelogs:'Логи боёв', logs:'События', actions:'Действия', cache:'Справочники', tools:'RPC-инструменты'})[name];
      if (logPollTimer) clearTimeout(logPollTimer);
      if (name === 'plan') loadPlan();
      if (name === 'ai') loadAiDashboard();
      if (name === 'battlelogs') loadBattleLogs();
      if (name === 'logs') loadEventLogs();
    }

    function toggleOutput() { document.getElementById('outputDrawer').classList.toggle('open'); }

    async function initApp() {
      authStatus.textContent = 'Открываю дашборд...';
      await loadAccounts();
      openApp({ method: 'web_dashboard' });
    }

    async function fetchJson(path, options = {}) {
      const res = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...options });
      return await res.json();
    }

    function applyCredentialHints(data) {
      const savedLogin = data && data.credentials && data.credentials.login;
      if (savedLogin && !document.getElementById('login').value) document.getElementById('login').value = savedLogin;
      document.getElementById('rememberLogin').checked = Boolean(data && data.credentials && data.credentials.saved);
    }

    function openAuth(error = '') {
      authenticated = false;
      authShell.classList.remove('hidden');
      appShell.classList.add('hidden');
      outputToggle.classList.add('hidden');
      outputDrawer.classList.add('hidden');
      if (logPollTimer) clearTimeout(logPollTimer);
      if (planPollTimer) clearTimeout(planPollTimer);
      authStatus.textContent = 'Войди, чтобы открыть дашборд.';
      authError.textContent = error && !error.includes('No saved credentials') ? error : '';
      statusEl.textContent = 'не в сети';
    }

    function openApp(data = {}) {
      authenticated = true;
      authShell.classList.add('hidden');
      appShell.classList.remove('hidden');
      outputToggle.classList.remove('hidden');
      outputDrawer.classList.remove('hidden');
      statusEl.textContent = 'дашборд';
      renderAccount(data);
      loadInitialData();
    }

    async function loadInitialData() {
      if (selectedAccountId) {
        loadPlayer();
        loadBreedOptions();
      }
      loadBreedLogs();
      loadCacheState();
      loadSavedData();
      loadBattleLogs();
      const attachedArena = await attachArenaJob();
      if (!attachedArena) arenaStatus();
      loadBattleLearning();
      loadAiDashboard();
      loadPlan();
      loadEventLogs();
    }

    async function attachArenaJob() {
      const data = await fetchJson('/api/arena-current?account_id=' + encodeURIComponent(selectedAccountId || ''));
      if (!data.ok || !data.job) {
        arenaJobId = null;
        return false;
      }
      arenaJobId = data.job.id;
      syncArenaControls(data.job);
      renderArenaLive(data.job);
      if (data.job.status === 'running') pollArenaJob();
      return true;
    }

    function syncArenaControls(job) {
      const mode = modeSelectValue(job.mode || '');
      if (mode) document.getElementById('arenaMode').value = mode;
      if (job.repeat) {
        document.getElementById('arenaRepeatCount').value = pick(job.repeat.count, 1);
        document.getElementById('arenaRepeatDelay').value = pick(job.repeat.delay_seconds, 0);
        document.getElementById('arenaStopOnError').value = job.repeat.stop_on_error ? '1' : '0';
      }
    }

    function modeSelectValue(mode) {
      const key = String(mode || '').toLowerCase();
      return ({default:'battle', battle:'battle', random:'random', platinum:'platinum', daily:'daily'})[key] || '';
    }

    async function loadAccounts() {
      const data = await fetchJson('/api/accounts');
      accounts = data.accounts || [];
      if (selectedAccountId && !accounts.some(account => account.id === selectedAccountId)) selectedAccountId = '';
      if (!selectedAccountId && accounts.length) selectedAccountId = accounts[0].id;
      localStorage.setItem('miscritsAccountId', selectedAccountId || '');
      renderAccountSelect();
      renderAccount();
      return accounts;
    }

    function renderAccountSelect() {
      const select = document.getElementById('accountSelect');
      select.innerHTML = '<option value="">Нет аккаунта Miscrits</option>' + accounts.map(account =>
        `<option value="${escapeHtml(account.id)}">${escapeHtml(account.label || account.login || account.id)}</option>`
      ).join('');
      select.value = selectedAccountId || '';
    }

    function selectedAccount() {
      return accounts.find(account => account.id === selectedAccountId) || null;
    }

    async function selectAccount(accountId) {
      selectedAccountId = accountId || '';
      localStorage.setItem('miscritsAccountId', selectedAccountId);
      renderAccount();
      await loadInitialData();
    }

    async function saveMiscritsAccount() {
      const data = await api('/api/account-save', { method: 'POST', body: JSON.stringify({
        id: selectedAccountId,
        label: document.getElementById('accountLabel').value,
        login: document.getElementById('accountLogin').value,
        password: document.getElementById('accountPassword').value
      })}, true);
      if (data.ok) {
        document.getElementById('accountPassword').value = '';
        selectedAccountId = data.account.id;
        accounts = data.accounts || [];
        localStorage.setItem('miscritsAccountId', selectedAccountId);
        renderAccountSelect();
        renderAccount();
        loadInitialData();
      }
    }

    async function removeMiscritsAccount() {
      if (!selectedAccountId) return;
      const data = await api('/api/account-remove', { method: 'POST', body: JSON.stringify({ id: selectedAccountId }) }, true);
      accounts = data.accounts || [];
      selectedAccountId = accounts.length ? accounts[0].id : '';
      localStorage.setItem('miscritsAccountId', selectedAccountId);
      renderAccountSelect();
      renderAccount();
      loadInitialData();
    }

    async function removeMiscritsAccountById(accountId) {
      if (!accountId) return;
      const data = await api('/api/account-remove', { method: 'POST', body: JSON.stringify({ id: accountId }) }, true);
      accounts = data.accounts || [];
      if (selectedAccountId === accountId) selectedAccountId = accounts.length ? accounts[0].id : '';
      localStorage.setItem('miscritsAccountId', selectedAccountId);
      renderAccountSelect();
      renderAccount();
      loadInitialData();
    }

    function renderAccount(data = {}) {
      const root = document.getElementById('accountSummary');
      const cardRoot = document.getElementById('accountCards');
      const account = selectedAccount();
      const credentials = account && account.credentials ? account.credentials : {};
      document.getElementById('accountLabel').value = account ? (account.label || '') : '';
      document.getElementById('accountLogin').value = account ? (account.login || '') : '';
      root.innerHTML = `
        <span class="chip good">веб-панель</span>
        ${account ? `<span class="chip">${escapeHtml(account.label || account.login)}</span><span class="chip">${escapeHtml(account.login)}</span><span class="chip ${credentials.decryptable ? 'good' : 'warn'}">${credentials.decryptable ? 'учётные данные готовы' : 'нужен пароль'}</span>${credentials.error ? `<span class="chip warn">${escapeHtml(credentials.error)}</span>` : ''}` : '<span class="chip warn">добавь аккаунт Miscrits для запуска задач</span>'}`;
      if (cardRoot) renderAccountCards(cardRoot);
    }

    function renderAccountCards(root) {
      if (!accounts.length) {
        root.innerHTML = '<div class="profile-card muted">Профили ещё не сохранены</div>';
        return;
      }
      root.innerHTML = accounts.map(account => {
        const active = account.id === selectedAccountId;
        const credentials = account.credentials || {};
        const title = account.label || account.login || account.id;
        return `<div class="profile-card ${active ? 'active' : ''}">
          <div class="profile-top">
            <div class="row" style="gap:10px">
              <div class="profile-avatar">${escapeHtml(String(title || '?').slice(0, 2).toUpperCase())}</div>
              <div>
                <strong>${escapeHtml(title)}</strong>
                <div class="muted">${escapeHtml(account.login || '')}</div>
              </div>
            </div>
            <span class="chip ${credentials.decryptable ? 'good' : 'warn'}">${credentials.decryptable ? 'готов' : 'заблокирован'}</span>
          </div>
          <div class="row">
            <button class="ghost" onclick="selectAccount(${quoteJs(account.id)})">${active ? 'Выбран' : 'Выбрать'}</button>
            <button class="ghost danger" onclick="removeMiscritsAccountById(${quoteJs(account.id)})">Удалить</button>
          </div>
        </div>`;
      }).join('');
    }

    async function checkStatus() {
      const data = await fetchJson('/api/status');
      statusEl.textContent = 'дашборд';
      renderAccount();
      return data;
    }

    async function login() {
      authError.textContent = '';
      authStatus.textContent = 'Выполняю вход...';
      const data = await api('/api/login', { method: 'POST', body: JSON.stringify({
        login: document.getElementById('login').value,
        password: document.getElementById('password').value,
        remember: document.getElementById('rememberLogin').checked
      })}, true);
      if (data.ok) {
        document.getElementById('password').value = '';
        openApp(data);
      } else {
        openAuth(data.error || 'Не удалось войти');
      }
    }

    async function loginSaved() {
      authError.textContent = '';
      authStatus.textContent = 'Использую сохранённые данные...';
      const data = await api('/api/login-saved', { method: 'POST' }, true);
      if (data.ok) openApp(data);
      else openAuth(data.error || 'Не удалось войти по сохранённым данным');
    }

    async function logout() {
      await api('/api/logout', { method: 'POST' }, true);
      openAuth('');
    }
    async function clearCredentials() {
      const data = await api('/api/credentials-clear', { method: 'POST' }, true);
      applyCredentialHints(data);
      if (authenticated) checkStatus();
    }

    async function loadPlayer() {
      if (!selectedAccountId) return;
      const data = await api('/api/player?account_id=' + encodeURIComponent(selectedAccountId));
      if (!data.ok || !data.player) return;
      currentPlayer = data.player;
      document.getElementById('goldMetric').textContent = pick(data.player.gold, '-');
      document.getElementById('gemsMetric').textContent = pick(data.player.gems, '-');
      document.getElementById('platMetric').textContent = pick(data.player.platinum, '-');
      document.getElementById('playerSummary').innerHTML = `
        <div class="player-card">
          <div class="player-avatar">${escapeHtml(String(data.player.username || '?').slice(0, 2).toUpperCase())}</div>
          <div>
            <strong style="font-size:22px">${escapeHtml(data.player.username || 'неизвестно')}</strong>
            <div class="chips" style="margin-top:10px">
              <span class="chip">Уровень ${pick(data.player.level, '-')}</span>
              <span class="chip">Добродетель ${pick(data.player.virtue, '-')}</span>
              <span class="chip">Золото ${pick(data.player.gold, '-')}</span>
              <span class="chip">Гемы ${pick(data.player.gems, '-')}</span>
              <span class="chip">Платина ${pick(data.player.platinum, '-')}</span>
            </div>
          </div>
        </div>`;
      loadSavedData();
    }

    async function loadSavedData() {
      const data = await api('/api/saved-data');
      const root = document.getElementById('savedData');
      if (!data.ok || !data.player_cached) {
        root.textContent = 'Сохранённых данных игрока ещё нет';
        return;
      }
      root.innerHTML = `
        <div class="chips">
          <span class="chip good">${data.miscrit_count || 0} мискритов</span>
          <span class="chip">сохранено ${formatTime(data.player_saved_at)}</span>
        </div>
        <div class="muted" style="margin-top:8px">${escapeHtml(data.miscrits_path || '')}</div>`;
    }

    async function loadBreedOptions() {
      if (!selectedAccountId) return;
      const data = await api('/api/breed-options?account_id=' + encodeURIComponent(selectedAccountId));
      if (!data.ok) return;
      document.getElementById('poolMetric').textContent = pick(data.candidate_count, '-');
      const targetSelect = document.getElementById('targetMid');
      const currentMid = targetSelect.value;
      targetSelect.innerHTML = '<option value="">Любой</option>' + data.targets.map(t =>
        `<option value="${t.mid}">${escapeHtml(t.name)} #${t.mid} (${escapeHtml(t.element || 'нет стихии')}, ${t.count})</option>`
      ).join('');
      targetSelect.value = currentMid;
      const elementSelect = document.getElementById('targetElement');
      const currentElement = elementSelect.value;
      elementSelect.innerHTML = '<option value="">Любая</option>' + data.elements.map(e => `<option value="${escapeHtml(e)}">${escapeHtml(e)}</option>`).join('');
      elementSelect.value = currentElement;
      renderTargets(data.targets);
    }

    async function loadCacheState() {
      const data = await api('/api/cache');
      if (data.ok) renderCache(data.references || []);
    }

    async function syncCache(force, all) {
      const data = await api('/api/cache-sync', { method: 'POST', body: JSON.stringify({ force, all }) }, true);
      if (data.ok) { loadCacheState(); loadBreedOptions(); }
    }

    async function syncAvatars(all = false) {
      const data = await api('/api/avatar-sync', {
        method: 'POST',
        body: JSON.stringify({ account_id: selectedAccountId, all, limit: all ? 0 : 0 })
      }, true);
      if (data.ok) {
        const ok = (data.assets || []).filter(item => item.ok).length;
        document.getElementById('cacheList').textContent = `Синхронизация аватаров завершена: ${ok}/${(data.assets || []).length}`;
      }
    }

    function breedSettings(dryRun = true) {
      return {
        account_id: selectedAccountId,
        dry_run: dryRun,
        target_mid: document.getElementById('targetMid').value || null,
        target_element: document.getElementById('targetElement').value || null,
        min_max_sum: Number(document.getElementById('minMaxSum').value || 15),
        allow_splus_parents: Number(document.getElementById('allowSplus').value || 0)
        ,max_breeds: Number(document.getElementById('maxBreeds').value || 1)
        ,delay_seconds: Number(document.getElementById('breedDelay').value || 0.4)
      };
    }

    async function breedPlan() {
      const data = await api('/api/breed-plan', { method: 'POST', body: JSON.stringify(breedSettings(true)) });
      renderPlan(data.plan);
    }

    async function breedOnce(dryRun) {
      const data = await api('/api/breed-once', { method: 'POST', body: JSON.stringify(breedSettings(dryRun)) }, !dryRun);
      renderPlan(data.plan);
      if (!dryRun) { loadBreedLogs(); loadPlayer(); loadBreedOptions(); }
    }

    async function autoBreed(dryRun) {
      const data = await api('/api/auto-breed', { method: 'POST', body: JSON.stringify(breedSettings(dryRun)) }, !dryRun);
      renderPlan(data.plan || (data.last && data.last.plan));
      if (!dryRun) { loadBreedLogs(); loadPlayer(); loadBreedOptions(); }
    }

    async function loadBreedLogs() {
      const data = await api('/api/breed-logs');
      if (data.ok) renderLogs(data.logs || []);
    }

    function action(name) { api('/api/action?name=' + encodeURIComponent(name) + '&account_id=' + encodeURIComponent(selectedAccountId), { method: 'POST' }, true).then(loadPlayer); }

    function rpc() {
      api('/api/rpc', { method: 'POST', body: JSON.stringify({
        method: document.getElementById('rpcMethod').value,
        payload: { ...JSON.parse(document.getElementById('rpcPayload').value || '{}'), _account_id: selectedAccountId }
      })}, true);
    }

    async function arenaStatus() {
      const mode = document.getElementById('arenaMode').value;
      if (!selectedAccountId) { renderArenaState({ ok: false, error: 'Сначала выбери аккаунт Miscrits.' }); return; }
      const data = await api('/api/arena-status?mode=' + encodeURIComponent(mode) + '&account_id=' + encodeURIComponent(selectedAccountId));
      renderArenaState(data);
    }

    async function arenaRun(dryRun) {
      const body = {
        account_id: selectedAccountId,
        mode: document.getElementById('arenaMode').value,
        timeout_seconds: Number(document.getElementById('arenaTimeout').value || 300),
        max_turns: Number(document.getElementById('arenaMaxTurns').value || 150),
        dry_run: dryRun,
        prepare: document.getElementById('arenaPrepare').value === '1',
        location_id: 1,
        area_id: 2,
        repeat_count: dryRun ? 1 : Number(document.getElementById('arenaRepeatCount').value || 1),
        repeat_delay_seconds: Number(document.getElementById('arenaRepeatDelay').value || 0),
        stop_on_error: document.getElementById('arenaStopOnError').value === '1'
      };
      if (!selectedAccountId) { renderArenaState({ ok: false, error: 'Сначала выбери аккаунт Miscrits.' }); return; }
      if (dryRun) {
        const data = await api('/api/arena-run', { method: 'POST', body: JSON.stringify(body) }, true);
        renderArenaState(data.status || data);
        loadPlayer();
        return;
      }
      const data = await api('/api/arena-start', { method: 'POST', body: JSON.stringify(body) }, true);
      if (!data.ok) {
        renderArenaState(data);
        loadEventLogs();
        return;
      }
      arenaJobId = data.job_id;
      renderArenaLive(data.job);
      loadEventLogs();
      pollArenaJob();
    }

    async function arenaStop() {
      if (!arenaJobId) return;
      const data = await api('/api/arena-stop', { method: 'POST', body: JSON.stringify({ id: arenaJobId }) }, true);
      if (data.ok && data.job) {
        renderArenaLive(data.job);
        loadEventLogs();
        pollArenaJob();
      }
    }

    async function pollArenaJob() {
      if (!arenaJobId) return;
      if (arenaPollTimer) clearTimeout(arenaPollTimer);
      const data = await fetchJson('/api/arena-live?id=' + encodeURIComponent(arenaJobId));
      if (data.ok && data.job) {
        renderArenaLive(data.job);
        if (data.job.status === 'running') {
          arenaPollTimer = setTimeout(pollArenaJob, 1000);
        } else {
          arenaJobId = null;
          loadPlayer();
          loadBattleLearning();
          loadAiDashboard();
          loadEventLogs();
        }
      } else if (!data.ok) {
        arenaJobId = null;
        arenaStatus();
        loadEventLogs();
      }
    }

    async function socketDoctor() {
      const data = await api('/api/socket-doctor', {}, true);
      renderArenaState(data);
    }

    async function loadBattleLearning() {
      const data = await api('/api/battle-learning');
      if (data.ok) renderBattleLearning(data);
    }

    async function loadAiDashboard() {
      const data = await fetchJson('/api/ai-dashboard');
      if (data.ok) renderAiDashboard(data);
    }

    async function saveAiWeight(category, key, inputId) {
      const input = document.getElementById(inputId);
      const weight = Number(input && input.value);
      const data = await api('/api/ai-weight', {
        method: 'POST',
        body: JSON.stringify({ category, key, weight })
      }, true);
      if (data.ok && data.dashboard) renderAiDashboard(data.dashboard);
    }

    async function loadPlan() {
      if (!selectedAccountId) {
        renderPlanState({ ok: false, error: 'Сначала выбери аккаунт Miscrits.' });
        return;
      }
      const data = await fetchJson('/api/plan?account_id=' + encodeURIComponent(selectedAccountId));
      if (data.ok) {
        applyPlanConfig(data.config || {});
        renderPlanState(data);
        syncPlanArenaToArena(data.job && data.job.active_arena);
        refreshBattleViewsFromPlan(data.job || {});
        if (data.job && data.job.status === 'running') schedulePlanPoll();
      }
    }

    function readPlanConfig() {
      return {
        enabled: document.getElementById('planEnabled').checked,
        tick_seconds: Number(document.getElementById('planTickSeconds').value || 20),
        battle_delay_seconds: Number(document.getElementById('planBattleDelay').value || 3),
        stop_on_error: false,
        blocks: planBlocks.map(block => ({ ...block })),
        fallback: {
          random_when_heal_blocked: document.getElementById('planFallbackRandom').checked,
          random_batch_wins: 1,
        },
      };
    }

    function applyPlanConfig(config) {
      const fallback = config.fallback || {};
      document.getElementById('planEnabled').checked = config.enabled !== false;
      document.getElementById('planTickSeconds').value = pick(config.tick_seconds, 20);
      document.getElementById('planBattleDelay').value = pick(config.battle_delay_seconds, 3);
      document.getElementById('planFallbackRandom').checked = fallback.random_when_heal_blocked !== false;
      planBlocks = normalizePlanBlocks(config.blocks || blocksFromLegacyPlanSteps(config.steps || {}));
      renderPlanBlocks();
    }

    function normalizePlanBlocks(blocks) {
      return (blocks || []).map((block, index) => {
        const type = String(block.type || '');
        if (type === 'wish') {
          const kind = String(block.kind || '').toLowerCase();
          return { id: block.id || `wish_${kind}_${index + 1}`, type, kind, enabled: block.enabled !== false };
        }
        const mode = String(block.mode || '').toLowerCase();
        const normalized = { id: block.id || `${mode}_arena_${index + 1}`, type: 'arena', mode, enabled: block.enabled !== false };
        if (mode === 'platinum') normalized.target_platinum = Number(block.target_platinum || 300);
        else {
          const fallback = mode === 'daily' ? 5 : 6;
          const goalMode = ['cycle_wins', 'arena_counter'].includes(String(block.goal_mode || '')) ? String(block.goal_mode) : 'arena_counter';
          normalized.goal_mode = goalMode;
          normalized.target_cycle_wins = Number(block.target_cycle_wins || block.target_wins || fallback);
          normalized.target_arena_wins = Number(block.target_arena_wins || block.target_wins || fallback);
          normalized.target_wins = goalMode === 'cycle_wins' ? normalized.target_cycle_wins : normalized.target_arena_wins;
        }
        return normalized;
      }).filter(block => block.type === 'wish' || block.type === 'arena');
    }

    function blocksFromLegacyPlanSteps(steps) {
      return [
        { id: 'wish_sk', type: 'wish', kind: 'sk', enabled: !steps.wish_sk || steps.wish_sk.enabled !== false },
        { id: 'wish_vi', type: 'wish', kind: 'vi', enabled: !steps.wish_vi || steps.wish_vi.enabled !== false },
        { id: 'daily_arena', type: 'arena', mode: 'daily', enabled: !steps.daily_arena || steps.daily_arena.enabled !== false, goal_mode: 'arena_counter', target_arena_wins: pick(steps.daily_arena && steps.daily_arena.target_wins, 5), target_wins: pick(steps.daily_arena && steps.daily_arena.target_wins, 5) },
        { id: 'platinum_arena', type: 'arena', mode: 'platinum', enabled: !steps.platinum_arena || steps.platinum_arena.enabled !== false, target_platinum: pick(steps.platinum_arena && steps.platinum_arena.target_platinum, 300) },
        { id: 'random_arena', type: 'arena', mode: 'random', enabled: !steps.random_arena || steps.random_arena.enabled !== false, goal_mode: 'arena_counter', target_arena_wins: pick(steps.random_arena && steps.random_arena.target_wins, 6), target_wins: pick(steps.random_arena && steps.random_arena.target_wins, 6) },
      ];
    }

    function addPlanBlock(type) {
      const selected = type || document.getElementById('planBlockType').value;
      const id = `${selected}_${Date.now()}`;
      const templates = {
        wish_sk: { id, type: 'wish', kind: 'sk', enabled: true },
        wish_vi: { id, type: 'wish', kind: 'vi', enabled: true },
        daily_arena: { id, type: 'arena', mode: 'daily', enabled: true, goal_mode: 'arena_counter', target_arena_wins: 5, target_cycle_wins: 5, target_wins: 5 },
        battle_arena: { id, type: 'arena', mode: 'battle', enabled: true, goal_mode: 'arena_counter', target_arena_wins: 5, target_cycle_wins: 5, target_wins: 5 },
        platinum_arena: { id, type: 'arena', mode: 'platinum', enabled: true, target_platinum: 300 },
        random_arena: { id, type: 'arena', mode: 'random', enabled: true, goal_mode: 'cycle_wins', target_arena_wins: 6, target_cycle_wins: 6, target_wins: 6 },
      };
      if (!templates[selected]) return;
      planBlocks.push(templates[selected]);
      renderPlanBlocks();
    }

    function movePlanBlock(index, delta) {
      const next = index + delta;
      if (next < 0 || next >= planBlocks.length) return;
      const [block] = planBlocks.splice(index, 1);
      planBlocks.splice(next, 0, block);
      renderPlanBlocks();
    }

    function removePlanBlock(index) {
      planBlocks.splice(index, 1);
      renderPlanBlocks();
    }

    function updatePlanBlock(index, key, value) {
      if (!planBlocks[index]) return;
      planBlocks[index][key] = value;
    }

    function setPlanArenaGoalMode(index, value) {
      const block = planBlocks[index];
      if (!block || block.type !== 'arena' || block.mode === 'platinum') return;
      block.goal_mode = value === 'cycle_wins' ? 'cycle_wins' : 'arena_counter';
      const fallback = block.mode === 'daily' ? 5 : 6;
      block.target_cycle_wins = Number(block.target_cycle_wins || block.target_wins || fallback);
      block.target_arena_wins = Number(block.target_arena_wins || block.target_wins || fallback);
      block.target_wins = block.goal_mode === 'cycle_wins' ? block.target_cycle_wins : block.target_arena_wins;
      renderPlanBlocks();
    }

    function updatePlanArenaTarget(index, value) {
      const block = planBlocks[index];
      if (!block || block.type !== 'arena' || block.mode === 'platinum') return;
      const numeric = Number(value || 0);
      if (block.goal_mode === 'cycle_wins') block.target_cycle_wins = numeric;
      else block.target_arena_wins = numeric;
      block.target_wins = numeric;
    }

    function planBlockLabel(block) {
      if (block.type === 'wish') return block.kind === 'sk' ? 'Wish SK' : 'Wish VI';
      return ({ daily:'Ежедневная арена', battle:'Обычная арена', platinum:'Платиновая арена', random:'Случайная арена' })[block.mode] || 'Арена';
    }

    function planBlockSummary(block) {
      if (block.type === 'wish') return 'Собрать один раз после ежедневного сброса';
      if (block.mode === 'platinum') return `До ${escapeHtml(block.target_platinum || 0)} платины`;
      if (block.goal_mode === 'cycle_wins') return `Выиграть ${escapeHtml(block.target_cycle_wins || 0)} боёв за цикл`;
      return `До счётчика ${escapeHtml(block.target_arena_wins || 0)} побед`;
    }

    function renderPlanBlocks() {
      const root = document.getElementById('planBlocks');
      if (!root) return;
      if (!planBlocks.length) {
        root.innerHTML = '<div class="log-card muted">В плане пока нет блоков.</div>';
        return;
      }
      root.innerHTML = planBlocks.map((block, index) => {
        const targetField = block.type === 'arena'
          ? block.mode === 'platinum'
            ? `<div class="field"><label>Целевая платина</label><input type="number" min="0" value="${escapeHtml(block.target_platinum)}" onchange="updatePlanBlock(${index}, 'target_platinum', Number(this.value || 0))"></div>`
            : `<div class="plan-block-goal">
                <div class="field">
                  <label>Тип цели</label>
                  <select onchange="setPlanArenaGoalMode(${index}, this.value)">
                    <option value="cycle_wins" ${block.goal_mode === 'cycle_wins' ? 'selected' : ''}>Победы за цикл</option>
                    <option value="arena_counter" ${block.goal_mode !== 'cycle_wins' ? 'selected' : ''}>Счётчик арены</option>
                  </select>
                </div>
                <div class="field">
                  <label>${block.goal_mode === 'cycle_wins' ? 'Победы за цикл' : 'Предел счётчика'}</label>
                  <input type="number" min="0" value="${escapeHtml(block.goal_mode === 'cycle_wins' ? block.target_cycle_wins : block.target_arena_wins)}" onchange="updatePlanArenaTarget(${index}, this.value)">
                </div>
              </div>`
          : '<div></div>';
        return `<div class="plan-block">
          <div class="plan-block-main">
            <span class="plan-block-index">${index + 1}</span>
            <label class="chip"><input type="checkbox" ${block.enabled !== false ? 'checked' : ''} onchange="updatePlanBlock(${index}, 'enabled', this.checked)"> включён</label>
            <div class="plan-block-title">
              <strong>${escapeHtml(planBlockLabel(block))}</strong>
              <span>${planBlockSummary(block)}</span>
            </div>
          </div>
          ${targetField}
          <div class="plan-block-actions">
            <button class="ghost" onclick="movePlanBlock(${index}, -1)">Вверх</button>
            <button class="ghost" onclick="movePlanBlock(${index}, 1)">Вниз</button>
            <button class="ghost" onclick="removePlanBlock(${index})">Удалить</button>
          </div>
        </div>`;
      }).join('');
    }

    async function savePlan() {
      if (!selectedAccountId) return renderPlanState({ ok: false, error: 'Сначала выбери аккаунт Miscrits.' });
      const data = await api('/api/plan-save', { method: 'POST', body: JSON.stringify({ account_id: selectedAccountId, config: readPlanConfig() }) }, true);
      if (data.ok) {
        applyPlanConfig(data.config || {});
        renderPlanState(data);
      }
    }

    async function startPlan() {
      await savePlan();
      const data = await api('/api/plan-start', { method: 'POST', body: JSON.stringify({ account_id: selectedAccountId, run_forever: true }) }, true);
      if (data.ok) {
        renderPlanState(data);
        schedulePlanPoll();
      }
    }

    async function runPlanOnce() {
      await savePlan();
      const data = await api('/api/plan-start', { method: 'POST', body: JSON.stringify({ account_id: selectedAccountId, run_forever: false }) }, true);
      if (data.ok) {
        renderPlanState(data);
        schedulePlanPoll();
      }
    }

    async function stopPlan() {
      if (!selectedAccountId) return;
      const data = await api('/api/plan-stop', { method: 'POST', body: JSON.stringify({ account_id: selectedAccountId }) }, true);
      if (data.ok) renderPlanState(data);
    }

    async function pollPlan() {
      if (!selectedAccountId) return;
      const data = await fetchJson('/api/plan-live?account_id=' + encodeURIComponent(selectedAccountId));
      if (data.ok) {
        renderPlanState(data);
        syncPlanArenaToArena(data.job && data.job.active_arena);
        refreshBattleViewsFromPlan(data.job || {});
        if (data.job && data.job.status === 'running') schedulePlanPoll();
      }
    }

    function schedulePlanPoll() {
      if (planPollTimer) clearTimeout(planPollTimer);
      const planEl = document.getElementById('plan');
      const planActive = planEl && planEl.classList.contains('active');
      planPollTimer = setTimeout(pollPlan, planActive ? 1500 : 5000);
    }

    function syncPlanArenaToArena(activeArena) {
      if (!activeArena || activeArena.status !== 'running') return;
      if (arenaJobId && arenaJobId !== activeArena.id && !String(arenaJobId).startsWith('plan-arena:')) return;
      arenaJobId = activeArena.id;
      syncArenaControls(activeArena);
      renderArenaLive(activeArena);
      pollArenaJob();
    }

    function refreshBattleViewsFromPlan(job = {}) {
      const events = Array.isArray(job.events) ? job.events : [];
      const complete = [...events].reverse().find(event => event && event.event === 'plan_arena_complete');
      if (!complete) return;
      const key = `${job.id || ''}:${complete.timestamp || ''}:${complete.mode || ''}:${complete.outcome || ''}`;
      if (key === lastPlanBattleRefreshKey) return;
      lastPlanBattleRefreshKey = key;
      loadBattleLearning();
      loadAiDashboard();
      loadPlayer();
      const battleLogsView = document.getElementById('battlelogs');
      if (battleLogsView && battleLogsView.classList.contains('active')) loadBattleLogs();
    }

    function renderPlanState(data) {
      const root = document.getElementById('planState');
      if (!data || !data.ok) {
        root.classList.add('muted');
        root.innerHTML = `<div class="log-card"><strong>План недоступен</strong><pre>${escapeHtml(JSON.stringify(data, null, 2))}</pre></div>`;
        return;
      }
      const state = data.state || (data.job && data.job.state) || {};
      const job = data.job || {};
      const progress = job.progress || {};
      const events = job.events || [];
      const arenas = state.arenas || {};
      const dailyArena = arenas.daily || {};
      const platinumArena = arenas.platinum || {};
      const randomArena = arenas.random || {};
      const configuredBlocks = normalizePlanBlocks((data.config && data.config.blocks) || planBlocks);
      root.classList.remove('muted');
      root.innerHTML = `
        <div class="log-card">
          <div class="log-head">
            <div><strong>${escapeHtml(job.status || 'сохранён')}</strong><div class="muted">${escapeHtml(progress.event || job.phase || 'ожидание')}</div></div>
            <div class="chips">
              <span class="chip">${escapeHtml(state.reset_key || '-')}</span>
              ${job.stop_requested ? '<span class="chip warn">запрошена остановка</span>' : ''}
            </div>
          </div>
          <div class="chips" style="margin-top:8px">
            <span class="chip ${state.wish_sk && state.wish_sk.done ? 'good' : ''}">Wish SK ${state.wish_sk && state.wish_sk.done ? 'готово' : 'ожидает'}</span>
            <span class="chip ${state.wish_vi && state.wish_vi.done ? 'good' : ''}">Wish VI ${state.wish_vi && state.wish_vi.done ? 'готово' : 'ожидает'}</span>
            <span class="chip">ежедневная ${escapeHtml(pick(dailyArena.goal_progress, state.daily_wins || 0))} / ${escapeHtml(dailyArena.target || 5)}</span>
            <span class="chip">платина ${escapeHtml(state.platinum_earned || 0)} / ${escapeHtml(platinumArena.target || platinumArena.cap_target || 300)}</span>
            <span class="chip">случайная ${escapeHtml(pick(randomArena.goal_progress, state.random_wins || 0))} / ${escapeHtml(randomArena.target || 6)}</span>
          </div>
          <div class="chips" style="margin-top:8px">
            ${configuredBlocks.map((block, index) => `<span class="chip ${block.enabled === false ? 'warn' : ''}">${index + 1}. ${escapeHtml(planBlockLabel(block))}</span>`).join('')}
          </div>
          ${job.error ? `<pre>${escapeHtml(job.error)}</pre>` : ''}
        </div>
        ${renderPlanArenaCards(arenas)}
        ${events.length ? `<div class="log-card"><div class="log-head"><strong>Последние события плана</strong><span class="chip">${events.length}</span></div><div class="logs" style="margin-top:10px;max-height:300px">${events.slice(-20).reverse().map(event => `
          <div class="parent">
            <div class="log-head"><strong>${escapeHtml(event.event || '')}</strong><span class="chip">${escapeHtml(event.phase || '')}</span></div>
            <div class="muted">${event.timestamp ? new Date(event.timestamp * 1000).toLocaleTimeString() : ''}</div>
          </div>`).join('')}</div></div>` : ''}`;
    }

    function renderPlanArenaCards(arenas = {}) {
      const modes = [
        ['daily', 'Ежедневная арена'],
        ['battle', 'Обычная арена'],
        ['platinum', 'Платиновая арена'],
        ['random', 'Случайная арена'],
      ];
      return `<div class="grid-3">${modes.map(([key, label]) => {
        const item = arenas[key] || {};
        const rewards = item.reward_steps || [];
        const rewardProgress = item.reward_progress || {};
        return `<div class="log-card">
        <div class="log-head"><strong>${label}</strong><span class="chip">${escapeHtml(pick(item.progress, 0))} / ${escapeHtml(item.target || item.cap_target || (key === 'platinum' ? 300 : '-'))}</span></div>
          <div class="chips" style="margin-top:8px">
            ${item.banned ? '<span class="chip warn">запрещена</span>' : ''}
            ${item.rewards_complete ? '<span class="chip good">все награды получены</span>' : ''}
            ${item.goal_mode === 'cycle_wins' ? '<span class="chip">цель: победы за цикл</span>' : ''}
            ${item.goal_mode === 'arena_counter' ? '<span class="chip">цель: счётчик арены</span>' : ''}
            ${item.time_left !== undefined && item.time_left !== null ? `<span class="chip">сброс через ${escapeHtml(item.time_left)}с</span>` : ''}
            ${item.pa_streak !== undefined && item.pa_streak !== null ? `<span class="chip">серия ${escapeHtml(item.pa_streak)}</span>` : ''}
            ${rewards.length ? `<span class="chip">награды ${rewards.map(escapeHtml).join(', ')}</span>` : '<span class="chip">награды не загружены</span>'}
          </div>
          ${renderArenaRewards(rewardProgress)}
        </div>`;
      }).join('')}</div>`;
    }

    async function loadBattleLogs() {
      const params = new URLSearchParams();
      for (const [id, key] of [
        ['battleLogMode', 'mode'],
        ['battleLogOutcome', 'outcome'],
        ['battleLogText', 'text'],
        ['battleLogLimit', 'limit'],
      ]) {
        const el = document.getElementById(id);
        if (el && el.value) params.set(key, el.value);
      }
      const data = await fetchJson('/api/battle-logs?' + params.toString());
      renderBattleLogs(data.ok ? (data.logs || []) : [], data.error || '');
    }

    function clearBattleLogFilters() {
      for (const id of ['battleLogMode', 'battleLogOutcome', 'battleLogText']) {
        const el = document.getElementById(id);
        if (el) el.value = '';
      }
      const limit = document.getElementById('battleLogLimit');
      if (limit) limit.value = 100;
      loadBattleLogs();
    }

    function renderBattleLogs(logs, error = '') {
      const root = document.getElementById('battleLogList');
      if (!root) return;
      if (error) {
        root.classList.add('muted');
        root.textContent = error;
        return;
      }
      if (!logs.length) {
        root.classList.add('muted');
        root.textContent = 'Нет логов боёв под текущие фильтры';
        return;
      }
      root.classList.remove('muted');
      root.innerHTML = logs.map(item => {
        const id = String(item.id || '');
        const outcome = String(item.outcome || 'unknown');
        const grade = item.grade_label ? `${escapeHtml(item.grade_label)} ${escapeHtml(pick(item.grade_score, ''))}` : '';
        return `<div class="log-card">
          <div class="log-head">
            <div>
              <strong>${escapeHtml(outcomeLabel(outcome))}</strong>
              <div class="muted">${formatTime(item.finished_at || item.started_at)} / ${escapeHtml(modeLabel(item.mode || ''))}</div>
            </div>
            <button class="ghost" onclick='loadBattleLogDetail(${JSON.stringify(id)})'>Открыть</button>
          </div>
          <div class="chips" style="margin-top:8px">
            <span class="chip">${escapeHtml(String(id).slice(0, 8))}</span>
            ${grade ? `<span class="chip">${grade}</span>` : ''}
            <span class="chip">ходы ${escapeHtml(pick(item.turns_sent, 0))}</span>
            <span class="chip">события ${escapeHtml(pick(item.event_count, 0))}</span>
            <span class="chip">действия ${escapeHtml(pick(item.decision_count, 0))}</span>
            <span class="chip">урон ${escapeHtml(pick(item.damage_sample_count, 0))}</span>
          </div>
        </div>`;
      }).join('');
    }

    async function loadBattleLogDetail(id) {
      const root = document.getElementById('battleLogDetail');
      if (root) {
        root.classList.add('muted');
        root.textContent = 'Загружаю лог боя...';
      }
      const data = await fetchJson('/api/battle-log?id=' + encodeURIComponent(id));
      if (!data.ok) {
        if (root) root.textContent = data.error || 'Лог боя не найден';
        return;
      }
      renderBattleLogDetail(data.battle || {});
    }

    function renderBattleLogDetail(battle) {
      const root = document.getElementById('battleLogDetail');
      if (!root) return;
      const grade = battle.grade || {};
      const post = battle.post_battle || {};
      const summary = post.summary || {};
      const timeline = battle.timeline || [];
      root.classList.remove('muted');
      root.innerHTML = `<div class="log-card">
        <div class="log-head">
          <div>
            <strong>${escapeHtml(outcomeLabel(battle.outcome || 'unknown'))}</strong>
            <div class="muted">${formatTime(battle.started_at)} - ${formatTime(battle.finished_at)}</div>
          </div>
          <div class="chips">
            <span class="chip">${escapeHtml(modeLabel(battle.mode || ''))}</span>
            <span class="chip">${escapeHtml(String(battle.id || '').slice(0, 8))}</span>
          </div>
        </div>
        <div class="chips" style="margin-top:8px">
          <span class="chip">${escapeHtml(grade.label || '')} ${escapeHtml(pick(grade.score, ''))}</span>
          <span class="chip">ходы ${escapeHtml(pick(battle.turns_sent, 0))}</span>
          <span class="chip">погибло союзников ${escapeHtml(pick(grade.ally_deaths, '-'))}</span>
          <span class="chip">погибло врагов ${escapeHtml(pick(grade.foe_deaths, '-'))}</span>
          <span class="chip">упущено добиваний ${escapeHtml(summary.missed_finishes || 0)}</span>
          <span class="chip">плохие смены ${escapeHtml(summary.useless_switches || 0)}</span>
        </div>
      </div>
      <div class="log-card">
        <div class="log-head"><strong>Хронология</strong><span class="chip">${timeline.length} событий</span></div>
        ${timeline.length ? timeline.map(renderBattleTimelineRow).join('') : '<div class="muted">События хронологии не сохранены</div>'}
      </div>`;
    }

    function renderBattleTimelineRow(row) {
      const detail = row.event === 'decision'
        ? `${escapeHtml(row.actor || '')} -> ${escapeHtml(row.action || '')}${row.target ? ' против ' + escapeHtml(row.target) : ''}${row.reason ? ' / ' + escapeHtml(aiReasonLabel(row.reason)) : ''}`
        : row.event === 'damage_sample'
          ? `${escapeHtml(row.actor || '')} использовал ${escapeHtml(row.action || '')} по ${escapeHtml(row.target || '')}: ${escapeHtml(row.actual_damage || 0)} урона, ожидалось ${escapeHtml(Number(row.expected_damage || 0).toFixed(1))}`
          : `opcode ${escapeHtml(pick(row.opcode, '-'))}`;
      return `<div class="log-card" style="margin-top:8px">
        <div class="log-head">
          <strong>${escapeHtml(row.event || 'событие')}</strong>
          <div class="chips"><span class="chip">ход ${escapeHtml(pick(row.turns, 0))}</span><span class="chip">${formatTime(row.timestamp)}</span></div>
        </div>
        <div class="muted" style="margin-top:6px">${detail}</div>
      </div>`;
    }

    async function loadEventLogs() {
      const params = new URLSearchParams();
      const logAccountScope = document.getElementById('logAccountScope');
      const accountFilter = logAccountScope && logAccountScope.value === 'selected' ? (selectedAccountId || '') : '';
      if (accountFilter) params.set('account_id', accountFilter);
      for (const [id, key] of [
        ['logLevel', 'level'],
        ['logCategory', 'category'],
        ['logInitiator', 'initiator'],
        ['logEvent', 'event'],
        ['logReason', 'reason'],
        ['logText', 'text'],
        ['logLimit', 'limit'],
      ]) {
        const el = document.getElementById(id);
        if (el && el.value) params.set(key, el.value);
      }
      const logSinceEl = document.getElementById('logSince');
      const logUntilEl = document.getElementById('logUntil');
      const since = dateTimeLocalToSeconds((logSinceEl && logSinceEl.value) || '');
      const until = dateTimeLocalToSeconds((logUntilEl && logUntilEl.value) || '');
      if (since) params.set('since', String(since));
      if (until) params.set('until', String(until));
      const data = await fetchJson('/api/logs?' + params.toString());
      if (data.ok) renderEventLogs(data.logs || [], data.status || {});
      scheduleLogPoll();
    }

    function scheduleLogPoll() {
      if (logPollTimer) clearTimeout(logPollTimer);
      const logsEl = document.getElementById('logs');
      const logsActive = logsEl && logsEl.classList.contains('active');
      if (authenticated && logsActive) logPollTimer = setTimeout(loadEventLogs, 5000);
    }

    function renderEventLogs(logs, status = {}) {
      const root = document.getElementById('systemLogs');
      const statusRoot = document.getElementById('logStatus');
      if (statusRoot) {
        statusRoot.innerHTML = `
          <span class="chip">показано ${logs.length}</span>
          ${status.exists ? `<span class="chip">файл ${Math.round(Number(status.size || 0) / 1024)} КБ</span>` : '<span class="chip warn">файла ещё нет</span>'}`;
      }
      if (!logs.length) {
        root.classList.add('muted');
        root.textContent = 'Нет событий под текущие фильтры';
        return;
      }
      root.classList.remove('muted');
      root.innerHTML = logs.map(entry => {
        const level = entry.level || 'info';
        const payload = entry.payload || {};
        return `<div class="log-card">
          <div class="log-head">
            <div>
              <strong>${escapeHtml(entry.event || 'событие')}</strong>
              <div class="muted">${formatTime(entry.timestamp)} ${entry.message ? ' / ' + escapeHtml(entry.message) : ''}</div>
            </div>
            <div class="chips">
              <span class="chip ${level === 'error' || level === 'warning' ? 'warn' : 'good'}">${escapeHtml(level)}</span>
              <span class="chip">${escapeHtml(entry.category || '')}</span>
              ${entry.phase ? `<span class="chip">${escapeHtml(entry.phase)}</span>` : ''}
              ${entry.initiator ? `<span class="chip">${escapeHtml(entry.initiator)}</span>` : ''}
            </div>
          </div>
          <div class="chips" style="margin-top:8px">
            ${entry.account_id ? `<span class="chip">аккаунт ${escapeHtml(entry.account_id)}</span>` : ''}
            ${entry.job_id ? `<span class="chip">задача ${escapeHtml(String(entry.job_id).slice(0, 8))}</span>` : ''}
            ${entry.mode ? `<span class="chip">${escapeHtml(modeLabel(entry.mode))}</span>` : ''}
            ${entry.reason ? `<span class="chip warn">${escapeHtml(entry.reason)}</span>` : ''}
          </div>
          ${renderLogPayload(payload)}
        </div>`;
      }).join('');
    }

    function renderLogPayload(payload) {
      if (!payload || !Object.keys(payload).length) return '';
      const loop = payload.loop || {};
      const result = payload.result || {};
      const battle = payload.battle || result.battle || {};
      const chips = [];
      if (loop.current || loop.completed !== undefined) {
        chips.push(`бой ${escapeHtml(loop.current || loop.completed || 0)} / ${loop.target === 0 || loop.infinite ? '∞' : escapeHtml(loop.target || loop.requested || '?')}`);
      }
      if (battle.outcome) chips.push(escapeHtml(outcomeLabel(battle.outcome)));
      if (battle.grade && battle.grade.label) chips.push(`${escapeHtml(battle.grade.label)} ${escapeHtml(pick(battle.grade.score, ''))}`);
      if (result.battle_count !== undefined) chips.push(`сохранено боёв: ${escapeHtml(result.battle_count)}`);
      if (!chips.length) {
        const compact = JSON.stringify(payload);
        return compact.length > 600 ? `<pre>${escapeHtml(compact.slice(0, 600))}...</pre>` : `<pre>${escapeHtml(JSON.stringify(payload, null, 2))}</pre>`;
      }
      return `<div class="chips" style="margin-top:8px">${chips.map(item => `<span class="chip">${item}</span>`).join('')}</div>`;
    }

    function fillLogLastHours(hours) {
      const now = new Date();
      const start = new Date(now.getTime() - Number(hours || 1) * 3600 * 1000);
      document.getElementById('logSince').value = toDateTimeLocal(start);
      document.getElementById('logUntil').value = toDateTimeLocal(now);
      loadEventLogs();
    }

    function clearLogFilters() {
      ['logAccountScope','logLevel','logCategory','logInitiator','logSince','logUntil','logEvent','logReason','logText'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.value = '';
      });
      document.getElementById('logLimit').value = '250';
      loadEventLogs();
    }

    function dateTimeLocalToSeconds(value) {
      if (!value) return 0;
      const ms = new Date(value).getTime();
      return Number.isFinite(ms) ? Math.floor(ms / 1000) : 0;
    }

    function toDateTimeLocal(date) {
      const pad = value => String(value).padStart(2, '0');
      return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
    }

    function renderPlan(plan) {
      const root = document.getElementById('planRoot');
      if (!plan) { root.className = 'plan-card muted'; root.textContent = 'Корректный план не найден'; return; }
      root.className = 'plan-card';
      root.innerHTML = `
        <div class="chips">
          <span class="chip good">макс. ${escapeHtml(plan.max_rating)}</span>
          <span class="chip">оценка ${plan.score}</span>
          <span class="chip">стоимость ${plan.cost}</span>
          <span class="chip">доминантный шанс ${Math.round((plan.dominant_chance || 0) * 100)}%</span>
        </div>
        <div class="parents">${(plan.parents || []).map(parentCard).join('')}</div>`;
    }

    function parentCard(p) {
      return miscritCard(p);
    }

    function renderTargets(items) {
      const root = document.getElementById('targetList');
      if (!items.length) { root.textContent = 'Нет кандидатов 1 уровня'; return; }
      root.innerHTML = items.slice(0, 80).map(t => miscritCard({
        ...t,
        id: t.count ? `${t.count} pcs` : '',
        rating: t.best_rating,
        rating_sum: t.best_rating_sum,
      }, { compact: true })).join('');
    }

    function renderLogs(logs) {
      const root = document.getElementById('breedLogs');
      if (!logs.length) { root.textContent = 'Логов скрещивания ещё нет'; return; }
      root.innerHTML = logs.map(log => {
        const child = log.child || {};
        const splus = isSplus(child);
        return `<div class="log-card ${splus ? 'splus-result' : ''}">
          <div class="log-head">
            <div><strong>${escapeHtml(child.name || ('#' + (child.mid || '?')))}</strong><div class="muted">${formatTime(log.timestamp_unix)}</div></div>
            <div class="chips">${splus ? '<span class="splus-banner">S+ создан</span>' : ''}<span class="chip good">${escapeHtml(child.rating || '?')}</span><span class="chip">${child.rating_sum || '?'}/18</span><span class="chip warn">${log.cost || 0} gold</span></div>
          </div>
          <div style="margin-top:10px">${miscritCard(child, { featuredSplus: splus })}</div>
          <div class="parents" style="margin-top:10px">${(log.parents || []).map(parentCard).join('')}</div>
        </div>`;
      }).join('');
    }

    function miscritCard(m = {}, options = {}) {
      const name = m.name || ('#' + (m.mid || m.id || '?'));
      const mid = pick(m.mid, '?');
      const rating = m.rating || m.best_rating || '?';
      const ratingSum = pick(pick(m.rating_sum, m.best_rating_sum), '');
      const rarity = rarityClass(m.rarity);
      const element = m.element || '';
      const image = m.img || m.image || m.avatar || m.icon || m.icon_url || m.thumbnail || miscritAssetUrl(m, 'avatars');
      const initials = name.replace(/^#/, '').split(/\\s+/).filter(Boolean).map(part => part[0]).join('').slice(0, 2).toUpperCase() || 'IMG';
      const stats = normalizeCardStats(m);
      return `<div class="miscrit-card ${rarity} ${options.compact ? 'compact' : ''} ${options.featuredSplus ? 'splus-card' : ''}">
        <div class="miscrit-avatar">
          ${image ? `<img src="${escapeHtml(image)}" alt="${escapeHtml(name)}">` : `<span>${escapeHtml(initials)}</span>`}
          <span class="rating-mark">${escapeHtml(shortRating(rating))}</span>
        </div>
        <div class="miscrit-main">
          <div class="miscrit-name">${escapeHtml(name)}</div>
          <div class="miscrit-mid">MID ${escapeHtml(mid)}</div>
          <div class="miscrit-subline">${escapeHtml(idLine(m, ratingSum))}</div>
        </div>
        <div class="element-orb" title="${escapeHtml(element || 'Нет стихии')}">${elementIcon(element, elementShort(element))}</div>
        ${statGrid(stats)}
      </div>`;
    }

    function rarityClass(rarity) {
      const key = String(rarity || 'common').toLowerCase().replace(/[^a-z]/g, '');
      if (key === 'legend' || key === 'legendary') return 'rarity-legend';
      if (['common', 'rare', 'epic', 'exotic'].includes(key)) return `rarity-${key}`;
      return 'rarity-common';
    }

    function normalizeCardStats(m = {}) {
      const src = m.stats || m;
      return {
        hp: Number(src.hp || 0),
        spd: Number(src.spd || 0),
        ea: Number(src.ea || 0),
        pa: Number(src.pa || 0),
        ed: Number(src.ed || 0),
        pd: Number(src.pd || 0),
      };
    }

    function statGrid(stats) {
      const keys = ['hp', 'spd', 'ea', 'pa', 'ed', 'pd'];
      const hasStats = keys.some(key => Number(stats[key] || 0) > 0);
      if (!hasStats) {
        return `<div class="miscrit-stat-grid">
          <div class="miscrit-stat">HP</div><div class="miscrit-stat">SPD</div><div class="miscrit-stat">EA</div>
          <div class="miscrit-stat">PA</div><div class="miscrit-stat">ED</div><div class="miscrit-stat">PD</div>
        </div>`;
      }
      return `<div class="miscrit-stat-grid">${keys.map(key => {
        const value = Number(stats[key] || 0);
        return `<div class="miscrit-stat v${value || 2}">${key.toUpperCase()} ${value || '-'}</div>`;
      }).join('')}</div>`;
    }

    function shortRating(rating) {
      const value = String(rating || '?').trim();
      if (!value) return '?';
      if (value.length <= 2) return value;
      return value[0].toUpperCase();
    }

    function elementShort(element) {
      const value = String(element || '').trim();
      return value ? value.slice(0, 2) : '-';
    }

    function isSplus(m = {}) {
      return String(m.rating || '').trim() === 'S+' || Number(m.rating_sum || 0) >= 18;
    }

    function elementAssetUrl(element) {
      const value = String(element || '').trim();
      return value ? `/api/element-asset?element=${encodeURIComponent(value)}` : '';
    }

    function elementIcon(element, fallback = '') {
      const src = elementAssetUrl(element);
      return src ? `<img src="${escapeHtml(src)}" alt="${escapeHtml(element)}">` : escapeHtml(fallback || '-');
    }

    function elementBadge(element) {
      const value = String(element || '').trim();
      if (!value) return '';
      return `<span class="element-badge">${elementIcon(value)}<span>${escapeHtml(value)}</span></span>`;
    }

    function idLine(m = {}, ratingSum = '') {
      const parts = [];
      if (m.id !== undefined && m.id !== null && m.id !== '') parts.push(`#${m.id}`);
      if (m.element) parts.push(m.element);
      if (m.rarity) parts.push(m.rarity);
      if (ratingSum !== undefined && ratingSum !== null && ratingSum !== '') parts.push(`${ratingSum}/18`);
      return parts.join(' / ');
    }

    function renderCache(items) {
      const root = document.getElementById('cacheList');
      if (!items.length) { root.textContent = 'Справочники не найдены'; return; }
      root.innerHTML = items.map(item => `<div class="log-card">
        <div class="log-head">
          <strong>${escapeHtml(item.name)}</strong>
          <span class="chip ${item.cached ? 'good' : 'warn'}">${item.cached ? 'в кэше' : 'отсутствует'}</span>
        </div>
        <div class="chips">
          <span class="chip">сервер ${escapeHtml(item.remote_version)}</span>
          <span class="chip">локально ${escapeHtml(pick(item.local_version, '-'))}</span>
        </div>
        <div class="muted">${escapeHtml(item.path)}</div>
      </div>`).join('');
    }

    function renderArenaState(data) {
      const root = document.getElementById('arenaState');
      if (!data || !data.ok) {
        root.innerHTML = `<div class="log-card"><strong>Не готово</strong><pre>${escapeHtml(JSON.stringify(data, null, 2))}</pre></div>`;
        return;
      }
      const validation = data.validation || {};
      const team = data.team || [];
      root.innerHTML = `
        <div class="log-card">
          <div class="log-head"><strong>${escapeHtml(modeLabel(data.mode || ''))}</strong><span class="chip ${validation.ok ? 'good' : 'warn'}">${validation.ok ? 'готово' : 'заблокировано'}</span></div>
          <div class="chips"><span class="chip">рейтинг команды ${pick(data.team_rating, '-')}</span>${(data.rules || []).map(r => `<span class="chip">${escapeHtml(r)}</span>`).join('')}</div>
        </div>
        <div class="parents">${team.map(t => `<div class="parent"><strong>${escapeHtml(t.name || ('#' + t.mid))}</strong><div class="muted">#${t.id} В· level ${t.level} В· ${escapeHtml(t.rarity || '')}</div><div class="chips"><span class="chip good">${escapeHtml(t.rating || '?')}</span><span class="chip">${escapeHtml(t.element || '')}</span></div></div>`).join('')}</div>`;
    }

    function renderArenaLive(job) {
      const root = document.getElementById('arenaState');
      root.classList.remove('muted');
      const progress = job.progress || {};
      const result = job.result || {};
      const battle = progress.battle || {};
      const draft = progress.draft || {};
      const outcome = progress.outcome || (result.battle && result.battle.outcome) || '';
      const grade = progress.grade || (result.battle && result.battle.grade) || {};
      const phase = job.phase || 'starting';
      const loop = progress.loop || result.loop || {};
      const summary = loop.summary || {};
      const target = Number(pick(pick(loop.target, loop.requested), pick(job.repeat && job.repeat.count, 1)));
      const current = Number(pick(loop.current, 1));
      const completed = Number(pick(pick(loop.completed, summary.total), (job.status !== 'running' && target === 1 ? 1 : 0)));
      root.innerHTML = `
        <div class="log-card">
          <div class="log-head">
            <div><strong>${escapeHtml(modeLabel(job.mode || ''))}</strong><div class="muted">${escapeHtml(progress.event || job.status || '')}</div></div>
            <div class="chips">
              <span class="chip ${job.status === 'running' ? 'warn' : 'good'}">${escapeHtml(job.status === 'running' ? 'выполняется' : job.status || '')}</span>
              ${job.source === 'plan' ? '<span class="chip">план</span>' : ''}
              ${job.stop_requested ? '<span class="chip warn">запрошена остановка</span>' : ''}
              <span class="chip">${target === 0 ? `бой ${current} / ∞` : `бой ${current} / ${target || 1}`}</span>
              <span class="chip">завершено ${completed}</span>
              ${outcome ? `<span class="chip ${outcome === 'victory' ? 'good' : 'warn'}">${escapeHtml(outcomeLabel(outcome))}</span>` : ''}
              ${grade.label ? `<span class="chip good">${escapeHtml(grade.label)} · ${escapeHtml(grade.score)}</span>` : ''}
            </div>
          </div>
          ${renderStageTrack(phase)}
          ${summary.total ? `<div class="chips" style="margin-top:10px">
            <span class="chip good">победы ${escapeHtml(summary.victories || 0)}</span>
            <span class="chip warn">поражения ${escapeHtml(summary.defeats || 0)}</span>
            <span class="chip">винрейт ${Math.round(Number(summary.win_rate || 0) * 100)}%</span>
            <span class="chip">средняя оценка ${Number(summary.average_score || 0).toFixed(1)}</span>
          </div>` : ''}
        </div>
        ${draft && (draft.pool || draft.bans || draft.picks || draft.prepare) ? renderDraft(draft) : ''}
        ${battle && (battle.player || battle.foe) ? renderBattle(battle) : ''}
        ${job.error ? `<div class="log-card"><strong>Ошибка</strong><pre>${escapeHtml(job.error)}</pre></div>` : ''}
        ${renderArenaEvents(job.events || [])}`;
    }

    function renderStageTrack(phase) {
      const normalized = ({starting:'searching', preparing:'searching', connecting:'searching', ready:'searching', cooldown:'searching', stopped:'finished', error:'finished'})[phase] || phase;
      const stages = [
        ['searching', 'Поиск'],
        ['draft', 'Драфт'],
        ['battle', 'Бой'],
        ['finished', 'Итог']
      ];
      const activeIndex = Math.max(0, stages.findIndex(([key]) => key === normalized));
      return `<div class="stage-track" style="margin-top:12px">${stages.map(([key, label], index) => {
        const cls = index < activeIndex ? 'done' : index === activeIndex ? 'active' : '';
        return `<div class="stage ${cls}">${label}</div>`;
      }).join('')}</div>`;
    }

    function renderDraft(draft) {
      return `<div class="log-card">
        <div class="log-head"><strong>Случайный драфт</strong><span class="chip">автопик</span></div>
        <div class="draft-grid" style="margin-top:10px">
          ${renderDraftColumn('Пул', draft.pool || [])}
          ${renderDraftColumn('Баны', draft.bans || [])}
          ${renderDraftColumn('Пики', draft.picks || [])}
        </div>
        ${draft.prepare ? `<pre>${escapeHtml(JSON.stringify(draft.prepare, null, 2))}</pre>` : ''}
      </div>`;
    }

    function renderDraftColumn(title, items) {
      const shown = (items || []).slice(0, 12);
      return `<div class="parent"><strong>${escapeHtml(title)}</strong><div class="chips" style="margin-top:8px">${
        shown.length ? shown.map(item => `<span class="chip">${escapeHtml(item.name || ('#' + item.mid))}</span>`).join('') : '<span class="muted">пусто</span>'
      }</div></div>`;
    }

    function renderBattle(battle) {
      const activeText = battle.active_side === 'player' ? 'Твой ход' : battle.active_side === 'foe' ? 'Ход соперника' : 'Ожидание';
      return `<div class="log-card">
        <div class="log-head">
          <strong>Бой</strong>
          <div class="chips"><span class="chip">${activeText}</span><span class="chip">ход ${pick(battle.turns, '-')}</span></div>
        </div>
        <div class="battle-board" style="margin-top:10px">
          ${renderBattleSide('Твоя команда', battle.player || {}, 'player')}
          ${renderBattleSide('Соперник', battle.foe || {}, 'foe')}
        </div>
      </div>`;
    }

    function renderBattleSide(title, side, sideKey = '') {
      const team = side.team || [];
      return `<div class="battle-side ${escapeHtml(sideKey)}">
        <div class="log-head"><strong>${escapeHtml(title)}</strong><span class="chip">${escapeHtml(side.username || '')}</span></div>
        <div class="battle-team">${team.length ? team.map(item => renderBattleMiscrit(item, sideKey)).join('') : '<span class="muted">Данные команды ещё не получены</span>'}</div>
      </div>`;
    }

    function renderBattleMiscrit(m, sideKey = '') {
      const maxHp = Number(m.max_hp || 0);
      const chp = Number(m.chp || 0);
      const pct = maxHp > 0 ? Math.max(0, Math.min(100, Math.round(chp / maxHp * 100))) : (m.dead ? 0 : 100);
      const name = m.name || ('#' + (m.mid || m.id || '?'));
      const icon = name.replace(/^#/, '').slice(0, 2).toUpperCase();
      const ids = [`mid ${escapeHtml(m.mid || '?')}`];
      if (m.id !== undefined && m.id !== null && Number(m.id) !== Number(m.mid || 0)) ids.push(`battle id ${escapeHtml(m.id)}`);
      m.battle_id = m.id;
      m.id = m.mid || m.id;
      return `<div class="miscrit-pill ${m.active ? 'active' : ''} ${m.dead ? 'dead' : ''}">
        <div class="miscrit-icon">${escapeHtml(icon)}</div>
        <div>
          <div class="log-head"><strong>${escapeHtml(name)}</strong><span class="chip">${m.active ? 'активен' : 'запас'}</span></div>
          <div class="muted">${ids.join(' / ')}</div>
          <div class="muted">#${escapeHtml(m.id || '?')} · ${escapeHtml(m.element || '')} · lvl ${escapeHtml(pick(m.level, '-'))}</div>
          <div class="hpbar"><div class="hpfill ${pct <= 30 ? 'low' : ''}" style="width:${pct}%"></div></div>
          <div class="muted">${chp}/${maxHp || '?'} HP</div>
        </div>
      </div>`;
    }

    function renderArenaEvents(events) {
      const recent = events.slice(-14).reverse();
      if (!recent.length) return '';
      return `<div class="log-card">
        <div class="log-head"><strong>Лог боя</strong><span class="chip">${events.length} событий</span></div>
        <div class="logs" style="margin-top:10px; max-height:300px">${recent.map(event => `
          <div class="parent">
            <div class="log-head"><strong>${escapeHtml(event.event || '')}</strong><span class="chip">${escapeHtml(event.phase || '')}</span></div>
            ${renderArenaEventDetail(event)}
            <div class="muted">${event.timestamp ? new Date(event.timestamp * 1000).toLocaleTimeString() : ''}</div>
          </div>`).join('')}</div>
      </div>`;
    }

    function renderArenaEventDetail(event) {
      if (event.event === 'auto_decision' && event.decision) {
        const decision = event.decision;
        const label = formatDecisionLabel(decision, event.battle || {});
        return `<div class="chips" style="margin:6px 0">
          <span class="chip good">${escapeHtml(label)}</span>
          <span class="chip">${escapeHtml(decision.reason || '')}</span>
          <span class="chip">${event.sent ? 'отправлено' : 'не отправлено'}</span>
        </div>`;
      }
      if (event.event === 'battle_state' && event.data) {
        return `<div class="chips" style="margin:6px 0">
          <span class="chip">opcode ${escapeHtml(pick(event.opcode, ''))}</span>
          <span class="chip">ход ${escapeHtml(pick(event.data.turns, ''))}</span>
          <span class="chip">${escapeHtml(event.data.next_turn || '')}</span>
        </div>`;
      }
      return '';
    }

    function renderArenaState(data) {
      const root = document.getElementById('arenaState');
      if (!data || !data.ok) {
        root.innerHTML = `<div class="arena-console"><div class="arena-scene">${renderSearchScene('Арена недоступна', data && data.error ? data.error : 'Состояние арены ещё не загружено')}</div></div>`;
        return;
      }
      const validation = data.validation || {};
      const team = data.team || [];
      root.innerHTML = `
        <div class="arena-console">
          <div class="arena-header">
            <div><strong>${escapeHtml(modeLabel(data.mode || ''))}</strong><div class="muted">${validation.ok ? 'можно искать бой' : 'заблокировано условиями'}</div></div>
            <div class="chips"><span class="chip ${validation.ok ? 'good' : 'warn'}">${validation.ok ? 'готово' : 'заблокировано'}</span><span class="chip">рейтинг команды ${pick(data.team_rating, '-')}</span></div>
          </div>
          <div class="arena-scene">
            ${team.length ? `<div class="battle-team">${team.map(item => renderBattleMiscrit({...item, active:false})).join('')}</div>` : renderSearchScene('Команда пуста', 'Проверка условий не вернула данные команды')}
            ${renderArenaRewards(data.reward_progress)}
            ${(data.rules || []).length ? `<details class="collapsible" style="margin-top:14px"><summary>Условия</summary><div class="collapsible-body"><div class="chips">${data.rules.map(r => `<span class="chip">${escapeHtml(r)}</span>`).join('')}</div></div></details>` : ''}
          </div>
        </div>`;
    }

    function renderArenaLive(job) {
      const root = document.getElementById('arenaState');
      root.classList.remove('muted');
      const progress = job.progress || {};
      const result = job.result || {};
      const battle = progress.battle || {};
      const draft = progress.draft || {};
      const outcome = progress.outcome || (result.battle && result.battle.outcome) || '';
      const grade = progress.grade || (result.battle && result.battle.grade) || {};
      const loop = progress.loop || result.loop || {};
      const summary = loop.summary || {};
      const target = Number(pick(pick(loop.target, loop.requested), pick(job.repeat && job.repeat.count, 1)));
      const current = Number(pick(loop.current, 1));
      const completed = Number(pick(pick(loop.completed, summary.total), (job.status !== 'running' && target === 1 ? 1 : 0)));
      const stage = arenaStage(job, progress, draft, battle, result, outcome);
      const rewardProgress = pick(
        progress.reward_progress,
        pick(progress.status && progress.status.reward_progress, result.status && result.status.reward_progress)
      );
      root.innerHTML = `
        <div class="arena-console">
          <div class="arena-header">
            <div><strong>${escapeHtml(modeLabel(job.mode || ''))}</strong><div class="muted">${escapeHtml(progress.event || job.status || '')}</div></div>
            <div class="chips">
              <span class="chip ${job.status === 'running' ? 'warn' : 'good'}">${escapeHtml(job.status === 'running' ? 'выполняется' : job.status || '')}</span>
              ${job.source === 'plan' ? '<span class="chip">план</span>' : ''}
              ${job.stop_requested ? '<span class="chip warn">запрошена остановка</span>' : ''}
              <span class="chip">${target === 0 ? `бой ${current} / ∞` : `бой ${current} / ${target || 1}`}</span>
              <span class="chip">завершено ${completed}</span>
              ${outcome ? `<span class="chip ${outcome === 'victory' ? 'good' : 'warn'}">${escapeHtml(outcomeLabel(outcome))}</span>` : ''}
              ${grade.label ? `<span class="chip good">${escapeHtml(grade.label)} / ${escapeHtml(grade.score)}</span>` : ''}
            </div>
          </div>
          ${renderStageTrack(stage)}
          <div class="arena-scene">${renderArenaScene(stage, {job, progress, result, battle, draft, outcome, grade, summary, current, target, completed})}</div>
          ${renderArenaRewards(rewardProgress)}
          ${job.error ? `<div class="log-card"><strong>Ошибка</strong><pre>${escapeHtml(job.error)}</pre></div>` : ''}
          ${renderArenaEvents(job.events || [])}
        </div>`;
    }

    function renderArenaRewards(progress) {
      const items = progress && Array.isArray(progress.items) ? progress.items : [];
      if (!items.length) return '';
      const totals = progress.totals || {};
      const claimed = progress.claimed_totals || {};
      const remaining = progress.remaining_totals || {};
      const currencies = Object.keys(totals);
      return `<section class="arena-rewards">
        <div class="log-head">
          <strong>Награды</strong>
          <div class="chips">
            <span class="chip">${escapeHtml(pick(progress.progress, 0))} / ${escapeHtml(pick(progress.target, 0))} побед</span>
          </div>
        </div>
        <div class="arena-reward-summary">
          ${currencies.map(currency => renderArenaRewardTotal(currency, claimed[currency] || 0, totals[currency] || 0, remaining[currency] || 0)).join('')}
        </div>
        <div class="arena-reward-list">
          ${items.map(renderArenaRewardItem).join('')}
        </div>
      </section>`;
    }

    function renderArenaRewardTotal(currency, claimed, total, remaining) {
      return `<div class="arena-reward-total ${escapeHtml(currency)}">
        <span class="muted">${escapeHtml(arenaRewardLabel(currency))}</span>
        <strong>${escapeHtml(claimed)} / ${escapeHtml(total)}</strong>
        <span class="muted">осталось ${escapeHtml(remaining)}</span>
      </div>`;
    }

    function renderArenaRewardItem(item = {}) {
      const currency = String(item.currency || '').toLowerCase();
      const name = currency === 'miscrit'
        ? (item.name || ('Мискрит #' + pick(item.mid, '?')))
        : `${item.amount || 0} ${arenaRewardLabel(currency)}`;
      const detail = item.claimed
        ? 'получено'
        : `осталось побед: ${pick(item.remaining_wins, 0)}`;
      const icon = arenaRewardIcon(item);
      return `<div class="arena-reward-item ${item.claimed ? 'claimed' : ''}">
        <div class="arena-reward-icon">${icon}</div>
        <div>
          <div class="log-head"><strong>${escapeHtml(name)}</strong><span class="chip ${item.claimed ? 'good' : ''}">${item.claimed ? 'получено' : 'заблокировано'}</span></div>
          <div class="muted">${escapeHtml(pick(item.threshold, '?'))} побед / ${escapeHtml(detail)}</div>
        </div>
      </div>`;
    }

    function arenaRewardIcon(item = {}) {
      const currency = String(item.currency || '').toLowerCase();
      if (currency === 'miscrit') {
        const img = miscritAssetUrl({mid:item.mid, evo:0}, 'avatars');
        return img ? `<img src="${escapeHtml(img)}" alt="">` : 'M';
      }
      if (currency === 'platinum') return 'P';
      if (currency === 'gems') return 'G';
      if (currency === 'gold') return '$';
      return escapeHtml(String(currency || '?').slice(0, 1).toUpperCase());
    }

    function arenaRewardLabel(currency) {
      const labels = {
        platinum: 'Платина',
        gems: 'Гемы',
        gold: 'Золото',
        miscrit: 'Мискриты',
        potion: 'Зелья',
        common_pack: 'Обычные наборы'
      };
      return labels[String(currency || '').toLowerCase()] || String(currency || 'Награда');
    }

    function arenaStage(job, progress, draft, battle, result, outcome) {
      const phase = String(job.phase || progress.phase || '').toLowerCase();
      if (phase === 'error' || phase === 'stopped') return 'finished';
      if (job.status === 'running' && battle && (battle.player || battle.foe)) return 'battle';
      if (job.status === 'running' && draft && (draft.pool || draft.bans || draft.picks || draft.prepare)) return 'draft';
      if (outcome || phase === 'finished' || String(progress.event || '').includes('complete')) return 'finished';
      if (battle && (battle.player || battle.foe)) return 'battle';
      if (draft && (draft.pool || draft.bans || draft.picks || draft.prepare)) return 'draft';
      return 'searching';
    }

    function renderArenaScene(stage, ctx) {
      if (stage === 'draft') return renderDraft(ctx.draft || {});
      if (stage === 'battle') return renderBattle(ctx.battle || {});
      if (stage === 'finished') return renderArenaResult(ctx);
      const label = ctx.job && ctx.job.phase === 'cooldown' ? 'Подготовка следующего боя' : 'Поиск игры';
      const detail = ctx.target === 0 ? `бой ${ctx.current} / ∞` : `бой ${ctx.current} / ${ctx.target || 1}`;
      return renderSearchScene(label, detail);
    }

    function renderStageTrack(phase) {
      const normalized = ({starting:'searching', preparing:'searching', connecting:'searching', ready:'searching', cooldown:'searching', stopped:'finished', error:'finished'})[phase] || phase;
      const stages = [
        ['searching', 'Поиск'],
        ['draft', 'Драфт'],
        ['battle', 'Бой'],
        ['finished', 'Итог']
      ];
      const activeIndex = Math.max(0, stages.findIndex(([key]) => key === normalized));
      return `<div class="stage-track" style="margin-top:12px">${stages.map(([key, label], index) => {
        const cls = index < activeIndex ? 'done' : index === activeIndex ? 'active' : '';
        return `<div class="stage ${cls}">${label}</div>`;
      }).join('')}</div>`;
    }

    function renderSearchScene(title, detail) {
      return `<div class="search-scene">
        <div>
          <div class="search-core">МАТЧ</div>
          <div style="margin-top:34px"><strong style="font-size:24px">${escapeHtml(title)}</strong></div>
          <div class="muted" style="margin-top:8px">${escapeHtml(detail || '')}</div>
        </div>
      </div>`;
    }

    function renderDraft(draft) {
      const pool = (draft.pool && draft.pool.length ? draft.pool : [...(draft.bans || []), ...(draft.picks || [])]).slice(0, 24);
      const bans = new Set((draft.bans || []).map(draftKey));
      const picks = new Set((draft.picks || []).map(draftKey));
      return `<div class="draft-board">
        <div class="log-head">
          <div><strong>Случайный драфт</strong><div class="muted">автобан / автопик</div></div>
          <div class="chips"><span class="chip warn">баны ${(draft.bans || []).length}</span><span class="chip good">пики ${(draft.picks || []).length}</span></div>
        </div>
        <div class="draft-card-grid">
          ${pool.length ? pool.map(item => renderDraftCard(item, bans.has(draftKey(item)), picks.has(draftKey(item)))).join('') : '<div class="muted">Жду пул драфта</div>'}
        </div>
        ${draft.prepare ? `<details class="collapsible"><summary>Подготовка драфта</summary><div class="collapsible-body"><pre>${escapeHtml(JSON.stringify(draft.prepare, null, 2))}</pre></div></details>` : ''}
      </div>`;
    }

    function draftKey(item = {}) {
      return String(pick(pick(item.mid, item.id), item.name) || '');
    }

    function renderDraftCard(item, banned, picked) {
      const name = item.name || ('#' + (item.mid || item.id || '?'));
      const img = item.image || item.icon || item.avatar || item.image_url || miscritAssetUrl(item, 'avatars');
      const rarity = rarityClass(item.rarity || item.rank_rarity || item.card_rarity || '');
      return `<div class="draft-card ${rarity} ${banned ? 'banned' : ''} ${picked ? 'picked' : ''}">
        <div class="draft-icon">${img ? `<img src="${escapeHtml(img)}" alt="">` : escapeHtml(String(name).slice(0, 2).toUpperCase())}</div>
        <strong>${escapeHtml(name)}</strong>
        <div class="chips">
          <span class="chip">MID ${escapeHtml(pick(pick(item.mid, item.id), '-'))}</span>
          ${item.element ? `<span class="chip">${elementBadge(item.element)}</span>` : ''}
        </div>
      </div>`;
    }

    function renderBattle(battle) {
      const activeText = battle.active_side === 'player' ? 'Твой ход' : battle.active_side === 'foe' ? 'Ход соперника' : 'Ожидание';
      return `<div>
        <div class="log-head">
          <strong>Бой</strong>
          <div class="chips"><span class="chip">${activeText}</span><span class="chip">ход ${pick(battle.turns, '-')}</span></div>
        </div>
        <div class="battle-board" style="margin-top:10px">
          ${renderBattleSide('Твоя команда', battle.player || {})}
          ${renderBattleSide('Соперник', battle.foe || {})}
        </div>
      </div>`;
    }

    function renderBattleSide(title, side, sideKey = '') {
      const team = side.team || [];
      if (!sideKey) sideKey = String(title || '').includes('Соп') || String(title || '').includes('РЎ') ? 'foe' : 'player';
      return `<div class="battle-side ${escapeHtml(sideKey)}">
        <div class="log-head"><strong>${escapeHtml(title)}</strong><span class="chip">${escapeHtml(side.username || '')}</span></div>
        <div class="battle-team">${team.length ? team.map(item => renderBattleMiscrit(item, sideKey)).join('') : '<span class="muted">Данные команды ещё не получены</span>'}</div>
      </div>`;
    }

    function renderBattleMiscrit(m, sideKey = '') {
      const maxHp = Number(m.max_hp || 0);
      const chp = Number(m.chp || 0);
      const pct = maxHp > 0 ? Math.max(0, Math.min(100, Math.round(chp / maxHp * 100))) : (m.dead ? 0 : 100);
      const name = m.name || ('#' + (m.mid || m.id || '?'));
      const icon = name.replace(/^#/, '').slice(0, 2).toUpperCase();
      const ids = [`mid ${escapeHtml(m.mid || '?')}`];
      const battleId = pick(m.battle_id, m.id);
      if (battleId !== undefined && battleId !== null && Number(battleId) !== Number(m.mid || 0)) ids.push(`battle id ${escapeHtml(battleId)}`);
      const displayId = m.mid || m.id;
      const img = m.image || m.icon || m.avatar || m.image_url || miscritAssetUrl(m, 'avatars');
      const rarity = rarityClass(m.rarity || m.rank_rarity || m.card_rarity || '');
      const damaged = miscritDamagedClass(m, sideKey, chp);
      return `<div class="battle-card ${rarity} ${m.active ? 'active' : ''} ${m.dead ? 'dead' : ''} ${damaged}">
        <div class="battle-avatar">${img ? `<img src="${escapeHtml(img)}" alt="">` : escapeHtml(icon)}</div>
        <div>
          <div class="log-head"><strong>${escapeHtml(name)}</strong><span class="chip">${m.active ? 'активен' : 'запас'}</span></div>
          <div class="muted">${ids.join(' / ')}</div>
          <div class="muted">#${escapeHtml(displayId || '?')} / ${elementBadge(m.element)} / ур. ${escapeHtml(pick(m.level, '-'))}</div>
          <div class="hpbar"><div class="hpfill ${pct <= 30 ? 'low' : ''}" style="width:${pct}%"></div></div>
          <div class="muted">${chp}/${maxHp || '?'} HP</div>
          ${renderStatusEffects(m)}
        </div>
      </div>`;
    }

    function miscritAssetUrl(m = {}, type = 'avatars') {
      const mid = Number(pick(pick(m.mid, m.mId), m.m) || 0);
      if (!mid) return '';
      const explicitEvo = pick(pick(m.evo, m.evo_id), m.evolution);
      const level = Number(pick(pick(m.level, m.l), 1) || 1);
      const evoRaw = explicitEvo === undefined || explicitEvo === null || explicitEvo === '' ? Math.floor(level / 10) : Number(explicitEvo);
      const evo = Math.max(0, Math.min(3, Number.isFinite(evoRaw) ? evoRaw : 0));
      return `/api/miscrit-asset?mid=${encodeURIComponent(mid)}&evo=${encodeURIComponent(evo)}&type=${encodeURIComponent(type)}`;
    }

    function miscritDamagedClass(m, sideKey, chp) {
      if (!sideKey) return '';
      const key = `${sideKey}:${pick(pick(pick(m.battle_id, m.id), m.mid), pick(m.name, '?'))}`;
      const previous = battleHpMemory.get(key);
      battleHpMemory.set(key, Number(chp || 0));
      return previous !== undefined && Number(chp || 0) < Number(previous || 0) ? 'damaged' : '';
    }

    function renderStatusEffects(m = {}) {
      const statuses = normalizeStatuses(pick(pick(m.statuses, m.status), pick(m.effects, [])));
      if (!statuses.length) return '';
      return `<div class="status-row">${statuses.slice(0, 6).map(item => {
        const key = String(item.key || item.name || item.type || item || '');
        const turns = item.turns || item.duration || '';
        return `<span class="status-badge ${statusBadgeClass(key)}">${escapeHtml(shortStatus(key))}${turns ? ` ${escapeHtml(turns)}` : ''}</span>`;
      }).join('')}</div>`;
    }

    function normalizeStatuses(raw) {
      if (Array.isArray(raw)) return raw.filter(Boolean);
      if (raw && typeof raw === 'object') {
        return Object.entries(raw).filter(([, value]) => Boolean(value)).map(([key, value]) => ({ key, turns: value }));
      }
      return raw ? [{ key: String(raw), turns: '' }] : [];
    }

    function shortStatus(key) {
      const value = String(key || '').replace(/_/g, ' ');
      const map = { sleep: 'сон', paralyze: 'парал', confuse: 'смят', poison: 'яд', burn: 'ожог', bleed: 'кровь', disease: 'болезнь', dot: 'дот', switchcurse: 'прокл', stun: 'оглуш', block: 'блок', negate: 'негация', buff: 'баф', debuff: 'дебаф' };
      const lower = value.toLowerCase();
      return map[lower] || value.slice(0, 10);
    }

    function statusBadgeClass(key) {
      const lower = String(key || '').toLowerCase();
      if (/(sleep|stun|freeze|blind|confuse|paralyze)/.test(lower)) return 'control';
      if (/(poison|burn|bleed|disease|switchcurse|debuff|slow|weak)/.test(lower)) return 'bad';
      if (/(buff|block|negate|immune|regen|heal)/.test(lower)) return 'good';
      return '';
    }

    function renderBattle(battle) {
      const activeText = battle.active_side === 'player' ? 'Твой ход' : battle.active_side === 'foe' ? 'Ход соперника' : 'Ожидание';
      return `<div>
        <div class="log-head">
          <strong>Бой</strong>
          <div class="chips"><span class="chip">${activeText}</span><span class="chip">ход ${pick(battle.turns, '-')}</span></div>
        </div>
        <div class="battle-board" style="margin-top:10px">
          ${renderBattleSide('Твоя команда', battle.player || {}, 'player', battle.active_side)}
          ${renderBattleSide('Соперник', battle.foe || {}, 'foe', battle.active_side)}
        </div>
      </div>`;
    }

    function renderBattleSide(title, side, sideKey = '', activeSide = '') {
      sideKey = sideKey || 'player';
      const team = orderedBattleTeam(side.team || [], sideKey);
      const turn = activeSide === sideKey ? 'turn' : '';
      return `<div class="battle-side ${escapeHtml(sideKey)} ${turn}">
        <div class="log-head"><strong>${escapeHtml(title)}</strong><span class="chip">${escapeHtml(side.username || '')}</span></div>
        <div class="battle-team">${team.length ? team.map(item => renderBattleMiscrit(item, sideKey)).join('') : '<span class="muted">Данные команды ещё не получены</span>'}</div>
      </div>`;
    }

    function orderedBattleTeam(team, sideKey) {
      if (!Array.isArray(team) || !team.length) return [];
      const keys = team.map(stableMiscritKey);
      let order = battleSlotMemory.get(sideKey) || [];
      const sameRoster = order.length === keys.length && keys.every(key => order.includes(key));
      if (!sameRoster) {
        const retained = order.filter(key => keys.includes(key));
        order = retained.length >= Math.min(2, keys.length)
          ? [...retained, ...keys.filter(key => !retained.includes(key))]
          : keys;
        battleSlotMemory.set(sideKey, order);
      }
      return [...team].sort((a, b) => order.indexOf(stableMiscritKey(a)) - order.indexOf(stableMiscritKey(b)));
    }

    function stableMiscritKey(m = {}) {
      return String(pick(pick(pick(m.battle_id, m.id), m.mid), pick(m.name, '?')));
    }

    function renderArenaResult(ctx) {
      const battleResult = (ctx.result && ctx.result.battle) || {};
      const outcome = ctx.outcome || battleResult.outcome || '';
      const grade = ctx.grade && Object.keys(ctx.grade).length ? ctx.grade : (battleResult.grade || {});
      const label = grade.label || (outcome === 'victory' ? 'Победа' : outcome === 'defeat' ? 'Поражение' : 'Итог боя');
      const score = pick(grade.score, '');
      const summary = ctx.summary || {};
      return `<div class="result-scene">
        <div>
          <div class="result-title ${outcome === 'victory' ? 'victory' : outcome === 'defeat' ? 'defeat' : ''}">${escapeHtml(label)}</div>
          <div class="chips" style="justify-content:center;margin-top:18px">
            ${outcome ? `<span class="chip ${outcome === 'victory' ? 'good' : 'warn'}">${escapeHtml(outcomeLabel(outcome))}</span>` : ''}
            ${score !== '' ? `<span class="chip">оценка ${escapeHtml(score)}</span>` : ''}
            ${summary.total ? `<span class="chip good">победы ${escapeHtml(summary.victories || 0)}</span><span class="chip warn">поражения ${escapeHtml(summary.defeats || 0)}</span><span class="chip">винрейт ${Math.round(Number(summary.win_rate || 0) * 100)}%</span>` : ''}
          </div>
        </div>
      </div>`;
    }

    function renderArenaEvents(events) {
      const recent = events.slice(-14).reverse();
      if (!recent.length) return '';
      return `<details class="collapsible">
        <summary>Лог боя <span class="chip">${events.length} событий</span></summary>
        <div class="collapsible-body logs" style="margin-top:10px; max-height:300px">${recent.map(event => `
          <div class="parent">
            <div class="log-head"><strong>${escapeHtml(event.event || '')}</strong><span class="chip">${escapeHtml(event.phase || '')}</span></div>
            ${renderArenaEventDetail(event)}
            <div class="muted">${event.timestamp ? new Date(event.timestamp * 1000).toLocaleTimeString() : ''}</div>
          </div>`).join('')}</div>
      </details>`;
    }

    function formatDecisionLabel(decision, battle) {
      const type = decision.type || 'action';
      const id = Number(decision.id || 0);
      if (type === 'ability') {
        const name = decision.debug && decision.debug.name ? decision.debug.name : '';
        return name ? `способность: ${name}` : `id способности ${id}`;
      }
      if (type === 'switch') {
        const team = battle && battle.player && Array.isArray(battle.player.team) ? battle.player.team : [];
        const target = team.find(m => Number(pick(m.battle_id, m.id)) === id);
        if (target) return `смена: ${target.name || ('mid ' + target.mid)} (${id})`;
        return `id смены в бою ${id}`;
      }
      return `${type} ${id || ''}`.trim();
    }

    function renderBattleLearning(data) {
      renderArenaWinrates(data);
      const root = document.getElementById('battleLearning');
      const recent = data.recent_battles || [];
      const actions = data.actions || [];
      const pairMatchups = data.pair_matchups || [];
      const opponentActions = data.opponent_actions || [];
      const opponentMatchups = data.opponent_matchups || [];
      const damage = data.damage_model || {};
      const damageBuckets = damage.top_buckets || [];
      root.classList.remove('muted');
      root.innerHTML = `
        <div class="log-card">
          <div class="log-head"><strong>Веса</strong><span class="chip">${data.battles || 0} боёв</span></div>
          <div class="chips" style="margin-top:8px">${actions.length ? actions.slice(0, 10).map(item => `<span class="chip">${escapeHtml(item.key)} ${Number(item.weight || 0).toFixed(2)}</span>`).join('') : '<span class="muted">Выученных весов ещё нет</span>'}</div>
        </div>
        <div class="log-card">
          <div class="log-head"><strong>Память matchup-ов</strong><span class="chip">${pairMatchups.length} пар</span></div>
          <div class="chips" style="margin-top:8px">${pairMatchups.length ? pairMatchups.slice(0, 12).map(item => `<span class="chip">${escapeHtml(item.key)} ${Number(item.weight || 0).toFixed(2)}</span>`).join('') : '<span class="muted">Память matchup-ов ещё пуста</span>'}</div>
        </div>
        <div class="log-card">
          <div class="log-head"><strong>Имитация соперника</strong><span class="chip">${opponentActions.length} действий</span></div>
          <div class="chips" style="margin-top:8px">${opponentActions.length ? opponentActions.slice(0, 10).map(item => `<span class="chip">${escapeHtml(item.key)} ${Number(item.weight || 0).toFixed(2)}</span>`).join('') : '<span class="muted">Паттерны соперника ещё не выучены</span>'}</div>
          <div class="chips" style="margin-top:8px">${opponentMatchups.length ? opponentMatchups.slice(0, 8).map(item => `<span class="chip">${escapeHtml(item.key)} ${Number(item.weight || 0).toFixed(2)}</span>`).join('') : ''}</div>
        </div>
        <div class="log-card">
          <div class="log-head"><strong>Формула урона</strong><span class="chip">${escapeHtml(damage.samples || 0)} примеров</span></div>
          <div class="chips" style="margin-top:8px">
            <span class="chip">глобально x${Number((damage.global && damage.global.scale) || 1).toFixed(3)}</span>
            <span class="chip">MAE ${Number(damage.mae || 0).toFixed(1)}</span>
            <span class="chip">MAPE ${Math.round(Number(damage.mape || 0) * 100)}%</span>
          </div>
          <div class="chips" style="margin-top:8px">${damageBuckets.length ? damageBuckets.slice(0, 12).map(item => `<span class="chip">${escapeHtml(item.key)} x${Number(item.scale || 1).toFixed(2)} (${escapeHtml(item.count || 0)})</span>`).join('') : '<span class="muted">Примеров урона ещё нет</span>'}</div>
        </div>
        ${recent.slice(0, 8).map(battle => `
          <div class="log-card">
            <div class="log-head">
              <strong>${escapeHtml(battle.grade && battle.grade.label || battle.outcome || 'battle')}</strong>
              <div class="chips"><span class="chip">${escapeHtml(modeLabel(battle.mode || ''))}</span><span class="chip">оценка ${escapeHtml(pick(battle.grade && battle.grade.score, '-'))}</span></div>
            </div>
            <div class="chips" style="margin-top:8px">
              <span class="chip">${escapeHtml(outcomeLabel(battle.outcome || ''))}</span>
              <span class="chip">ходы ${escapeHtml(pick(battle.turns_sent, 0))}</span>
              <span class="chip">погибло союзников ${escapeHtml(pick(battle.grade && battle.grade.ally_deaths, '-'))}</span>
              <span class="chip">hp ${Math.round(Number(battle.grade && battle.grade.player_hp_retained || 0) * 100)}%</span>
            </div>
            ${renderPostBattle(battle.post_battle || {})}
          </div>`).join('')}`;
    }

    function renderAiDashboard(data = {}) {
      const root = document.getElementById('aiDashboard');
      if (!root) return;
      const recent = data.recent_series || [];
      const post = data.post_metrics || {};
      const logic = data.logic || {};
      const reasons = data.decision_reasons || [];
      const actions = data.decision_actions || [];
      const damage = data.damage_model || {};
      const editable = data.editable_weights || {};
      root.classList.remove('muted');
      root.innerHTML = `
        <div class="ai-metrics">
          <div class="ai-metric"><span class="muted">Изучено боёв</span><strong>${escapeHtml(data.battles || 0)}</strong></div>
          <div class="ai-metric"><span class="muted">Всего в истории</span><strong>${escapeHtml(data.history_total || 0)}</strong></div>
          <div class="ai-metric"><span class="muted">Bucket-ы игрока</span><strong>${escapeHtml(logic.player_learning_buckets || 0)}</strong></div>
          <div class="ai-metric"><span class="muted">Bucket-ы соперника</span><strong>${escapeHtml(logic.opponent_learning_buckets || 0)}</strong></div>
          <div class="ai-metric"><span class="muted">Примеры урона</span><strong>${escapeHtml(damage.samples || 0)}</strong></div>
          <div class="ai-metric"><span class="muted">Ошибка урона MAPE</span><strong>${Math.round(Number(damage.mape || 0) * 100)}%</strong></div>
        </div>
        <div class="grid-2">
          <div class="log-card ai-chart">
            <div class="log-head"><strong>Качество последних боёв</strong><span class="chip">${recent.length}</span></div>
            ${renderAiSparkline(recent)}
            <div class="ai-summary-metrics">
              <div class="ai-summary-metric">
                <span class="muted">Упущено добиваний</span>
                <strong>${escapeHtml(post.missed_finishes || 0)}</strong>
              </div>
              <div class="ai-summary-metric">
                <span class="muted">Бесполезные смены</span>
                <strong>${escapeHtml(post.useless_switches || 0)}</strong>
              </div>
              <div class="ai-summary-metric">
                <span class="muted">Потеряно урона</span>
                <strong>${escapeHtml(Number(post.lost_damage || 0).toFixed(1))}</strong>
              </div>
            </div>
          </div>
          <div class="log-card ai-chart">
            <div class="log-head"><strong>Причины решений</strong><span class="chip">${reasons.length}</span></div>
            ${renderAiBars(reasons, 'count', false, aiReasonLabel)}
          </div>
        </div>
        <div class="grid-2">
          <div class="log-card ai-chart">
            <div class="log-head"><strong>Использование действий</strong><span class="chip">${actions.length}</span></div>
            ${renderAiBars(actions, 'count', false, aiActionLabel)}
          </div>
          <div class="log-card ai-chart">
            <div class="log-head"><strong>Модель урона</strong><span class="chip">глобально x${Number((damage.global && damage.global.scale) || 1).toFixed(3)}</span></div>
            ${renderAiBars((damage.top_buckets || []).slice(0, 12), 'scale', true)}
          </div>
        </div>
        <div class="log-card">
          <div class="log-head"><strong>Редактируемые веса</strong><span class="chip">ручные правки применяются сразу</span></div>
          <div class="ai-weight-grid" style="margin-top:12px">
            ${renderAiWeightGroups(editable)}
          </div>
        </div>`;
    }

    function renderAiSparkline(items = []) {
      if (!items.length) return '<span class="muted">Боёв ещё нет</span>';
      return `<div class="ai-sparkline">${items.map(item => {
        const score = Math.max(0, Math.min(100, Number(item.score || 0)));
        const height = Math.max(6, score);
        return `<div class="ai-spark ${escapeHtml(item.outcome || '')}" title="${escapeHtml(modeLabel(item.mode || ''))} ${escapeHtml(outcomeLabel(item.outcome || ''))} / ${score.toFixed(1)}" style="height:${height}%"></div>`;
      }).join('')}</div>`;
    }

    function renderAiBars(items = [], valueKey = 'count', centered = false, labeler = value => value) {
      if (!items.length) return '<span class="muted">Данных ещё нет</span>';
      const values = items.map(item => Math.abs(Number(item[valueKey] || 0)));
      const max = Math.max(1, ...values);
      return `<div class="ai-bars">${items.slice(0, 12).map(item => {
        const raw = Number(item[valueKey] || 0);
        const pct = Math.max(2, Math.round(Math.abs(raw) / max * 100));
        const cls = centered ? (raw >= 1 ? 'good' : 'warn') : '';
        const shown = valueKey === 'scale' ? `x${raw.toFixed(2)}` : String(raw);
        return `<div class="ai-bar">
          <span class="ai-weight-key">${escapeHtml(labeler(item.key || ''))}</span>
          <div class="ai-bar-track"><div class="ai-bar-fill ${cls}" style="width:${pct}%"></div></div>
          <span>${escapeHtml(shown)}</span>
        </div>`;
      }).join('')}</div>`;
    }

    function renderAiWeightGroups(groups = {}) {
      const order = [
        ['actions', 'Веса собственных действий'],
        ['reasons', 'Веса причин решений'],
        ['matchups', 'Веса собственных matchup-ов'],
        ['pair_matchups', 'Память пар matchup-ов'],
        ['opponent_actions', 'Веса действий соперника'],
        ['opponent_matchups', 'Веса matchup-ов соперника'],
        ['opponent_pair_matchups', 'Память пар соперника'],
      ];
      return order.map(([key, label]) => {
        const rows = groups[key] || [];
        return `<details class="collapsible">
          <summary>${escapeHtml(label)} <span class="chip">${rows.length}</span></summary>
          <div class="collapsible-body ai-weight-table">
            ${rows.length ? rows.slice(0, 80).map((item, index) => renderAiWeightRow(key, item, index)).join('') : '<span class="muted">Выученных весов ещё нет</span>'}
          </div>
        </details>`;
      }).join('');
    }

    function renderAiWeightRow(category, item, index) {
      const inputId = `aiWeight_${category}_${index}`;
      return `<div class="ai-weight-row">
        <span class="ai-weight-key" title="${escapeHtml(item.key || '')}">${escapeHtml(aiWeightKeyLabel(category, item.key || ''))}</span>
        <span class="chip">n ${escapeHtml(item.count || 0)}</span>
        <input id="${escapeHtml(inputId)}" type="number" min="-24" max="24" step="0.05" value="${Number(item.weight || 0).toFixed(3)}">
        <button class="ghost" onclick="saveAiWeight(${quoteJs(category)}, ${quoteJs(item.key || '')}, ${quoteJs(inputId)})">Сохранить</button>
      </div>`;
    }

    function renderArenaWinrates(data = {}) {
      const root = document.getElementById('arenaGaugeGrid');
      if (!root) return;
      const battles = data.recent_battles || [];
      const arenaStats = data.arena_stats || {};
      const modes = [
        ['total', 'Все арены'],
        ['battle', 'Обычная'],
        ['daily', 'Ежедневная'],
        ['platinum', 'Платиновая'],
        ['random', 'Случайная'],
      ];
      const stats = modes.map(([key, label]) => {
        const aggregate = arenaStats[key] || {};
        if (aggregate.battles !== undefined) {
          const wins = Number(aggregate.wins || 0);
          const losses = Number(aggregate.losses || 0);
          const total = wins + losses;
          const percent = total ? Math.round(wins / total * 100) : Math.round(Number(aggregate.win_rate || 0) * 100);
          return { key, label, wins, losses, total, percent, battles: Number(aggregate.battles || 0), unknown: Number(aggregate.unknown || 0) };
        }
        const scoped = key === 'total' ? battles : battles.filter(item => String(item.mode || '').toLowerCase() === key);
        const wins = scoped.filter(item => item.outcome === 'victory').length;
        const losses = scoped.filter(item => item.outcome === 'defeat').length;
        const total = wins + losses;
        const percent = total ? Math.round(wins / total * 100) : 0;
        return { key, label, wins, losses, total, percent, battles: scoped.length, unknown: Math.max(0, scoped.length - total) };
      });
      root.innerHTML = stats.map(item => renderWinrateGauge(item)).join('');
    }

    function renderWinrateGauge(item) {
      const win = Math.max(0, Math.min(100, Number(item.percent || 0)));
      const loss = Math.max(0, 100 - win);
      const lossDeg = item.total ? (loss * 1.8).toFixed(1) : '0';
      const totalDeg = item.total ? '180' : '0';
      return `<div class="gauge-card">
        <strong>${escapeHtml(item.label)}</strong>
        <div class="gauge-arc" style="--win:${win};--loss:${loss};--loss-deg:${lossDeg}deg;--total-deg:${totalDeg}deg">
          <div class="gauge-value">${win}%</div>
        </div>
        <div class="chips" style="justify-content:center">
          <span class="chip good">П ${escapeHtml(item.wins)}</span>
          <span class="chip warn">ПР ${escapeHtml(item.losses)}</span>
          <span class="chip">всего ${escapeHtml(item.battles || item.total || 0)}</span>
        </div>
      </div>`;
    }

    function renderPostBattle(post) {
      const summary = post.summary || {};
      if (!post || !Object.keys(summary).length) return '';
      const losing = post.suspected_losing_move || {};
      return `<div class="chips" style="margin-top:8px">
        <span class="chip ${post.severity === 'ok' ? 'good' : 'warn'}">${escapeHtml(post.severity || 'разбор')}</span>
        <span class="chip">упущено добиваний ${escapeHtml(summary.missed_finishes || 0)}</span>
        <span class="chip">плохие смены ${escapeHtml(summary.useless_switches || 0)}</span>
        <span class="chip">потеряно урона ${Number(summary.lost_damage || 0).toFixed(1)}</span>
        ${losing.turns !== undefined ? `<span class="chip warn">проигрышный ход ${escapeHtml(losing.turns)}</span>` : ''}
      </div>`;
    }

    function statLine(stats = {}) {
      return `<div class="stat-line">${['hp','spd','ea','pa','ed','pd'].map(k => `<span class="stat v${stats[k] || 1}">${k.toUpperCase()} ${stats[k] || '-'}</span>`).join('')}</div>`;
    }

    function formatTime(ts) {
      if (!ts) return '';
      return new Date(ts * 1000).toLocaleString();
    }

    function escapeHtml(value) {
      return String(pick(value, '')).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    }

    function quoteJs(value) {
      return JSON.stringify(String(pick(value, ''))).replace(/"/g, '&quot;');
    }

    initApp();
  </script>
</body>
</html>
"""
