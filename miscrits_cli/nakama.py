from __future__ import annotations

import base64
import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ClientConfig, DEFAULT_CONFIG
from .storage import clear_session, load_session, save_session


class MiscritsError(RuntimeError):
    pass


SAFE_HTTP_RETRY_ATTEMPTS = 3
SAFE_HTTP_RETRY_DELAY_SECONDS = 0.35


@dataclass
class RpcResult:
    success: bool
    data: Any = None
    code: int = 0
    raw: Any = None


class MiscritsClient:
    def __init__(self, config: ClientConfig = DEFAULT_CONFIG, session_file: Path | None = None, credentials_file: Path | None = None) -> None:
        self.config = config
        self.session_file = session_file
        self.credentials_file = credentials_file
        self.session = load_session(session_file) if session_file else load_session()

    def is_logged_in(self) -> bool:
        return bool(self.session.get("token"))

    def login(self, login_name: str, password: str) -> dict[str, Any]:
        email = login_name if "@" in login_name and "." in login_name else ""
        username = "" if email else login_name
        query = {"create": "false"}
        if username:
            query["username"] = username
        path = "/v2/account/authenticate/email?" + urllib.parse.urlencode(query)
        payload = {"email": email, "password": password}
        response = self._request_json(
            "POST",
            path,
            payload,
            auth=self._basic_auth_header(),
        )
        token = response.get("token")
        refresh_token = response.get("refresh_token")
        if not token:
            raise MiscritsError("Login response did not include a session token.")
        self.session = {
            "token": token,
            "refresh_token": refresh_token,
            "login": login_name,
            "saved_at": int(time.time()),
        }
        self._save_session()
        return self.session

    def login_saved(self) -> dict[str, Any]:
        from .credentials import load_credentials

        credentials = load_credentials(self.credentials_file) if self.credentials_file else load_credentials()
        if not credentials:
            raise MiscritsError("No saved credentials. Run login with --remember first.")
        return self.login(credentials["login"], credentials["password"])

    def logout(self) -> None:
        clear_session(self.session_file) if self.session_file else clear_session()
        self.session = {}

    def refresh(self) -> bool:
        refresh_token = self.session.get("refresh_token")
        if not refresh_token:
            return False
        try:
            response = self._request_json(
                "POST",
                "/v2/account/session/refresh",
                {"token": refresh_token},
                auth=self._basic_auth_header(),
            )
        except MiscritsError as exc:
            if "HTTP 401" in str(exc) or "Refresh token invalid or expired" in str(exc):
                return False
            raise
        token = response.get("token")
        if not token:
            return False
        self.session["token"] = token
        self.session["refresh_token"] = response.get("refresh_token", refresh_token)
        self.session["saved_at"] = int(time.time())
        self._save_session()
        return True

    def _save_session(self) -> None:
        save_session(self.session, self.session_file) if self.session_file else save_session(self.session)

    def ensure_realtime_session(self, force_login: bool = False) -> None:
        if force_login:
            try:
                self.login_saved()
                return
            except Exception:
                if not self.is_logged_in():
                    raise
        if not self.is_logged_in():
            self.login_saved()
            return
        if token_would_expire_in(str(self.session.get("token", "")), 90):
            if self.refresh():
                return
            self.login_saved()

    def ensure_session(self) -> None:
        if not self.is_logged_in():
            self.login_saved()
            return
        if token_would_expire_in(str(self.session.get("token", "")), 90):
            if self.refresh():
                return
            self.login_saved()

    def rpc(self, method: str, payload: dict[str, Any] | None = None) -> RpcResult:
        payload = payload or {}
        try:
            return self._rpc_once(method, payload)
        except MiscritsError as exc:
            if "HTTP 401" in str(exc) and self.refresh():
                return self._rpc_once(method, payload)
            if "HTTP 401" in str(exc):
                try:
                    self.login_saved()
                except Exception:
                    raise
                return self._rpc_once(method, payload)
            raise

    def get_player(self) -> RpcResult:
        result = self.rpc("get_player")
        if result.success and isinstance(result.data, dict):
            from .player_store import save_player_snapshot

            save_player_snapshot(result.data)
        return result

    def heal_team(self) -> RpcResult:
        return self.rpc("heal_team")

    def get_arena(self, kind: str) -> RpcResult:
        clean = str(kind or "").strip().lower()
        if clean not in {"battle", "random", "platinum", "daily"}:
            raise MiscritsError(f"Unknown arena kind: {kind}")
        return self.rpc(f"get_{clean}_arena")

    def update_team(self, team_ids: list[int]) -> RpcResult:
        if not team_ids:
            raise MiscritsError("Team cannot be empty.")
        return self.rpc("update_team", {"team": [int(item) for item in team_ids]})

    def enchant_ability(self, miscrit_id: int, ability_id: int, currency: str = "gold") -> RpcResult:
        return self.rpc(
            "enchant_ability",
            {"abilityId": int(ability_id), "miscritId": int(miscrit_id), "currency": str(currency or "gold")},
        )

    def update_location(self, location_id: int, area_id: int) -> RpcResult:
        return self.rpc("update_location", {"locationId": int(location_id), "areaId": int(area_id)})

    def wish(self, kind: str) -> RpcResult:
        methods = {
            "sk": "wish_sk",
            "vi": "wish_vi",
            "xmas": "wish_xmas",
        }
        if kind not in methods:
            raise MiscritsError(f"Unknown wish kind: {kind}")
        return self.rpc(methods[kind])

    def create_battle(self, battle_type: str, payload: dict[str, Any] | None = None) -> RpcResult:
        return self.rpc("create_battle", {"type": battle_type, "payload": payload or {}})

    def breed(self, miscrit_ids: list[int]) -> RpcResult:
        if len(miscrit_ids) != 3:
            raise MiscritsError("Breed requires exactly 3 miscrit ids.")
        return self.rpc("breed", {"miscrits": [int(item) for item in miscrit_ids]})

    def fetch_cdn_json(self, name: str, version: int | None = None) -> Any:
        suffix = f"?v={version}" if version is not None else ""
        url = f"{self.config.cdn_url.rstrip('/')}/{name}{suffix}"
        body = self._urlopen("GET", url, None, self._cdn_headers())
        return json.loads(body.decode("utf-8"))

    def fetch_cdn_bytes(self, path_or_url: str, accept: str = "*/*") -> bytes:
        if str(path_or_url).startswith(("http://", "https://")):
            url = str(path_or_url)
        else:
            url = f"{self.config.cdn_url.rstrip('/')}/{str(path_or_url).lstrip('/')}"
        headers = self._cdn_headers()
        headers["Accept"] = accept
        return self._urlopen("GET", url, None, headers)

    def request_info(self, path: str = "/v2/account/authenticate/email") -> dict[str, Any]:
        url = f"{self.config.api_url.rstrip('/')}{path}"
        return {
            "url": url,
            "headers": self._base_headers("<auth omitted>"),
            "note": "Login uses POST JSON body and Basic auth with the Nakama server key.",
        }

    def _rpc_once(self, method: str, payload: dict[str, Any]) -> RpcResult:
        self._require_session()
        response = self._request_json(
            "POST",
            f"/v2/rpc/{urllib.parse.quote(method)}",
            json.dumps(payload, separators=(",", ":")),
            auth=f"Bearer {self.session['token']}",
        )
        rpc_payload = response.get("payload", "{}")
        if isinstance(rpc_payload, str):
            parsed = json.loads(rpc_payload or "{}")
        else:
            parsed = rpc_payload
        data = parsed.get("data", {})
        if isinstance(data, str):
            data = json.loads(data or "{}")
        return RpcResult(
            success=bool(parsed.get("success", False)),
            data=data,
            code=int(parsed.get("code", 0) or 0),
            raw=parsed,
        )

    def _require_session(self) -> None:
        login_error: Exception | None = None
        if not self.is_logged_in():
            try:
                self.login_saved()
            except Exception as exc:
                login_error = exc
        if not self.is_logged_in():
            if login_error:
                raise MiscritsError(f"Could not use saved credentials: {login_error}") from login_error
            raise MiscritsError("Not logged in. Run: python -m miscrits_cli login <user> <password>")

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Any,
        auth: str,
        raw_string_body: bool = False,
    ) -> dict[str, Any]:
        url = f"{self.config.api_url.rstrip('/')}{path}"
        body = payload if raw_string_body else json.dumps(payload, separators=(",", ":"))
        data = body.encode("utf-8")
        headers = self._base_headers(auth)
        response_body = self._urlopen(method, url, data, headers)
        if not response_body:
            return {}
        return json.loads(response_body.decode("utf-8"))

    def _urlopen(
        self,
        method: str,
        url: str,
        data: bytes | None,
        headers: dict[str, str],
    ) -> bytes:
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        attempts = SAFE_HTTP_RETRY_ATTEMPTS if method.upper() == "GET" else 1
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                message = exc.read().decode("utf-8", errors="replace")
                raise MiscritsError(f"HTTP {exc.code}: {message}") from exc
            except urllib.error.URLError as exc:
                if attempt < attempts and is_transient_network_error(exc):
                    time.sleep(SAFE_HTTP_RETRY_DELAY_SECONDS * attempt)
                    continue
                raise MiscritsError(network_error_message(exc, retried=attempts > 1)) from exc
        raise MiscritsError("Network error: request failed without a response.")

    def _basic_auth_header(self) -> str:
        token = base64.b64encode(f"{self.config.server_key}:".encode("utf-8")).decode("ascii")
        return f"Basic {token}"

    def _base_headers(self, auth: str) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": auth,
            "User-Agent": self.config.user_agent,
            "X-Godot-Engine": "4.3.stable",
            "X-Miscrits-Version": self.config.client_version,
        }
        if self.config.api_host_header:
            headers["Host"] = self.config.api_host_header
        return headers

    def _cdn_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "User-Agent": self.config.user_agent,
            "X-Godot-Engine": "4.3.stable",
            "X-Miscrits-Version": self.config.client_version,
        }


def is_transient_network_error(exc: BaseException) -> bool:
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(reason, (ssl.SSLEOFError, TimeoutError, ConnectionResetError, ConnectionAbortedError)):
        return True
    text = str(reason).strip().lower()
    return any(
        marker in text
        for marker in (
            "unexpected_eof_while_reading",
            "connection reset",
            "connection aborted",
            "timed out",
            "temporarily unavailable",
        )
    )


def network_error_message(exc: urllib.error.URLError, retried: bool) -> str:
    if is_transient_network_error(exc):
        suffix = " after retrying" if retried else ""
        return f"Network error: temporary connection failure{suffix}: {exc.reason}"
    return f"Network error: {exc.reason}"


def token_would_expire_in(token: str, seconds: int) -> bool:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8"))
        exp = int(data.get("exp", 0) or 0)
        return exp <= int(time.time()) + int(seconds)
    except (IndexError, ValueError, json.JSONDecodeError):
        return True
