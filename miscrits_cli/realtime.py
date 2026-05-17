from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import ssl
import struct
import time
import urllib.parse
from typing import Any

from .config import ClientConfig, DEFAULT_CONFIG
from .nakama import MiscritsError


class NakamaRealtime:
    def __init__(self, token: str, config: ClientConfig = DEFAULT_CONFIG) -> None:
        self.token = token
        self.config = config
        self._sock: socket.socket | ssl.SSLSocket | None = None
        self._cid = 0

    def __enter__(self) -> "NakamaRealtime":
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def connect(self) -> None:
        if self._sock is not None:
            return
        scheme = self.config.socket_scheme
        host = self.config.socket_host
        port = self.config.socket_port
        query = urllib.parse.urlencode({"lang": "en", "status": "true", "token": self.token})
        path = f"/ws?{query}"
        raw = socket.create_connection((host, port), timeout=self.config.timeout_seconds)
        if scheme == "wss":
            raw = ssl.create_default_context().wrap_socket(raw, server_hostname=host)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            f"User-Agent: {self.config.user_agent}\r\n"
            "X-Godot-Engine: 4.3.stable\r\n"
            f"X-Miscrits-Version: {self.config.client_version}\r\n"
            "\r\n"
        )
        raw.sendall(request.encode("ascii"))
        response = self._read_http_response(raw)
        if response.startswith("HTTP/1.1 401") or response.startswith("HTTP/1.0 401"):
            raw.close()
            raise MiscritsError(
                "WebSocket handshake failed: HTTP 401 Unauthorized. "
                "The realtime server rejected the session token; login again with --remember. "
                "If HTTP login succeeds but this continues, HTTP API and websocket may be on different backend clusters."
            )
        if not response:
            raw.close()
            raise MiscritsError(
                "WebSocket handshake failed: empty response. "
                "The realtime server usually closes like this when the session token is invalid; "
                "run login again or save credentials with --remember."
            )
        if " 101 " not in response.split("\r\n", 1)[0]:
            raw.close()
            raise MiscritsError(f"WebSocket handshake failed: {response.splitlines()[0] if response else 'empty response'}")
        accept = self._header(response, "sec-websocket-accept")
        expected = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
        ).decode("ascii")
        if accept and accept != expected:
            raw.close()
            raise MiscritsError("WebSocket handshake returned an invalid accept key.")
        self._sock = raw

    def close(self) -> None:
        if self._sock is None:
            return
        try:
            self._send_frame(b"", opcode=0x8)
        except OSError:
            pass
        try:
            self._sock.close()
        finally:
            self._sock = None

    def add_matchmaker(self, mode: str) -> dict[str, Any]:
        return self.request(
            "matchmaker_add",
            {
                "query": "*",
                "min_count": 2,
                "max_count": 2,
                "string_properties": {"type": mode},
                "numeric_properties": {},
            },
        ).get("matchmaker_ticket", {})

    def remove_matchmaker(self, ticket: str) -> dict[str, Any]:
        return self.request("matchmaker_remove", {"ticket": ticket})

    def join_match(self, match_id: str | None = None, token: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if token:
            payload["token"] = token
        elif match_id:
            payload["match_id"] = match_id
        else:
            raise ValueError("join_match requires match_id or token")
        return self.request("match_join", payload).get("match", {})

    def send_match_state(self, match_id: str, op_code: int, data: dict[str, Any]) -> None:
        encoded = base64.b64encode(json.dumps(data, separators=(",", ":")).encode("utf-8")).decode("ascii")
        self.send({"match_data_send": {"match_id": match_id, "op_code": int(op_code), "data": encoded}})

    def request(self, key: str, payload: dict[str, Any]) -> dict[str, Any]:
        cid = self._next_cid()
        self.send({"cid": cid, key: payload})
        while True:
            message = self.receive()
            if message.get("cid") == cid:
                if "error" in message:
                    raise MiscritsError(f"Realtime error: {message['error']}")
                return message

    def send(self, payload: dict[str, Any]) -> None:
        self._require_socket()
        self._send_frame(json.dumps(payload, separators=(",", ":")).encode("utf-8"), opcode=0x1)

    def receive(self, timeout: float | None = None) -> dict[str, Any]:
        self._require_socket()
        if timeout is not None:
            self._sock.settimeout(timeout)
        try:
            while True:
                opcode, payload = self._read_frame()
                if opcode == 0x1:
                    return json.loads(payload.decode("utf-8"))
                if opcode == 0x8:
                    code = 0
                    reason = ""
                    if len(payload) >= 2:
                        code = struct.unpack("!H", payload[:2])[0]
                        reason = payload[2:].decode("utf-8", errors="replace")
                    detail = f" code={code}" if code else ""
                    if reason:
                        detail += f" reason={reason}"
                    raise MiscritsError(f"Realtime socket closed by server.{detail}")
                if opcode == 0x9:
                    self._send_frame(payload, opcode=0xA)
        finally:
            if timeout is not None and self._sock is not None:
                self._sock.settimeout(self.config.timeout_seconds)

    def wait_for(self, key: str, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for {key}.")
            message = self.receive(timeout=remaining)
            if key in message:
                return message[key]
            if "error" in message:
                raise MiscritsError(f"Realtime error: {message['error']}")

    def _next_cid(self) -> str:
        self._cid += 1
        return str(self._cid)

    def _require_socket(self) -> None:
        if self._sock is None:
            raise MiscritsError("Realtime socket is not connected.")

    def _send_frame(self, payload: bytes, opcode: int) -> None:
        self._require_socket()
        first = 0x80 | opcode
        mask_bit = 0x80
        length = len(payload)
        if length < 126:
            header = struct.pack("!BB", first, mask_bit | length)
        elif length <= 0xFFFF:
            header = struct.pack("!BBH", first, mask_bit | 126, length)
        else:
            header = struct.pack("!BBQ", first, mask_bit | 127, length)
        mask = os.urandom(4)
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        self._sock.sendall(header + mask + masked)

    def _read_frame(self) -> tuple[int, bytes]:
        self._require_socket()
        first_two = self._recv_exact(2)
        first, second = first_two[0], first_two[1]
        opcode = first & 0x0F
        length = second & 0x7F
        masked = bool(second & 0x80)
        if length == 126:
            length = struct.unpack("!H", self._recv_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._recv_exact(8))[0]
        mask = self._recv_exact(4) if masked else b""
        payload = self._recv_exact(length) if length else b""
        if masked:
            payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        return opcode, payload

    def _recv_exact(self, size: int) -> bytes:
        self._require_socket()
        chunks = bytearray()
        while len(chunks) < size:
            chunk = self._sock.recv(size - len(chunks))
            if not chunk:
                raise MiscritsError("Realtime socket closed.")
            chunks.extend(chunk)
        return bytes(chunks)

    @staticmethod
    def _read_http_response(sock: socket.socket | ssl.SSLSocket) -> str:
        data = bytearray()
        while b"\r\n\r\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data.extend(chunk)
        return data.decode("iso-8859-1", errors="replace")

    @staticmethod
    def _header(response: str, name: str) -> str:
        needle = name.lower() + ":"
        for line in response.split("\r\n")[1:]:
            if line.lower().startswith(needle):
                return line.split(":", 1)[1].strip()
        return ""


def decode_match_data(message: dict[str, Any]) -> dict[str, Any]:
    state = message.get("match_data", message)
    raw = state.get("data", "")
    decoded = base64.b64decode(raw).decode("utf-8") if raw else "{}"
    try:
        data = json.loads(decoded or "{}")
    except json.JSONDecodeError:
        data = decoded
    return {
        "match_id": state.get("match_id", ""),
        "op_code": int(state.get("op_code", 0) or 0),
        "data": data,
        "presence": state.get("presence", {}),
    }


def diagnose_socket_endpoints(token: str, config: ClientConfig = DEFAULT_CONFIG) -> list[dict[str, Any]]:
    candidates = [
        (config.socket_scheme, config.socket_host, config.socket_port, "/ws"),
        ("ws", "63.183.56.199", 7350, "/ws"),
        ("ws", "67.213.121.59", 7350, "/ws"),
        ("wss", "worldofmiscrits.com", 443, "/ws"),
        ("ws", "worldofmiscrits.com", 80, "/ws"),
    ]
    seen = set()
    out = []
    for scheme, host, port, path_root in candidates:
        key = (scheme, host, port, path_root)
        if key in seen:
            continue
        seen.add(key)
        out.append(_diagnose_one_endpoint(scheme, host, port, path_root, token, config))
    return out


def _diagnose_one_endpoint(
    scheme: str,
    host: str,
    port: int,
    path_root: str,
    token: str,
    config: ClientConfig,
) -> dict[str, Any]:
    query = urllib.parse.urlencode({"lang": "en", "status": "true", "token": token})
    path = f"{path_root}?{query}"
    result: dict[str, Any] = {"scheme": scheme, "host": host, "port": port, "path": path_root}
    raw: socket.socket | ssl.SSLSocket | None = None
    try:
        raw = socket.create_connection((host, port), timeout=8)
        if scheme == "wss":
            raw = ssl.create_default_context().wrap_socket(raw, server_hostname=host)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            f"User-Agent: {config.user_agent}\r\n"
            "X-Godot-Engine: 4.3.stable\r\n"
            f"X-Miscrits-Version: {config.client_version}\r\n"
            "\r\n"
        )
        raw.sendall(request.encode("ascii"))
        raw.settimeout(8)
        response = _read_probe_response(raw)
        result["ok"] = response.startswith("HTTP/1.1 101") or response.startswith("HTTP/1.0 101")
        result["status_line"] = response.splitlines()[0] if response else "empty response"
        result["empty_response"] = not bool(response)
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if raw is not None:
            try:
                raw.close()
            except OSError:
                pass
    return result


def _read_probe_response(sock: socket.socket | ssl.SSLSocket) -> str:
    data = bytearray()
    while b"\r\n\r\n" not in data and len(data) < 4096:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data.extend(chunk)
    return data.decode("iso-8859-1", errors="replace")
