from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from .config import DATA_DIR
from .storage import ensure_data_dir


EVENT_LOG_FILE = DATA_DIR / "event_log.jsonl"
MAX_LOG_SCAN_LINES = 5000
TAIL_CHUNK_SIZE = 65536
_LOG_LOCK = threading.RLock()


def log_event(
    event: str,
    *,
    category: str = "system",
    level: str = "info",
    source: str = "cli",
    initiator: str = "cli",
    account_id: str = "",
    job_id: str = "",
    mode: str = "",
    phase: str = "",
    reason: str = "",
    message: str = "",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry = {
        "timestamp": time.time(),
        "event": str(event or "event"),
        "category": str(category or "system"),
        "level": str(level or "info"),
        "source": str(source or "cli"),
        "initiator": str(initiator or source or "cli"),
        "account_id": str(account_id or ""),
        "job_id": str(job_id or ""),
        "mode": str(mode or ""),
        "phase": str(phase or ""),
        "reason": str(reason or ""),
        "message": str(message or ""),
        "payload": payload or {},
    }
    with _LOG_LOCK:
        ensure_data_dir()
        EVENT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with EVENT_LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
    return entry


def load_events(
    *,
    limit: int = 250,
    level: str = "",
    category: str = "",
    event: str = "",
    account_id: str = "",
    job_id: str = "",
    source: str = "",
    initiator: str = "",
    reason: str = "",
    text: str = "",
    since: float = 0.0,
    until: float = 0.0,
) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit or 250), 1000))
    rows = _read_recent_rows()
    filters = {
        "level": level,
        "category": category,
        "event": event,
        "account_id": account_id,
        "job_id": job_id,
        "source": source,
        "initiator": initiator,
        "reason": reason,
    }
    needle = str(text or "").strip().lower()
    result: list[dict[str, Any]] = []
    for row in reversed(rows):
        ts = _float(row.get("timestamp"))
        if since and ts < since:
            continue
        if until and ts > until:
            continue
        filtered = False
        for key, value in filters.items():
            if not value:
                continue
            field = str(row.get(key, "")).lower()
            wanted = str(value).lower()
            if key in {"event", "reason"}:
                filtered = wanted not in field
            else:
                filtered = field != wanted
            if filtered:
                break
        if filtered:
            continue
        if needle and needle not in json.dumps(row, ensure_ascii=False).lower():
            continue
        result.append(row)
        if len(result) >= limit:
            break
    return result


def event_log_status() -> dict[str, Any]:
    if not EVENT_LOG_FILE.exists():
        return {"path": str(EVENT_LOG_FILE), "exists": False, "size": 0}
    return {"path": str(EVENT_LOG_FILE), "exists": True, "size": EVENT_LOG_FILE.stat().st_size}


def _read_recent_rows() -> list[dict[str, Any]]:
    if not EVENT_LOG_FILE.exists():
        return []
    with _LOG_LOCK:
        lines = _tail_lines(EVENT_LOG_FILE, MAX_LOG_SCAN_LINES)
    rows: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _tail_lines(path: Path, max_lines: int) -> list[str]:
    if max_lines <= 0:
        return []
    blocks: list[bytes] = []
    line_count = 0
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        while position > 0 and line_count <= max_lines:
            read_size = min(TAIL_CHUNK_SIZE, position)
            position -= read_size
            handle.seek(position)
            block = handle.read(read_size)
            blocks.append(block)
            line_count += block.count(b"\n")
    if not blocks:
        return []
    text = b"".join(reversed(blocks)).decode("utf-8", errors="replace")
    return text.splitlines()[-max_lines:]


def _float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0
