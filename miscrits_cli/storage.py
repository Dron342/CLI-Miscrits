from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from .config import DATA_DIR, SESSION_FILE


_SAVE_LOCK = threading.RLock()


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    with _SAVE_LOCK:
        ensure_data_dir()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        replace_with_retry(tmp_path, path)


def replace_with_retry(tmp_path: Path, path: Path) -> None:
    last_error: PermissionError | None = None
    for attempt in range(8):
        try:
            tmp_path.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    raise last_error if last_error is not None else PermissionError(path)


def load_session(path: Path = SESSION_FILE) -> dict[str, Any]:
    return load_json(path, {})


def save_session(session: dict[str, Any], path: Path = SESSION_FILE) -> None:
    save_json(path, session)


def clear_session(path: Path = SESSION_FILE) -> None:
    if path.exists():
        path.unlink()
