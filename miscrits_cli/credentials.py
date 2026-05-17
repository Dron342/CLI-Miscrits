from __future__ import annotations

import base64
import ctypes
import sys
from ctypes import wintypes
from typing import Any
from pathlib import Path

from .config import DATA_DIR
from .storage import load_json, save_json


CREDENTIALS_FILE = DATA_DIR / "credentials.json"


class CredentialError(RuntimeError):
    pass


class DATA_BLOB(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def save_credentials(login: str, password: str, path: Path = CREDENTIALS_FILE) -> None:
    if not login or not password:
        raise CredentialError("Login and password are required.")
    payload = {
        "login": login,
        "password": _protect_text(password),
        "method": "windows-dpapi" if sys.platform == "win32" else "plain-unsupported",
    }
    save_json(path, payload)


def load_credentials(path: Path = CREDENTIALS_FILE) -> dict[str, str]:
    payload = load_json(path, {})
    if not isinstance(payload, dict) or not payload:
        return {}
    login = str(payload.get("login", "") or "")
    protected = str(payload.get("password", "") or "")
    if not login or not protected:
        return {}
    return {"login": login, "password": _unprotect_text(protected)}


def clear_credentials(path: Path = CREDENTIALS_FILE) -> None:
    if path.exists():
        path.unlink()


def credentials_status(path: Path = CREDENTIALS_FILE) -> dict[str, Any]:
    payload = load_json(path, {})
    saved = isinstance(payload, dict) and bool(payload.get("login")) and bool(payload.get("password"))
    decryptable = False
    error = ""
    if saved:
        try:
            load_credentials(path)
            decryptable = True
        except Exception as exc:
            error = str(exc)
    return {
        "saved": saved,
        "decryptable": decryptable,
        "error": error,
        "login": payload.get("login") if isinstance(payload, dict) else None,
        "method": payload.get("method") if isinstance(payload, dict) else None,
        "path": str(path),
    }


def _protect_text(text: str) -> str:
    if sys.platform != "win32":
        raise CredentialError("Credential saving is only implemented with Windows DPAPI.")
    raw = text.encode("utf-8")
    blob_in = _blob_from_bytes(raw)
    blob_out = DATA_BLOB()
    if not ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(blob_in),
        None,
        None,
        None,
        None,
        0,
        ctypes.byref(blob_out),
    ):
        raise CredentialError("CryptProtectData failed.")
    try:
        encrypted = ctypes.string_at(blob_out.pbData, blob_out.cbData)
        return base64.b64encode(encrypted).decode("ascii")
    finally:
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)


def _unprotect_text(encoded: str) -> str:
    if sys.platform != "win32":
        raise CredentialError("Credential loading is only implemented with Windows DPAPI.")
    encrypted = base64.b64decode(encoded.encode("ascii"))
    blob_in = _blob_from_bytes(encrypted)
    blob_out = DATA_BLOB()
    if not ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(blob_in),
        None,
        None,
        None,
        None,
        0,
        ctypes.byref(blob_out),
    ):
        raise CredentialError("CryptUnprotectData failed.")
    try:
        raw = ctypes.string_at(blob_out.pbData, blob_out.cbData)
        return raw.decode("utf-8")
    finally:
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)


def _blob_from_bytes(data: bytes) -> DATA_BLOB:
    buffer = ctypes.create_string_buffer(data)
    blob = DATA_BLOB(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
    blob._buffer = buffer  # keep the backing buffer alive for the Windows API call
    return blob
