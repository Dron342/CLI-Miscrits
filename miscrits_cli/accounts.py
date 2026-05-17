from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from typing import Any

from .config import DATA_DIR
from .credentials import clear_credentials, credentials_status, load_credentials, save_credentials
from .storage import clear_session, load_json, save_json


ACCOUNTS_FILE = DATA_DIR / "accounts.json"
ACCOUNTS_DIR = DATA_DIR / "accounts"


def list_accounts() -> list[dict[str, Any]]:
    payload = load_json(ACCOUNTS_FILE, {"accounts": []})
    accounts = payload.get("accounts", []) if isinstance(payload, dict) else []
    if not isinstance(accounts, list):
        return []
    return [account_public(item) for item in accounts if isinstance(item, dict)]


def get_account(account_id: str) -> dict[str, Any] | None:
    for account in raw_accounts():
        if str(account.get("id", "")) == str(account_id):
            return account
    return None


def save_account(login: str, password: str, label: str = "", account_id: str = "") -> dict[str, Any]:
    if not login or not password:
        raise ValueError("Login and password are required.")
    accounts = raw_accounts()
    now = int(time.time())
    existing = None
    if account_id:
        existing = next((item for item in accounts if str(item.get("id", "")) == account_id), None)
    if not existing:
        existing = next((item for item in accounts if str(item.get("login", "")).lower() == login.lower()), None)
    if existing:
        existing["login"] = login
        existing["label"] = label or existing.get("label") or login
        existing["updated_at"] = now
        account = existing
    else:
        account = {
            "id": unique_account_id(login),
            "login": login,
            "label": label or login,
            "created_at": now,
            "updated_at": now,
        }
        accounts.append(account)
    account_dir(account["id"]).mkdir(parents=True, exist_ok=True)
    path = credentials_path(account["id"])
    save_credentials(login, password, path)
    load_credentials(path)
    save_json(ACCOUNTS_FILE, {"accounts": accounts})
    return account_public(account)


def remove_account(account_id: str) -> bool:
    accounts = raw_accounts()
    kept = [item for item in accounts if str(item.get("id", "")) != str(account_id)]
    if len(kept) == len(accounts):
        return False
    clear_credentials(credentials_path(account_id))
    clear_session(session_path(account_id))
    save_json(ACCOUNTS_FILE, {"accounts": kept})
    return True


def account_public(account: dict[str, Any]) -> dict[str, Any]:
    account_id = str(account.get("id", ""))
    return {
        "id": account_id,
        "login": account.get("login", ""),
        "label": account.get("label") or account.get("login", ""),
        "created_at": account.get("created_at", 0),
        "updated_at": account.get("updated_at", 0),
        "credentials": credentials_status(credentials_path(account_id)) if account_id else {},
    }


def raw_accounts() -> list[dict[str, Any]]:
    payload = load_json(ACCOUNTS_FILE, {"accounts": []})
    accounts = payload.get("accounts", []) if isinstance(payload, dict) else []
    return accounts if isinstance(accounts, list) else []


def account_dir(account_id: str) -> Path:
    return ACCOUNTS_DIR / safe_id(account_id)


def session_path(account_id: str) -> Path:
    return account_dir(account_id) / "session.json"


def credentials_path(account_id: str) -> Path:
    return account_dir(account_id) / "credentials.json"


def unique_account_id(login: str) -> str:
    base = safe_id(login.split("@")[0] or "account")
    existing = {str(item.get("id", "")) for item in raw_accounts()}
    account_id = base
    while not account_id or account_id in existing:
        account_id = f"{base}-{uuid.uuid4().hex[:6]}"
    return account_id


def safe_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value).strip().lower()).strip("-._")
    return cleaned[:48] or "account"
