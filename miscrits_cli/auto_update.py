from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .config import DATA_DIR, ROOT_DIR
from .storage import load_json, save_json


AUTO_UPDATE_RESUME_FILE = DATA_DIR / "auto_update_resume.json"


def save_resume_state(payload: dict[str, Any]) -> None:
    save_json(AUTO_UPDATE_RESUME_FILE, payload)


def load_resume_state() -> dict[str, Any]:
    data = load_json(AUTO_UPDATE_RESUME_FILE, {})
    return data if isinstance(data, dict) else {}


def consume_resume_state() -> dict[str, Any]:
    payload = load_resume_state()
    if AUTO_UPDATE_RESUME_FILE.exists():
        AUTO_UPDATE_RESUME_FILE.unlink()
    return payload


def clear_resume_state() -> None:
    if AUTO_UPDATE_RESUME_FILE.exists():
        AUTO_UPDATE_RESUME_FILE.unlink()


def git_update_capability(repo_dir: Path = ROOT_DIR) -> dict[str, Any]:
    if not (repo_dir / ".git").exists():
        return {"ok": False, "reason": "not_git_checkout"}
    try:
        branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir)
        status = run_git(["status", "--porcelain", "--untracked-files=no"], repo_dir)
    except RuntimeError as exc:
        return {"ok": False, "reason": "git_unavailable", "error": str(exc)}
    if branch == "HEAD":
        return {"ok": False, "reason": "detached_head"}
    if status.strip():
        return {"ok": False, "reason": "dirty_worktree"}
    return {"ok": True, "branch": branch}


def install_git_tag(tag: str, repo_dir: Path = ROOT_DIR) -> dict[str, Any]:
    capability = git_update_capability(repo_dir)
    if not capability.get("ok"):
        return capability
    clean_tag = str(tag or "").strip()
    if not clean_tag:
        return {"ok": False, "reason": "missing_tag"}
    try:
        run_git(["fetch", "origin", "tag", clean_tag], repo_dir)
        run_git(["merge", "--ff-only", clean_tag], repo_dir)
    except RuntimeError as exc:
        return {"ok": False, "reason": "git_update_failed", "error": str(exc)}
    return {"ok": True, "tag": clean_tag, "branch": capability.get("branch", "")}


def restart_current_process() -> None:
    os.execv(sys.executable, [sys.executable, *sys.argv])


def restart_cli_process(args: list[str]) -> None:
    os.execv(sys.executable, [sys.executable, "-m", "miscrits_cli", *args])


def resume_payload(intents: list[dict[str, Any]], target_tag: str, current_version: str) -> dict[str, Any]:
    return {
        "created_at": time.time(),
        "target_tag": target_tag,
        "from_version": current_version,
        "intents": intents,
    }


def run_git(args: list[str], repo_dir: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(message or f"git {' '.join(args)} failed with code {completed.returncode}")
    return (completed.stdout or "").strip()
