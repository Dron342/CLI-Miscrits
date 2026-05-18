from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from .config import DATA_DIR, ROOT_DIR
from .storage import load_json, save_json


AUTO_UPDATE_RESUME_FILE = DATA_DIR / "auto_update_resume.json"
DEFAULT_UPDATE_REPOSITORY = "Dron342/CLI-Miscrits"
UPDATE_REPOSITORY_ENV = "MISCRITS_CLI_UPDATE_REPOSITORY"
VERSION_TAG_RE = re.compile(r"^v?\d+\.\d+\.\d+$")


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
    repo_dir = Path(repo_dir).resolve()
    if not (repo_dir / ".git").exists():
        return {
            "ok": True,
            "method": "archive",
            "install_dir": str(repo_dir),
            "reason": "not_git_checkout",
        }
    try:
        branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir)
        status = run_git(["status", "--porcelain", "--untracked-files=no"], repo_dir)
    except RuntimeError as exc:
        return {"ok": False, "reason": "git_unavailable", "error": str(exc)}
    if branch == "HEAD":
        return {"ok": False, "reason": "detached_head"}
    if status.strip():
        return {"ok": False, "reason": "dirty_worktree"}
    return {"ok": True, "method": "git", "branch": branch, "repo_dir": str(repo_dir)}


def install_git_tag(tag: str, repo_dir: Path = ROOT_DIR) -> dict[str, Any]:
    capability = git_update_capability(repo_dir)
    if not capability.get("ok"):
        return capability
    clean_tag = str(tag or "").strip()
    if not clean_tag:
        return {"ok": False, "reason": "missing_tag"}
    if not VERSION_TAG_RE.match(clean_tag):
        return {"ok": False, "reason": "invalid_tag", "tag": clean_tag}
    if capability.get("method") == "archive":
        return install_archive_tag(clean_tag, Path(str(capability.get("install_dir") or repo_dir)))
    try:
        checkout_dir = Path(str(capability.get("repo_dir") or repo_dir))
        run_git(["fetch", "origin", "tag", clean_tag], checkout_dir)
        run_git(["merge", "--ff-only", clean_tag], checkout_dir)
    except RuntimeError as exc:
        return {"ok": False, "reason": "git_update_failed", "error": str(exc)}
    return {"ok": True, "method": "git", "tag": clean_tag, "branch": capability.get("branch", "")}


def install_archive_tag(tag: str, install_dir: Path = ROOT_DIR) -> dict[str, Any]:
    repository = os.environ.get(UPDATE_REPOSITORY_ENV, DEFAULT_UPDATE_REPOSITORY).strip()
    if not repository or "/" not in repository:
        return {"ok": False, "reason": "invalid_repository", "repository": repository}
    install_dir = Path(install_dir).resolve()
    archive_url = f"https://codeload.github.com/{repository}/zip/refs/tags/{tag}"
    try:
        with tempfile.TemporaryDirectory(prefix="cli-miscrits-update-") as tmp:
            archive_path = Path(tmp) / "source.zip"
            download_file(archive_url, archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                extract_archive_safely(archive, Path(tmp))
            roots = [path for path in Path(tmp).iterdir() if path.is_dir()]
            source_root = next((path for path in roots if (path / "miscrits_cli" / "__init__.py").exists()), None)
            if source_root is None:
                return {"ok": False, "reason": "invalid_archive", "tag": tag}
            copy_release_files(source_root, install_dir)
    except Exception as exc:  # noqa: BLE001 - returned to UI/event log as update diagnostic.
        return {"ok": False, "reason": "archive_update_failed", "error": str(exc), "tag": tag}
    return {"ok": True, "method": "archive", "tag": tag, "install_dir": str(install_dir)}


def download_file(url: str, target: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/zip",
            "User-Agent": "CLI-Miscrits-auto-update",
        },
    )
    with urllib.request.urlopen(request, timeout=30.0) as response:
        target.write_bytes(response.read())


def extract_archive_safely(archive: zipfile.ZipFile, target_dir: Path) -> None:
    target_dir = target_dir.resolve()
    for member in archive.infolist():
        destination = (target_dir / member.filename).resolve()
        try:
            destination.relative_to(target_dir)
        except ValueError:
            raise RuntimeError(f"unsafe archive path: {member.filename}")
    archive.extractall(target_dir)


def copy_release_files(source_root: Path, install_dir: Path) -> None:
    package_source = source_root / "miscrits_cli"
    package_target = install_dir / "miscrits_cli"
    if not package_target.exists():
        raise RuntimeError(f"package target not found: {package_target}")
    shutil.copytree(package_source, package_target, dirs_exist_ok=True)
    for filename in ("README.md", "pyproject.toml"):
        source = source_root / filename
        target = install_dir / filename
        if source.exists() and target.exists():
            shutil.copy2(source, target)


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
