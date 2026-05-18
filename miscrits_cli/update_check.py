from __future__ import annotations

import json
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import __version__


DEFAULT_REPOSITORY = "Dron342/CLI-Miscrits"
TAG_PATTERN = re.compile(r"^v?(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")


def check_for_updates(repository: str = DEFAULT_REPOSITORY, timeout: float = 5.0) -> dict[str, Any]:
    repository = normalize_repository(repository)
    checked_at = time.time()
    try:
        latest_tag = latest_semver_tag(repository, timeout=timeout)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "repository": repository,
            "current_version": __version__,
            "latest_version": "",
            "latest_tag": "",
            "update_available": False,
            "checked_at": checked_at,
            "error": str(exc) or exc.__class__.__name__,
        }
    latest_version = version_from_tag(latest_tag)
    return {
        "ok": True,
        "repository": repository,
        "current_version": __version__,
        "latest_version": latest_version,
        "latest_tag": latest_tag,
        "update_available": compare_versions(latest_version, __version__) > 0,
        "checked_at": checked_at,
        "source": "github_tags",
    }


def latest_semver_tag(repository: str, timeout: float = 5.0) -> str:
    url = f"https://api.github.com/repos/{repository}/tags?per_page=100"
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"CLI-Miscrits/{__version__}",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("GitHub tags response is not a list.")
    tags = [str(item.get("name", "")) for item in payload if isinstance(item, dict)]
    semver_tags = [tag for tag in tags if parse_version(version_from_tag(tag)) is not None]
    if not semver_tags:
        raise ValueError("No semantic version tags found.")
    return max(semver_tags, key=lambda tag: parse_version(version_from_tag(tag)) or (-1, -1, -1))


def normalize_repository(repository: str) -> str:
    clean = str(repository or "").strip().strip("/")
    if clean.count("/") != 1:
        raise ValueError("Repository must use owner/name format.")
    return clean


def version_from_tag(tag: str) -> str:
    return str(tag or "").strip().removeprefix("v")


def compare_versions(left: str, right: str) -> int:
    left_version = parse_version(left)
    right_version = parse_version(right)
    if left_version is None or right_version is None:
        raise ValueError("Versions must use MAJOR.MINOR.PATCH format.")
    return (left_version > right_version) - (left_version < right_version)


def parse_version(value: str) -> tuple[int, int, int] | None:
    match = TAG_PATTERN.match(str(value or "").strip())
    if not match:
        return None
    return tuple(int(match.group(key)) for key in ("major", "minor", "patch"))
