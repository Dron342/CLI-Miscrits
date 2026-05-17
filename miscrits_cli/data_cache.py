from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DATA_DIR
from .nakama import MiscritsClient, MiscritsError
from .storage import load_json, save_json


CACHE_DIR = DATA_DIR / "cache"
CACHE_INDEX_FILE = CACHE_DIR / "cache.json"
CACHE_VERSIONS_FILE = CACHE_DIR / "versions.json"
JSON_DIR = CACHE_DIR / "json"
DEFAULT_REFERENCE_FILES = ("miscrits.json",)


@dataclass(frozen=True)
class CachedReference:
    name: str
    path: Path
    version: int | str | None
    refreshed: bool


class DataCache:
    def __init__(self, client: MiscritsClient) -> None:
        self.client = client

    def sync_index(self) -> dict[str, Any]:
        index = self.client.fetch_cdn_json("cache.json")
        if not isinstance(index, dict):
            raise RuntimeError("CDN cache.json did not return an object.")
        save_json(CACHE_INDEX_FILE, index)
        return index

    def load_index(self, refresh: bool = False) -> dict[str, Any]:
        cached = load_json(CACHE_INDEX_FILE, {})
        if refresh or not isinstance(cached, dict) or not cached:
            return self.sync_index()
        return cached

    def list_known_references(self, refresh_index: bool = False) -> list[dict[str, Any]]:
        index = self.load_index(refresh_index)
        versions = self._load_versions()
        names = set(DEFAULT_REFERENCE_FILES)
        for key in index.keys():
            text = str(key)
            if text.endswith(".json"):
                names.add(text)
        return [
            {
                "name": name,
                "remote_version": self.get_cache_key(name, index),
                "local_version": versions.get(name),
                "cached": self._json_path(name).exists(),
                "path": str(self._json_path(name)),
            }
            for name in sorted(names)
        ]

    def get_json(self, name: str, force: bool = False, refresh_index: bool = False) -> Any:
        normalized = normalize_reference_name(name)
        index = self.load_index(refresh_index)
        versions = self._load_versions()
        remote_version = self.get_cache_key(normalized, index)
        path = self._json_path(normalized)
        local_version = versions.get(normalized)
        if not force and path.exists() and local_version == remote_version:
            return load_json(path, None)
        payload = self._fetch_reference_payload(
            normalized,
            int(remote_version) if is_int_like(remote_version) else None,
        )
        save_json(path, payload)
        versions[normalized] = remote_version
        save_json(CACHE_VERSIONS_FILE, versions)
        return payload

    def sync_references(
        self,
        names: list[str] | None = None,
        force: bool = False,
        refresh_index: bool = True,
        include_all_index_json: bool = False,
    ) -> dict[str, Any]:
        index = self.load_index(refresh_index)
        selected = set(normalize_reference_name(name) for name in (names or DEFAULT_REFERENCE_FILES))
        if include_all_index_json:
            for key in index.keys():
                text = str(key)
                if text.endswith(".json"):
                    selected.add(text)
        results = []
        for name in sorted(selected):
            try:
                before = self._json_path(name).exists()
                self.get_json(name, force=force, refresh_index=False)
                results.append(
                    {
                        "name": name,
                        "ok": True,
                        "version": self.get_cache_key(name, index),
                        "refreshed": force or not before,
                        "path": str(self._json_path(name)),
                    }
                )
            except Exception as exc:
                results.append({"name": name, "ok": False, "error": str(exc)})
        return {"ok": all(item.get("ok") for item in results), "references": results}

    def get_cache_key(self, name: str, index: dict[str, Any] | None = None) -> int | str:
        index = index if index is not None else self.load_index(False)
        normalized = normalize_reference_name(name)
        candidates = [normalized, normalized.removesuffix(".json")]
        for candidate in candidates:
            if candidate in index:
                return index[candidate]
        return 1

    def _load_versions(self) -> dict[str, Any]:
        versions = load_json(CACHE_VERSIONS_FILE, {})
        return versions if isinstance(versions, dict) else {}

    def _json_path(self, name: str) -> Path:
        normalized = normalize_reference_name(name)
        return JSON_DIR / normalized

    def _fetch_reference_payload(self, name: str, version: int | None) -> Any:
        try:
            return self.client.fetch_cdn_json(name, version)
        except MiscritsError as exc:
            if "HTTP 404" not in str(exc):
                raise
            if not self.client.is_logged_in():
                raise RuntimeError(
                    f"{name} is not available on CDN and RPC fallback requires login."
                ) from exc
            method = "get_" + name.removesuffix(".json")
            result = self.client.rpc(method, {})
            if not result.success:
                raise RuntimeError(f"{method} RPC failed: {result.raw}") from exc
            return result.data


def normalize_reference_name(name: str) -> str:
    clean = str(name).strip().replace("\\", "/").split("/")[-1]
    if not clean:
        raise ValueError("Reference name is empty.")
    if not clean.endswith(".json"):
        clean += ".json"
    return clean


def is_int_like(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False
