from __future__ import annotations

import re
import threading
import urllib.parse
from pathlib import Path
from typing import Any

from .config import DATA_DIR
from .data_cache import DataCache
from .nakama import MiscritsClient


ASSET_DIR = DATA_DIR / "cache" / "assets"
MISCRIT_ASSET_TYPES = {"avatars": "_avatar", "miscrits": "_back", "house": "_house"}
_ASSET_LOCK = threading.RLock()


class AssetCache:
    def __init__(self, client: MiscritsClient) -> None:
        self.client = client
        self.references = DataCache(client)

    def miscrit_asset_info(self, mid: int, evo: int = 0, asset_type: str = "avatars") -> dict[str, Any]:
        asset_type = normalize_asset_type(asset_type)
        metadata = self._miscrit_metadata(int(mid))
        if not metadata:
            raise ValueError(f"Unknown miscrit mid: {mid}")
        url_filename = miscrit_asset_filename(metadata, evo, asset_type)
        local_filename = safe_filename(url_filename)
        path = ASSET_DIR / asset_type / local_filename
        url = f"{self.client.config.cdn_url.rstrip('/')}/{asset_type}/{urllib.parse.quote(url_filename)}"
        return {
            "mid": int(mid),
            "evo": max(0, min(3, int(evo or 0))),
            "type": asset_type,
            "filename": local_filename,
            "cdn_filename": url_filename,
            "path": path,
            "url": url,
            "cached": path.exists(),
        }

    def ensure_miscrit_asset(self, mid: int, evo: int = 0, asset_type: str = "avatars", force: bool = False) -> Path:
        info = self.miscrit_asset_info(mid, evo, asset_type)
        path = Path(info["path"])
        with _ASSET_LOCK:
            if path.exists() and not force:
                return path
            path.parent.mkdir(parents=True, exist_ok=True)
            body = self.client.fetch_cdn_bytes(str(info["url"]), self._image_headers()["Accept"])
            if not looks_like_image(body):
                raise RuntimeError(f"CDN did not return an image for mid {mid}.")
            tmp_path = path.with_suffix(f"{path.suffix}.{threading.get_ident()}.tmp")
            tmp_path.write_bytes(body)
            tmp_path.replace(path)
            return path

    def sync_miscrit_assets(
        self,
        mids: list[int] | None = None,
        asset_type: str = "avatars",
        evo: int = 0,
        force: bool = False,
        include_all: bool = False,
        limit: int = 0,
    ) -> dict[str, Any]:
        selected = self._select_mids(mids, include_all)
        if limit > 0:
            selected = selected[:limit]
        results = []
        for mid in selected:
            try:
                before = self.miscrit_asset_info(mid, evo, asset_type)
                path = self.ensure_miscrit_asset(mid, evo, asset_type, force)
                results.append(
                    {
                        "mid": mid,
                        "ok": True,
                        "cached": before["cached"] and not force,
                        "path": str(path),
                    }
                )
            except Exception as exc:
                results.append({"mid": mid, "ok": False, "error": str(exc)})
        return {"ok": all(item.get("ok") for item in results), "assets": results}

    def sync_miscrit_asset_specs(
        self,
        specs: list[dict[str, Any]],
        asset_type: str = "avatars",
        force: bool = False,
        limit: int = 0,
    ) -> dict[str, Any]:
        seen: set[tuple[int, int]] = set()
        selected: list[tuple[int, int]] = []
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            try:
                mid = int(spec.get("mid", spec.get("mId", spec.get("m", 0))) or 0)
                evo_value = spec.get("evo", spec.get("evo_id", spec.get("evolution")))
                if evo_value in (None, ""):
                    evo = int(spec.get("level", spec.get("l", 1)) or 1) // 10
                else:
                    evo = int(evo_value or 0)
            except (TypeError, ValueError):
                continue
            key = (mid, max(0, min(3, evo)))
            if mid > 0 and key not in seen:
                seen.add(key)
                selected.append(key)
        if limit > 0:
            selected = selected[:limit]
        results = []
        for mid, evo in selected:
            try:
                before = self.miscrit_asset_info(mid, evo, asset_type)
                path = self.ensure_miscrit_asset(mid, evo, asset_type, force)
                results.append({"mid": mid, "evo": evo, "ok": True, "cached": before["cached"] and not force, "path": str(path)})
            except Exception as exc:
                results.append({"mid": mid, "evo": evo, "ok": False, "error": str(exc)})
        return {"ok": all(item.get("ok") for item in results), "assets": results}

    def _select_mids(self, mids: list[int] | None, include_all: bool) -> list[int]:
        if mids:
            return sorted({int(mid) for mid in mids if int(mid) > 0})
        if include_all:
            return sorted(self._metadata_by_mid().keys())
        return []

    def _miscrit_metadata(self, mid: int) -> dict[str, Any]:
        return self._metadata_by_mid().get(int(mid), {})

    def _metadata_by_mid(self) -> dict[int, dict[str, Any]]:
        items = self.references.get_json("miscrits.json", refresh_index=False) or []
        if not isinstance(items, list):
            return {}
        out: dict[int, dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            mid = int(item.get("id", item.get("mId", item.get("mid", 0))) or 0)
            if mid > 0:
                out[mid] = item
        return out

    def _image_headers(self) -> dict[str, str]:
        return {"Accept": "image/avif,image/webp,image/png,image/*,*/*;q=0.8"}


def miscrit_asset_filename(metadata: dict[str, Any], evo: int, asset_type: str) -> str:
    asset_type = normalize_asset_type(asset_type)
    names = metadata.get("names", [])
    evo = max(0, min(3, int(evo or 0)))
    if isinstance(names, list) and names:
        index = min(evo, len(names) - 1)
        name = str(names[index] or names[0])
    else:
        name = str(metadata.get("name", metadata.get("display_name", metadata.get("id", ""))) or "")
    clean = name.replace(" ", "_").lower()
    return f"{clean}{MISCRIT_ASSET_TYPES[asset_type]}.png"


def normalize_asset_type(value: str) -> str:
    asset_type = str(value or "avatars").strip().lower()
    if asset_type not in MISCRIT_ASSET_TYPES:
        raise ValueError(f"Unsupported miscrit asset type: {value}")
    return asset_type


def safe_filename(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value))
    return clean.strip("._") or "asset.png"


def looks_like_image(body: bytes) -> bool:
    return body.startswith(b"\x89PNG\r\n\x1a\n") or body.startswith(b"\xff\xd8\xff") or body.startswith(b"RIFF")
