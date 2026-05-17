from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SESSION_FILE = DATA_DIR / "session.json"


@dataclass(frozen=True)
class ClientConfig:
    api_url: str = "https://worldofmiscrits.com:443"
    api_host_header: str = ""
    cdn_url: str = "https://cdn.worldofmiscrits.com"
    socket_host: str = "63.183.56.199"
    socket_port: int = 7350
    socket_scheme: str = "ws"
    server_key: str = "a1c737cc188f54ab3658ba5da0e12ee5"
    client_version: str = "2.4.0"
    user_agent: str = "GodotEngine/4.3.stable Miscrits/2.4.0"
    timeout_seconds: float = 30.0


DEFAULT_CONFIG = ClientConfig()
