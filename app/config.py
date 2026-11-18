"""Runtime configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: str) -> list[str]:
    return [
        item.strip() for item in os.getenv(name, default).split(",") if item.strip()
    ]


@dataclass(frozen=True)
class Settings:
    """API and server settings (see README for the variables)."""

    host: str = field(default_factory=lambda: os.getenv("POWERPULSE_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("POWERPULSE_PORT", "8000")))
    reload: bool = field(default_factory=lambda: _env_bool("POWERPULSE_RELOAD", False))
    log_level: str = field(
        default_factory=lambda: os.getenv("POWERPULSE_LOG_LEVEL", "info").lower()
    )
    cors_origins: list[str] = field(
        default_factory=lambda: _env_list(
            "POWERPULSE_CORS_ORIGINS", DEFAULT_CORS_ORIGINS
        )
    )
    skip_startup_training: bool = field(
        default_factory=lambda: _env_bool("POWERPULSE_SKIP_STARTUP_TRAINING", False)
    )


def get_settings() -> Settings:
    """Settings read from the current environment."""
    return Settings()
