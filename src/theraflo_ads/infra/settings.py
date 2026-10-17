"""Process settings, read from THERAFLO_ADS_* environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"THERAFLO_ADS_{name}", default)


@dataclass(frozen=True)
class Settings:
    env: str = field(default_factory=lambda: _env("ENV", "development"))
    backend: str = field(default_factory=lambda: _env("BACKEND", "memory"))
    database_url: str = field(
        default_factory=lambda: _env(
            "DATABASE_URL", "postgresql+pg8000://theraflo:theraflo@127.0.0.1:5432/theraflo_ads"
        )
    )
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", "redis://127.0.0.1:6379/0"))
    amqp_url: str = field(
        default_factory=lambda: _env("AMQP_URL", "amqp://theraflo:theraflo@127.0.0.1:5672/%2F")
    )
    lake_path: Path = field(default_factory=lambda: Path(_env("LAKE_PATH", "data/lake")))
    model_dir: Path = field(default_factory=lambda: Path(_env("MODEL_DIR", "models")))
    # Per-dependency timeouts (design doc: Timeouts). Redis sits on the serving path.
    redis_timeout_s: float = field(default_factory=lambda: float(_env("REDIS_TIMEOUT_S", "0.05")))
    db_pool_size: int = field(default_factory=lambda: int(_env("DB_POOL_SIZE", "10")))
    worker_max_attempts: int = field(default_factory=lambda: int(_env("WORKER_MAX_ATTEMPTS", "6")))

    @property
    def production(self) -> bool:
        return self.env.lower() == "production"
