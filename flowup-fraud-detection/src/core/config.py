"""Application configuration loaded from environment variables.

Uses pydantic-settings to provide type-safe, validated configuration
with sensible defaults. All paths use pathlib for cross-platform compatibility.
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        model_path: Filesystem path to the trained model artifact (.joblib).
        log_level: Logging verbosity level.
        threshold: Decision threshold for fraud classification (0.0–1.0).
        redis_url: Redis connection URL for optional caching.
        app_port: Port the application listens on.
        app_host: Host the application binds to.
        model_version: Semantic version of the deployed model.
    """

    model_path: Path = Path("models/fraud_model.joblib")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    threshold: float = 0.5
    redis_url: str = "redis://redis:6379/0"
    app_port: int = 8000
    app_host: str = "0.0.0.0"
    model_version: str = "1.0.0"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


def get_settings() -> Settings:
    """Create and return a Settings instance.

    Returns:
        Settings: Validated application settings.
    """
    return Settings()
