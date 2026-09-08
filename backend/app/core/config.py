"""
core/config.py

Application configuration loaded from environment variables.
All configurable values are defined here — never scattered through the code.

Usage:
    from app.core.config import settings
    print(settings.max_prompt_length)
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings.  Values are read from environment variables or
    from the .env file in the backend directory.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Server ────────────────────────────────────────────────────────────────
    app_name: str = "PromptOptAI Analysis API"
    app_version: str = "1.0.0"
    debug: bool = False

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Set to your Next.js dev origin; overrideable via FRONTEND_URL env var.
    frontend_url: str = "http://localhost:3000"

    # ── Prompt validation ─────────────────────────────────────────────────────
    # Maximum number of characters accepted in a single prompt submission.
    # This can be tuned in .env without a code change.
    max_prompt_length: int = 100_000

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    # Registry key for the default tokenizer (see analysis/tokenizer.py)
    default_tokenizer: str = "tiktoken"

    # ── Intent Classifier ─────────────────────────────────────────────────────
    # Configurable selection: 'tfidf' (E1 baseline), 'semantic' (E2 SentenceTransformer), or 'hybrid' (E3)
    intent_classifier_type: str = "tfidf"
    intent_hybrid_alpha: float = 0.5

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return Settings()


# Convenience alias used throughout the app
settings = get_settings()
