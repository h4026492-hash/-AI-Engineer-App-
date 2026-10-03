"""Application settings.

Single source of truth for configuration, loaded from environment variables and
an optional `.env` file. Import ``get_settings`` (cached) rather than
constructing ``Settings`` directly so the whole process agrees on one config.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Deployment environment. Drives debug/verbosity defaults."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Provider(StrEnum):
    """Which model backend the service runs against."""

    ECHO = "echo"
    OPENAI = "openai"


class Settings(BaseSettings):
    """Runtime configuration.

    Every field maps to an uppercase env var of the same name (e.g. ``RAG_TOP_K``).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application ---
    app_name: str = "ai-engineer-app"
    app_env: Environment = Environment.DEVELOPMENT
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
    log_format: str = "console"
    cors_origins: str = "http://localhost:3000,http://localhost:8000"
    request_timeout_seconds: float = Field(default=60.0, gt=0)

    # --- Providers ---
    llm_provider: Provider = Provider.ECHO
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_base_url: str = ""

    # --- Retrieval ---
    rag_chunk_size: int = Field(default=800, ge=100, le=8000)
    rag_chunk_overlap: int = Field(default=120, ge=0)
    rag_top_k: int = Field(default=4, ge=1, le=50)
    rag_min_score: float = Field(default=0.05, ge=0.0, le=1.0)
    rag_embed_dim: int = Field(default=512, ge=64, le=8192)

    # --- Persistence ---
    vector_store_path: str = "data/vectorstore"
    # Seed data/sample_docs into an empty index on boot, so a fresh checkout
    # answers questions immediately. Set to false in production.
    seed_on_startup: bool = True

    # --- Guardrails ---
    max_ingest_chars: int = Field(default=200_000, ge=100)
    max_question_chars: int = Field(default=2000, ge=10)
    rate_limit_per_minute: int = Field(default=60, ge=0)

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        return value.upper()

    @field_validator("log_format")
    @classmethod
    def _normalise_log_format(cls, value: str) -> str:
        normalised = value.lower()
        if normalised not in {"console", "json"}:
            msg = f"log_format must be 'console' or 'json', got {value!r}"
            raise ValueError(msg)
        return normalised

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins parsed from the comma-separated env var."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env is Environment.PRODUCTION

    def public_config(self) -> dict[str, Any]:
        """Config that is safe to expose over HTTP. Never includes secrets."""
        return {
            "app_name": self.app_name,
            "app_env": self.app_env.value,
            "debug": self.debug,
            "log_level": self.log_level,
            "llm_provider": self.llm_provider.value,
            "openai_api_key_set": bool(self.openai_api_key),
            "openai_chat_model": self.openai_chat_model,
            "openai_embedding_model": self.openai_embedding_model,
            "rag": {
                "chunk_size": self.rag_chunk_size,
                "chunk_overlap": self.rag_chunk_overlap,
                "top_k": self.rag_top_k,
                "min_score": self.rag_min_score,
                "embed_dim": self.rag_embed_dim,
            },
            "indexed_chunks": None,  # filled in by the caller from the live index
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance (cached)."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the settings cache. Used by tests that patch the environment."""
    get_settings.cache_clear()
