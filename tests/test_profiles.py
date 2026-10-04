"""Loopback and offline-mode gating for the synthetic profile UX."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def make_settings(tmp_path: Path, **updates: Any) -> Settings:
    values: dict[str, Any] = {
        "_env_file": None,
        "app_env": "development",
        "llm_provider": "echo",
        "vector_store_path": str(tmp_path / "vectorstore"),
        "seed_on_startup": False,
        "rate_limit_per_minute": 0,
    }
    values.update(updates)
    return Settings(**values)  # type: ignore[call-arg]


def _profile_flag(settings: Settings, *, host: str, client_ip: str) -> bool:
    application = create_app(settings)
    with TestClient(
        application,
        base_url=f"http://{host}",
        client=(client_ip, 50100),
    ) as client:
        return bool(client.get("/service-info").json()["local_synthetic_profiles_available"])


def test_synthetic_profiles_only_show_on_offline_loopback_development(
    tmp_path: Path,
) -> None:
    local_development = make_settings(tmp_path / "local")
    external_development = make_settings(tmp_path / "external")
    production = make_settings(tmp_path / "production", app_env="production")
    hosted_development = make_settings(
        tmp_path / "hosted",
        llm_provider="openai",
        openai_api_key="test-key-only-used-for-construction",
    )

    assert (
        _profile_flag(
            local_development,
            host="localhost",
            client_ip="127.0.0.1",
        )
        is True
    )
    assert (
        _profile_flag(
            external_development,
            host="demo.example.test",
            client_ip="192.0.2.10",
        )
        is False
    )
    assert (
        _profile_flag(
            external_development,
            host="localhost",
            client_ip="192.0.2.10",
        )
        is False
    )
    assert (
        _profile_flag(
            external_development,
            host="demo.example.test",
            client_ip="127.0.0.1",
        )
        is False
    )
    assert (
        _profile_flag(
            production,
            host="localhost",
            client_ip="127.0.0.1",
        )
        is False
    )
    assert (
        _profile_flag(
            hosted_development,
            host="localhost",
            client_ip="127.0.0.1",
        )
        is False
    )
