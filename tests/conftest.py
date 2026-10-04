"""Autouse test isolation: security env vars cleared so local .env values never leak in."""

import pytest

from atlas.config import get_settings

SECURITY_ENV_VARS = (
    "API_KEYS_JSON",
    "API_KEY_PEPPER",
    "METRICS_BEARER_TOKEN",
    "ADMIN_TOKEN",
    "APP_ENV",
)


@pytest.fixture(autouse=True)
def clean_security_env(monkeypatch):
    for var in SECURITY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
