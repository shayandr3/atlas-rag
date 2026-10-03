"""Typed application settings via pydantic-settings. Grows per milestone."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"
    port: int = 10000


@lru_cache
def get_settings() -> Settings:
    return Settings()
