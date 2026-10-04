"""Typed application settings via pydantic-settings. Grows per milestone."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"
    port: int = 10000

    # LLM (M1)
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str = ""
    # Extra JSON body merged into every OpenAI-compatible request,
    # e.g. {"thinking": {"type": "disabled"}} for Z.AI GLM models.
    llm_extra_body_json: str = ""
    llm_cheap_model: str = ""
    llm_strong_model: str = ""
    llm_fallback_model: str = ""

    # Qdrant (M1)
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "atlas_chunks_v1"
    corpus_version: str = "v1"

    # Retrieval models (M1)
    embed_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"

    # Pipeline limits (M1)
    max_context_tokens: int = 6000
    max_output_tokens: int = 700
    retrieval_top_k: int = 8

    # Timeouts (M1)
    llm_timeout_seconds: float = 60.0
    qdrant_timeout_seconds: float = 10.0

    def validate_prod(self) -> list[str]:
        """Config problems that make a prod boot unsafe; empty list when clean (spec §2.3)."""
        if self.app_env != "prod":
            return []
        missing: list[str] = []
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        if self.llm_provider == "openai" and not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not self.llm_cheap_model:
            missing.append("LLM_CHEAP_MODEL")
        if not self.llm_strong_model:
            missing.append("LLM_STRONG_MODEL")
        if not self.qdrant_url:
            missing.append("QDRANT_URL")
        if not self.qdrant_api_key:
            missing.append("QDRANT_API_KEY")
        return missing


@lru_cache
def get_settings() -> Settings:
    return Settings()
