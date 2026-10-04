"""Thin in-house LLM adapter over the official SDK (spec §5): no chains, no LiteLLM.

Features: per-call timeout, bounded retries on transient errors, token/cost accounting.
Streaming, prompt-caching hooks and the structured-JSON helper arrive with later milestones.
"""

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from anthropic import (
    APIConnectionError,
    APITimeoutError,
    AsyncAnthropic,
    InternalServerError,
    RateLimitError,
)
from anthropic.types import TextBlock
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from atlas.config import Settings
from atlas.llm.pricing import PricingTable

logger = logging.getLogger("atlas.llm")

_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: float


class LLM(Protocol):
    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int,
    ) -> LLMResponse: ...


class AnthropicLLM:
    def __init__(self, api_key: str, pricing: PricingTable, timeout_seconds: float = 60.0) -> None:
        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout_seconds)
        self._pricing = pricing

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential_jitter(initial=0.5, max=8.0),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int,
    ) -> LLMResponse:
        resp = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        usage = resp.usage
        cached_tokens = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        text = "".join(block.text for block in resp.content if isinstance(block, TextBlock))
        cost_usd = self._pricing.cost_usd(
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=cached_tokens,
        )
        logger.info(
            "llm_complete model=%s in=%d out=%d cached=%d cost_usd=%.6f",
            model,
            usage.input_tokens,
            usage.output_tokens,
            cached_tokens,
            cost_usd,
        )
        return LLMResponse(
            text=text,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost_usd,
        )


_RETRYABLE_OPENAI_NAMES = {
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
}


def _is_retryable_openai(exc: BaseException) -> bool:
    """Match openai SDK retryables without importing openai at module scope."""
    return type(exc).__name__ in _RETRYABLE_OPENAI_NAMES and type(exc).__module__.startswith(
        "openai"
    )


def _cached_tokens(usage: object) -> int:
    hit = getattr(usage, "prompt_cache_hit_tokens", None)  # DeepSeek usage shape
    if hit is not None:
        return int(hit)
    details = getattr(usage, "prompt_tokens_details", None)  # OpenAI usage shape
    return int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0


class OpenAICompatLLM:
    """OpenAI-compatible chat APIs (OpenAI, DeepSeek, ...) via the official openai SDK."""

    def __init__(
        self,
        api_key: str,
        pricing: PricingTable,
        timeout_seconds: float = 60.0,
        base_url: str = "",
        http_client: Any | None = None,
    ) -> None:
        # Lazy import keeps the openai SDK off the boot path unless this provider is used.
        from openai import AsyncOpenAI

        self._client: Any = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout_seconds,
            http_client=http_client,
        )
        self._pricing = pricing

    @retry(
        retry=retry_if_exception(_is_retryable_openai),
        wait=wait_exponential_jitter(initial=0.5, max=8.0),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def complete(self, *, system: str, user: str, model: str, max_tokens: int) -> LLMResponse:
        resp = await self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = resp.usage
        input_tokens = int(usage.prompt_tokens) if usage else 0
        output_tokens = int(usage.completion_tokens) if usage else 0
        cached_tokens = _cached_tokens(usage) if usage else 0
        cost_usd = self._pricing.cost_usd(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
        )
        text = resp.choices[0].message.content or "" if resp.choices else ""
        logger.info(
            "llm_complete model=%s in=%d out=%d cached=%d cost_usd=%.6f",
            model,
            input_tokens,
            output_tokens,
            cached_tokens,
            cost_usd,
        )
        return LLMResponse(
            text=text,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost_usd,
        )


def build_llm(settings: Settings, pricing: PricingTable) -> LLM:
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set; copy .env.example to .env and fill it in"
            )
        return AnthropicLLM(
            api_key=settings.anthropic_api_key,
            pricing=pricing,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set; LLM_PROVIDER=openai requires it (e.g. DeepSeek)"
        )
    return OpenAICompatLLM(
        api_key=settings.openai_api_key,
        pricing=pricing,
        timeout_seconds=settings.llm_timeout_seconds,
        base_url=settings.openai_base_url,
    )
