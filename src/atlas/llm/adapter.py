"""Thin in-house LLM adapter over the official SDK (spec §5): no chains, no LiteLLM.

Features: per-call timeout, bounded retries on transient errors, token/cost accounting.
Streaming, prompt-caching hooks and the structured-JSON helper arrive with later milestones.
"""

import logging
from dataclasses import dataclass
from typing import Protocol

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
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

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
