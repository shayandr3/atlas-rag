"""Chaos-style ladder tests (spec §12): kill each dependency, assert the rung behaves."""

import dataclasses

import pytest
from prometheus_client import REGISTRY
from tests.fakes import (
    DyingRetriever,
    FakeEmbedder,
    FakeLLM,
    FakeReranker,
    FakeRetriever,
    make_chunks,
)

from atlas.cache.client import FailOpenRedis
from atlas.cache.service import CacheService
from atlas.config import Settings
from atlas.llm.adapter import LLMResponse
from atlas.pipeline.ask import run_ask
from atlas.resilience.breaker import Breakers
from atlas.resilience.errors import UpstreamUnavailable
from atlas.retrieval.chunks import Chunk


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def _broken_cache() -> CacheService:
    class BrokenClient:
        def __getattr__(self, name):
            async def _raise(*args, **kwargs):
                raise RuntimeError("down")

            return _raise

    return CacheService(Settings(_env_file=None), FailOpenRedis(client=BrokenClient()))


class PickyLLM:
    """Strong model is down; the fallback model answers."""

    def __init__(self) -> None:
        self.models_used: list[str] = []

    async def complete(self, *, system: str, user: str, model: str, max_tokens: int) -> LLMResponse:
        self.models_used.append(model)
        if model == "strong-x":
            raise RuntimeError("strong model 500")
        return LLMResponse(
            text="Fallback answer [S1].",
            model=model,
            input_tokens=50,
            output_tokens=10,
            cached_tokens=0,
            cost_usd=0.0,
        )


class DeadLLM:
    async def complete(self, *, system: str, user: str, model: str, max_tokens: int) -> LLMResponse:
        raise RuntimeError("provider blackholed")


async def test_rung1_reranker_down_falls_back_to_retrieval_order() -> None:
    result = await run_ask(
        "what is rag?",
        retriever=FakeRetriever(make_chunks()),
        llm=FakeLLM("Answer [S1]."),
        settings=_settings(),
        reranker=FakeReranker(fail=True).rerank,
        breakers=Breakers.build(),
        request_id="r1",
    )

    assert result.answer == "Answer [S1]."
    assert result.degraded == ["rerank"]
    assert (REGISTRY.get_sample_value("atlas_degraded_total", {"rung": "rerank"}) or 0) >= 1


async def test_rung2_strong_model_down_fallback_model_answers() -> None:
    settings = _settings(llm_strong_model="strong-x", llm_fallback_model="fallback-x")
    llm = PickyLLM()

    result = await run_ask(
        "what is rag?",
        retriever=FakeRetriever(make_chunks()),
        llm=llm,
        settings=settings,
        breakers=Breakers.build(),
        request_id="r2",
    )

    assert result.answer == "Fallback answer [S1]."
    assert "llm_fallback" in result.degraded
    assert "strong-x" in llm.models_used and "fallback-x" in llm.models_used


async def test_rung3_llm_down_extractive_fallback_with_citations() -> None:
    result = await run_ask(
        "what is rag?",
        retriever=FakeRetriever(make_chunks()),
        llm=DeadLLM(),
        settings=_settings(),
        breakers=Breakers.build(),
        request_id="r3",
    )

    assert result.degraded and "llm_extractive" in result.degraded
    assert result.abstained is False
    assert len(result.citations) == 2  # top evidence served with citations
    assert "Evidence one." in result.answer


async def test_rung4_qdrant_down_serves_from_cache_then_503() -> None:
    settings = _settings(semantic_cache_enabled=False)
    cache = CacheService(
        settings,
        FailOpenRedis(client=__import__("fakeredis").aioredis.FakeRedis()),
        embedder=FakeEmbedder(),
    )
    retriever = DyingRetriever(make_chunks())

    first = await run_ask(
        "cached question?",
        retriever=retriever,
        llm=FakeLLM("Answer [S1]."),
        settings=settings,
        cache=cache,
        breakers=Breakers.build(),
        request_id="r4a",
    )
    assert first.cache_status == "miss"

    second = await run_ask(
        "cached question?",
        retriever=retriever,
        llm=FakeLLM("Answer [S1]."),
        settings=settings,
        cache=cache,
        breakers=Breakers.build(),
        request_id="r4b",
    )
    assert second.cache_status == "hit"  # served entirely from cache while qdrant dead

    with pytest.raises(UpstreamUnavailable) as excinfo:
        await run_ask(
            "uncached different question?",
            retriever=retriever,
            llm=FakeLLM("Answer [S1]."),
            settings=settings,
            cache=cache,
            breakers=Breakers.build(),
            request_id="r4c",
        )
    assert excinfo.value.retry_after == 30


async def test_rung5_redis_down_bypasses_cache_still_answers() -> None:
    result = await run_ask(
        "q",
        retriever=FakeRetriever(make_chunks()),
        llm=FakeLLM("Answer [S1]."),
        settings=_settings(),
        cache=_broken_cache(),
        breakers=Breakers.build(),
        request_id="r5",
    )

    assert result.cache_status == "miss"
    assert result.answer == "Answer [S1]."


def _chunks() -> list[Chunk]:
    return [dataclasses.replace(c, score=0.5) for c in make_chunks()]


async def test_grade_loop_through_broken_qdrant_raises_upstream() -> None:
    settings = _settings(grade_skip_threshold=0.99)
    retriever = DyingRetriever(_chunks())
    retriever.calls = 1  # dependency already dead

    with pytest.raises(UpstreamUnavailable):
        await run_ask(
            "what is rag?",
            retriever=retriever,
            llm=FakeLLM("Answer [S1]."),
            settings=settings,
            breakers=Breakers.build(),
            request_id="r6",
        )
