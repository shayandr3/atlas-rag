import asyncio

import pytest
from fakeredis import aioredis as fakeredis_aioredis
from prometheus_client import REGISTRY
from tests.fakes import FakeEmbedder, FakeLLM, FakeRetriever, make_chunks

from atlas.cache.client import FailOpenRedis
from atlas.cache.keys import CacheKeys, config_fingerprint, norm_query
from atlas.cache.service import CacheService
from atlas.config import Settings
from atlas.pipeline.ask import run_ask


def make_cache(**setting_overrides) -> tuple[CacheService, Settings]:
    settings = Settings(_env_file=None, **setting_overrides)
    client = fakeredis_aioredis.FakeRedis()
    cache = CacheService(settings, FailOpenRedis(client=client), embedder=FakeEmbedder())
    return cache, settings


def sample_value(name: str, labels: dict[str, str]) -> float | None:
    return REGISTRY.get_sample_value(name, labels)


def test_norm_query_is_stable() -> None:
    assert norm_query("  What   is RAG? ") == "what is rag?"
    keys = CacheKeys(Settings(_env_file=None))
    assert keys.response("What is RAG?") == keys.response("what   is RAG?")


def test_keys_are_versioned_by_corpus_and_config() -> None:
    keys_a = CacheKeys(Settings(_env_file=None, corpus_version="v1"))
    keys_b = CacheKeys(Settings(_env_file=None, corpus_version="v2"))
    keys_c = CacheKeys(Settings(_env_file=None, max_output_tokens=999))
    assert keys_a.response("q") != keys_b.response("q")
    assert keys_a.response("q") != keys_c.response("q")
    assert config_fingerprint(Settings(_env_file=None)) != config_fingerprint(
        Settings(_env_file=None, llm_strong_model="other")
    )


async def test_exact_hit_skips_llm_and_reports_cost_saved() -> None:
    cache, settings = make_cache()
    llm = FakeLLM("Answer [S1].")
    retriever = FakeRetriever(make_chunks())

    first = await run_ask("q", retriever=retriever, llm=llm, settings=settings, cache=cache)
    second = await run_ask("q", retriever=retriever, llm=llm, settings=settings, cache=cache)

    assert first.cache_status == "miss"
    assert second.cache_status == "hit"
    assert second.answer == "Answer [S1]."
    assert llm.calls == 1
    saved = sample_value("atlas_cache_cost_saved_usd_total", {"layer": "response"})
    assert saved is not None and saved >= 0.001
    assert (
        sample_value("atlas_cache_requests_total", {"layer": "response", "result": "hit"}) or 0
    ) >= 1


async def test_single_flight_collapses_concurrent_misses() -> None:
    cache, settings = make_cache()
    llm = FakeLLM("Answer [S1].", delay_s=0.3)
    retriever = FakeRetriever(make_chunks())

    results = await asyncio.gather(
        run_ask("same query", retriever=retriever, llm=llm, settings=settings, cache=cache),
        run_ask("same query", retriever=retriever, llm=llm, settings=settings, cache=cache),
    )

    assert llm.calls == 1
    statuses = sorted(r.cache_status for r in results)
    assert statuses == ["hit", "miss"]


async def test_semantic_hit_on_paraphrase_with_similar_vector() -> None:
    cache, settings = make_cache()
    llm = FakeLLM("Answer [S1].")
    retriever = FakeRetriever(make_chunks())

    await run_ask(
        "what is hybrid retrieval", retriever=retriever, llm=llm, settings=settings, cache=cache
    )
    paraphrased = await run_ask(
        "hybrid retrieval — what is it?",
        retriever=retriever,
        llm=llm,
        settings=settings,
        cache=cache,
    )

    assert paraphrased.cache_status == "semantic"
    assert llm.calls == 1


async def test_semantic_skips_time_sensitive_queries() -> None:
    cache, settings = make_cache()
    llm = FakeLLM("Answer [S1].")
    retriever = FakeRetriever(make_chunks())

    await run_ask("what is reranking", retriever=retriever, llm=llm, settings=settings, cache=cache)
    second = await run_ask(
        "the latest news on reranking", retriever=retriever, llm=llm, settings=settings, cache=cache
    )

    assert second.cache_status == "miss"
    assert llm.calls == 2


async def test_embedding_cache_l0() -> None:
    cache, _ = make_cache()
    embedder = FakeEmbedder()
    cache._embedder = embedder

    await cache.embed_query("hello")
    await cache.embed_query("hello")

    assert embedder.calls == 1


class BrokenClient:
    def __getattr__(self, name):
        async def _raise(*args, **kwargs):
            raise RuntimeError("redis down")

        return _raise


async def test_fail_open_on_dead_redis() -> None:
    settings = Settings(_env_file=None)
    cache = CacheService(settings, FailOpenRedis(client=BrokenClient()), embedder=FakeEmbedder())
    llm = FakeLLM("Answer [S1].")
    retriever = FakeRetriever(make_chunks())

    result = await run_ask("q", retriever=retriever, llm=llm, settings=settings, cache=cache)

    assert result.cache_status == "miss"
    assert llm.calls == 1

    for _ in range(6):
        await cache.get_response("q")
    assert cache.redis_up is False  # down-circuit trips, later ops skip instantly


def test_normal_settings_use_cache_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.semantic_cache_threshold == pytest.approx(0.95)
    assert settings.cache_exact_ttl_hours == 24
