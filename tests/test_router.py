import pytest
from tests.fakes import FakeLLM

from atlas.config import Settings
from atlas.pipeline.router import route_query

_JSON_ROUTE = '{"route": "multi_hop", "confidence": 0.8}'


def _settings() -> Settings:
    return Settings(_env_file=None)


async def test_greeting_routes_without_llm() -> None:
    llm = FakeLLM("should not be called")

    decision = await route_query("hello!", llm, _settings())

    assert decision.route == "no_retrieval"
    assert llm.calls == 0


async def test_short_query_is_simple_without_llm() -> None:
    llm = FakeLLM("should not be called")

    decision = await route_query("what is rag?", llm, _settings())

    assert decision.route == "simple"
    assert llm.calls == 0


async def test_comparative_query_confirmed_by_classifier() -> None:
    llm = FakeLLM(_JSON_ROUTE)

    decision = await route_query(
        "compare dense retrieval and BM25: how did each evolve after 2020?",
        llm,
        _settings(),
    )

    assert decision.route == "multi_hop"
    assert llm.calls == 1


async def test_classifier_failure_falls_back_to_heuristic() -> None:
    llm = FakeLLM("not json at all")

    decision = await route_query(
        "compare hybrid and dense retrieval and explain how they differ",
        llm,
        _settings(),
    )

    assert decision.route == "multi_hop"  # regex heuristic candidate stands


async def test_route_decision_is_cached() -> None:
    from tests.test_cache import make_cache

    cache, _ = make_cache()
    llm = FakeLLM(_JSON_ROUTE)
    query = "compare A and B and explain how one influenced the other"

    first = await route_query(query, llm, _settings(), cache)
    second = await route_query(query, llm, _settings(), cache)

    assert first.route == "multi_hop"
    assert second.route == "multi_hop"
    assert second.reason == "cached"
    assert llm.calls == 1


def test_unallowed_route_rejected() -> None:
    from atlas.pipeline.router import _ALLOWED

    assert "simple" in _ALLOWED and "multi_hop" in _ALLOWED
    with pytest.raises(AssertionError):
        assert "make_coffee" in _ALLOWED
