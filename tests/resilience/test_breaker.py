import asyncio

import pytest
from prometheus_client import REGISTRY

from atlas.resilience.breaker import Breakers, CircuitBreaker, CircuitOpenError


async def test_breaker_opens_after_threshold() -> None:
    br = CircuitBreaker("llm", failure_threshold=3, window_s=60, reset_timeout_s=0.05)

    async def fail() -> None:
        raise RuntimeError("dependency dead")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await br.call(fail)

    assert br.state == "open"
    with pytest.raises(CircuitOpenError):
        await br.call(fail)  # open circuit fails fast, fn never runs


async def test_breaker_half_open_probe_success_closes() -> None:
    br = CircuitBreaker("llm", failure_threshold=2, window_s=60, reset_timeout_s=0.05)

    async def fail() -> None:
        raise RuntimeError("dead")

    async def ok() -> str:
        return "fine"

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await br.call(fail)
    await asyncio.sleep(0.06)

    assert br.state == "half_open"
    assert await br.call(ok) == "fine"
    assert br.state == "closed"


async def test_breaker_half_open_probe_failure_reopens() -> None:
    br = CircuitBreaker("qdrant", failure_threshold=2, window_s=60, reset_timeout_s=0.05)

    async def fail() -> None:
        raise RuntimeError("still dead")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await br.call(fail)
    await asyncio.sleep(0.06)
    assert br.state == "half_open"

    with pytest.raises(RuntimeError):
        await br.call(fail)
    assert br.state == "open"


def test_breaker_state_gauge_exported() -> None:
    Breakers.build()
    value = REGISTRY.get_sample_value("atlas_circuit_state", {"dependency": "llm"})
    assert value is not None
    assert value == 0  # closed


async def test_breakers_build_gives_four_dependencies() -> None:
    breakers = Breakers.build()
    assert {b.name for b in (breakers.llm, breakers.qdrant, breakers.redis, breakers.rerank)} == {
        "llm",
        "qdrant",
        "redis",
        "rerank",
    }
