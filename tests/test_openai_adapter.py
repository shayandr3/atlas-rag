import httpx
import pytest

from atlas.config import Settings
from atlas.llm.adapter import OpenAICompatLLM, build_llm
from atlas.llm.pricing import ModelPrice, PricingTable


def test_build_llm_selects_openai_compat() -> None:
    settings = Settings(_env_file=None, llm_provider="openai", openai_api_key="k")

    llm = build_llm(settings, PricingTable({}))

    assert isinstance(llm, OpenAICompatLLM)


def test_build_llm_openai_provider_requires_key() -> None:
    settings = Settings(_env_file=None, llm_provider="openai", openai_api_key="")

    with pytest.raises(RuntimeError):
        build_llm(settings, PricingTable({}))


async def test_openai_compat_adapter_maps_usage_and_cost() -> None:
    """Injected MockTransport instead of respx: the openai SDK's own client
    bypasses respx's global patching, so requests would leak to the network."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "id": "1",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "pong"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100000,
                    "completion_tokens": 10000,
                    "total_tokens": 110000,
                    "prompt_cache_hit_tokens": 50000,
                    "prompt_cache_miss_tokens": 50000,
                },
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.deepseek.com"
    )
    table = PricingTable(
        {
            "deepseek-flash": ModelPrice(
                input_per_1m_usd=0.30, cached_input_per_1m_usd=0.006, output_per_1m_usd=1.20
            )
        }
    )
    llm = OpenAICompatLLM("k", table, base_url="https://api.deepseek.com", http_client=client)
    try:
        response = await llm.complete(system="s", user="u", model="deepseek-flash", max_tokens=8)
    finally:
        await client.aclose()

    assert len(seen) == 1
    assert response.text == "pong"
    assert response.input_tokens == 100000
    assert response.cached_tokens == 50000
    assert response.output_tokens == 10000
    # (50000 miss * 0.30 + 50000 hit * 0.006 + 10000 out * 1.20) / 1e6
    assert response.cost_usd == pytest.approx((15000.0 + 300.0 + 12000.0) / 1e6)
