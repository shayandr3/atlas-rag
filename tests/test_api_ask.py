from fastapi.testclient import TestClient
from tests.fakes import FakeLLM, FakeRetriever, make_chunks

from atlas.config import Settings, get_settings
from atlas.llm.pricing import PricingTable
from atlas.main import create_app
from atlas.services import Services


def make_services() -> Services:
    return Services(
        settings=Settings(_env_file=None),
        pricing=PricingTable({}),
        embedder=None,  # type: ignore[arg-type]
        repo=None,  # type: ignore[arg-type]
        llm=FakeLLM("Answer [S1]."),
        retriever=FakeRetriever(make_chunks()),
    )


def test_ask_endpoint_returns_answer_citations_usage(monkeypatch) -> None:
    monkeypatch.setenv("API_KEYS_JSON", "")  # dev open mode: no keys configured
    get_settings.cache_clear()
    client = TestClient(create_app(services=make_services()))

    response = client.post("/v1/ask", json={"query": "what is retrieval?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Answer [S1]."
    assert body["citations"][0]["id"] == "c1"
    assert body["usage"]["input_tokens"] == 100
    assert body["usage"]["cost_usd"] == 0.001
    assert body["route"] == "simple"
    assert body["trace"] is None
    assert body["request_id"]


def test_ask_endpoint_rejects_empty_query(monkeypatch) -> None:
    monkeypatch.setenv("API_KEYS_JSON", "")
    get_settings.cache_clear()
    client = TestClient(create_app(services=make_services()))

    response = client.post("/v1/ask", json={"query": ""})

    assert response.status_code == 422


def test_readyz_reports_not_configured_without_qdrant(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "")
    get_settings.cache_clear()
    try:
        client = TestClient(create_app())
        response = client.get("/readyz")
        assert response.status_code == 503
        assert response.json()["qdrant"] == "not_configured"
    finally:
        get_settings.cache_clear()
