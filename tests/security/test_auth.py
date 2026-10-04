from fastapi.testclient import TestClient

from atlas.config import Settings, get_settings
from atlas.security.auth import hash_key, load_keys, resolve_principal, verify_key


def _keys_json(raw: str, pepper: str = "pepper") -> str:
    import json

    return json.dumps([{"id": "k1", "sha256_hash": hash_key(raw, pepper), "rpm": 5}])


def test_hash_is_peppered_and_stable() -> None:
    assert hash_key("raw", "p1") != hash_key("raw", "p2")
    assert hash_key("raw", "p1") == hash_key("raw", "p1")


def test_verify_key_roundtrip() -> None:
    records = load_keys(_keys_json("atlas_secret"))
    assert len(records) == 1

    assert verify_key("atlas_secret", records, "pepper") is records[0]
    assert verify_key("wrong", records, "pepper") is None
    assert verify_key("atlas_secret", records, "other-pepper") is None


def test_malformed_keys_json_is_tolerated() -> None:
    assert load_keys("not json") == []
    assert load_keys('[{"no_id": true}]') == []


def test_resolve_principal_anonymous_only_in_dev() -> None:
    dev = Settings(_env_file=None, app_env="dev")
    prod = Settings(_env_file=None, app_env="prod")

    principal = resolve_principal(None, dev)
    assert principal is not None and principal.anonymous
    assert resolve_principal(None, prod) is None


def test_resolve_principal_invalid_bearer_rejected() -> None:
    settings = Settings(_env_file=None, api_keys_json=_keys_json("right"), api_key_pepper="pepper")

    assert resolve_principal("right", settings) is not None
    assert resolve_principal("wrong", settings) is None


def test_api_ask_requires_key_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("API_KEYS_JSON", _keys_json("atlas_right"))
    monkeypatch.setenv("API_KEY_PEPPER", "pepper")
    get_settings.cache_clear()
    try:
        from tests.test_api_ask import make_services

        from atlas.main import create_app

        client = TestClient(create_app(services=make_services()))

        assert client.post("/v1/ask", json={"query": "q"}).status_code == 401
        assert (
            client.post(
                "/v1/ask", json={"query": "q"}, headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 401
        )
        ok = client.post(
            "/v1/ask", json={"query": "q"}, headers={"Authorization": "Bearer atlas_right"}
        )
        assert ok.status_code == 200
    finally:
        get_settings.cache_clear()
