from fastapi.testclient import TestClient

from atlas.main import app, create_app


def test_healthz_returns_ok() -> None:
    client = TestClient(create_app())

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_module_app_is_the_factory_product() -> None:
    assert app.title == "atlas-rag"
    assert app.version == "0.1.0"
