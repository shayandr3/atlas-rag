"""FastAPI app factory."""

from fastapi import FastAPI

from atlas import __version__
from atlas.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="atlas-rag",
        version=__version__,
        docs_url=None if settings.app_env == "prod" else "/docs",
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Liveness probe: cheap, no dependency checks (readiness lands with /readyz in M1)."""
        return {"status": "ok"}

    return app


app = create_app()
