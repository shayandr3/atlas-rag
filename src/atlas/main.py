"""FastAPI app factory."""

from fastapi import FastAPI

from atlas import __version__
from atlas.api.routes import router
from atlas.config import get_settings
from atlas.services import Services


def create_app(services: Services | None = None) -> FastAPI:
    settings = get_settings()
    problems = settings.validate_prod()
    if problems:
        raise RuntimeError(f"unsafe prod config: missing {', '.join(problems)}")
    app = FastAPI(
        title="atlas-rag",
        version=__version__,
        docs_url=None if settings.app_env == "prod" else "/docs",
    )
    app.state.services = services
    app.include_router(router)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Liveness probe: cheap, no dependency checks (readiness is /readyz)."""
        return {"status": "ok"}

    return app


app = create_app()
