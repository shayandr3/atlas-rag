"""FastAPI app factory."""

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.middleware("http")
    async def security_envelope(
        request: Request, call_next: Callable[[Request], Awaitable[JSONResponse]]
    ) -> JSONResponse:
        # request hygiene (spec §11.1): body-size ceiling + security headers
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.max_body_bytes:
            return JSONResponse({"detail": "request too large"}, status_code=413)
        response: JSONResponse = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'"
        if settings.app_env == "prod":
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Liveness probe: cheap, no dependency checks (readiness is /readyz)."""
        return {"status": "ok"}

    return app


app = create_app()
