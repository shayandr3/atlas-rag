"""FastAPI app factory."""

import sys
import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from atlas import __version__
from atlas.api.routes import router
from atlas.config import get_settings
from atlas.observability.metrics import (
    BUILD_INFO,
    HTTP_DURATION,
    HTTP_REQUESTS,
    INFLIGHT,
    PROCESS_RSS,
)
from atlas.resilience.errors import AtlasError, ValidationFailed
from atlas.services import Services

if sys.platform == "win32":  # stdlib-only RSS reading; psutil stays out of the image

    def _rss_bytes() -> int:
        import ctypes
        import ctypes.wintypes as wintypes

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        k32 = ctypes.windll.kernel32
        k32.K32GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PMC),
            wintypes.DWORD,
        ]
        k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        pmc = PMC()
        pmc.cb = ctypes.sizeof(pmc)
        if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return 0
        return int(pmc.WorkingSetSize)

else:

    def _rss_bytes() -> int:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


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
    BUILD_INFO.labels(version=__version__, corpus_version=settings.corpus_version).set(1)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.exception_handler(AtlasError)
    async def atlas_problem_handler(request: Request, exc: AtlasError) -> JSONResponse:
        """RFC 9457 problem+json with stable codes; no internals leak (spec §12)."""
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
        return JSONResponse(
            status_code=exc.status,
            content=exc.problem(),
            media_type="application/problem+json",
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        problem = ValidationFailed("invalid request body").problem()
        return JSONResponse(status_code=422, content=problem, media_type="application/problem+json")

    @app.middleware("http")
    async def observability_envelope(
        request: Request, call_next: Callable[[Request], Awaitable[JSONResponse]]
    ) -> JSONResponse:
        # request hygiene (spec §11.1) + full HTTP instrumentation (spec §13.2)
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.max_body_bytes:
            HTTP_REQUESTS.labels(route=request.url.path, method=request.method, status="413").inc()
            return JSONResponse({"detail": "request too large"}, status_code=413)
        route = request.url.path
        INFLIGHT.inc()
        started = time.perf_counter()
        try:
            response: JSONResponse = await call_next(request)
        finally:
            INFLIGHT.dec()
        elapsed = time.perf_counter() - started
        HTTP_REQUESTS.labels(
            route=route, method=request.method, status=str(response.status_code)
        ).inc()
        HTTP_DURATION.labels(route=route).observe(elapsed)
        PROCESS_RSS.set(_rss_bytes())
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
