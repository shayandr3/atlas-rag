"""/v1/ask, /readyz, /metrics, /admin/* — with the M4 security layer (spec §11)."""

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import REGISTRY, generate_latest
from pydantic import BaseModel, Field

from atlas.config import get_settings
from atlas.llm.pricing import PricingError
from atlas.observability.metrics import BUDGET_REMAINING, LLM_COST, RATE_LIMITED
from atlas.pipeline.ask import AskResult, Usage, run_ask
from atlas.resilience.errors import (
    AtlasError,
    BudgetExceeded,
    GuardBlocked,
    Internal,
    RateLimited,
    Unauthorized,
    UpstreamUnavailable,
)
from atlas.security.auth import Principal, check_token, resolve_principal
from atlas.security.budget import BudgetLedger
from atlas.security.guards import input_guard
from atlas.security.ratelimit import ip_of
from atlas.services import Services

logger = logging.getLogger("atlas.api")
router = APIRouter()
_bearer_scheme = HTTPBearer(auto_error=False)


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    conversation_id: str | None = None
    options: dict[str, Any] | None = None


class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: float


class CitationOut(BaseModel):
    id: str
    title: str
    section: str
    url: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    route: str
    abstained: bool
    usage: UsageOut
    request_id: str
    cache: str = "miss"
    trace: None = None


def _get_services(request: Request) -> Services:
    services: Services | None = request.app.state.services
    if services is None:
        services = Services.build(get_settings())
        request.app.state.services = services
    return services


def _principal(request: Request, creds: HTTPAuthorizationCredentials | None) -> Principal:
    principal = resolve_principal(creds.credentials if creds else None, get_settings())
    if principal is None:
        raise Unauthorized("invalid or missing API key")
    return principal


def _to_response(result: AskResult) -> AskResponse:
    return AskResponse(
        answer=result.answer,
        citations=[
            CitationOut(id=c.id, title=c.title, section=c.section, url=c.url, snippet=c.snippet)
            for c in result.citations
        ],
        route=result.route,
        abstained=result.abstained,
        usage=UsageOut(
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            cached_tokens=result.usage.cached_tokens,
            cost_usd=result.usage.cost_usd,
        ),
        request_id=result.request_id,
        cache=result.cache_status,
    )


@router.post("/v1/ask", response_model=AskResponse)
async def ask(
    payload: AskRequest,
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> AskResponse:
    services = _get_services(request)
    principal = _principal(request, creds)
    settings = services.settings

    limiter = services.rate_limiter
    scope_key = f"key:{principal.id}" if not principal.anonymous else f"ip:{await ip_of(request)}"
    allowed, retry_after = await limiter.allow(scope_key, principal.rpm)
    if not allowed:
        RATE_LIMITED.labels(scope="key" if not principal.anonymous else "ip").inc()
        raise RateLimited(retry_after=retry_after)

    ledger = BudgetLedger(settings, services.cache._redis if services.cache else None)
    budget_ok, reason = await ledger.check(principal.id, principal.daily_budget_usd)
    if not budget_ok:
        raise BudgetExceeded(reason)

    verdict = await input_guard(payload.query, services.llm, settings)
    if verdict.blocked:
        logger.info("input blocked rules=%s query_len=%d", verdict.rules, len(payload.query))
        raise GuardBlocked("input blocked by the guard")

    try:
        result = await run_ask(
            verdict.redacted_query,
            retriever=services.retriever,
            llm=services.llm,
            settings=settings,
            cache=services.cache,
            reranker=services.reranker,
            graph=services.graph,
            breakers=services.breakers,
        )
    except AtlasError:
        raise
    except PricingError as exc:
        logger.error("pricing misconfigured: %s", exc)
        raise Internal("cost accounting failed") from exc
    except Exception:
        logger.exception("ask pipeline failed")
        raise UpstreamUnavailable("upstream unavailable; try again shortly") from None

    if result.usage.cost_usd > 0:
        LLM_COST.labels(model=settings.llm_strong_model).inc(result.usage.cost_usd)
    await ledger.charge(principal.id, result.usage.cost_usd)
    daily_spent = await ledger.spent(f"budget:global:daily:{time.strftime('%Y%m%d')}")
    BUDGET_REMAINING.labels(scope="global_daily").set(
        max(settings.global_daily_budget_usd - daily_spent, 0.0)
    )
    return _to_response(result)


def _zero_usage() -> Usage:
    return Usage(input_tokens=0, output_tokens=0, cached_tokens=0, cost_usd=0.0)


@router.get("/readyz")
async def readyz() -> Response:
    settings = get_settings()
    if not settings.qdrant_url:
        return Response(
            content='{"status":"not_ready","qdrant":"not_configured"}',
            status_code=503,
            media_type="application/json",
        )
    from atlas.retrieval.qdrant_repo import QdrantRepo

    repo = QdrantRepo(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
        timeout_seconds=settings.qdrant_timeout_seconds,
    )
    reachable = await repo.ping()
    body = (
        '{"status":"ok","qdrant":"reachable"}'
        if reachable
        else '{"status":"degraded","qdrant":"unreachable"}'
    )
    return Response(
        content=body, status_code=200 if reachable else 503, media_type="application/json"
    )


@router.get("/metrics")
async def metrics(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> Response:
    settings = get_settings()
    token = creds.credentials if creds else None
    dev_open = settings.app_env == "dev" and not settings.metrics_bearer_token
    if not dev_open and not check_token(token, settings.metrics_bearer_token):
        raise Unauthorized("metrics token required")
    return Response(content=generate_latest(REGISTRY), media_type="text/plain; version=0.0.4")


@router.post("/admin/cache/flush")
async def cache_flush(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict[str, Any]:
    settings = get_settings()
    if not check_token(creds.credentials if creds else None, settings.admin_token):
        raise Unauthorized("admin token required")
    services = _get_services(request)
    if services.cache is None:
        return {"flushed": 0}
    flushed = 0
    try:
        async for key in services.cache._redis._client.scan_iter(match="resp:*"):
            await services.cache._redis._client.delete(key)
            flushed += 1
        async for key in services.cache._redis._client.scan_iter(match="sem:*"):
            await services.cache._redis._client.delete(key)
            flushed += 1
    except Exception:
        logger.warning("cache flush failed (fail-open)", exc_info=True)
    return {"flushed": flushed}


@router.get("/admin/stats")
async def admin_stats(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict[str, Any]:
    settings = get_settings()
    if not check_token(creds.credentials if creds else None, settings.admin_token):
        raise Unauthorized("admin token required")
    services = _get_services(request)
    ledger = BudgetLedger(settings, services.cache._redis if services.cache else None)
    daily = await ledger.spent(f"budget:global:daily:{time.strftime('%Y%m%d')}")
    monthly = await ledger.spent(f"budget:global:monthly:{time.strftime('%Y%m')}")
    return {
        "corpus_version": settings.corpus_version,
        "cache_up": services.cache.redis_up if services.cache else False,
        "global_daily_spent_usd": round(daily, 6),
        "global_monthly_spent_usd": round(monthly, 6),
        "global_daily_limit_usd": settings.global_daily_budget_usd,
        "global_monthly_limit_usd": settings.global_monthly_budget_usd,
    }


@router.post("/admin/budget/reset")
async def budget_reset(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict[str, Any]:
    settings = get_settings()
    if not check_token(creds.credentials if creds else None, settings.admin_token):
        raise Unauthorized("admin token required")
    services = _get_services(request)
    ledger = BudgetLedger(settings, services.cache._redis if services.cache else None)
    return {"cleared": await ledger.reset_all()}
