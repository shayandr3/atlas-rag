"""/v1/ask and /readyz. Auth, rate limits and guards arrive in M4 (spec §11, §17)."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from atlas.config import get_settings
from atlas.llm.pricing import PricingError
from atlas.pipeline.ask import AskResult, run_ask
from atlas.retrieval.qdrant_repo import QdrantRepo
from atlas.services import Services

logger = logging.getLogger("atlas.api")
router = APIRouter()


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
    trace: None = None


def _get_services(request: Request) -> Services:
    services: Services | None = request.app.state.services
    if services is None:
        services = Services.build(get_settings())
        request.app.state.services = services
    return services


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
    )


@router.post("/v1/ask", response_model=AskResponse)
async def ask(payload: AskRequest, request: Request) -> AskResponse:
    services = _get_services(request)
    try:
        result = await run_ask(
            payload.query,
            retriever=services.retriever,
            llm=services.llm,
            settings=services.settings,
        )
    except PricingError as exc:
        logger.error("pricing misconfigured: %s", exc)
        raise HTTPException(status_code=500, detail=f"cost accounting failed: {exc}") from exc
    except Exception:
        logger.exception("ask pipeline failed")
        raise HTTPException(
            status_code=503, detail="upstream unavailable; try again shortly"
        ) from None
    return _to_response(result)


@router.get("/readyz")
async def readyz() -> Response:
    settings = get_settings()
    if not settings.qdrant_url:
        return Response(
            content='{"status":"not_ready","qdrant":"not_configured"}',
            status_code=503,
            media_type="application/json",
        )
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
