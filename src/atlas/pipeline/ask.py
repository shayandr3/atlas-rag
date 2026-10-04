"""Query pipeline steps + sequential orchestration (spec §8).

Steps are framework-free functions shared by the sequential runner below and the
LangGraph wiring (pipeline/graph.py, spec §4.1). M3 adds: adaptive routing,
cross-encoder reranking with degradation, CRAG corrective grading, multi-hop
decomposition, and sampled faithfulness.
"""

import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from atlas.cache.service import CacheService
from atlas.config import Settings
from atlas.llm.adapter import LLMResponse
from atlas.llm.prompts import ANSWER_SYSTEM, CANARY_TOKEN
from atlas.observability.metrics import DEGRADED_TOTAL
from atlas.pipeline.faithfulness import maybe_check_faithfulness
from atlas.pipeline.grade import grade_chunks, rewrite_query, should_skip_grading
from atlas.pipeline.router import RouteDecision, route_query
from atlas.resilience.breaker import Breakers
from atlas.resilience.errors import UpstreamUnavailable
from atlas.retrieval.chunks import Chunk
from atlas.retrieval.embedders import QueryEmbedding
from atlas.retrieval.rerank import _cap_per_doc

logger = logging.getLogger("atlas.pipeline")

_CITATION_RE = re.compile(r"\[S(\d+)\]")
_CHARS_PER_TOKEN = 4
_SNIPPET_CHARS = 280

RerankerFn = Callable[[str, list[Chunk]], Any]  # async (query, chunks) -> list[Chunk]


@dataclass(frozen=True)
class Citation:
    id: str
    title: str
    section: str
    url: str
    snippet: str


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: float


@dataclass(frozen=True)
class AskResult:
    answer: str
    citations: list[Citation]
    route: str
    abstained: bool
    usage: Usage
    request_id: str
    cache_status: str = "miss"
    degraded: list[str] = field(default_factory=list)


class Retriever(Protocol):
    async def search(
        self, query: str, *, limit: int, embedding: QueryEmbedding | None = None
    ) -> list[Chunk]: ...


class LLM(Protocol):
    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int,
    ) -> LLMResponse: ...


def pack_context(chunks: list[Chunk], max_tokens: int) -> str:
    """Token-budgeted packing; each chunk labelled [S1]..[Sn] with title and section."""
    budget = max_tokens * _CHARS_PER_TOKEN
    blocks: list[str] = []
    used = 0
    for index, chunk in enumerate(chunks, start=1):
        block = f"[S{index}] {chunk.title} › {chunk.section_path}\n{chunk.text}"
        if used + len(block) > budget and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def extract_labels(answer: str) -> set[int]:
    return {int(match) for match in _CITATION_RE.findall(answer)}


def build_citations(answer: str, chunks: list[Chunk]) -> list[Citation]:
    by_label = dict(enumerate(chunks, start=1))
    citations: list[Citation] = []
    for label in sorted(extract_labels(answer)):
        chunk = by_label.get(label)
        if chunk is None:
            continue  # unknown [S#] dropped; full citation guard lands in M4
        citations.append(
            Citation(
                id=chunk.chunk_id,
                title=chunk.title,
                section=chunk.section_path,
                url=chunk.source_url,
                snippet=chunk.text[:_SNIPPET_CHARS],
            )
        )
    return citations


def result_payload(result: AskResult) -> dict[str, Any]:
    return {
        "answer": result.answer,
        "citations": [asdict(c) for c in result.citations],
        "route": result.route,
        "abstained": result.abstained,
        "usage": asdict(result.usage),
        "degraded": list(result.degraded),
    }


def result_from_payload(payload: dict[str, Any], rid: str, cache_status: str) -> AskResult:
    usage_raw = payload.get("usage") or {}
    return AskResult(
        answer=str(payload.get("answer", "")),
        citations=[Citation(**c) for c in payload.get("citations", [])],
        route=str(payload.get("route", "simple")),
        abstained=bool(payload.get("abstained", False)),
        usage=Usage(
            input_tokens=int(usage_raw.get("input_tokens", 0)),
            output_tokens=int(usage_raw.get("output_tokens", 0)),
            cached_tokens=int(usage_raw.get("cached_tokens", 0)),
            cost_usd=float(usage_raw.get("cost_usd", 0.0)),
        ),
        request_id=rid,
        cache_status=cache_status,
        degraded=[str(d) for d in payload.get("degraded", [])],
    )


def _zero_usage() -> Usage:
    return Usage(input_tokens=0, output_tokens=0, cached_tokens=0, cost_usd=0.0)


def _extractive_answer(chunks: list[Chunk], route: str, degraded: list[str], rid: str) -> AskResult:
    """Ladder rung 3: LLM unavailable -> serve top evidence snippets with citations."""
    DEGRADED_TOTAL.labels(rung="extractive").inc()
    top = chunks[:3]
    snippets = "\n\n".join(f"[S{i}] {c.title}: {c.text[:300]}" for i, c in enumerate(top, start=1))
    answer = (
        "Answer generation is temporarily unavailable; here are the most relevant "
        f"passages found for your question:\n\n{snippets}"
    )
    cited = [
        Citation(
            id=c.chunk_id,
            title=c.title,
            section=c.section_path,
            url=c.source_url,
            snippet=c.text[:_SNIPPET_CHARS],
        )
        for c in top
    ]
    return AskResult(
        answer=answer,
        citations=cited,
        route=route,
        abstained=False,
        usage=_zero_usage(),
        request_id=rid,
        degraded=degraded,
    )


# ---- steps -----------------------------------------------------------------------


async def step_cache_lookup(query: str, cache: CacheService | None) -> AskResult | None:
    if cache is None:
        return None
    payload = await cache.get_response(query)
    if payload is not None:
        return result_from_payload(payload, uuid.uuid4().hex, cache_status="hit")
    return None


async def step_embed(query: str, cache: CacheService | None) -> QueryEmbedding | None:
    if cache is None:
        return None
    return await cache.embed_query(query)


async def step_semantic(
    query: str, emb: QueryEmbedding | None, cache: CacheService | None
) -> AskResult | None:
    if cache is None:
        return None
    payload = await cache.get_semantic(query, emb)
    if payload is not None:
        return result_from_payload(payload, uuid.uuid4().hex, cache_status="semantic")
    return None


def step_route(query: str, llm: Any, settings: Settings, cache: CacheService | None) -> Any:
    """Async adaptive routing (heuristic + optional cached classifier)."""
    return route_query(query, llm, settings, cache)


def canned_response(decision: RouteDecision, rid: str) -> AskResult:
    answers = {
        "no_retrieval": (
            "Hello! Ask me anything about this demo's arXiv corpus on LLMs and retrieval."
        ),
        "out_of_scope": (
            "This demo only answers questions about the ingested arXiv corpus "
            "(LLMs, retrieval, ML)."
        ),
        "unsafe": "This request was blocked by the input guard.",
    }
    answer = answers.get(decision.route, answers["out_of_scope"])
    return AskResult(
        answer=answer,
        citations=[],
        route=decision.route,
        abstained=decision.route != "no_retrieval",
        usage=_zero_usage(),
        request_id=rid,
    )


async def step_retrieve(
    query: str,
    retriever: Retriever,
    settings: Settings,
    cache: CacheService | None,
    emb: QueryEmbedding | None,
    reranker: RerankerFn | None,
    breakers: Breakers | None = None,
) -> tuple[list[Chunk], list[str]]:
    """L3-cached retrieval + rerank with diversity cap. Returns (chunks, degraded)."""
    limit = _retrieval_limit(settings, reranker)
    if cache is not None:
        cached = await cache.get_chunks(query, limit)
        if cached is not None:
            return [Chunk(**c) for c in cached], []
    if breakers is not None:
        try:
            chunks = await breakers.qdrant.call(retriever.search, query, limit=limit, embedding=emb)
        except Exception as exc:
            raise UpstreamUnavailable("retrieval is unavailable") from exc
    else:
        chunks = await retriever.search(query, limit=limit, embedding=emb)
    if not chunks:
        return [], []
    if reranker is None:
        ranked = chunks[: settings.retrieval_top_k]
        degraded: list[str] = []
    else:
        try:
            ranked = await reranker(query, chunks[: settings.rerank_top_in])
            ranked = _cap_per_doc(ranked, per_doc=3)[: settings.rerank_top_out]
            degraded = []
        except Exception:
            logger.warning("reranker failed; falling back to retrieval order", exc_info=True)
            ranked, degraded = chunks[: settings.retrieval_top_k], ["rerank"]
            DEGRADED_TOTAL.labels(rung="rerank").inc()
    if cache is not None and not degraded:
        await cache.put_chunks(query, limit, [asdict(c) for c in ranked])
    return ranked, degraded


async def step_grade_loop(
    query: str,
    chunks: list[Chunk],
    llm: Any,
    settings: Settings,
    retriever: Retriever,
    cache: CacheService | None,
    emb: QueryEmbedding | None,
    reranker: RerankerFn | None,
    degraded: list[str] | None = None,
    breakers: Breakers | None = None,
) -> tuple[list[Chunk], list[str], bool]:
    """CRAG loop: grade → rewrite+retrieve → grade, bounded by MAX_CORRECTIVE_LOOPS.

    Grading is skipped when the top score is above the calibrated threshold.
    Returns (chunks, degraded, enough_evidence).
    """
    flags = list(degraded or [])
    if chunks and should_skip_grading(chunks[0].score, settings):
        return chunks, flags, True
    outcome = await grade_chunks(query, chunks, llm, settings)
    loops = 0
    while not outcome.enough and loops < settings.max_corrective_loops:
        rewritten = await rewrite_query(query, outcome.feedback, llm, settings)
        chunks, degraded2 = await step_retrieve(
            rewritten, retriever, settings, cache, emb, reranker, breakers
        )
        flags = list({*flags, *degraded2})
        if not chunks:
            return [], flags, False
        outcome = await grade_chunks(rewritten, chunks, llm, settings)
        loops += 1
    return chunks, flags, outcome.enough


async def step_generate(
    query: str,
    chunks: list[Chunk],
    llm: Any,
    settings: Settings,
    route: str,
    degraded: list[str],
    rid: str,
    breakers: Breakers | None = None,
) -> AskResult:
    context = pack_context(chunks, settings.max_context_tokens)
    user_msg = f"<sources>\n{context}\n</sources>\n\nQuestion: {query}"

    async def call(model: str) -> Any:
        return await llm.complete(
            system=ANSWER_SYSTEM,
            user=user_msg,
            model=model,
            max_tokens=settings.max_output_tokens,
        )

    attempts = [settings.llm_strong_model]
    if settings.llm_fallback_model and settings.llm_fallback_model != settings.llm_strong_model:
        attempts.append(settings.llm_fallback_model)

    response: Any = None
    for index, model in enumerate(attempts):
        try:
            if index == 0 and breakers is not None:
                response = await breakers.llm.call(call, model)
            else:
                response = await call(model)
            if index > 0:
                degraded = [*degraded, "llm_fallback"]
                DEGRADED_TOTAL.labels(rung="fallback_model").inc()
            break
        except Exception:
            logger.warning("generate failed on model %s", model, exc_info=True)
    if response is None:
        return _extractive_answer(chunks, route, [*degraded, "llm_extractive"], rid)

    usage = Usage(
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cached_tokens=response.cached_tokens,
        cost_usd=response.cost_usd,
    )
    answer = response.text
    if CANARY_TOKEN in answer:
        return AskResult(
            answer="The answer was withheld because it failed the leak check.",
            citations=[],
            route=route,
            abstained=True,
            usage=usage,
            request_id=rid,
            degraded=degraded,
        )
    return AskResult(
        answer=answer,
        citations=build_citations(answer, chunks),
        route=route,
        abstained=False,
        usage=usage,
        request_id=rid,
        degraded=degraded,
    )


async def _abstain_with_evidence(
    query: str, chunks: list[Chunk], route: str, degraded: list[str], rid: str
) -> AskResult:
    cited = [
        Citation(
            id=c.chunk_id,
            title=c.title,
            section=c.section_path,
            url=c.source_url,
            snippet=c.text[:_SNIPPET_CHARS],
        )
        for c in chunks[:3]
    ]
    found = "; ".join(c.title for c in chunks[:3]) or "nothing relevant"
    return AskResult(
        answer=(
            "I couldn't find enough evidence in the corpus to answer this question confidently. "
            f"Closest material found: {found}."
        ),
        citations=cited,
        route=route,
        abstained=True,
        usage=_zero_usage(),
        request_id=rid,
        degraded=degraded,
    )


async def _store_result(
    cache: CacheService | None, query: str, emb: QueryEmbedding | None, result: AskResult
) -> None:
    if cache is None:
        return
    payload = result_payload(result)
    await cache.put_response(query, payload)
    await cache.put_semantic(query, emb, payload)


def _retrieval_limit(settings: Settings, reranker: RerankerFn | None) -> int:
    return settings.rerank_top_in if reranker is not None else settings.retrieval_top_k


async def run_simple(
    query: str,
    *,
    retriever: Retriever,
    llm: Any,
    settings: Settings,
    cache: CacheService | None,
    emb: QueryEmbedding | None,
    reranker: RerankerFn | None,
    rid: str,
    breakers: Breakers | None = None,
) -> AskResult:
    chunks, degraded = await step_retrieve(
        query, retriever, settings, cache, emb, reranker, breakers
    )
    if not chunks:
        return AskResult(
            answer="I couldn't find any relevant evidence in the corpus for this question.",
            citations=[],
            route="simple",
            abstained=True,
            usage=_zero_usage(),
            request_id=rid,
            degraded=degraded,
        )
    chunks, degraded, enough = await step_grade_loop(
        query, chunks, llm, settings, retriever, cache, emb, reranker, degraded, breakers
    )
    if not enough:
        return await _abstain_with_evidence(query, chunks, "simple", degraded, rid)
    result = await step_generate(query, chunks, llm, settings, "simple", degraded, rid, breakers)
    maybe_check_faithfulness(result, chunks, llm, settings)
    return result


async def run_ask(
    query: str,
    *,
    retriever: Retriever,
    llm: Any,
    settings: Settings,
    cache: CacheService | None = None,
    reranker: RerankerFn | None = None,
    graph: Any = None,
    breakers: Breakers | None = None,
    request_id: str | None = None,
) -> AskResult:
    """Sequential orchestration; delegates to the LangGraph runner when one is wired."""
    if graph is not None:
        graphed: AskResult = await graph.run(query, request_id=request_id)
        return graphed

    rid = request_id or uuid.uuid4().hex
    cached = await step_cache_lookup(query, cache)
    if cached is not None:
        return cached

    decision = await route_query(query, llm, settings, cache)
    if decision.route in ("no_retrieval", "out_of_scope", "unsafe"):
        result = canned_response(decision, rid)
        await _store_result(cache, query, None, result)
        return result

    emb = await step_embed(query, cache)
    semantic = await step_semantic(query, emb, cache)
    if semantic is not None:
        return semantic
    if cache is not None and not await cache.acquire(query):
        payload = await cache.wait_for_response(query)
        if payload is not None:
            return result_from_payload(payload, rid, cache_status="hit")

    if decision.route == "multi_hop":
        from atlas.pipeline.multihop import run_multi_hop  # local import: no cycle

        result = await run_multi_hop(
            query,
            retriever=retriever,
            llm=llm,
            settings=settings,
            cache=cache,
            emb=emb,
            reranker=reranker,
            request_id=rid,
        )
    else:
        result = await run_simple(
            query,
            retriever=retriever,
            llm=llm,
            settings=settings,
            cache=cache,
            emb=emb,
            reranker=reranker,
            rid=rid,
            breakers=breakers,
        )

    await _store_result(cache, query, emb, result)
    return result
