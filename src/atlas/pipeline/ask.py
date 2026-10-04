"""M1 ask pipeline: hybrid retrieve → pack → generate with citations (spec §8.5-§8.7).

M2 adds the cache lifecycle (spec §9): L1 exact → L2 semantic → single-flight →
L3 retrieval → generate → store. The domain↔cache payload mapping lives here so
the cache layer itself stays domain-free.
"""

import re
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from atlas.cache.service import CacheService
from atlas.config import Settings
from atlas.llm.adapter import LLMResponse
from atlas.llm.prompts import ANSWER_SYSTEM, CANARY_TOKEN
from atlas.retrieval.chunks import Chunk
from atlas.retrieval.embedders import QueryEmbedding

_CITATION_RE = re.compile(r"\[S(\d+)\]")
_CHARS_PER_TOKEN = 4
_SNIPPET_CHARS = 280


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
    )


async def run_ask(
    query: str,
    *,
    retriever: Retriever,
    llm: LLM,
    settings: Settings,
    cache: CacheService | None = None,
    request_id: str | None = None,
) -> AskResult:
    rid = request_id or uuid.uuid4().hex
    emb: QueryEmbedding | None = None

    if cache is not None:
        payload = await cache.get_response(query)
        if payload is not None:
            return result_from_payload(payload, rid, cache_status="hit")
        emb = await cache.embed_query(query)
        payload = await cache.get_semantic(query, emb)
        if payload is not None:
            return result_from_payload(payload, rid, cache_status="semantic")
        if not await cache.acquire(query):
            payload = await cache.wait_for_response(query)
            if payload is not None:
                return result_from_payload(payload, rid, cache_status="hit")

    chunks = await _retrieve(query, retriever, settings, cache, emb)

    async def finish(result: AskResult) -> AskResult:
        if cache is not None:
            payload = result_payload(result)
            await cache.put_response(query, payload)
            await cache.put_semantic(query, emb, payload)
        return result

    if not chunks:
        return await finish(
            AskResult(
                answer="I couldn't find any relevant evidence in the corpus for this question.",
                citations=[],
                route="simple",
                abstained=True,
                usage=Usage(input_tokens=0, output_tokens=0, cached_tokens=0, cost_usd=0.0),
                request_id=rid,
            )
        )

    context = pack_context(chunks, settings.max_context_tokens)
    user_msg = f"<sources>\n{context}\n</sources>\n\nQuestion: {query}"
    response = await llm.complete(
        system=ANSWER_SYSTEM,
        user=user_msg,
        model=settings.llm_strong_model,
        max_tokens=settings.max_output_tokens,
    )
    usage = Usage(
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cached_tokens=response.cached_tokens,
        cost_usd=response.cost_usd,
    )

    answer = response.text
    if CANARY_TOKEN in answer:
        return await finish(
            AskResult(
                answer="The answer was withheld because it failed the leak check.",
                citations=[],
                route="simple",
                abstained=True,
                usage=usage,
                request_id=rid,
            )
        )

    return await finish(
        AskResult(
            answer=answer,
            citations=build_citations(answer, chunks),
            route="simple",
            abstained=False,
            usage=usage,
            request_id=rid,
        )
    )


async def _retrieve(
    query: str,
    retriever: Retriever,
    settings: Settings,
    cache: CacheService | None,
    emb: QueryEmbedding | None,
) -> list[Chunk]:
    if cache is not None:
        cached = await cache.get_chunks(query, settings.retrieval_top_k)
        if cached is not None:
            return [Chunk(**c) for c in cached]
    chunks = await retriever.search(query, limit=settings.retrieval_top_k, embedding=emb)
    if cache is not None:
        await cache.put_chunks(query, settings.retrieval_top_k, [asdict(c) for c in chunks])
    return chunks
