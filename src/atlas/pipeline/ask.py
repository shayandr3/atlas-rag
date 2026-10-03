"""M1 ask pipeline: hybrid retrieve → pack → generate with citations (spec §8.5-§8.7).

Router, reranking, corrective grading and multi-hop arrive in M3; this module keeps the
node functions plain so the later graph wiring only orchestrates them.
"""

import re
import uuid
from dataclasses import dataclass
from typing import Protocol

from atlas.config import Settings
from atlas.llm.adapter import LLMResponse
from atlas.llm.prompts import ANSWER_SYSTEM, CANARY_TOKEN
from atlas.retrieval.chunks import Chunk

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


class Retriever(Protocol):
    async def search(self, query: str, *, limit: int) -> list[Chunk]: ...


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


async def run_ask(
    query: str,
    *,
    retriever: Retriever,
    llm: LLM,
    settings: Settings,
    request_id: str | None = None,
) -> AskResult:
    rid = request_id or uuid.uuid4().hex
    chunks = await retriever.search(query, limit=settings.retrieval_top_k)
    if not chunks:
        return AskResult(
            answer="I couldn't find any relevant evidence in the corpus for this question.",
            citations=[],
            route="simple",
            abstained=True,
            usage=Usage(input_tokens=0, output_tokens=0, cached_tokens=0, cost_usd=0.0),
            request_id=rid,
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
        return AskResult(
            answer="The answer was withheld because it failed the leak check.",
            citations=[],
            route="simple",
            abstained=True,
            usage=usage,
            request_id=rid,
        )

    return AskResult(
        answer=answer,
        citations=build_citations(answer, chunks),
        route="simple",
        abstained=False,
        usage=usage,
        request_id=rid,
    )
