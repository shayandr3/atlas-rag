"""Multi-hop decomposition and sequential execution (spec §8.2).

The cheap model decomposes the question into ordered sub-questions with dependencies
(≤ MAX_HOPS); each sub-question is retrieved and answered with the cheap model, with
earlier answers substituted into dependent ones. The strong model sees all accumulated
evidence only for the final answer. Hard stop on hop/budget limits.
"""

from dataclasses import dataclass, field
from typing import Any

from atlas.config import Settings
from atlas.llm.structured import StructuredOutputError, complete_json
from atlas.pipeline.ask import AskResult, Usage, build_citations, pack_context
from atlas.pipeline.faithfulness import maybe_check_faithfulness
from atlas.pipeline.grade import grade_chunks, rewrite_query
from atlas.retrieval.chunks import Chunk
from atlas.retrieval.embedders import QueryEmbedding

_DECOMPOSE_PROMPT = """Break this research question into the minimal sequence of sub-questions
(2-{max_hops}) that must be answered in order, each answerable from one paper's passages.

Question: {query}

Strict JSON only: [{{"id": "Q1", "question": "...", "depends_on": []}}, ...]
Later sub-questions may list earlier ids in depends_on."""


@dataclass(frozen=True)
class SubQuestion:
    id: str
    question: str
    depends_on: list[str] = field(default_factory=list)


async def decompose(query: str, llm: Any, settings: Settings) -> list[SubQuestion]:
    """Decompose; any failure falls back to a single sub-question (= plain retrieval)."""
    try:
        raw = await complete_json(
            llm,
            system="You output only valid JSON.",
            user=_DECOMPOSE_PROMPT.format(query=query, max_hops=settings.max_hops),
            model=settings.llm_cheap_model,
            max_tokens=500,
        )
        subs = [
            SubQuestion(
                id=str(item.get("id", f"Q{i}")),
                question=str(item.get("question", "")).strip(),
                depends_on=[str(d) for d in item.get("depends_on", [])],
            )
            for i, item in enumerate(raw if isinstance(raw, list) else [], start=1)
            if isinstance(item, dict) and str(item.get("question", "")).strip()
        ]
    except (StructuredOutputError, AttributeError, TypeError, ValueError):
        subs = []
    if not subs:
        subs = [SubQuestion(id="Q1", question=query, depends_on=[])]
    return subs[: settings.max_hops]


def _with_dependencies(sq: SubQuestion, answers: dict[str, str]) -> str:
    if not sq.depends_on:
        return sq.question
    prior = " ".join(f"[{dep}] {answers[dep]}" for dep in sq.depends_on if dep in answers)
    return f"{sq.question}\nContext from earlier steps: {prior}" if prior else sq.question


async def run_multi_hop(
    query: str,
    *,
    retriever: Any,
    llm: Any,
    settings: Settings,
    cache: Any = None,
    emb: QueryEmbedding | None = None,
    reranker: Any = None,
    request_id: str = "",
) -> AskResult:
    from atlas.pipeline.ask import step_retrieve  # local import: no cycle

    sub_questions = await decompose(query, llm, settings)
    answers: dict[str, str] = {}
    evidence: list[Chunk] = []
    subanswers: list[str] = []
    degraded: list[str] = []

    for sq in sub_questions:
        question = _with_dependencies(sq, answers)
        chunks, degraded2 = await step_retrieve(question, retriever, settings, cache, emb, reranker)
        degraded = list({*degraded, *degraded2})
        if not chunks:
            answers[sq.id] = "no evidence found"
            continue
        evidence.extend(chunks)
        outcome = await grade_chunks(question, chunks, llm, settings)
        if not outcome.enough:
            rewritten = await rewrite_query(question, outcome.feedback, llm, settings)
            chunks, degraded2 = await step_retrieve(
                rewritten, retriever, settings, cache, emb, reranker
            )
            degraded = list({*degraded, *degraded2})
            evidence.extend(chunks)
            if not chunks:
                answers[sq.id] = "no evidence found"
                continue
        context = pack_context(chunks, settings.max_context_tokens)
        sub_answer = await llm.complete(
            system="Answer strictly from the sources; cite [S1]-style labels.",
            user=f"<sources>\n{context}\n</sources>\n\nQuestion: {question}",
            model=settings.llm_cheap_model,
            max_tokens=300,
        )
        answers[sq.id] = sub_answer.text
        subanswers.append(f"{sq.id}: {sub_answer.text}")

    if not evidence:
        return AskResult(
            answer="I couldn't find any relevant evidence in the corpus for this question.",
            citations=[],
            route="multi_hop",
            abstained=True,
            usage=Usage(input_tokens=0, output_tokens=0, cached_tokens=0, cost_usd=0.0),
            request_id=request_id,
            degraded=degraded,
        )

    context = pack_context(evidence[: settings.retrieval_top_k], settings.max_context_tokens)
    sub_text = "\n".join(subanswers) or "none"
    final = await llm.complete(
        system="Answer strictly from the sources; cite [S1]-style labels.",
        user=(
            f"<sources>\n{context}\n</sources>\n\n"
            f"Step-by-step findings:\n{sub_text}\n\nFinal question: {query}"
        ),
        model=settings.llm_strong_model,
        max_tokens=settings.max_output_tokens,
    )
    usage = Usage(
        input_tokens=final.input_tokens,
        output_tokens=final.output_tokens,
        cached_tokens=final.cached_tokens,
        cost_usd=final.cost_usd,
    )
    result = AskResult(
        answer=final.text,
        citations=build_citations(final.text, evidence[: settings.retrieval_top_k]),
        route="multi_hop",
        abstained=False,
        usage=usage,
        request_id=request_id,
        degraded=degraded,
    )
    maybe_check_faithfulness(result, evidence[: settings.retrieval_top_k], llm, settings)
    return result
