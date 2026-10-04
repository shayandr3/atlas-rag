"""CRAG-style corrective retrieval grading (spec §8.4).

One batched cheap-LLM call grades all candidate chunks. If evidence is insufficient,
the query is rewritten using the grader's feedback and retrieval retries, bounded by
MAX_CORRECTIVE_LOOPS. Grading is skipped when the top reranker score is already above
a calibrated confidence threshold (saves a call).
"""

from dataclasses import dataclass
from typing import Any

from atlas.config import Settings
from atlas.llm.structured import StructuredOutputError, complete_json
from atlas.retrieval.chunks import Chunk

_GRADE_PROMPT = """You grade retrieved evidence for a question.
For EACH source return a verdict:
- "relevant": contains information needed to answer
- "partial": touches the topic but misses the key fact
- "irrelevant": unrelated
Strict JSON array only: [{{"id": "S1", "verdict": "..."}}, ...]

Question: {query}
Sources:
{sources}"""

_REWRITE_PROMPT = """A retrieval system found insufficient evidence for this question.

Question: {query}
Grader feedback: {feedback}

Rewrite the question into a better search query for an arXiv research-paper corpus.
Strict JSON only: {{"query": "..."}}"""


@dataclass(frozen=True)
class GradeOutcome:
    enough: bool
    relevant_count: int
    feedback: str


def _source_block(query: str, chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[S{i}] {c.title}: {c.text[:600]}" for i, c in enumerate(chunks, start=1))


async def grade_chunks(
    query: str, chunks: list[Chunk], llm: Any, settings: Settings
) -> GradeOutcome:
    try:
        raw = await complete_json(
            llm,
            system="You output only valid JSON.",
            user=_GRADE_PROMPT.format(query=query, sources=_source_block(query, chunks)),
            model=settings.llm_cheap_model,
            max_tokens=400,
        )
        verdicts = {
            str(item.get("id")): str(item.get("verdict", "irrelevant"))
            for item in raw
            if isinstance(item, dict)
        }
    except (StructuredOutputError, AttributeError, TypeError, ValueError):
        # safe default (spec §12): treat grading as "proceed"
        return GradeOutcome(enough=True, relevant_count=-1, feedback="grader unavailable")

    relevant = sum(1 for v in verdicts.values() if v == "relevant")
    partial = sum(1 for v in verdicts.values() if v == "partial")
    feedback = (
        "; ".join(f"{k}: {v}" for k, v in sorted(verdicts.items()) if v != "relevant")
        or "sources graded relevant"
    )
    return GradeOutcome(
        enough=relevant >= 1 or (relevant + partial) >= 2,
        relevant_count=relevant,
        feedback=feedback,
    )


async def rewrite_query(query: str, feedback: str, llm: Any, settings: Settings) -> str:
    try:
        raw = await complete_json(
            llm,
            system="You output only valid JSON.",
            user=_REWRITE_PROMPT.format(query=query, feedback=feedback[:400]),
            model=settings.llm_cheap_model,
            max_tokens=120,
        )
        rewritten = str(raw.get("query", "")).strip()
        return rewritten or query
    except (StructuredOutputError, AttributeError, TypeError, ValueError):
        return query


def should_skip_grading(top_score: float, settings: Settings) -> bool:
    return top_score >= settings.grade_skip_threshold
