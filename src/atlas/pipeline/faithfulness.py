"""Sampled online faithfulness check (spec §8.7): never blocks the response.

A cheap-LLM judge scores whether each claim in the answer is supported by the cited
evidence. Sampling is probabilistic (FAITHFULNESS_SAMPLE_RATE); the score is logged
and lands in a Prometheus histogram in M6. Fire-and-forget by design.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING, Any

from atlas.config import Settings
from atlas.observability.metrics import FAITHFULNESS

if TYPE_CHECKING:
    from atlas.pipeline.ask import AskResult
    from atlas.retrieval.chunks import Chunk

logger = logging.getLogger("atlas.faithfulness")

_background_tasks: set[asyncio.Task[Any]] = set()

_JUDGE_PROMPT = """You judge whether an answer is faithful to its cited evidence.
Score 0.0-1.0: fraction of factual claims actually supported by the evidence.
Strict JSON only: {{"score": 0.0, "unsupported_claims": ["..."]}}

Answer: {answer}

Evidence:
{evidence}"""


def _evidence_block(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[S{i}] {c.text[:500]}" for i, c in enumerate(chunks, start=1))


async def judge_faithfulness(
    result: AskResult, chunks: list[Chunk], llm: Any, settings: Settings
) -> float | None:
    from atlas.llm.structured import StructuredOutputError, complete_json

    try:
        raw = await complete_json(
            llm,
            system="You output only valid JSON.",
            user=_JUDGE_PROMPT.format(
                answer=result.answer[:1500], evidence=_evidence_block(chunks[:6])
            ),
            model=settings.llm_cheap_model,
            max_tokens=250,
        )
        score = max(0.0, min(1.0, float(raw.get("score", -1.0))))
        FAITHFULNESS.observe(score)
        logger.info("faithfulness score=%.2f route=%s", score, result.route)
        return score
    except (StructuredOutputError, AttributeError, TypeError, ValueError):
        return None


def maybe_check_faithfulness(
    result: AskResult, chunks: list[Chunk], llm: Any, settings: Settings
) -> None:
    """Sampled, fire-and-forget: a failure or slowness here must never affect the answer."""
    if settings.faithfulness_sample_rate <= 0 or not chunks:
        return
    if random.random() >= settings.faithfulness_sample_rate:
        return
    task = asyncio.create_task(judge_faithfulness(result, chunks, llm, settings))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
