"""Pluggable cross-encoder reranker (spec §8.3): local ONNX via fastembed, degradable.

Errors propagate as RerankError so the pipeline can fall back to RRF order and mark
`degraded` (resilience ladder rung 1, spec §12).
"""

import asyncio
import logging
from typing import Any

from atlas.config import Settings
from atlas.retrieval.chunks import Chunk

logger = logging.getLogger("atlas.retrieval")


class RerankError(Exception):
    pass


def _cap_per_doc(chunks: list[Chunk], per_doc: int) -> list[Chunk]:
    """MMR-lite diversity: keep at most `per_doc` chunks of the same document."""
    counts: dict[str, int] = {}
    kept: list[Chunk] = []
    for chunk in chunks:
        n = counts.get(chunk.doc_id, 0)
        if n < per_doc:
            kept.append(chunk)
            counts[chunk.doc_id] = n + 1
    return kept


class FastembedReranker:
    """Cross-encoder scoring in a worker thread; model downloads on first use."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any = None

    def _ensure(self) -> Any:
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(model_name=self._model_name)
        return self._model

    async def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        model = self._ensure()

        def _score() -> list[float]:
            docs = [c.text for c in chunks]
            return [float(s) for s in model.rerank(query, docs)]

        try:
            scores = await asyncio.to_thread(_score)
        except Exception as exc:
            raise RerankError(f"cross-encoder failed: {exc}") from exc
        if len(scores) != len(chunks):
            raise RerankError("reranker returned wrong number of scores")
        ranked = [c for _, c in sorted(zip(scores, chunks, strict=True), key=lambda p: -p[0])]
        return ranked


async def rerank_chunks(
    query: str,
    chunks: list[Chunk],
    settings: Settings,
) -> tuple[list[Chunk], list[str]]:
    """Rerank + diversity cap + top_out cut. Returns (chunks, degraded_flags)."""
    if not chunks or settings.rerank_backend == "none":
        return chunks[: settings.rerank_top_out], []
    try:
        reranker = FastembedReranker(settings.rerank_model)
        ranked = await reranker.rerank(query, chunks[: settings.rerank_top_in])
        diversified = _cap_per_doc(ranked, per_doc=3)
        return diversified[: settings.rerank_top_out], []
    except RerankError:
        logger.warning("reranker failed; falling back to RRF order", exc_info=True)
        return chunks[: settings.rerank_top_out], ["rerank"]
