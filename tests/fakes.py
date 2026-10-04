"""Shared test fakes: no network, no keys, no models."""

import asyncio

from atlas.llm.adapter import LLMResponse
from atlas.retrieval.chunks import Chunk
from atlas.retrieval.embedders import QueryEmbedding


def make_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="c1",
            doc_id="d1",
            title="Title One",
            section_path="body",
            text="Evidence one.",
            source_url="https://arxiv.org/abs/d1",
            score=1.0,
        ),
        Chunk(
            chunk_id="c2",
            doc_id="d2",
            title="Title Two",
            section_path="body",
            text="Evidence two.",
            source_url="https://arxiv.org/abs/d2",
            score=0.9,
        ),
    ]


class FakeRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.calls = 0

    async def search(
        self, query: str, *, limit: int, embedding: QueryEmbedding | None = None
    ) -> list[Chunk]:
        self.calls += 1
        return self.chunks[:limit]


class FakeLLM:
    def __init__(
        self,
        text: str = "",
        delay_s: float = 0.0,
        responses: list[str] | None = None,
    ) -> None:
        self.text = text
        self.calls = 0
        self.delay_s = delay_s
        self.last_user = ""
        self._responses = list(responses) if responses else None

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int,
    ) -> LLMResponse:
        self.calls += 1
        self.last_user = user
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        body = self._responses.pop(0) if self._responses else self.text
        return LLMResponse(
            text=body,
            model=model,
            input_tokens=100,
            output_tokens=20,
            cached_tokens=0,
            cost_usd=0.001,
        )


class FakeReranker:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    async def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("reranker exploded")
        return list(reversed(chunks))


class FakeEmbedder:
    """Constant vector by default (semantic cache always hits); optional per-text vectors."""

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self.calls = 0
        self._vectors = vectors or {}

    async def embed_query(self, text: str) -> QueryEmbedding:
        self.calls += 1
        dense = self._vectors.get(text, [1.0, 0.0, 0.0, 0.0])
        return QueryEmbedding(dense=dense, sparse_indices=[0], sparse_values=[1.0])


class DyingRetriever:
    """Serves once, then the dependency 'dies' (chaos kill)."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.calls = 0

    async def search(
        self, query: str, *, limit: int, embedding: QueryEmbedding | None = None
    ) -> list[Chunk]:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("qdrant blackholed")
        return self.chunks[:limit]
