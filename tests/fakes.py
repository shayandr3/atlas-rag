"""Shared test fakes: no network, no keys, no models."""

from atlas.llm.adapter import LLMResponse
from atlas.retrieval.chunks import Chunk


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

    async def search(self, query: str, *, limit: int) -> list[Chunk]:
        self.calls += 1
        return self.chunks[:limit]


class FakeLLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0
        self.last_user = ""

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
        return LLMResponse(
            text=self.text,
            model=model,
            input_tokens=100,
            output_tokens=20,
            cached_tokens=0,
            cost_usd=0.001,
        )
