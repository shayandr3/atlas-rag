"""Local ONNX embedders via fastembed (free): dense bge-small + sparse BM25.

Models download on first use locally; in the serving image they are baked in at
build time so cold start is load-only (spec §16.1). Imports are deferred so the app
boot path does not pay for fastembed until an embedding is actually needed.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class QueryEmbedding:
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


class Embedder(Protocol):
    async def embed_query(self, text: str) -> QueryEmbedding: ...


class FastembedEmbedder:
    def __init__(self, embed_model: str, sparse_model: str) -> None:
        self._embed_model = embed_model
        self._sparse_model = sparse_model
        self._dense: Any = None
        self._sparse: Any = None
        self._lock = asyncio.Lock()

    async def _ensure_models(self) -> tuple[Any, Any]:
        async with self._lock:
            if self._dense is None:
                from fastembed import TextEmbedding

                self._dense = TextEmbedding(model_name=self._embed_model)
            if self._sparse is None:
                from fastembed import SparseTextEmbedding

                self._sparse = SparseTextEmbedding(model_name=self._sparse_model)
            return self._dense, self._sparse

    async def embed_query(self, text: str) -> QueryEmbedding:
        embeddings = await self.embed_passages([text])
        return embeddings[0]

    async def embed_passages(self, texts: list[str]) -> list[QueryEmbedding]:
        dense_model, sparse_model = await self._ensure_models()

        def _embed() -> list[QueryEmbedding]:
            out: list[QueryEmbedding] = []
            for dense_vec, sparse_vec in zip(
                dense_model.embed(texts), sparse_model.embed(texts), strict=True
            ):
                out.append(
                    QueryEmbedding(
                        dense=[float(x) for x in dense_vec],
                        sparse_indices=[int(i) for i in sparse_vec.indices],
                        sparse_values=[float(v) for v in sparse_vec.values],
                    )
                )
            return out

        return await asyncio.to_thread(_embed)
