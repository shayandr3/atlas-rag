"""Hybrid retrieval source: embed the query, search Qdrant, normalize to Chunks."""

from atlas.retrieval.chunks import Chunk
from atlas.retrieval.embedders import Embedder, QueryEmbedding
from atlas.retrieval.qdrant_repo import QdrantRepo


class HybridRetriever:
    def __init__(self, embedder: Embedder, repo: QdrantRepo, mode: str = "hybrid") -> None:
        self._embedder = embedder
        self._repo = repo
        self._mode = mode

    async def search(
        self, query: str, *, limit: int, embedding: QueryEmbedding | None = None
    ) -> list[Chunk]:
        if embedding is None:
            embedding = await self._embedder.embed_query(query)
        points = await self._repo.search(
            dense=embedding.dense,
            sparse_indices=embedding.sparse_indices,
            sparse_values=embedding.sparse_values,
            limit=limit,
            mode=self._mode,
        )
        chunks: list[Chunk] = []
        for point in points:
            payload = dict(point.payload or {})
            chunks.append(
                Chunk(
                    chunk_id=str(payload.get("chunk_id", "")),
                    doc_id=str(payload.get("doc_id", "")),
                    title=str(payload.get("title", "")),
                    section_path=str(payload.get("section_path", "")),
                    text=str(payload.get("text", "")),
                    source_url=str(payload.get("source_url", "")),
                    score=float(point.score),
                )
            )
        return chunks
