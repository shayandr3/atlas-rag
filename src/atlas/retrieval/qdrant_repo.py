"""Qdrant Cloud wrapper: collection management and hybrid search (spec §7.4).

Hybrid = dense prefetch + BM25 sparse prefetch fused with RRF. `mode` also exposes
dense-only and sparse-only for the ablation runner (spec §15.3).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

if TYPE_CHECKING:
    from qdrant_client import models

logger = logging.getLogger("atlas.retrieval")


class QdrantRepo:
    def __init__(
        self, url: str, api_key: str, collection: str, timeout_seconds: float = 10.0
    ) -> None:
        # Imported lazily: qdrant_client can pull fastembed/onnxruntime, which must stay
        # off the app boot path (memory budget, spec §0.5).
        from qdrant_client import AsyncQdrantClient

        self._client: Any = AsyncQdrantClient(
            url=url, api_key=api_key or None, timeout=int(timeout_seconds)
        )
        self.collection = collection

    async def ensure_collection(self, dense_dim: int) -> None:
        from qdrant_client import models

        existing = await self._client.get_collections()
        if any(c.name == self.collection for c in existing.collections):
            return
        await self._client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=dense_dim,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8, quantile=0.99, always_ram=True
                        )
                    ),
                )
            },
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            on_disk_payload=True,
        )
        for field in ("doc_id", "categories", "published_date"):
            await self._client.create_payload_index(
                self.collection, field, models.PayloadSchemaType.KEYWORD
            )
        logger.info("created qdrant collection %s (dense_dim=%d)", self.collection, dense_dim)

    @retry(
        retry=retry_if_exception_type((Exception,)),
        wait=wait_fixed(3),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def search(
        self,
        *,
        dense: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        limit: int,
        mode: str = "hybrid",
    ) -> list[models.ScoredPoint]:
        from qdrant_client import models

        if mode == "dense":
            result = await self._client.query_points(
                collection_name=self.collection,
                query=dense,
                using="dense",
                limit=limit,
                with_payload=True,
            )
        elif mode == "sparse":
            result = await self._client.query_points(
                collection_name=self.collection,
                query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                using="bm25",
                limit=limit,
                with_payload=True,
            )
        else:
            result = await self._client.query_points(
                collection_name=self.collection,
                prefetch=[
                    models.Prefetch(query=dense, using="dense", limit=40),
                    models.Prefetch(
                        query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                        using="bm25",
                        limit=40,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        return list(result.points)

    async def upsert(self, points: list[models.PointStruct]) -> None:
        await self._client.upsert(collection_name=self.collection, points=points, wait=True)

    async def count(self) -> int:
        result = await self._client.count(collection_name=self.collection, exact=True)
        return int(result.count)

    async def ping(self) -> bool:
        try:
            await self._client.get_collections()
        except Exception:
            logger.exception("qdrant ping failed")
            return False
        return True
