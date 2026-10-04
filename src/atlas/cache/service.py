"""Cache service: L0 embeddings, L1 exact responses, L2 semantic, L3 retrieval (spec §9).

Domain objects stay in the pipeline: this layer speaks normalized queries, bytes and
plain-JSON payloads, and reports every lookup to the metrics registry.
"""

import asyncio
import base64
import logging
import random
import re
import time
from typing import Any

import numpy as np
from orjson import dumps, loads

from atlas.cache.client import FailOpenRedis
from atlas.cache.keys import CacheKeys, norm_query
from atlas.cache.semantic import SemanticIndex, encode_vector
from atlas.config import Settings
from atlas.observability.metrics import CACHE_COST_SAVED, CACHE_REQUESTS
from atlas.retrieval.embedders import Embedder, QueryEmbedding

logger = logging.getLogger("atlas.cache")

_TIME_SENSITIVE = re.compile(
    r"\b(today|tonight|now|latest|newest|current|currently|recent|recently"
    r"|breaking|news|price|stock|weather|forecast|2025|2026|2027)\b"
)
_MAX_PAYLOAD_BYTES = 16_000
_LOCK_TTL_MS = 30_000
_WAIT_POLL_S = 0.2


def is_time_sensitive(query: str) -> bool:
    return bool(_TIME_SENSITIVE.search(norm_query(query)))


class CacheService:
    def __init__(
        self,
        settings: Settings,
        redis: FailOpenRedis,
        embedder: Embedder | None = None,
    ) -> None:
        self._settings = settings
        self._redis = redis
        self._embedder = embedder
        self._keys = CacheKeys(settings)
        self._sem_index: SemanticIndex | None = None

    @property
    def redis_up(self) -> bool:
        return not self._redis._is_down()

    # ---- L0: query embeddings -------------------------------------------------

    async def embed_query(self, query: str) -> QueryEmbedding | None:
        """L0-cached query embedding; None when no embedder is wired (cache bypassed)."""
        if self._embedder is None:
            return None
        key = self._keys.embedding(query)
        cached = await self._redis.get(key)
        if cached is not None:
            CACHE_REQUESTS.labels(layer="embedding", result="hit").inc()
            try:
                return self._decode_embedding(cached)
            except Exception:
                logger.warning("embedding cache decode failed", exc_info=True)
        CACHE_REQUESTS.labels(layer="embedding", result="miss").inc()
        emb = await self._embedder.embed_query(query)
        await self._redis.set(
            key, self._encode_embedding(emb), int(self._settings.cache_embed_ttl_days * 86400)
        )
        return emb

    @staticmethod
    def _encode_embedding(emb: QueryEmbedding) -> bytes:
        return dumps(
            {
                "d": base64.b64encode(encode_vector(emb.dense)).decode(),
                "i": emb.sparse_indices,
                "v": emb.sparse_values,
            }
        )

    @staticmethod
    def _decode_embedding(blob: bytes) -> QueryEmbedding:
        entry = loads(blob)
        dense = np.frombuffer(base64.b64decode(entry["d"]), dtype=np.float32)
        return QueryEmbedding(
            dense=[float(x) for x in dense],
            sparse_indices=list(entry["i"]),
            sparse_values=list(entry["v"]),
        )

    # ---- L1: exact response cache ----------------------------------------------

    async def get_response(self, query: str, *, record: bool = True) -> dict[str, Any] | None:
        cached = await self._redis.get(self._keys.response(query))
        if cached is None:
            if record:
                CACHE_REQUESTS.labels(layer="response", result="miss").inc()
            return None
        if record:
            CACHE_REQUESTS.labels(layer="response", result="hit").inc()
        payload: dict[str, Any] = loads(cached)
        cost = float(payload.get("usage", {}).get("cost_usd", 0.0))
        CACHE_COST_SAVED.labels(layer="response").inc(cost)
        return payload

    async def put_response(self, query: str, payload: dict[str, Any]) -> None:
        blob = dumps(payload)
        if len(blob) > _MAX_PAYLOAD_BYTES:
            return
        if payload.get("abstained"):
            ttl = self._settings.cache_abstention_ttl_minutes * 60
        else:
            ttl = int(self._settings.cache_exact_ttl_hours * 3600 * random.uniform(0.9, 1.1))
        await self._redis.set(self._keys.response(query), blob, ttl)

    # ---- L2: semantic cache ------------------------------------------------------

    def _semantic_index(self) -> SemanticIndex:
        if self._sem_index is None:
            self._sem_index = SemanticIndex(
                threshold=self._settings.semantic_cache_threshold,
                max_entries=self._settings.semantic_cache_max_entries,
            )
        return self._sem_index

    async def get_semantic(self, query: str, emb: QueryEmbedding | None) -> dict[str, Any] | None:
        if not self._settings.semantic_cache_enabled or emb is None or is_time_sensitive(query):
            return None
        index = self._semantic_index()
        if not index._order:  # lazy local rebuild from the durable hash
            raw = await self._redis.hgetall(self._keys.semantic())
            index.load(raw)
        field = index.lookup(np.asarray(emb.dense, dtype=np.float32))
        if field is None:
            CACHE_REQUESTS.labels(layer="semantic", result="miss").inc()
            return None
        cached = await self._redis.get(str(field))
        if cached is None:
            CACHE_REQUESTS.labels(layer="semantic", result="miss").inc()
            return None
        CACHE_REQUESTS.labels(layer="semantic", result="hit").inc()
        payload: dict[str, Any] = loads(cached)
        CACHE_COST_SAVED.labels(layer="semantic").inc(
            float(payload.get("usage", {}).get("cost_usd", 0.0))
        )
        return payload

    async def put_semantic(
        self, query: str, emb: QueryEmbedding | None, payload: dict[str, Any]
    ) -> None:
        if (
            not self._settings.semantic_cache_enabled
            or emb is None
            or payload.get("abstained")
            or payload.get("route") != "simple"
            or is_time_sensitive(query)
        ):
            return
        answer_key = self._keys.response(query)
        if await self._redis.set(
            answer_key, dumps(payload), int(self._settings.cache_exact_ttl_hours * 3600)
        ):
            field = answer_key
            vec_blob = encode_vector(emb.dense)
            evicted = self._semantic_index().add(field, vec_blob)
            await self._redis.hset(
                self._keys.semantic(),
                field,
                # base64: orjson cannot serialize raw bytes
                dumps({"vec": base64.b64encode(vec_blob).decode(), "answer": answer_key}),
                int(self._settings.cache_exact_ttl_hours * 3600),
            )
            if evicted is not None:
                await self._redis.hdel(self._keys.semantic(), evicted)

    # ---- L3: retrieval cache -------------------------------------------------------

    async def get_chunks(self, query: str, limit: int) -> list[dict[str, Any]] | None:
        cached = await self._redis.get(self._keys.retrieval(query, limit))
        if cached is None:
            CACHE_REQUESTS.labels(layer="retrieval", result="miss").inc()
            return None
        CACHE_REQUESTS.labels(layer="retrieval", result="hit").inc()
        return list(loads(cached))

    async def put_chunks(self, query: str, limit: int, chunks: list[dict[str, Any]]) -> None:
        await self._redis.set(
            self._keys.retrieval(query, limit),
            dumps(chunks),
            int(self._settings.cache_retrieval_ttl_hours * 3600),
        )

    # ---- single-flight (spec §9) -----------------------------------------------------

    async def acquire(self, query: str) -> bool:
        """True → this request computes. False → another request is already on it."""
        return await self._redis.set_nx(self._keys.lock(query), _LOCK_TTL_MS)

    async def wait_for_response(self, query: str, timeout_s: float = 8.0) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            await asyncio.sleep(_WAIT_POLL_S)
            payload = await self.get_response(query, record=False)
            if payload is not None:
                return payload
        return None
