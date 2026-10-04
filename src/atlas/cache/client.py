"""Fail-open async Redis wrapper (spec §9): a cache must never fail a request.

Errors are swallowed (logged, counted via the down-circuit); GETs return None, writes
no-op. After repeated consecutive failures the wrapper trips "down" for a minute so a
dead Redis does not add per-op socket timeouts to every request. Single-flight locks
fail OPEN too: if Redis is down, the caller simply computes.
"""

import logging
import time
from typing import Any

logger = logging.getLogger("atlas.cache")

_MAX_CONSECUTIVE_FAILURES = 5
_DOWN_SECONDS = 60.0


class FailOpenRedis:
    def __init__(self, url: str = "", *, client: Any = None, socket_timeout: float = 2.0) -> None:
        if client is not None:
            self._client: Any = client
        else:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(
                url,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_timeout,
            )
        self._failures = 0
        self._down_until = 0.0

    def _is_down(self) -> bool:
        return time.time() < self._down_until

    def _ok(self) -> None:
        self._failures = 0

    def _fail(self) -> None:
        self._failures += 1
        if self._failures >= _MAX_CONSECUTIVE_FAILURES:
            self._down_until = time.time() + _DOWN_SECONDS
            self._failures = 0
            logger.warning("redis marked down for %.0fs (fail-open)", _DOWN_SECONDS)

    async def get(self, key: str) -> bytes | None:
        if self._is_down():
            return None
        try:
            value = await self._client.get(key)
        except Exception:
            logger.warning("cache get failed (fail-open)", exc_info=True)
            self._fail()
            return None
        self._ok()
        return value if isinstance(value, bytes) else None

    async def set(self, key: str, value: bytes, ttl_seconds: int) -> bool:
        if self._is_down():
            return False
        try:
            await self._client.set(key, value, ex=ttl_seconds)
        except Exception:
            logger.warning("cache set failed (fail-open)", exc_info=True)
            self._fail()
            return False
        self._ok()
        return True

    async def delete(self, key: str) -> bool:
        if self._is_down():
            return False
        try:
            await self._client.delete(key)
        except Exception:
            self._fail()
            return False
        self._ok()
        return True

    async def hgetall(self, key: str) -> dict[bytes, bytes]:
        if self._is_down():
            return {}
        try:
            data = await self._client.hgetall(key)
        except Exception:
            logger.warning("cache hgetall failed (fail-open)", exc_info=True)
            self._fail()
            return {}
        self._ok()
        return data or {}

    async def hset(self, key: str, field: str, value: bytes, ttl_seconds: int) -> bool:
        if self._is_down():
            return False
        try:
            pipe = self._client.pipeline()
            pipe.hset(key, field, value)
            pipe.expire(key, ttl_seconds)
            await pipe.execute()
        except Exception:
            self._fail()
            return False
        self._ok()
        return True

    async def hdel(self, key: str, field: str) -> bool:
        if self._is_down():
            return False
        try:
            await self._client.hdel(key, field)
        except Exception:
            self._fail()
            return False
        self._ok()
        return True

    async def set_nx(self, key: str, ttl_ms: int) -> bool:
        """Single-flight lock. Fails OPEN (True) when Redis is unusable: compute."""
        if self._is_down():
            return True
        try:
            acquired = await self._client.set(key, b"1", nx=True, px=ttl_ms)
        except Exception:
            self._fail()
            return True
        self._ok()
        return bool(acquired)

    async def ping(self) -> bool:
        if self._is_down():
            return False
        try:
            await self._client.ping()
        except Exception:
            self._fail()
            return False
        self._ok()
        return True
