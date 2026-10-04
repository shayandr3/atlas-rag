"""Sliding-window rate limiting (spec §11.1): Redis ZSET primary, in-process fallback.

Fail SAFE, not open: when Redis is down the in-process window still enforces the limit
for the single serving worker (Render free runs exactly one worker).
"""

import time
from collections import defaultdict, deque
from typing import Any

from atlas.cache.client import FailOpenRedis


class RateLimiter:
    def __init__(self, redis: FailOpenRedis | None = None, window_s: int = 60) -> None:
        self._redis = redis
        self._window = window_s
        self._local: dict[str, deque[float]] = defaultdict(deque)
        self._local_keys_cap = 10_000

    async def allow(self, key: str, limit: int) -> tuple[bool, int]:
        """Returns (allowed, retry_after_seconds)."""
        now = time.time()
        if limit <= 0:
            return False, self._window
        if self._redis is not None and not self._redis._is_down():
            allowed, retry = await self._redis_window(key, limit, now)
            if allowed is not None:
                return allowed, retry
        return self._local_window(key, limit, now)

    async def _redis_window(self, key: str, limit: int, now: float) -> tuple[bool | None, int]:
        assert self._redis is not None  # guarded by allow()
        zkey = f"ratelimit:{key}"
        member = f"{now}:{id(now)}"
        try:
            pipe = self._redis._client.pipeline()
            pipe.zremrangebyscore(zkey, 0, now - self._window)
            pipe.zcard(zkey)
            pipe.zadd(zkey, {member: now})
            pipe.expire(zkey, self._window * 2)
            results = await pipe.execute()
        except Exception:
            self._redis._fail()
            return None, 0
        self._redis._ok()
        count = int(results[1])
        if count >= limit:
            return False, self._window
        return True, 0

    def _local_window(self, key: str, limit: int, now: float) -> tuple[bool, int]:
        window = self._local[key]
        while window and window[0] <= now - self._window:
            window.popleft()
        if len(window) >= limit:
            return False, int(self._window - (now - window[0])) + 1
        window.append(now)
        if len(self._local) > self._local_keys_cap:
            self._local.clear()
        return True, 0


async def ip_of(request: Any) -> str:
    forwarded: str = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    client = getattr(request, "client", None)
    host: str = client.host if client else "unknown"
    return host
