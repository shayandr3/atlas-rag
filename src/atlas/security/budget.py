"""USD budget enforcement (spec §10.4): per-key daily, global daily and monthly.

Redis INCRFLOAT counters are the fast path; an in-process dict is the fail-safe
fallback (single worker). The provider-side spend limit remains the final backstop.
"""

import time
from typing import Any

from atlas.cache.client import FailOpenRedis
from atlas.config import Settings


def _day() -> str:
    return time.strftime("%Y%m%d")


def _month() -> str:
    return time.strftime("%Y%m")


def _scopes(settings: Settings) -> dict[str, float]:
    return {
        f"key:{'x'}": 0.0,  # placeholder; real key scopes are per-id
    }


class BudgetLedger:
    def __init__(self, settings: Settings, redis: FailOpenRedis | None) -> None:
        self._settings = settings
        self._redis = redis
        self._local: dict[str, float] = {}
        self._local_stamp: dict[str, str] = {}

    def _key_daily(self, key_id: str) -> str:
        return f"budget:key:{key_id}:{_day()}"

    @property
    def _global_daily(self) -> str:
        return f"budget:global:daily:{_day()}"

    @property
    def _global_monthly(self) -> str:
        return f"budget:global:monthly:{_month()}"

    async def spent(self, scope: str) -> float:
        if self._redis is not None and not self._redis._is_down():
            try:
                value = await self._redis._client.get(scope)
                self._redis._ok()
                if value is not None:
                    return float(value)
                return 0.0
            except Exception:
                self._redis._fail()
        return self._local.get(scope, 0.0)

    async def charge(self, key_id: str, usd: float) -> None:
        if usd <= 0:
            return
        for scope in (self._key_daily(key_id), self._global_daily, self._global_monthly):
            await self._incr(scope, usd)

    async def _incr(self, scope: str, usd: float) -> None:
        if self._redis is not None and not self._redis._is_down():
            try:
                pipe = self._redis._client.pipeline()
                pipe.incrbyfloat(scope, usd)
                ttl = 172_800 if ":daily:" in scope else 2_678_400
                pipe.expire(scope, ttl)
                await pipe.execute()
                self._redis._ok()
                return
            except Exception:
                self._redis._fail()
        self._local[scope] = self._local.get(scope, 0.0) + usd

    async def check(self, key_id: str, key_daily_limit: float) -> tuple[bool, str]:
        """(allowed, reason). Global exhaustion is a service-wide condition."""
        key_spent = await self.spent(self._key_daily(key_id))
        if key_spent >= key_daily_limit:
            return False, "key_daily_budget_exceeded"
        daily_spent = await self.spent(self._global_daily)
        if daily_spent >= self._settings.global_daily_budget_usd:
            return False, "global_daily_budget_exceeded"
        monthly_spent = await self.spent(self._global_monthly)
        if monthly_spent >= self._settings.global_monthly_budget_usd:
            return False, "global_monthly_budget_exceeded"
        return True, ""

    async def reset_all(self) -> int:
        """Admin op: clear every budget counter. Returns scopes cleared."""
        scopes = list(self._local.keys())
        self._local.clear()
        if self._redis is None:
            return len(scopes)
        cleared = 0
        try:
            async for scope in self._redis._client.scan_iter(match="budget:*"):
                await self._redis._client.delete(scope)
                cleared += 1
        except Exception:
            self._redis._fail()
        return cleared


def budget_snapshot(settings: Settings, ledger: BudgetLedger, key_id: str) -> dict[str, Any]:
    """Sync view for /admin/stats (best-effort; async fills the real numbers)."""
    return {
        "key_id": key_id,
        "global_daily_limit_usd": settings.global_daily_budget_usd,
        "global_monthly_limit_usd": settings.global_monthly_budget_usd,
    }
