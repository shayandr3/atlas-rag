from fakeredis import aioredis as fakeredis_aioredis

from atlas.cache.client import FailOpenRedis
from atlas.config import Settings
from atlas.security.budget import BudgetLedger
from atlas.security.ratelimit import RateLimiter


async def test_rate_limit_allows_then_denies() -> None:
    limiter = RateLimiter(FailOpenRedis(client=fakeredis_aioredis.FakeRedis()))

    results = [await limiter.allow("k", 3) for _ in range(5)]

    assert [ok for ok, _ in results] == [True, True, True, False, False]
    assert results[3][1] > 0  # retry-after hint


async def test_rate_limit_local_fallback_when_redis_dead() -> None:
    class BrokenClient:
        def __getattr__(self, name):
            async def _raise(*args, **kwargs):
                raise RuntimeError("down")

            return _raise

    limiter = RateLimiter(FailOpenRedis(client=BrokenClient()))

    results = [await limiter.allow("k", 2) for _ in range(4)]

    assert [ok for ok, _ in results] == [True, True, False, False]


async def test_budget_check_and_charge() -> None:
    ledger = BudgetLedger(
        Settings(_env_file=None, global_daily_budget_usd=1.0, global_monthly_budget_usd=10.0),
        FailOpenRedis(client=fakeredis_aioredis.FakeRedis()),
    )

    ok, _ = await ledger.check("k1", key_daily_limit=0.5)
    assert ok is True

    await ledger.charge("k1", 0.6)

    ok, reason = await ledger.check("k1", key_daily_limit=0.5)
    assert ok is False
    assert reason == "key_daily_budget_exceeded"


async def test_global_monthly_budget_blocks() -> None:
    ledger = BudgetLedger(
        Settings(_env_file=None, global_daily_budget_usd=20.0, global_monthly_budget_usd=10.0),
        FailOpenRedis(client=fakeredis_aioredis.FakeRedis()),
    )

    await ledger.charge("k1", 11.0)

    ok, reason = await ledger.check("k1", key_daily_limit=100.0)
    assert ok is False
    assert reason == "global_monthly_budget_exceeded"


async def test_budget_local_fallback_when_redis_dead() -> None:
    class BrokenClient:
        def __getattr__(self, name):
            async def _raise(*args, **kwargs):
                raise RuntimeError("down")

            return _raise

    ledger = BudgetLedger(
        Settings(_env_file=None, global_daily_budget_usd=0.01, global_monthly_budget_usd=1.0),
        FailOpenRedis(client=BrokenClient()),
    )

    await ledger.charge("k1", 0.02)
    ok, reason = await ledger.check("k1", key_daily_limit=1.0)
    assert ok is False
    assert reason == "global_daily_budget_exceeded"


async def test_budget_reset() -> None:
    ledger = BudgetLedger(
        Settings(_env_file=None),
        FailOpenRedis(client=fakeredis_aioredis.FakeRedis()),
    )
    await ledger.charge("k1", 0.5)
    cleared = await ledger.reset_all()

    assert cleared >= 3  # key daily + global daily + global monthly
    ok, _ = await ledger.check("k1", key_daily_limit=0.1)
    assert ok is True
