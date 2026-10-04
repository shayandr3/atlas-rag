"""M2/M4 metrics: cache and security slices of the spec §13.2 catalogue (full set in M6)."""

from prometheus_client import Counter, Gauge

CACHE_REQUESTS = Counter(
    "atlas_cache_requests_total",
    "Cache lookups by layer and result",
    ["layer", "result"],
)

CACHE_COST_SAVED = Counter(
    "atlas_cache_cost_saved_usd_total",
    "USD of LLM spend avoided by cache hits",
    ["layer"],
)

GUARD_BLOCKS = Counter(
    "atlas_guard_blocks_total",
    "Guard blocks by guard and rule",
    ["guard", "rule"],
)

RATE_LIMITED = Counter(
    "atlas_rate_limited_total",
    "Rate-limited requests by scope",
    ["scope"],
)

LLM_COST = Counter(
    "atlas_llm_cost_usd_total",
    "LLM spend by model",
    ["model"],
)

BUDGET_REMAINING = Gauge(
    "atlas_budget_remaining_usd",
    "Budget headroom by scope (global_daily/global_monthly)",
    ["scope"],
)
