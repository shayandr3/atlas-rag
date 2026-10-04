"""M2 metrics: the cache slice of the spec §13.2 catalogue (full catalogue lands in M6)."""

from prometheus_client import Counter

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
