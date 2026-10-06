"""Full Prometheus metric catalogue (spec §13.2).

Cardinality discipline: labels are enums bounded by config (route/stage/model/rung),
never user ids, query text or chunk ids. Budget ≤ ~1,500 active series (cap 10k).
"""

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
)

# --- HTTP ------------------------------------------------------------------
HTTP_REQUESTS = Counter(
    "atlas_http_requests_total",
    "HTTP requests by route, method and status",
    ["route", "method", "status"],
)
HTTP_DURATION = Histogram(
    "atlas_http_request_duration_seconds",
    "HTTP request latency",
    ["route"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)

# --- pipeline ---------------------------------------------------------------
PIPELINE_REQUESTS = Counter(
    "atlas_pipeline_requests_total",
    "Pipeline outcomes by route",
    ["pipeline_route", "outcome"],
)
STAGE_DURATION = Histogram(
    "atlas_stage_duration_seconds",
    "Per-stage latency",
    ["stage"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 15, 60),
)
TTFT = Histogram(
    "atlas_time_to_first_token_seconds",
    "Time to first token (streaming)",
    ["pipeline_route"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)
HOPS = Histogram(
    "atlas_hops",
    "Multi-hop sub-questions per request",
    buckets=(1, 2, 3, 4, 5),
)
CORRECTIVE_LOOPS = Histogram(
    "atlas_corrective_loops",
    "Corrective retries per request",
    buckets=(0, 1, 2, 3),
)
ABSTENTIONS = Counter(
    "atlas_abstentions_total",
    "Abstentions by reason",
    ["reason"],
)
RETRIEVAL_TOP_SCORE = Histogram(
    "atlas_retrieval_top_score",
    "Top retrieval score by stage",
    ["stage"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
FAITHFULNESS = Histogram(
    "atlas_faithfulness_score",
    "Sampled answer faithfulness (judge score)",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# --- LLM / cache / security (M2/M4 additions kept here as one catalogue) ----
LLM_TOKENS = Counter(
    "atlas_llm_tokens_total",
    "LLM tokens by model, node and kind",
    ["model", "node", "kind"],
)
LLM_COST = Counter(
    "atlas_llm_cost_usd_total",
    "LLM spend by model",
    ["model"],
)
LLM_ERRORS = Counter(
    "atlas_llm_errors_total",
    "LLM call failures by model and type",
    ["model", "error_type"],
)
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
BUDGET_REMAINING = Gauge(
    "atlas_budget_remaining_usd",
    "Budget headroom by scope (global_daily/global_monthly)",
    ["scope"],
)
CIRCUIT_STATE = Gauge(
    "atlas_circuit_state",
    "Circuit breaker state: 0 closed, 1 half_open, 2 open",
    ["dependency"],
)
DEGRADED_TOTAL = Counter(
    "atlas_degraded_total",
    "Degradation-ladder activations by rung",
    ["rung"],
)
INFLIGHT = Gauge(
    "atlas_inflight_requests",
    "Requests currently executing",
)
PROCESS_RSS = Gauge(
    "atlas_process_rss_bytes",
    "Process resident memory (watch vs the 512 MB cap)",
)
BUILD_INFO = Gauge(
    "atlas_build_info",
    "Build provenance (value 1; labels carry version info)",
    ["version", "corpus_version"],
)
