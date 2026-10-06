# Runbook (spec §13.3)

For each alert: likely cause → first checks → mitigation. All dashboards referenced
live in Grafana (local: `docker compose -f docker-compose.monitoring.yml up -d` →
http://localhost:3000).

## AtlasHighErrorRate (5xx > 5% for 5m)

1. **First checks:** which route/status? (`atlas_http_requests_total` by `status`, `route`).
   Is `AtlasCircuitOpen` firing too? Then a dependency is down — jump to that runbook row.
2. **Likely causes:** Qdrant Cloud suspended (free tier), LLM provider outage, rate-limit
   masquerading as 5xx (should be 429 — if you see 5xx, check the limiter's Redis fallback).
3. **Mitigation:** if Qdrant: resume the cluster in the cloud console (or hit `/readyz` after
   the keep-alive scrape resumes it). If the LLM provider: the degradation ladder already
   serves extractive answers; consider pausing the demo key until the provider recovers.

## AtlasHighLatency (p95 > 30s for 10m)

1. **First checks:** stage p95 panel — which stage exploded? `retrieve` → Qdrant;
   `generate` → LLM provider. Also check cold start: Render spins down after 15 min idle
   (~1 min wake-up shows as one huge sample, then normal).
2. **Mitigation:** none needed for a cold start (keep-alive scrape prevents it — ADR in M7);
   if `generate` is slow, the free-tier model may be contended — the fallback model path
   already covers hard failures; optionally lower `rerank_top_in`.

## AtlasBudgetLow (daily budget < 20%)

1. **First checks:** `/admin/stats` (admin token) — spend by scope; who burned it: per-key
   counters (`budget:key:*` in Redis).
2. **Mitigation:** lower `DEFAULT_KEY_DAILY_BUDGET_USD` or the global limit and redeploy;
   the kill switch returns 429s automatically at exhaustion. The provider-side spend cap is
   the last backstop.

## AtlasCircuitOpen (any breaker open > 5m)

1. **First checks:** which dependency (`atlas_circuit_state` by `dependency`)?
   - `llm`: provider outage → rung 2/3 keep the service degraded-but-up.
   - `qdrant`: cluster suspended → resume it; breaker half-opens automatically after
     `reset_timeout_s`.
2. **Mitigation:** fix the dependency; the breaker recovers on its own. If it keeps
   reopening, check whether failures are timeouts (raise the per-dependency timeout) or
   auth (expired key).

## AtlasMemoryHigh (RSS > 450 MB)

1. **First checks:** RSS panel trend (leak vs step change). A step change after a deploy:
   new model or bigger batch. Slow climb: leak — capture `atlas_inflight_requests` first.
2. **Mitigation:** on the free tier this precedes an OOM kill: reduce `rerank_top_in`,
   disable the reranker (`RERANK_BACKEND=none`), redeploy. ADR 0004 records the model
   memory baseline.

## AtlasAbstentionSpike (abstentions > 50% for 15m)

1. **First checks:** guard blocks panel — are we under a prompt-injection campaign?
   Retrieval top-score panel — did the corpus change (`CORPUS_VERSION` bump without
   re-ingestion?)?
2. **Mitigation:** if a bad corpus re-index: re-run ingestion and bump `CORPUS_VERSION`.
   If queries are genuinely out-of-corpus: expected behavior, raise the alert threshold.

## Where alerts route

Alertmanager/Grafana notification channels are configured per deployment (email for the
human owner; a Slack webhook is the natural second hop). Local dev: alerts visible in
Prometheus UI → http://localhost:9090/alerts.
