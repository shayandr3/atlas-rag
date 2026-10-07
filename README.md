---
title: Atlas RAG
emoji: "🗺️"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 10000
pinned: false
---
# atlas-rag

A production-grade, observable, cost-aware, secured RAG API + demo UI, engineered to run inside
a free-tier envelope (Render 512 MB / 0.1 vCPU, Qdrant Cloud free, Grafana Cloud free):
hybrid retrieval (dense + BM25 → RRF), cross-encoder reranking, corrective retrieval grading,
multi-hop decomposition, adaptive routing, layered Redis caching, hard USD budgets, and
Prometheus/Grafana observability.

Full build spec: [PRODUCTION_RAG_PROJECT_SPEC.md](PRODUCTION_RAG_PROJECT_SPEC.md).

> **Status:** M0–M7 complete — deployed on Render (free tier), observability on Grafana Cloud. Live demo: this README updates with the URL at M8 sign-off.
>
> **Numbers policy:** latency, cost, recall and faithfulness figures appear in this README only
> when produced by a recorded eval/load run. Anything unmeasured says `TBD (not measured)`.

## Milestones

| M  | Scope                                                                    | Status |
|----|--------------------------------------------------------------------------|--------|
| M0 | Repo, tooling, CI skeleton, pre-commit, ADR 0001                         | ✅     |
| M1 | Ingestion + Qdrant hybrid retrieval + LLM adapter + basic `/v1/ask`      | ⬜     |
| M2 | Redis caches (L0–L3), single-flight, fail-open                           | ✅     |
| M3 | Reranker, router, corrective loop, multi-hop, LangGraph + memory go/no-go | ✅    |
| M4 | Security layer (auth, rate limits, guards, red-team suite, budgets)      | ✅     |
| M5 | Resilience (breakers, degradation ladder, chaos tests)                   | ✅     |
| M6 | Observability (metrics catalogue, dashboards, alerts; LangSmith optional, deferred) | ✅    |
| M7 | Dockerize + Render deploy + Grafana Cloud + keep-alive ADR               | ✅     |
| M8 | Eval at scale, load test, RESULTS.md, final README                       | ⬜     |

## Development

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash; Python 3.12+
pip install -e ".[dev]"
pre-commit run --all-files                              # ruff + mypy + gitleaks
pytest -q
docker compose up -d                                    # local Qdrant + Redis
```

CI runs ruff, mypy (strict), pytest, gitleaks and pip-audit on every push/PR
(`.github/workflows/ci.yml`).

## Security

Auth (peppered key hashes), rate limits, layered prompt-injection guards, canary leak
detection, PII redaction, USD budgets — mapped to the OWASP LLM Top 10 in
[docs/security.md](docs/security.md), with a 43-prompt red-team suite:
**93% block rate, 0% false positives** (measured 2026-10-04).

## Deployment

Single Docker image (multi-stage, non-root, ONNX models baked at build so cold start is
load-only; the cross-encoder is deliberately excluded on the free tier — [ADR 0004](docs/adr/0004-langgraph-memory-go-no-go.md)).

**Live platform: Hugging Face Spaces (Docker)** — Render's free tier demanded card
verification the operator could not provide, so the image deploys as a HF Space instead
(same Dockerfile, no card required, more RAM than Render free) — [ADR 0006](docs/adr/0006-hosting-hf-spaces.md).
No managed Redis on the platform: caches fail open and the rate limiter uses its in-process
fallback (single worker, sound by design). Keep-alive strategy:
[ADR 0005](docs/adr/0005-keep-alive-grafana-scrape.md) (Grafana Cloud scrape).
Post-deploy smoke: `DEPLOY_URL=... DEPLOY_KEY=... python scripts/smoke_deploy.py`.
Serving config: `RERANK_BACKEND=none`, 1 uvicorn worker, port 10000 (`app_port`).
The spec-conformant [Render Blueprint](render.yaml) is kept for reference.

## Observability

Full Prometheus catalogue (26 metric families, low-cardinality labels, series-budget
test) with a pre-provisioned local stack:

```bash
docker compose -f docker-compose.monitoring.yml up -d   # Prometheus :9090, Grafana :3000
uvicorn atlas.main:app --port 10000
```

Grafana loads the `atlas-rag` dashboard automatically (traffic, stage latency, cost per
query, cache hit ratio, abstentions/faithfulness, guard blocks, circuit states, RSS vs
the 512 MB cap). Alerts + runbook: `ops/prometheus/alerts.yml`, `docs/runbook.md`.

## Docs

- ADRs: [docs/adr/](docs/adr/) — 0001: one flagship project first
- Measured results: [docs/RESULTS.md](docs/RESULTS.md) — TBD (not measured)
