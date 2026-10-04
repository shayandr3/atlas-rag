# atlas-rag

A production-grade, observable, cost-aware, secured RAG API + demo UI, engineered to run inside
a free-tier envelope (Render 512 MB / 0.1 vCPU, Qdrant Cloud free, Grafana Cloud free):
hybrid retrieval (dense + BM25 → RRF), cross-encoder reranking, corrective retrieval grading,
multi-hop decomposition, adaptive routing, layered Redis caching, hard USD budgets, and
Prometheus/Grafana observability.

Full build spec: [PRODUCTION_RAG_PROJECT_SPEC.md](PRODUCTION_RAG_PROJECT_SPEC.md).

> **Status:** M0 — repo skeleton, tooling, CI. Live demo: TBD (not deployed yet).
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
| M4 | Security layer (auth, rate limits, guards, red-team suite, budgets)      | ⬜     |
| M5 | Resilience (breakers, degradation ladder, chaos tests)                   | ⬜     |
| M6 | Observability (metrics catalogue, dashboards, alerts, optional LangSmith) | ⬜    |
| M7 | Dockerize + Render deploy + Grafana Cloud + keep-alive ADR               | ⬜     |
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

## Docs

- ADRs: [docs/adr/](docs/adr/) — 0001: one flagship project first
- Measured results: [docs/RESULTS.md](docs/RESULTS.md) — TBD (not measured)
