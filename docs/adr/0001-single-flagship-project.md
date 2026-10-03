# ADR 0001: Build one flagship project before any second project

- Date: 2026-10-03
- Status: accepted
- Deciders: project owner with Claude Code

## Context

The goal is a portfolio that demonstrates production-grade LLM engineering: a live, measured,
secured RAG system under real free-tier constraints (Render 512 MB / 0.1 vCPU, Qdrant Cloud
free, Grafana Cloud free). Options considered: (a) one deep flagship, (b) two shallower
projects in parallel, (c) fold GraphRAG into the flagship.

## Decision

Build ONE flagship (`atlas-rag`, this spec) end-to-end — live, measured, secured, documented —
before starting anything else. GraphRAG is deferred to an optional Project B (spec §19) that
reuses the chassis extracted from this repo (auth, cache, metrics, resilience, CI) over a
different corpus.

## Alternatives

- Two projects in parallel: halves the depth available for each; neither would gather the
  measured evidence (ablations, chaos tests, live dashboards) that makes a RAG portfolio
  credible to reviewers.
- GraphRAG inside the flagship: LLM entity/relation extraction is expensive, adds storage, and
  risks the 512 MB memory ceiling on the serving instance.

## Consequences

- Depth over breadth: one polished system with honest, measured results.
- Domain modules must stay framework-free and cleanly adapter-wired so the chassis remains
  extractable (this also serves the LangGraph fallback plan in spec §4.1).
- Project B is blocked until the flagship is live and documented.
