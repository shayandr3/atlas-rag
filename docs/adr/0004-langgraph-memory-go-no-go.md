# ADR 0004: LangGraph memory go/no-go — KEEP, the models are the constraint

- Date: 2026-10-04
- Status: accepted
- Measurement: `scripts/measure_rss.py` (peak Working Set, Windows; identical components
  and query in both modes; scripted LLM so the numbers are reproducible and free)

## Context

Spec §4.1: keep LangGraph only if the serving process stays under ~400 MB RSS with it.
The serving stack loads three ONNX models (bge-small dense, Qdrant/bm25 sparse,
jina-reranker-v1-tiny-en) plus onnxruntime.

## Measurement (dev machine, 2026-10-04)

| Stage                          | without LangGraph | with LangGraph | delta |
|--------------------------------|-------------------|----------------|-------|
| after imports                  | 81 MB             | 81 MB          | ~0    |
| after models loaded            | 535 MB (peak 572) | 549 MB (peak 591) | +14 MB |
| peak during one real query     | 1,087 MB          | 1,155 MB       | +68 MB |

## Decision

**Keep LangGraph.** Its marginal cost (+14 MB resident, +68 MB transient peak) is not the
binding constraint. The honest finding is that the local model set itself exceeds the
400 MB envelope before any query runs — on both configurations.

## Consequences

- The 400 MB rule's intent (surviving a 512 MB container) must be met in M7 by the
  deployment configuration, not by dropping LangGraph: `rerank_backend=none` on the free
  Render instance (rerank stays in the dev/eval path), batch-size discipline, and a fresh
  measurement on Render itself (its own ADR/runbook entry).
- The sequential runner remains available behind the same step functions; switching
  orchestrators is a config change, not a rewrite.
- Model memory mitigations to evaluate in M7: int8-quantized dense model, lazy reranker
  loading, and Qdrant Cloud inference for query embeddings.
