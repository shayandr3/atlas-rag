# RESULTS

All numbers below come from real runs, with command, commit and date recorded (spec §0.3).

## Retrieval ablation on the golden set (M1/M3)

- Run date: 2026-10-04, command:
  `python eval/retrieval_metrics.py --golden eval/golden/golden_draft.jsonl --modes hybrid,dense,sparse --k 5`
  and `python eval/run_ablation.py --golden eval/golden/golden_draft.jsonl --k 5`
- Corpus: 60 arXiv papers (cs.CL/cs.IR/cs.LG) → 2,558 contextualized (cheap mode) chunks in Qdrant
  (`atlas_chunks_v1`; dense `BAAI/bge-small-en-v1.5` + sparse `Qdrant/bm25`, RRF fusion,
  cross-encoder `jinaai/jina-reranker-v1-tiny-en`)
- Golden set: 131 questions (`golden_draft.jsonl`, LLM-generated, gold ids resolved from source
  chunks; **pending human review**). Scored: n=96 content questions (72 single_hop, 16 multi_hop,
  8 comparative), k=5.

| config          | recall_chunk | recall_doc | MRR chunk | MRR doc | nDCG@5 chunk | nDCG@5 doc |
|-----------------|--------------|------------|-----------|---------|--------------|------------|
| dense only      | 0.484        | 0.740      | 0.438     | 0.783   | 0.429        | 0.729      |
| BM25 only       | 0.661        | 0.849      | 0.539     | 0.852   | 0.539        | 0.804      |
| hybrid (RRF)    | 0.661        | 0.833      | 0.569     | 0.872   | 0.565        | 0.817      |
| hybrid + rerank | **0.734**    | 0.875      | **0.644** | **0.885** | **0.624**  | 0.824      |

### Findings (honest, small-corpus caveats apply)

- **Reranking is the biggest single win**: chunk Recall@5 +0.073 and chunk MRR +0.075 over
  hybrid alone, with doc-level recall improving too (0.833 → 0.875).
- Hybrid (RRF) clearly beats dense-only (chunk MRR +0.13); BM25 is a strong baseline on this
  small technical corpus and slightly leads hybrid on doc recall.
- `section`/`full` contextual modes, cache warm/cold and LLM-judged answer quality (correctness,
  faithfulness, abstention precision) are later-milestone ablation axes — not yet measured.
- Caveats: draft golden set (human review pending); questions were seeded from the same corpus
  they retrieve against, which inflates absolute numbers.

## Pipeline probe (paced free-tier sample, 2026-10-04)

`run_ablation.py --llm-sample 2 --pace 75`: corrective on/off answered both sampled single-hop
questions (latency 4.6–21.5 s); two multi-hop decompositions completed (75–93 s, multiple paced
LLM hops). Zero USD (free tier). These are mechanism checks, not quality measurements —
LLM-judged correctness/faithfulness come with the M8 eval.

## Memory: LangGraph go/no-go (ADR 0004)

`scripts/measure_rss.py` (peak Working Set, Windows; identical components + scripted LLM):

| stage                    | without LangGraph | with LangGraph | delta   |
|--------------------------|-------------------|----------------|---------|
| after imports            | 81 MB             | 81 MB          | ~0      |
| after models loaded      | 535 MB (peak 572) | 549 MB (peak 591) | +14 MB |
| peak during one query    | 1,087 MB          | 1,155 MB       | +68 MB  |

**Verdict: keep LangGraph** (+14 MB resident / +68 MB transient is not the constraint).
The honest finding: the three-model ONNX stack itself exceeds the spec's 400 MB envelope
before any query — the M7 deployment will run `rerank_backend=none` on the free Render
instance, with a fresh on-Render measurement recorded there.

## Cost

- LLM: Z.AI `glm-4.7-flash` free tier — golden set generation (14 calls), pipeline probes and
  every live smoke test cost **$0.0000** total. The binding constraint is free-tier rate limits
  (paced calls, patient retries).
- Everything else (embedding, reranking, retrieval metrics, ablation retrieval axes) is local
  ONNX / Qdrant free tier: $0.
