# RESULTS

All numbers below come from real runs, with command, commit and date recorded (spec §0.3).

## Retrieval metrics on the golden set (M1)

- Run date: 2026-10-04, commit: post-62c8d36 (eval tooling + qdrant search retry)
- Command: `python eval/retrieval_metrics.py --golden eval/golden/golden_draft.jsonl --modes hybrid,dense,sparse --k 5`
- Corpus: 60 arXiv papers (cs.CL/cs.IR/cs.LG) → 2,558 contextualized (cheap mode) chunks in Qdrant
  (`atlas_chunks_v1`, named vectors: dense `BAAI/bge-small-en-v1.5` + sparse `Qdrant/bm25`, RRF fusion)
- Golden set: 131 questions (`golden_draft.jsonl`, LLM-generated with gold ids resolved from the
  source chunks; **pending human review**). Scored here: n=96 content questions
  (72 single_hop, 16 multi_hop, 8 comparative); unanswerable/paraphrase rows are excluded
  (answer-level eval and cache tuning, later milestones). k=5.

| mode          | MRR chunk | MRR doc | nDCG@5 chunk | nDCG@5 doc | Recall@5 chunk | Recall@5 doc |
|---------------|-----------|---------|--------------|------------|----------------|--------------|
| hybrid (RRF)  | **0.559** | **0.882** | **0.557**  | **0.825**  | **0.661**      | 0.833        |
| dense only    | 0.438     | 0.783   | 0.429        | 0.729      | 0.484          | 0.740        |
| sparse (BM25) | 0.539     | 0.852   | 0.539        | 0.804      | 0.661          | **0.849**    |

### Findings (honest, small-corpus caveats apply)

- Hybrid (RRF) beats dense-only decisively: chunk MRR +0.12, chunk Recall@5 +0.18.
- On this 60-paper technical corpus, BM25 is a strong baseline: hybrid ≈ BM25 at chunk level;
  hybrid wins doc-level MRR, BM25 edges doc-level recall. Dense (bge-small) lags on its own.
- Caveats: draft golden set (LLM-generated, human review pending); small corpus inflates absolute
  numbers; questions were seeded from the same corpus they retrieve against. Reranking,
  contextual (`section`/`full`) and corrective/multi-hop axes are M3 ablation work — the
  comparison table above is the dense/BM25/hybrid slice only.

## Cost

- LLM: Z.AI `glm-4.7-flash` free tier — generation of the golden set (14 calls) cost **$0.0000**
  (pricing.yaml records the true unit cost of 0; the binding constraint is free-tier rate limits).
- Everything else (embedding, retrieval metrics) is local ONNX / Qdrant free tier: $0.
