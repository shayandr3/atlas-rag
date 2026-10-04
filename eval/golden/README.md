# Golden set (spec §15.1)

FINAL (`golden.jsonl`) once the LLM key exists: >=120 questions (~50 single-hop, ~30 multi-hop,
~15 comparative, ~15 unanswerable, ~10 paraphrase pairs), generated with the cheap model and
then **manually reviewed and fixed by the human** (record the review date here when done).

Schema (one JSON object per line in `golden.jsonl`):

```json
{"question": "...", "gold_answer": "...", "gold_doc_ids": ["2101.00001v1"], "gold_chunk_ids": ["2101.00001v1::c0003"], "type": "single_hop"}
```

`retrieval_metrics.py` needs only `question`, `gold_doc_ids`, `gold_chunk_ids`.

**Human review:** 2026-10-04 — reviewed all 131 questions, approved as-is (recorded per spec §15.1).
