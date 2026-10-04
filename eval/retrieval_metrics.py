"""Retrieval metrics over the golden set: Recall@k, MRR, nDCG@k at chunk and doc level.

Free — no LLM, no spend. Requires QDRANT_URL/QDRANT_API_KEY in .env and the corpus ingested.
Usage:
    python eval/retrieval_metrics.py --golden eval/golden/golden.jsonl --modes hybrid,dense,sparse
"""

import argparse
import asyncio
import json
import math
from pathlib import Path
from typing import Any

from atlas.config import get_settings
from atlas.retrieval.embedders import FastembedEmbedder
from atlas.retrieval.hybrid import HybridRetriever
from atlas.retrieval.qdrant_repo import QdrantRepo


def load_golden(path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


def metrics_for(
    item: dict[str, Any], chunk_ids: list[str], doc_ids: list[str], k: int
) -> dict[str, float]:
    gold_chunks: list[str] = item.get("gold_chunk_ids") or []
    gold_docs: list[str] = item.get("gold_doc_ids") or []
    out: dict[str, float] = {}
    for name, gold, got in (
        ("recall_chunk", gold_chunks, chunk_ids),
        ("recall_doc", gold_docs, doc_ids),
    ):
        out[name] = (len(set(gold) & set(got)) / len(gold)) if gold else 0.0
    for name, gold, got in (("mrr", gold_chunks, chunk_ids), ("mrr_doc", gold_docs, doc_ids)):
        out[name] = next((1.0 / (rank + 1) for rank, g in enumerate(got[:k]) if g in gold), 0.0)
    for name, gold, got in (
        ("ndcg_chunk", gold_chunks, chunk_ids),
        ("ndcg_doc", gold_docs, doc_ids),
    ):
        dcg = sum(1.0 / math.log2(rank + 2) for rank, g in enumerate(got[:k]) if g in gold)
        ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(gold), k)))
        out[name] = dcg / ideal if ideal else 0.0
    return out


async def evaluate(modes: list[str], golden: list[dict[str, Any]], k: int) -> None:
    settings = get_settings()
    embedder = FastembedEmbedder(settings.embed_model, settings.sparse_model)
    repo = QdrantRepo(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
        timeout_seconds=settings.qdrant_timeout_seconds,
    )
    for mode in modes:
        retriever = HybridRetriever(embedder, repo, mode=mode)
        sums: dict[str, float] = {}
        for item in golden:
            chunks = await retriever.search(item["question"], limit=k)
            scores = metrics_for(
                item,
                [c.chunk_id for c in chunks],
                [c.doc_id for c in chunks],
                k,
            )
            for name, value in scores.items():
                sums[name] = sums.get(name, 0.0) + value
        n = len(golden)
        averages = ", ".join(f"{name}={value / n:.3f}" for name, value in sorted(sums.items()))
        print(f"mode={mode} n={n} k={k}: {averages}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default="eval/golden/golden.jsonl")
    parser.add_argument("--modes", default="hybrid,dense,sparse")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--types",
        default="single_hop,multi_hop,comparative",
        help="types to score (unanswerable/paraphrase are answer-level and cache-tuning items)",
    )
    args = parser.parse_args()
    golden = load_golden(args.golden)
    wanted = {t.strip() for t in args.types.split(",")}
    golden = [g for g in golden if not wanted or g.get("type", "single_hop") in wanted]
    if not golden:
        print("golden set is empty — generate and review it first (spec §15.1)")
        return
    asyncio.run(evaluate(args.modes.split(","), golden, args.k))


if __name__ == "__main__":
    main()
