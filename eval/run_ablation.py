"""Ablation runner (spec §15.3): same golden set across configurations.

Retrieval axes run over the FULL content golden set (LLM-free):
    dense | sparse | hybrid | hybrid+rerank
Pipeline axes run on a small paced sample (free-tier rate limits, ADR 0003):
    corrective on/off, multi-hop decomposition
Writes results to stdout; `--write` appends the measured tables
into docs/RESULTS.md.

Usage:
    python eval/run_ablation.py --golden eval/golden/golden_draft.jsonl --k 5
    python eval/run_ablation.py --llm-sample 2 --pace 75 --write
"""

import argparse
import asyncio
import time
from pathlib import Path
from typing import Any

from retrieval_metrics import load_golden, metrics_for  # same-dir helper module

from atlas.config import get_settings
from atlas.retrieval.embedders import FastembedEmbedder
from atlas.retrieval.hybrid import HybridRetriever
from atlas.retrieval.qdrant_repo import QdrantRepo


async def retrieval_axis(
    modes: list[str], golden: list[dict[str, Any]], k: int, with_rerank: bool
) -> dict[str, float]:
    settings = get_settings()
    embedder = FastembedEmbedder(settings.embed_model, settings.sparse_model)
    repo = QdrantRepo(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
        timeout_seconds=30,
    )
    reranker = None
    if with_rerank:
        from atlas.retrieval.rerank import FastembedReranker, _cap_per_doc

        model = FastembedReranker(settings.rerank_model)

        async def reranker(query: str, chunks: list[Any]) -> list[Any]:
            ranked = await model.rerank(query, chunks)
            return _cap_per_doc(ranked, per_doc=3)

    sums: dict[str, float] = {}
    for item in golden:
        if with_rerank:
            emb = await embedder.embed_query(item["question"])
            points = await repo.search(
                dense=emb.dense,
                sparse_indices=emb.sparse_indices,
                sparse_values=emb.sparse_values,
                limit=settings.rerank_top_in,
                mode="hybrid",
            )
            chunks = [
                type(
                    "C",
                    (),
                    {
                        "chunk_id": str(p.payload.get("chunk_id", "")),
                        "doc_id": str(p.payload.get("doc_id", "")),
                        "text": str(p.payload.get("text", "")),
                    },
                )()
                for p in points
            ]
            ranked = await reranker(item["question"], chunks)
            chunk_ids = [c.chunk_id for c in ranked]
            doc_ids = [c.doc_id for c in ranked]
        else:
            retriever = HybridRetriever(embedder, repo, mode=modes[0])
            found = await retriever.search(item["question"], limit=k)
            chunk_ids = [c.chunk_id for c in found]
            doc_ids = [c.doc_id for c in found]
        for name, value in metrics_for(item, chunk_ids, doc_ids, k).items():
            sums[name] = sums.get(name, 0.0) + value
    n = len(golden)
    return {name: value / n for name, value in sums.items()}


async def pipeline_probe(
    sample: int, pace: float, golden: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Small paced sample: corrective on/off on single_hop, multi-hop on multi_hop items."""
    from atlas.llm.structured import StructuredOutputError  # noqa: F401
    from atlas.pipeline.ask import run_ask
    from atlas.pipeline.multihop import run_multi_hop
    from atlas.services import Services

    svc = Services.build(get_settings())
    single = [g for g in golden if g["type"] == "single_hop"][:sample]
    multi = [g for g in golden if g["type"] == "multi_hop"][: min(2, sample)]
    rows: list[dict[str, Any]] = []

    for label, settings, questions in (
        ("simple+corrective", get_settings(), single),
        ("simple (no corrective)", _no_corrective(get_settings()), single),
    ):
        for item in questions:
            started = time.monotonic()
            result = await run_ask(
                item["question"],
                retriever=svc.retriever,
                llm=svc.llm,
                settings=settings,
                cache=None,
                reranker=svc.reranker,
            )
            rows.append(
                {
                    "axis": label,
                    "type": item["type"],
                    "abstained": result.abstained,
                    "degraded": result.degraded,
                    "out_tokens": result.usage.output_tokens,
                    "cost_usd": result.usage.cost_usd,
                    "latency_s": round(time.monotonic() - started, 1),
                }
            )
            await asyncio.sleep(pace)

    for item in multi:
        started = time.monotonic()
        result = await run_multi_hop(
            item["question"],
            retriever=svc.retriever,
            llm=svc.llm,
            settings=get_settings(),
            cache=None,
            request_id="ablation",
        )
        rows.append(
            {
                "axis": "multi_hop decomposition",
                "type": item["type"],
                "abstained": result.abstained,
                "degraded": result.degraded,
                "out_tokens": result.usage.output_tokens,
                "cost_usd": result.usage.cost_usd,
                "latency_s": round(time.monotonic() - started, 1),
            }
        )
        await asyncio.sleep(pace)
    return rows


def _no_corrective(settings: Any) -> Any:
    settings.grade_skip_threshold = 0.0  # top score always >= 0 -> grading skipped
    return settings


def append_results(table: str) -> None:
    """Sync helper: blocking file IO must stay out of the async body (ASYNC240)."""
    results = Path("docs/RESULTS.md")
    stamp = time.strftime("%Y-%m-%d")
    text = results.read_text(encoding="utf-8")
    text += "\n## Ablation run (" + stamp + ")\n\n```\n" + table + "\n```\n"
    results.write_text(text, encoding="utf-8")
    print("appended to docs/RESULTS.md")


async def main_async(args: argparse.Namespace) -> None:
    content_types = {"single_hop", "multi_hop", "comparative"}
    golden = [g for g in load_golden(args.golden) if g.get("type") in content_types]
    if not golden:
        print("golden set empty — generate it first (eval/generate_golden.py)")
        return
    print(f"golden content questions: {len(golden)}, k={args.k}")

    rows: list[tuple[str, dict[str, float]]] = []
    axes = (("dense only", "dense"), ("BM25 only", "sparse"), ("hybrid (RRF)", "hybrid"))
    for label, mode in axes:
        started = time.monotonic()
        metrics = await retrieval_axis([mode], golden, args.k, with_rerank=False)
        rows.append((label, metrics))
        print(f"{label}: done in {time.monotonic() - started:.0f}s")
    started = time.monotonic()
    metrics = await retrieval_axis(["hybrid"], golden, args.k, with_rerank=True)
    rows.append(("hybrid + rerank", metrics))
    print(f"hybrid + rerank: done in {time.monotonic() - started:.0f}s")

    header = "| config | " + " | ".join(rows[0][1].keys()) + " |"
    sep = "|" + "---|" * (len(rows[0][1]) + 1)
    lines = [header, sep]
    for label, metrics in rows:
        lines.append("| " + label + " | " + " | ".join(f"{v:.3f}" for v in metrics.values()) + " |")
    table = "\n".join(lines)
    print("\n" + table)

    if args.llm_sample > 0:
        probe_rows = await pipeline_probe(args.llm_sample, args.pace, golden)
        print("\npipeline probe (paced, free-tier):")
        for row in probe_rows:
            print(
                f"  {row['axis']}: abstained={row['abstained']} degraded={row['degraded']}"
                f" latency={row['latency_s']}s out={row['out_tokens']}tok"
                f" cost=${row['cost_usd']:.4f}"
            )

    if args.write:
        append_results(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default="eval/golden/golden_draft.jsonl")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--llm-sample", type=int, default=0, help="pipeline-probe questions per axis (0 skips)"
    )
    parser.add_argument("--pace", type=float, default=75.0, help="seconds between LLM calls")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
