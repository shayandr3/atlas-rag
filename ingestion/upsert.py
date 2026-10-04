"""Embed contextualized chunks and upsert to Qdrant. Free: local ONNX models only (spec §7.4)."""

import argparse
import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import models
from qdrant_client.http.exceptions import ResponseHandlingException
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from atlas.config import get_settings
from atlas.retrieval.embedders import FastembedEmbedder
from atlas.retrieval.qdrant_repo import QdrantRepo


def point_id(chunk_id: str) -> str:
    """Deterministic UUID so re-ingestion upserts idempotently (Qdrant ids must be UUID/int)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"atlas-rag:{chunk_id}"))


@retry(
    retry=retry_if_exception_type((ResponseHandlingException,)),
    wait=wait_fixed(5),
    stop=stop_after_attempt(4),
    reraise=True,
)
async def _upsert_batch(repo: QdrantRepo, points: list[models.PointStruct]) -> None:
    """Free-cluster writes can exceed short timeouts; batches are idempotent, so retry."""
    await repo.upsert(points)


async def run(args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    settings = get_settings()
    est_kb = sum(len(str(r.get("text", ""))) for r in records) // 1024
    print(f"{len(records)} chunks (~{est_kb} KB text)")
    if args.dry_run:
        print("dry-run: no network calls made")
        return

    collection = args.collection or settings.qdrant_collection
    embedder = FastembedEmbedder(settings.embed_model, settings.sparse_model)
    repo = QdrantRepo(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=collection,
        timeout_seconds=settings.qdrant_timeout_seconds,
    )

    sample = await embedder.embed_query(records[0]["context_prefix"] + "\n" + records[0]["text"])
    await repo.ensure_collection(len(sample.dense))
    print(f"collection {collection!r} ready (dense_dim={len(sample.dense)})")

    done = 0
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        embeddings = await embedder.embed_passages(
            [str(r["context_prefix"]) + "\n" + str(r["text"]) for r in batch]
        )
        points = [
            models.PointStruct(
                id=point_id(str(r["chunk_id"])),
                vector={
                    "dense": emb.dense,
                    "bm25": models.SparseVector(
                        indices=emb.sparse_indices, values=emb.sparse_values
                    ),
                },
                payload={**r, "corpus_version": settings.corpus_version},
            )
            for r, emb in zip(batch, embeddings, strict=True)
        ]
        await _upsert_batch(repo, points)
        done += len(points)
        print(f"upserted {done}/{len(records)}")
    print(f"done: collection {collection!r} now has {await repo.count()} points")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="infile", default="data/chunks/chunks_ctx.jsonl")
    parser.add_argument("--collection", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    records: list[dict[str, Any]] = [
        json.loads(line)
        for line in Path(args.infile).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    asyncio.run(run(args, records))


if __name__ == "__main__":
    main()
