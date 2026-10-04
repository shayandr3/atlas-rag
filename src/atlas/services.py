"""Lazy service container so /healthz stays cheap and tests can inject fakes."""

from dataclasses import dataclass
from typing import Any

from atlas.cache.client import FailOpenRedis
from atlas.cache.service import CacheService
from atlas.config import Settings
from atlas.llm.adapter import LLM, build_llm
from atlas.llm.pricing import PricingTable
from atlas.retrieval.embedders import FastembedEmbedder
from atlas.retrieval.hybrid import HybridRetriever
from atlas.retrieval.qdrant_repo import QdrantRepo
from atlas.retrieval.rerank import FastembedReranker


@dataclass
class Services:
    settings: Settings
    pricing: PricingTable
    embedder: FastembedEmbedder
    repo: QdrantRepo
    llm: LLM
    retriever: HybridRetriever
    cache: CacheService | None = None
    reranker: Any = None
    graph: Any = None

    @classmethod
    def build(cls, settings: Settings) -> "Services":
        problems = settings.validate_prod()
        if problems:
            raise RuntimeError(f"unsafe config: missing {', '.join(problems)}")
        if not settings.qdrant_url:
            raise RuntimeError("QDRANT_URL is not set; copy .env.example to .env and fill it in")
        pricing = PricingTable.load()
        embedder = FastembedEmbedder(settings.embed_model, settings.sparse_model)
        repo = QdrantRepo(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection=settings.qdrant_collection,
            timeout_seconds=settings.qdrant_timeout_seconds,
        )
        llm = build_llm(settings, pricing)
        reranker: Any = None
        if settings.rerank_backend == "local_onnx":
            model = FastembedReranker(settings.rerank_model)
            reranker = model.rerank
        cache: CacheService | None = None
        if settings.redis_url:
            cache = CacheService(
                settings,
                FailOpenRedis(settings.redis_url, socket_timeout=settings.cache_socket_timeout),
                embedder=embedder,
            )
        graph: Any = None
        try:
            from atlas.pipeline.graph import build_graph_runner

            graph = build_graph_runner(
                retriever=HybridRetriever(embedder, repo),
                llm=llm,
                settings=settings,
                cache=cache,
                reranker=reranker,
            )
        except Exception:
            # LangGraph unavailable or wiring failed: the sequential runner covers it
            pass
        return cls(
            settings=settings,
            pricing=pricing,
            embedder=embedder,
            repo=repo,
            llm=llm,
            retriever=HybridRetriever(embedder, repo),
            cache=cache,
            reranker=reranker,
            graph=graph,
        )
