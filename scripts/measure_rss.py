"""Measure serving-process RSS with and without LangGraph (spec §4.1 go/no-go).

Run twice and compare:
    python scripts/measure_rss.py --mode without
    python scripts/measure_rss.py --mode with
Both modes load the identical serving components (fastembed dense+sparse, cross-encoder
reranker, Qdrant client) and run one real retrieval+pipeline query with a scripted LLM;
the only difference is the LangGraph import + graph wiring + graph execution path.
Peak Working Set (Windows) / ru_maxrss (Linux) is the go/no-go number.
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas.config import get_settings
from atlas.pipeline.ask import run_ask
from atlas.retrieval.chunks import Chunk


def _rss_windows() -> tuple[int, int]:
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    k32 = ctypes.windll.kernel32
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    pmc = PMC()
    pmc.cb = ctypes.sizeof(pmc)
    if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return int(pmc.WorkingSetSize), int(pmc.PeakWorkingSetSize)


def memory() -> dict[str, int]:
    if sys.platform == "win32":
        current, peak = _rss_windows()
        return {"rss": current, "peak": peak}
    import resource

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    return {"rss": peak, "peak": peak}


class ScriptedLLM:
    """Deterministic stand-in for the LLM so the measurement is free and reproducible."""

    async def complete(self, *, system: str, user: str, model: str, max_tokens: int) -> Any:
        return type(
            "R",
            (),
            {
                "text": "ok [S1].",
                "input_tokens": 10,
                "output_tokens": 2,
                "cached_tokens": 0,
                "cost_usd": 0.0,
            },
        )()


@dataclass
class ProbeResult:
    answer: str


async def run_probe(mode: str) -> dict[str, Any]:
    report: dict[str, Any] = {"mode": mode}
    report["after_imports"] = memory()

    settings = get_settings()
    from atlas.retrieval.embedders import FastembedEmbedder
    from atlas.retrieval.hybrid import HybridRetriever
    from atlas.retrieval.qdrant_repo import QdrantRepo
    from atlas.retrieval.rerank import FastembedReranker

    embedder = FastembedEmbedder(settings.embed_model, settings.sparse_model)
    repo = QdrantRepo(settings.qdrant_url, settings.qdrant_api_key, settings.qdrant_collection, 30)
    retriever = HybridRetriever(embedder, repo)
    reranker = FastembedReranker(settings.rerank_model).rerank
    llm = ScriptedLLM()

    if mode == "with":
        from atlas.pipeline.graph import build_graph_runner

        graph = build_graph_runner(
            retriever=retriever, llm=llm, settings=settings, cache=None, reranker=reranker
        )
    else:
        graph = None

    await embedder.embed_query("retrieval augmented generation probe")
    await reranker(
        "probe",
        [
            Chunk(
                chunk_id="x",
                doc_id="d",
                title="t",
                section_path="s",
                text="some text",
                source_url="",
            )
        ],
    )
    report["after_models_loaded"] = memory()

    await run_ask(
        "What is retrieval-augmented generation?",
        retriever=retriever,
        llm=llm,
        settings=settings,
        reranker=reranker,
        graph=graph,
    )
    report["peak_during_query"] = memory()
    report["langgraph_in_modules"] = any("langgraph" in m for m in sys.modules)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["with", "without"], required=True)
    args = parser.parse_args()
    report = asyncio.run(run_probe(args.mode))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
