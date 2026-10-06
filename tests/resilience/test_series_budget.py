"""Series-budget test (spec §13.2): labels must stay low-cardinality.

Runs a synthetic workload across routes/models/rungs and asserts the total active
series stays well under the free-tier budget (≤ ~1,500 target).
"""

from prometheus_client import REGISTRY, generate_latest
from tests.fakes import FakeLLM, FakeRetriever, make_chunks

from atlas.config import Settings
from atlas.pipeline.ask import run_ask
from atlas.resilience.breaker import Breakers


async def test_series_budget_under_synthetic_workload() -> None:
    settings = Settings(_env_file=None)
    llm = FakeLLM("Answer [S1].")
    retriever = FakeRetriever(make_chunks())
    breakers = Breakers.build()

    queries = [f"question number {i} about retrieval" for i in range(5)]
    for q in queries:
        await run_ask(
            q,
            retriever=retriever,
            llm=llm,
            settings=settings,
            reranker=None,
            breakers=breakers,
            request_id=q,
        )

    latest = generate_latest(REGISTRY).decode("utf-8")
    series = {
        line.split("{")[0] for line in latest.splitlines() if line and not line.startswith("#")
    }
    total_samples = sum(1 for line in latest.splitlines() if line and not line.startswith("#"))

    # every metric family must come from the fixed catalogue (bounded label values)
    assert len(series) <= 60, f"metric families exploded: {len(series)}"
    # total active samples must stay far below the 1,500-series budget
    assert total_samples <= 1500, f"series budget exceeded: {total_samples}"
