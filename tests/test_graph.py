import pytest
from langgraph.errors import GraphRecursionError
from tests.fakes import FakeLLM, FakeReranker, FakeRetriever, make_chunks

from atlas.config import Settings
from atlas.pipeline.graph import build_graph_runner


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def _runner(llm: FakeLLM, settings: Settings | None = None, **kw):
    return build_graph_runner(
        retriever=FakeRetriever(make_chunks()), llm=llm, settings=settings or _settings(), **kw
    )


def test_graph_has_expected_structure() -> None:
    runner = _runner(FakeLLM("Answer [S1]."))
    graph = runner._app.get_graph()

    node_names = (
        {n.name for n in graph.nodes.values()}
        if hasattr(graph.nodes, "values")
        else set(graph.nodes)
    )
    expected = {
        "cache_lookup",
        "route",
        "embed",
        "semantic",
        "acquire",
        "retrieve",
        "grade",
        "transform",
        "multi_hop",
        "generate",
        "cache_store",
    }
    assert expected <= node_names


async def test_graph_simple_path_end_to_end() -> None:
    runner = _runner(FakeLLM("Answer [S1]."))

    result = await runner.run("what is rag?", request_id="rid-1")

    assert result.answer == "Answer [S1]."
    assert result.route == "simple"
    assert result.cache_status == "miss"
    assert result.request_id == "rid-1"


async def test_graph_greeting_short_circuits() -> None:
    llm = FakeLLM("should not be called")
    runner = _runner(llm)

    result = await runner.run("thanks!")

    assert result.route == "no_retrieval"
    assert llm.calls == 0


async def test_graph_corrective_loop_is_bounded_and_abstains() -> None:
    import dataclasses

    settings = _settings(grade_skip_threshold=0.99)  # force grading (fake scores are 1.0/0.9)
    llm = FakeLLM(
        responses=[
            '{"id": "S1", "verdict": "irrelevant"}',
            '{"query": "rewritten once"}',
            '{"id": "S1", "verdict": "irrelevant"}',
            '{"query": "rewritten twice"}',
            '{"id": "S1", "verdict": "irrelevant"}',
        ]
    )
    chunks = [dataclasses.replace(c, score=0.5) for c in make_chunks()]
    retriever = FakeRetriever(chunks)
    runner = build_graph_runner(retriever=retriever, llm=llm, settings=settings)

    result = await runner.run("obscure question?")

    assert result.abstained is True
    assert llm.calls == 5  # grade, rewrite, grade, rewrite, grade — then stop


async def test_graph_reranker_degradation() -> None:
    settings = _settings()
    runner = build_graph_runner(
        retriever=FakeRetriever(make_chunks()),
        llm=FakeLLM("Answer [S1]."),
        settings=settings,
        reranker=FakeReranker(fail=True).rerank,
    )

    result = await runner.run("what is rag?")

    assert result.degraded == ["rerank"]
    assert result.answer == "Answer [S1]."


async def test_graph_recursion_limit_is_enforced() -> None:
    runner = _runner(FakeLLM("Answer [S1]."))

    state = {
        "query": "what is rag?",
        "request_id": "rid",
        "degraded": [],
        "corrective_loops": 0,
        "result": None,
    }
    with pytest.raises(GraphRecursionError):
        await runner._app.ainvoke(state, config={"recursion_limit": 2})
