"""LangGraph orchestration of the pipeline (spec §4.1): graph wiring only.

Node bodies are the framework-free step functions in pipeline/ask.py; this module
binds them into a StateGraph with conditional edges for routing and the corrective
loop, plus an explicit recursion limit as a second safety net. A GraphRunner exposes
.run(query) -> AskResult; the sequential runner in ask.py is the fallback.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Any, TypedDict

from atlas.config import Settings
from atlas.pipeline.ask import (
    AskResult,
    RerankerFn,
    Retriever,
    canned_response,
    result_from_payload,
    result_payload,
    step_cache_lookup,
    step_embed,
    step_generate,
    step_grade_loop,
    step_retrieve,
    step_route,
    step_semantic,
)
from atlas.pipeline.faithfulness import maybe_check_faithfulness
from atlas.retrieval.embedders import QueryEmbedding

logger = logging.getLogger("atlas.graph")


class AskState(TypedDict, total=False):
    query: str
    request_id: str
    route: str
    emb: QueryEmbedding | None
    chunks: list[Any]
    degraded: list[str]
    corrective_loops: int
    result: AskResult | None


@dataclass
class GraphDeps:
    retriever: Retriever
    llm: Any
    settings: Settings
    cache: Any = None
    reranker: RerankerFn | None = None
    breakers: Any = None


class GraphRunner:
    def __init__(self, app: Any, deps: GraphDeps) -> None:
        self._app = app
        self._deps = deps

    async def run(self, query: str, request_id: str | None = None) -> AskResult:
        state: AskState = {
            "query": query,
            "request_id": request_id or uuid.uuid4().hex,
            "degraded": [],
            "corrective_loops": 0,
            "result": None,
        }
        limit = max(
            25,
            self._deps.settings.max_hops * 2 + self._deps.settings.max_corrective_loops * 2 + 8,
        )
        final = await self._app.ainvoke(state, config={"recursion_limit": limit})
        result: AskResult | None = final.get("result")
        if result is None:  # should not happen; defensive
            raise RuntimeError("graph finished without a result")
        return result


def build_graph_runner(
    *,
    retriever: Retriever,
    llm: Any,
    settings: Settings,
    cache: Any = None,
    reranker: RerankerFn | None = None,
    breakers: Any = None,
) -> GraphRunner:
    from langgraph.graph import END, START, StateGraph

    deps = GraphDeps(
        retriever=retriever,
        llm=llm,
        settings=settings,
        cache=cache,
        reranker=reranker,
        breakers=breakers,
    )

    async def cache_lookup(state: AskState) -> dict[str, Any]:
        result = await step_cache_lookup(state["query"], deps.cache)
        return {"result": result}

    async def route(state: AskState) -> dict[str, Any]:
        decision = await step_route(state["query"], deps.llm, deps.settings, deps.cache)
        update: dict[str, Any] = {"route": decision.route}
        if decision.route in ("no_retrieval", "out_of_scope", "unsafe"):
            update["result"] = canned_response(decision, state["request_id"])
        return update

    async def embed(state: AskState) -> dict[str, Any]:
        emb = await step_embed(state["query"], deps.cache)
        return {"emb": emb}

    async def semantic(state: AskState) -> dict[str, Any]:
        result = await step_semantic(state["query"], state.get("emb"), deps.cache)
        return {"result": result} if result is not None else {}

    async def acquire(state: AskState) -> dict[str, Any]:
        if deps.cache is not None and not await deps.cache.acquire(state["query"]):
            payload = await deps.cache.wait_for_response(state["query"])
            if payload is not None:
                return {
                    "result": result_from_payload(payload, state["request_id"], cache_status="hit")
                }
        return {}

    async def retrieve(state: AskState) -> dict[str, Any]:
        chunks, degraded = await step_retrieve(
            state["query"],
            deps.retriever,
            deps.settings,
            deps.cache,
            state.get("emb"),
            deps.reranker,
            deps.breakers,
        )
        return {"chunks": chunks, "degraded": degraded}

    async def grade(state: AskState) -> dict[str, Any]:
        chunks, degraded, enough = await step_grade_loop(
            state["query"],
            state.get("chunks", []),
            deps.llm,
            deps.settings,
            deps.retriever,
            deps.cache,
            state.get("emb"),
            deps.reranker,
            state.get("degraded", []),
            deps.breakers,
        )
        update: dict[str, Any] = {
            "chunks": chunks,
            "degraded": degraded,
            "corrective_loops": state.get("corrective_loops", 0) + 1,
        }
        if not enough:
            from atlas.pipeline.ask import _abstain_with_evidence

            update["result"] = await _abstain_with_evidence(
                state["query"], chunks, state.get("route", "simple"), degraded, state["request_id"]
            )
        return update

    async def transform(state: AskState) -> dict[str, Any]:
        return {}  # decomposition happens inside the multi-hop node (bounded by MAX_HOPS)

    async def multi_hop(state: AskState) -> dict[str, Any]:
        from atlas.pipeline.multihop import run_multi_hop

        result = await run_multi_hop(
            state["query"],
            retriever=deps.retriever,
            llm=deps.llm,
            settings=deps.settings,
            cache=deps.cache,
            emb=state.get("emb"),
            reranker=deps.reranker,
            request_id=state["request_id"],
        )
        return {"result": result, "degraded": result.degraded}

    async def generate(state: AskState) -> dict[str, Any]:
        result = await step_generate(
            state["query"],
            state.get("chunks", []),
            deps.llm,
            deps.settings,
            state.get("route", "simple"),
            state.get("degraded", []),
            state["request_id"],
            deps.breakers,
        )
        maybe_check_faithfulness(result, state.get("chunks", []), deps.llm, deps.settings)
        return {"result": result}

    async def cache_store(state: AskState) -> dict[str, Any]:
        result = state.get("result")
        if result is not None and deps.cache is not None:
            await deps.cache.put_response(state["query"], result_payload(result))
            await deps.cache.put_semantic(state["query"], state.get("emb"), result_payload(result))
        return {}

    def after_cache_lookup(state: AskState) -> str:
        return "done" if state.get("result") is not None else "route"

    def after_route(state: AskState) -> str:
        if state.get("result") is not None:
            return "done"
        return "transform" if state.get("route") == "multi_hop" else "embed"

    def after_semantic(state: AskState) -> str:
        return "done" if state.get("result") is not None else "acquire"

    def after_acquire(state: AskState) -> str:
        return "done" if state.get("result") is not None else "retrieve"

    def after_grade(state: AskState) -> str:
        if state.get("result") is not None:
            return "done"
        return "generate"

    builder = StateGraph(AskState)
    builder.add_node("cache_lookup", cache_lookup)
    builder.add_node("route", route)
    builder.add_node("embed", embed)
    builder.add_node("semantic", semantic)
    builder.add_node("acquire", acquire)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade)
    builder.add_node("transform", transform)
    builder.add_node("multi_hop", multi_hop)
    builder.add_node("generate", generate)
    builder.add_node("cache_store", cache_store)

    builder.add_edge(START, "cache_lookup")
    builder.add_conditional_edges(
        "cache_lookup", after_cache_lookup, {"done": END, "route": "route"}
    )
    builder.add_conditional_edges(
        "route", after_route, {"done": END, "embed": "embed", "transform": "transform"}
    )
    builder.add_edge("embed", "semantic")
    builder.add_conditional_edges("semantic", after_semantic, {"done": END, "acquire": "acquire"})
    builder.add_conditional_edges("acquire", after_acquire, {"done": END, "retrieve": "retrieve"})
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", after_grade, {"done": END, "generate": "generate"})
    builder.add_edge("transform", "multi_hop")
    builder.add_edge("multi_hop", "generate")
    builder.add_edge("generate", "cache_store")
    builder.add_edge("cache_store", END)

    return GraphRunner(builder.compile(), deps)
