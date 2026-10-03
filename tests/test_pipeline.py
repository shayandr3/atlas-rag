from tests.fakes import FakeLLM, FakeRetriever, make_chunks

from atlas.config import Settings
from atlas.llm.prompts import CANARY_TOKEN
from atlas.pipeline.ask import extract_labels, pack_context, run_ask
from atlas.retrieval.chunks import Chunk


def _settings() -> Settings:
    return Settings(_env_file=None)


async def test_run_ask_maps_citations_and_usage() -> None:
    llm = FakeLLM("Point one [S1]. Point two [S2]. Ghost [S9].")

    result = await run_ask(
        "what is rag?",
        retriever=FakeRetriever(make_chunks()),
        llm=llm,
        settings=_settings(),
    )

    assert [c.id for c in result.citations] == ["c1", "c2"]
    assert result.usage.input_tokens == 100
    assert result.usage.cost_usd == 0.001
    assert result.abstained is False
    assert result.route == "simple"
    assert result.request_id


async def test_abstains_without_evidence_and_skips_llm() -> None:
    llm = FakeLLM("should not be called")

    result = await run_ask("anything", retriever=FakeRetriever([]), llm=llm, settings=_settings())

    assert result.abstained is True
    assert llm.calls == 0
    assert result.usage.cost_usd == 0.0


async def test_canary_leak_withholds_answer() -> None:
    llm = FakeLLM(f"leaking {CANARY_TOKEN} oops")

    result = await run_ask(
        "q", retriever=FakeRetriever(make_chunks()), llm=llm, settings=_settings()
    )

    assert result.abstained is True
    assert CANARY_TOKEN not in result.answer
    assert result.usage.cost_usd == 0.001


def test_pack_context_respects_budget() -> None:
    chunks = [
        Chunk(
            chunk_id=f"c{i}",
            doc_id="d",
            title="T",
            section_path="body",
            text="word " * 60,
            source_url="",
        )
        for i in range(20)
    ]

    context = pack_context(chunks, max_tokens=50)

    labels = extract_labels(context)
    assert labels and max(labels) < 20
