from tests.fakes import FakeLLM

from atlas.config import Settings
from atlas.pipeline.multihop import SubQuestion, _with_dependencies, decompose


def _settings() -> Settings:
    return Settings(_env_file=None)


async def test_decompose_parses_subquestions() -> None:
    llm = FakeLLM(
        '[{"id": "Q1", "question": "first", "depends_on": []},'
        ' {"id": "Q2", "question": "second", "depends_on": ["Q1"]}]'
    )

    subs = await decompose("compare A and B", llm, _settings())

    assert [s.id for s in subs] == ["Q1", "Q2"]
    assert subs[1].depends_on == ["Q1"]


async def test_decompose_garbage_falls_back_to_single() -> None:
    llm = FakeLLM("cannot comply")

    subs = await decompose("compare A and B", llm, _settings())

    assert len(subs) == 1
    assert subs[0].question == "compare A and B"


def test_dependencies_are_substituted() -> None:
    sq = SubQuestion(id="Q2", question="What changed?", depends_on=["Q1"])
    text = _with_dependencies(sq, {"Q1": "earlier finding"})

    assert "earlier finding" in text
    assert "What changed?" in text


def test_decompose_caps_hops() -> None:
    llm = FakeLLM(
        '[{"id": "Q1", "question": "a", "depends_on": []},'
        ' {"id": "Q2", "question": "b", "depends_on": []},'
        ' {"id": "Q3", "question": "c", "depends_on": []},'
        ' {"id": "Q4", "question": "d", "depends_on": []}]'
    )

    import asyncio

    subs = asyncio.run(decompose("q", llm, _settings()))
    assert len(subs) <= 3  # MAX_HOPS default
