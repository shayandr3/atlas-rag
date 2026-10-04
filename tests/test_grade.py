from tests.fakes import FakeLLM

from atlas.config import Settings
from atlas.pipeline.grade import grade_chunks, rewrite_query, should_skip_grading
from atlas.retrieval.chunks import Chunk


def _settings() -> Settings:
    return Settings(_env_file=None)


def _chunk(score: float = 0.5) -> Chunk:
    return Chunk(
        chunk_id="c1",
        doc_id="d1",
        title="T",
        section_path="body",
        text="Some evidence text.",
        source_url="",
        score=score,
    )


async def test_grade_parses_verdicts() -> None:
    llm = FakeLLM('[{"id": "S1", "verdict": "relevant"}]')

    outcome = await grade_chunks("q", [_chunk()], llm, _settings())

    assert outcome.enough is True
    assert outcome.relevant_count == 1


async def test_grade_garbage_json_defaults_to_proceed() -> None:
    llm = FakeLLM("I cannot do that")

    outcome = await grade_chunks("q", [_chunk()], llm, _settings())

    assert outcome.enough is True
    assert outcome.relevant_count == -1


async def test_grade_insufficient_feedback() -> None:
    llm = FakeLLM('[{"id": "S1", "verdict": "irrelevant"}]')

    outcome = await grade_chunks("q", [_chunk()], llm, _settings())

    assert outcome.enough is False
    assert "S1" in outcome.feedback


async def test_rewrite_falls_back_to_original() -> None:
    llm = FakeLLM("no json here")

    rewritten = await rewrite_query("original q", "nothing relevant", llm, _settings())

    assert rewritten == "original q"


def test_skip_threshold() -> None:
    assert should_skip_grading(0.99, _settings()) is True
    assert should_skip_grading(0.5, _settings()) is False
