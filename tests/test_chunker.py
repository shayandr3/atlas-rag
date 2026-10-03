from ingestion.chunk import chunk_text


def test_long_text_is_split_with_overlap() -> None:
    text = " ".join(f"Sentence number {i} adds a few words here." for i in range(200))

    chunks = chunk_text(text, max_tokens=50, overlap_pct=15)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 50 * 4 + 100  # one overshooting sentence is tolerated
    assert chunks[1][:25] in chunks[0]  # overlap carries the tail sentences forward


def test_short_text_is_one_chunk() -> None:
    assert chunk_text("Hello world.") == ["Hello world."]


def test_empty_text_is_no_chunks() -> None:
    assert chunk_text("") == []
