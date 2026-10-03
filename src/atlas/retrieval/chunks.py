"""A normalized chunk as returned by a retriever."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section_path: str
    text: str
    source_url: str
    score: float = 0.0
