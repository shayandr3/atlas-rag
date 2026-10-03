"""Chunk papers into ~max-token overlapping chunks (spec §7.2).

M1 uses sentence windows over the PDF text; section-heading awareness is refined in a
later increment (PDF extraction loses headings — worth an ADR when improved).
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_CHARS_PER_TOKEN = 4


def split_sentences(text: str) -> list[str]:
    flat = " ".join(text.split())
    return [s for s in _SENT_RE.split(flat) if s]


def chunk_text(text: str, max_tokens: int = 300, overlap_pct: int = 15) -> list[str]:
    max_chars = max_tokens * _CHARS_PER_TOKEN
    overlap_chars = int(max_chars * overlap_pct / 100)
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in split_sentences(text):
        if current and size + len(sentence) > max_chars:
            chunks.append(" ".join(current))
            tail: list[str] = []
            tail_size = 0
            if overlap_chars > 0:
                # always carry at least the last sentence forward so consecutive
                # chunks genuinely overlap, even when one sentence > overlap budget
                for idx, sent in enumerate(reversed(current)):
                    if idx > 0 and tail_size + len(sent) > overlap_chars:
                        break
                    tail.insert(0, sent)
                    tail_size += len(sent) + 1
            current, size = tail, tail_size
        current.append(sentence)
        size += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="infile", default="data/raw/papers.jsonl")
    parser.add_argument("--out", default="data/chunks/chunks.jsonl")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--overlap-pct", type=int, default=15)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total_chars = 0
    n_chunks = 0
    with out.open("w", encoding="utf-8") as fh:
        for line in Path(args.infile).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record: dict[str, Any] = json.loads(line)
            for idx, chunk in enumerate(
                chunk_text(record["text"], args.max_tokens, args.overlap_pct)
            ):
                payload = {
                    "chunk_id": f"{record['doc_id']}::c{idx:04d}",
                    "doc_id": record["doc_id"],
                    "title": record["title"],
                    "section_path": "body",
                    "chunk_idx": idx,
                    "text": chunk,
                    "context_prefix": "",
                    "source_url": record["source_url"],
                    "published_date": record["published"],
                    "token_count": len(chunk) // _CHARS_PER_TOKEN,
                }
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                n_chunks += 1
                total_chars += len(chunk)
    print(f"wrote {n_chunks} chunks to {out} (~{total_chars // 1024} KB text)")


if __name__ == "__main__":
    main()
