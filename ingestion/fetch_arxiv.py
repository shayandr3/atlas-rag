"""Fetch arXiv papers and extract PDF text into JSONL. Local-only, free (no LLM, spec §7.1)."""

import argparse
import json
from pathlib import Path
from typing import Any

import arxiv
import fitz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="cat:cs.CL OR cat:cs.IR OR cat:cs.LG")
    parser.add_argument("--max-papers", type=int, default=100)
    parser.add_argument("--out", default="data/raw/papers.jsonl")
    parser.add_argument("--pdf-dir", default="data/raw/pdfs")
    return parser.parse_args()


def pdf_to_text(pdf_path: str | Path) -> str:
    with fitz.open(pdf_path) as doc:
        return "\n".join(page.get_text() for page in doc)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pdf_dir = Path(args.pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client()
    search = arxiv.Search(
        query=args.query,
        max_results=args.max_papers,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    written = 0
    skipped = 0
    with out.open("w", encoding="utf-8") as fh:
        for paper in client.results(search):
            try:
                pdf_path = paper.download_pdf(dirpath=str(pdf_dir))
                text = pdf_to_text(pdf_path)
            except Exception as exc:
                print(f"skip {paper.get_short_id()}: {exc}")
                skipped += 1
                continue
            record: dict[str, Any] = {
                "doc_id": paper.get_short_id(),
                "title": paper.title,
                "authors": [str(a) for a in paper.authors],
                "published": str(paper.published.date()) if paper.published else "",
                "categories": paper.categories,
                "abstract": paper.summary,
                "source_url": paper.entry_id,
                "text": text,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    print(f"wrote {written} papers to {out} (skipped {skipped})")


if __name__ == "__main__":
    main()
