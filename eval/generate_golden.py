"""Generate the golden QA set from the ingested corpus (spec §15.1).

Batches many questions per LLM call and paces calls to respect free-tier rate limits
(ADR 0003). Writes a DRAFT for human review: eval/golden/golden_draft.jsonl.
Every question's gold ids are resolved from the sources shown to the model, so the
ground truth comes from the corpus, not from the model's memory.

Usage:
    python eval/generate_golden.py --batches 12 --pace 90 --max-usd 1.0
"""

import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import Any

from atlas.config import get_settings
from atlas.llm.adapter import build_llm
from atlas.llm.pricing import PricingTable

CONTENT_PROMPT = """\
You write evaluation questions for a RAG system over an arXiv corpus about LLMs,
retrieval and machine learning.

Below are 3 sources. Write exactly 8 questions as a strict JSON array (no markdown fences):
[
  {{"question": "...", "type": "single_hop", "uses": "S1", "gold_answer": "one short sentence"}},
  ... 4 single_hop questions (each grounded in exactly ONE source, named via "uses"),
  ... 2 multi_hop questions ("uses": "S1+S3" - answering must require linking a fact
      from each named source),
  ... 2 comparative questions ("uses": "S2+S3" - comparing or aggregating across sources)
]
Rules: answerable ONLY from the given sources; never mention "source", "S1" or "document"
in question text; vary the phrasing and difficulty.

Sources:
{sources}"""

UNANSWERABLE_PROMPT = """\
An evaluation corpus contains only arXiv papers about LLMs, retrieval and ML.

Write {n} realistic questions a user might ask such a system that the corpus CANNOT answer
(topics like history, medicine, law, sports, cooking, travel, personal advice, recent events).
Strict JSON array of strings only, no markdown:
["question one", "question two", ...]"""

PARAPHRASE_PROMPT = """\
Write {n} pairs of questions with identical meaning but different wording
(for tuning a semantic cache). Topics: RAG, dense retrieval, reranking, LLM agents.
Strict JSON array, no markdown:
[{{"a": "...", "b": "..."}}, ...]"""


def open_out(path: Path) -> Any:
    """Sync helper: ASYNC230 forbids blocking open() inside the async body."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def load_best_chunk_per_doc(path: str) -> dict[str, dict[str, Any]]:
    """One representative (longest mid-index) chunk per document, for question seeding."""
    best: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec: dict[str, Any] = json.loads(line)
        doc = str(rec["doc_id"])
        cur = best.get(doc)
        if cur is None or int(rec.get("token_count", 0)) > int(cur.get("token_count", 0)):
            best[doc] = rec
    return best


def parse_json_array(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    start = min([i for i in (cleaned.find("["), cleaned.find("{")) if i != -1], default=-1)
    if start == -1:
        raise ValueError("no JSON found")
    candidate = cleaned[start:]
    if candidate.endswith("]"):
        return json.loads(candidate)
    # trim trailing prose after the last closing bracket/brace
    for end_char in ("]", "}", ")"):
        idx = candidate.rfind(end_char)
        if idx != -1:
            return json.loads(candidate[: idx + 1])
    return json.loads(candidate)


async def ask_json(
    llm: Any, prompt: str, model: str, max_usd: float, state: dict[str, float]
) -> Any:
    if state["cost"] > max_usd:
        raise SystemExit(f"--max-usd {max_usd} reached; stopping (spent ${state['cost']:.4f})")
    last_err: Exception | None = None
    for attempt in range(2):
        resp = await llm.complete(
            system="You output only valid JSON.",
            user=prompt
            if attempt == 0
            else prompt + "\n\nYour previous reply was not valid JSON. Output ONLY the JSON.",
            model=model,
            max_tokens=2000,
        )
        state["cost"] += resp.cost_usd
        try:
            return parse_json_array(resp.text)
        except (ValueError, json.JSONDecodeError) as exc:
            last_err = exc
    raise ValueError(f"model output not parseable as JSON: {last_err}")


def resolve_gold(uses: str, sources: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    doc_ids: list[str] = []
    chunk_ids: list[str] = []
    for token in uses.replace(" ", "").split("+"):
        idx = int(token[1:]) - 1
        src = sources[idx]
        doc_ids.append(str(src["doc_id"]))
        chunk_ids.append(str(src["chunk_id"]))
    return doc_ids, chunk_ids


async def main_async(args: argparse.Namespace) -> None:
    settings = get_settings()
    pricing = PricingTable.load()
    llm = build_llm(settings, pricing)
    state = {"cost": 0.0}
    model = settings.llm_cheap_model
    out = Path(args.out)

    docs = load_best_chunk_per_doc(args.chunks)
    doc_ids = sorted(docs)
    random.Random(7).shuffle(doc_ids)
    print(f"{len(doc_ids)} docs available; generating from {args.batches * 3} of them")

    written = 0
    with open_out(out) as fh:

        def emit(item: dict[str, Any]) -> None:
            nonlocal written
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1

        batches = [doc_ids[i : i + 3] for i in range(0, len(doc_ids) - 2, 3)][: args.batches]
        for bi, batch in enumerate(batches, start=1):
            sources = []
            for si, doc_id in enumerate(batch, start=1):
                rec = docs[doc_id]
                sources.append(
                    {
                        "label": f"S{si}",
                        "doc_id": doc_id,
                        "chunk_id": rec["chunk_id"],
                        "text": str(rec["text"])[:1400],
                    }
                )
            src_text = "\n\n".join(
                f"[{s['label']}] doc_id={s['doc_id']} chunk_id={s['chunk_id']}\n{s['text']}"
                for s in sources
            )
            try:
                items = await ask_json(
                    llm, CONTENT_PROMPT.format(n=8, sources=src_text), model, args.max_usd, state
                )
            except ValueError as exc:
                print(f"batch {bi}: skipped ({exc})")
                continue
            for item in items:
                if not isinstance(item, dict) or item.get("type") not in {
                    "single_hop",
                    "multi_hop",
                    "comparative",
                }:
                    continue
                try:
                    gold_docs, gold_chunks = resolve_gold(str(item.get("uses", "")), sources)
                except (ValueError, IndexError):
                    continue
                emit(
                    {
                        "question": str(item.get("question", "")).strip(),
                        "gold_answer": str(item.get("gold_answer", "")),
                        "gold_doc_ids": gold_docs,
                        "gold_chunk_ids": gold_chunks,
                        "type": item["type"],
                    }
                )
            print(
                f"batch {bi}/{len(batches)}: ok (total {written} questions, ${state['cost']:.4f})"
            )
            await asyncio.sleep(args.pace)

        unans = await ask_json(llm, UNANSWERABLE_PROMPT.format(n=15), model, args.max_usd, state)
        for q in unans[:15]:
            if isinstance(q, str) and q.strip():
                emit(
                    {
                        "question": q.strip(),
                        "gold_answer": "",
                        "gold_doc_ids": [],
                        "gold_chunk_ids": [],
                        "type": "unanswerable",
                    }
                )
        print(f"unanswerable done (total {written}, ${state['cost']:.4f})")
        await asyncio.sleep(args.pace)

        pairs = await ask_json(llm, PARAPHRASE_PROMPT.format(n=10), model, args.max_usd, state)
        for gi, pair in enumerate(pairs[:10]):
            if isinstance(pair, dict) and pair.get("a") and pair.get("b"):
                for variant in ("a", "b"):
                    emit(
                        {
                            "question": str(pair[variant]).strip(),
                            "gold_answer": "",
                            "gold_doc_ids": [],
                            "gold_chunk_ids": [],
                            "type": "paraphrase",
                            "group": f"p{gi:02d}",
                        }
                    )
        print(f"paraphrases done (total {written}, ${state['cost']:.4f})")

    print(f"WROTE {written} questions -> {out} (total LLM cost ${state['cost']:.4f})")
    print("HUMAN REVIEW REQUIRED before renaming to golden.jsonl (spec §15.1)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", default="data/chunks/chunks_ctx.jsonl")
    parser.add_argument("--out", default="eval/golden/golden_draft.jsonl")
    parser.add_argument(
        "--batches", type=int, default=12, help="content batches (3 sources, 8 questions each)"
    )
    parser.add_argument(
        "--pace", type=float, default=90.0, help="seconds between LLM calls (free-tier pacing)"
    )
    parser.add_argument("--max-usd", type=float, default=1.0)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
