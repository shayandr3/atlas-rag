"""Prepend situating context to chunks before embedding (spec §7.3).

M1 implements --mode cheap only (no LLM cost). The `section` and `full` modes need the
cheap-LLM key, --dry-run cost estimation and --max-usd enforcement; they arrive once the
human has created the LLM key (spec §2.2 #1).
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="infile", default="data/chunks/chunks.jsonl")
    parser.add_argument("--out", default="data/chunks/chunks_ctx.jsonl")
    parser.add_argument("--mode", choices=["cheap", "section", "full"], default="cheap")
    args = parser.parse_args()

    if args.mode != "cheap":
        print(f"mode {args.mode!r} needs the cheap-LLM key and cost caps; not implemented yet")
        sys.exit(2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for line in Path(args.infile).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            record["context_prefix"] = f"{record['title']} › {record['section_path']}"
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    print(f"contextualized {n} chunks (mode=cheap) -> {out}")


if __name__ == "__main__":
    main()
