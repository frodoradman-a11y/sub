"""Build the character vocabulary for the CTC/TDT heads (and the tiny test LLM).

    python -m subasr.tools.build_vocab --manifest train.jsonl --output vocab.txt --min-count 2
"""

from __future__ import annotations

import argparse

from subasr.data import read_manifest
from subasr.text.tokenizer import LLM_SPECIALS, CharTokenizer


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--llm", action="store_true", help="add the special tokens of the tiny LLM")
    args = ap.parse_args(argv)
    texts = [item["text"] for m in args.manifest for item in read_manifest(m)]
    kwargs = {"specials": LLM_SPECIALS} if args.llm else {}
    tok = CharTokenizer.build(texts, args.min_count, **kwargs)
    tok.save(args.output)
    print(f"{len(tok)} tokens -> {args.output}")


if __name__ == "__main__":
    main()
