#!/usr/bin/env python3
"""Tokenise the expanded all-season corpus for stage-2 pretraining."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kiyotaka.tokenizer import TokenizerWrapper  # noqa: E402

PROC = ROOT / "data" / "processed"


def encode(text: str, tok: TokenizerWrapper, path: Path) -> int:
    ids = tok.encode(text, add_special_tokens=False) + [tok.end_id]
    np.asarray(ids, dtype=np.uint16).tofile(path)
    return len(ids)


def main() -> int:
    tok = TokenizerWrapper(PROC / "tokenizer.json")
    wiki = (PROC / "domain_extra.txt").read_text(encoding="utf-8")

    train = (PROC / "corpus_v2_train.txt").read_text(encoding="utf-8")
    val = (PROC / "corpus_v2_val.txt").read_text(encoding="utf-8")
    test = (PROC / "corpus_v2_test.txt").read_text(encoding="utf-8")

    chat_parts = []
    sft = PROC / "sft_train.jsonl"
    if sft.exists():
        for line in open(sft, encoding="utf-8"):
            r = json.loads(line)
            chat_parts.append(
                f"<|user|>\n{r['user']}\n<|assistant|>\n{r['assistant']}<|end|>\n"
            )
    chat = "\n".join(chat_parts)

    n_all = encode(wiki + "\n" + train + "\n" + chat, tok, PROC / "domain_v2_all.bin")
    n_train = encode(train, tok, PROC / "domain_v2_train.bin")
    n_val = encode(val, tok, PROC / "domain_v2_val.bin")
    n_test = encode(test, tok, PROC / "domain_v2_test.bin")
    print(f"domain_v2_all : {n_all:,} tokens")
    print(f"domain_v2_train: {n_train:,}")
    print(f"domain_v2_val  : {n_val:,}")
    print(f"domain_v2_test : {n_test:,}")
    print(f"chat examples  : {len(chat_parts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
