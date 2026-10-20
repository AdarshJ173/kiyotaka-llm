#!/usr/bin/env python3
"""Append chat-formatted SFT text to the stage-2 domain corpus.

The template tokens (<|user|>, <|assistant|>, <|end|>) never occur in raw
dialogue, so the model must see them once before SFT. This makes stage 2 a
warm start for the chat format without changing any assistant target.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kiyotaka.tokenizer import TokenizerWrapper  # noqa: E402

PROC = ROOT / "data" / "processed"


def main() -> int:
    tok = TokenizerWrapper(PROC / "tokenizer.json")
    domain = (PROC / "domain_extra.txt").read_text(encoding="utf-8")
    corpus = (PROC / "cote_corpus.txt").read_text(encoding="utf-8")
    parts = [domain, corpus]
    n = 0
    path = PROC / "sft_train.jsonl"
    if path.exists():
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            parts.append(
                f"<|user|>\n{r['user']}\n<|assistant|>\n{r['assistant']}<|end|>\n"
            )
            n += 1
    text = "\n".join(parts)
    ids = tok.encode(text, add_special_tokens=False) + [tok.end_id]
    arr = np.asarray(ids, dtype=np.uint16)
    arr.tofile(PROC / "domain_all.bin")
    print(f"domain_all.bin: {len(ids):,} tokens ({n} chat examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
