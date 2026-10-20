#!/usr/bin/env python3
"""Build the offline retrieval index from the SFT training pairs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kiyotaka.rag import MODEL, mean_pool  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(ROOT / "data/processed/sft_train.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/processed/rag_index.npz"))
    args = ap.parse_args()

    rows = []
    seen = set()
    for line in open(args.pairs, encoding="utf-8"):
        r = json.loads(line)
        key = re.sub(r"\s+", " ", r["user"].lower().strip())
        if key in seen:
            continue
        seen.add(key)
        rows.append({"user": r["user"], "assistant": r["assistant"]})
    print(f"{len(rows)} unique exchanges")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            batch = [r["user"] for r in rows[i : i + 64]]
            enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state
            out.append(mean_pool(hidden, enc["attention_mask"]).cpu().numpy())
    emb = np.concatenate(out, axis=0).astype(np.float32)
    np.savez_compressed(args.out, emb=emb, rows=json.dumps(rows))
    print(f"wrote {emb.shape} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
