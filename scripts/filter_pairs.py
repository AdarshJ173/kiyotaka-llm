#!/usr/bin/env python3
"""Filter synthetic Q&A pairs by semantic relevance.

The generator can attach a canon line to a question it does not actually
answer. We embed question and answer with a small sentence-transformer and keep
pairs above a cosine threshold, which removes most mismatches without any
hand-labelling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@torch.no_grad()
def embed(texts: list[str], tok, model, device, batch_size=64) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        out.append(pooled.cpu().numpy())
    return np.concatenate(out, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--suffix", default=".filtered.jsonl")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()
    print(f"embedder on {device}")

    for path_str in args.inputs:
        path = Path(path_str)
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        if not rows:
            continue
        q = embed([r["user"] for r in rows], tok, model, device)
        a = embed([r["assistant"] for r in rows], tok, model, device)
        sims = (q * a).sum(-1)
        keep = sims >= args.threshold
        out = path.with_suffix("").with_suffix("")
        out = out.parent / (out.name + args.suffix)
        with open(out, "w", encoding="utf-8") as fh:
            for r, s, k in zip(rows, sims, keep):
                if k:
                    r["relevance"] = float(s)
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(
            f"{path.name}: {len(rows)} -> {int(keep.sum())} kept "
            f"(sim mean {sims.mean():.3f}, p50 {np.percentile(sims,50):.3f}, "
            f"p90 {np.percentile(sims,90):.3f}) -> {out}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
