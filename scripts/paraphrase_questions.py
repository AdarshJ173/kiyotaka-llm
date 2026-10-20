#!/usr/bin/env python3
"""Paraphrase user turns to improve intent generalisation.

For every curated Q&A pair we ask a local instruct model for several natural
rewrites of the *question*, keeping the original answer. This teaches the small
model that many phrasings map to the same character response.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

SYSTEM = (
    "Rewrite the user's message in 3 different natural ways. Keep the same "
    "intent and tone. Do not add new topics, names, or details. Output only the "
    "three rewrites, one per line, without numbering."
)


def clean(line: str) -> str | None:
    line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"').strip("'")
    if not line or len(line.split()) < 2 or len(line.split()) > 32:
        return None
    if not re.search(r"[?.!…]$", line):
        line += "?"
    return line


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--inputs", nargs="+", default=[
        str(ROOT / "data/manual/persona_qa_v2.jsonl"),
        str(ROOT / "data/manual/persona_qa.jsonl"),
    ])
    ap.add_argument("--output", default=str(ROOT / "data/processed/manual_paraphrases.jsonl"))
    ap.add_argument("--batch-size", type=int, default=24)
    args = ap.parse_args()

    rows = []
    seen = set()
    for p in args.inputs:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            key = r["user"].lower().strip()
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
    print(f"{len(rows)} unique curated prompts")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0}
    ).eval()
    tpl = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "{prompt}"}],
        tokenize=False,
        add_generation_prompt=True,
    )

    out = []
    seen_out = set()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts = [tpl.replace("{prompt}", f'Message: "{r["user"]}"\nRewrites:') for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(0)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=96, do_sample=True, temperature=0.9,
                top_p=0.95, repetition_penalty=1.05,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for r, raw in zip(batch, texts):
            for line in raw.split("\n"):
                q = clean(line)
                if not q:
                    continue
                key = q.lower()
                if key in seen_out or key == r["user"].lower().strip():
                    continue
                seen_out.add(key)
                out.append(
                    {
                        "user": q,
                        "assistant": r["assistant"],
                        "source": "paraphrase",
                        "original": r["user"],
                    }
                )
        if (start // args.batch_size) % 5 == 0:
            print(f"  {start+len(batch)}/{len(rows)} | kept {len(out)}", flush=True)

    random.Random(0).shuffle(out)
    with open(args.output, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} paraphrases -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
