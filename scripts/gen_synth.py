#!/usr/bin/env python3
"""Generate diverse user prompts for canon Ayanokoji replies with a local
instruct model (Qwen2.5-1.5B-Instruct). Only the *user* side is synthetic;
every assistant target stays an unmodified canon line, so the character's
voice is never polluted by the generator.

Fully local: no API keys, no network at generation time.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

SYSTEM = (
    "You write user messages for a roleplay chatbot. Given one canon reply by "
    "Kiyotaka Ayanokoji, write ONE natural question (5-25 words) from a "
    "classmate that this reply answers. Do not mention names, quotes, scenes, "
    "episodes, anime, or roleplay. Output only the question.\n"
    "Example:\nCanon reply: \"People are tools. I don't need friends.\"\n"
    "Question: Why do you keep everyone at a distance?"
)

BAD = re.compile(
    r"ayanok|ayanoko|quote|anime|episode|scene|screenplay|transcript|classroom of the elite|"
    r"character|roleplay|role-play|chatbot|canon|reply|message|user",
    re.I,
)


def context_text(row: dict) -> str:
    lines = []
    for c in row["context"][-2:]:
        tag = " (inner thought)" if c["monologue"] else ""
        lines.append(f"{c['speaker']}{tag}: {c['text']}")
    scene = f"Scene {row['scene']}." if row["scene"] else ""
    return scene + " " + " ".join(lines)


def clean_output(text: str) -> str | None:
    text = text.strip().split("\n")[0].strip().strip('"').strip("'").strip()
    text = re.sub(r"^(User|Message|Question)\s*:\s*", "", text, flags=re.I)
    if not text or BAD.search(text):
        return None
    words = text.split()
    if not (4 <= len(words) <= 30):
        return None
    if not text.endswith("?"):
        return None
    if not text[0].isupper():
        return None
    if re.search(r"[*\[\]{}<>|\]\[]|\?\?|!!", text):
        return None
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--input", default=str(PROC / "kiyo_lines.jsonl"))
    ap.add_argument("--output", default=str(PROC / "synth_questions.jsonl"))
    ap.add_argument("--per-line", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=48)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    # Weight monologues a bit higher: they carry his worldview.
    jobs: list[dict] = []
    for r in rows:
        if len(r["text"].split()) < 2:
            continue
        n = args.per_line + (1 if r["monologue"] else 0)
        jobs.extend([r] * n)
    random.Random(0).shuffle(jobs)
    print(f"{len(rows)} canon lines -> {len(jobs)} prompt jobs")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0}
    ).eval()
    print(f"generator on {next(model.parameters()).device}")

    out: list[dict] = []
    seen: set[str] = set()
    tpl = tok.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "{prompt}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start : start + args.batch_size]
        prompts = []
        for r in batch:
            p = (
                f"Canon reply: \"{r['text']}\"\n"
                f"Context: {context_text(r)}\n"
                f"Write the user's message:"
            )
            prompts.append(tpl.replace("{prompt}", p))
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(0)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                repetition_penalty=1.05,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for r, raw in zip(batch, texts):
            q = clean_output(raw)
            if not q:
                continue
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "episode": r["episode"],
                    "scene": r["scene"],
                    "source_text": r["text"],
                    "monologue": r["monologue"],
                    "user": q,
                    "assistant": r["text"],
                    "source": "synth_prompt",
                }
            )
        if (start // args.batch_size) % 10 == 0:
            print(f"  {start + len(batch)}/{len(jobs)} jobs, {len(out)} kept", flush=True)

    with open(args.output, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} pairs -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
