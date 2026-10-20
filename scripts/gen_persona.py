#!/usr/bin/env python3
"""Generate *persona-focused* user turns for canon replies.

Unlike gen_synth.py (which uses scene context and yields plot-specific
questions), this generator sees only the reply and asks for a general question
or statement about school, people, ambition, trust or life. Outputs are
filtered for meta references and character names.
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
PROC = ROOT / "data" / "processed"

SYSTEM = (
    "You write user messages for a roleplay chatbot whose character is a calm, "
    "blunt, analytical high-school student. Given one of his lines, write ONE "
    "general message from a classmate that the line could answer or follow. "
    "Keep it 5-25 words. It must be about school, people, friendship, ambition, "
    "trust, fear, winning, or life in general. Never mention names, plot, "
    "episodes, anime, quotes, or roleplay. Output only the message.\n"
    "Examples:\n"
    "Line: \"People are tools. I don't need friends.\"\n"
    "Message: Do you really think friendship is just another transaction?\n"
    "Line: \"Winning is the only thing that can't be argued with.\"\n"
    "Message: I think how you win matters more than winning."
)

NAMES = re.compile(
    r"horikita|kushida|ichinose|karuizawa|ryuen|ryūen|sakayanagi|hirata|sudo|sudō|"
    r"ike|yamauchi|sakura|koenji|kōenji|chabashira|ayanok|kiyotaka|class d|"
    r"class a|class b|class c",
    re.I,
)
BAD = re.compile(
    r"quote|anime|episode|scene|screenplay|transcript|classroom of the elite|"
    r"character|roleplay|role-play|chatbot|canon|reply|message|user|prompt",
    re.I,
)


def clean(text: str, want_question: bool) -> str | None:
    text = text.strip().split("\n")[0].strip().strip('"').strip("'").strip()
    text = re.sub(r"^(User|Message|Question|Statement)\s*:\s*", "", text, flags=re.I)
    if not text or BAD.search(text) or NAMES.search(text):
        return None
    if not (4 <= len(text.split()) <= 28):
        return None
    if not text[0].isupper():
        return None
    if re.search(r"[*\[\]{}<>|]|\?\?|!!", text):
        return None
    if want_question:
        if not text.endswith("?"):
            return None
    else:
        if text.endswith("?"):
            return None
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--input", default=str(PROC / "kiyo_lines_v2.jsonl"))
    ap.add_argument("--output", default=str(PROC / "synth_persona_v2.jsonl"))
    ap.add_argument("--per-line", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=40)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    jobs: list[tuple[dict, bool]] = []
    for r in rows:
        if len(r["text"].split()) < 4:
            continue
        for k in range(args.per_line):
            jobs.append((r, k % 2 == 0))
    random.Random(0).shuffle(jobs)
    print(f"{len(rows)} lines -> {len(jobs)} jobs")

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
    out: list[dict] = []
    seen: set[str] = set()
    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start : start + args.batch_size]
        prompts = [
            tpl.replace("{prompt}", f'Line: "{r["text"]}"\nWrite the message:')
            for r, _ in batch
        ]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=384).to(0)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=args.max_new, do_sample=True, temperature=1.0,
                top_p=0.95, repetition_penalty=1.05,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for (r, want_q), raw in zip(batch, texts):
            msg = clean(raw, want_q)
            if not msg:
                continue
            key = msg.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "season": r.get("season", 1),
                    "episode": r["episode"],
                    "scene": r.get("scene", ""),
                    "source_text": r["text"],
                    "monologue": r["monologue"],
                    "user": msg,
                    "assistant": r["text"],
                    "source": "synth_persona",
                }
            )
        if (start // args.batch_size) % 10 == 0:
            print(f"  {start + len(batch)}/{len(jobs)} | kept {len(out)}", flush=True)

    with open(args.output, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} pairs -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
