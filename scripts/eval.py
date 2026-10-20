#!/usr/bin/env python3
"""Evaluate the character model: held-out perplexity, style metrics, optional
local LLM judge, and base-vs-SFT ablation.

All scoring is local. The judge is Qwen2.5-1.5B-Instruct used only as a rater,
never as a source of training targets.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kiyotaka.generate import generate_reply, load_model  # noqa: E402
from src.kiyotaka.tokenizer import TokenizerWrapper  # noqa: E402
from scripts.train import TokenLoader  # noqa: E402

FIRST_PERSON = re.compile(r"\b(i|me|my|mine|myself)\b", re.I)
HEDGE = re.compile(r"\b(maybe|perhaps|probably|possibly|i think|might|could)\b", re.I)
CERTAIN = re.compile(r"\b(is|are|will|must|can't|won't|don't|never|always)\b", re.I)
EMOTION = re.compile(
    r"\b(angry|sad|happy|afraid|scared|love|hate|cry|joy|fear|anxious|excited|lonely)\b", re.I
)


def word_stats(texts: list[str]) -> dict:
    lens = [len(t.split()) for t in texts]
    sents = [len(re.findall(r"[.!?]+", t)) + 1 for t in texts]
    words = re.findall(r"[a-z']+", " ".join(texts).lower())
    counts = Counter(words)
    total = max(1, len(words))

    def rate(rx):
        return sum(len(rx.findall(t)) for t in texts) / max(1, len(texts))

    def distinct(n: int):
        grams = []
        for t in texts:
            w = re.findall(r"[a-z']+", t.lower())
            grams += [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]
        return len(set(grams)) / max(1, len(grams))

    rep = 0
    rep_total = 0
    for t in texts:
        w = re.findall(r"[a-z']+", t.lower())
        grams = [tuple(w[i : i + 3]) for i in range(len(w) - 2)]
        rep_total += len(grams)
        rep += len(grams) - len(set(grams))

    return {
        "n": len(texts),
        "mean_words": float(np.mean(lens)) if lens else 0.0,
        "median_words": float(np.median(lens)) if lens else 0.0,
        "p90_words": float(np.percentile(lens, 90)) if lens else 0.0,
        "mean_sentences": float(np.mean(sents)) if sents else 0.0,
        "question_rate": sum(t.strip().endswith("?") for t in texts) / max(1, len(texts)),
        "exclamation_rate": sum("!" in t for t in texts) / max(1, len(texts)),
        "first_person_per_reply": rate(FIRST_PERSON),
        "hedge_per_reply": rate(HEDGE),
        "certain_per_reply": rate(CERTAIN),
        "emotion_per_reply": rate(EMOTION),
        "distinct_1": distinct(1),
        "distinct_2": distinct(2),
        "trigram_repeat_rate": rep / max(1, rep_total),
        "vocab_size": len(counts),
        "term_char_rate": sum(bool(re.search(r"[.!?…][\"']?$", t.strip())) for t in texts)
        / max(1, len(texts)),
    }


@torch.no_grad()
def perplexity(model, loader, iters, device) -> dict:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = loader.batch(device)
        with torch.autocast("cuda", dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
            _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
    mean = float(np.mean(losses))
    return {"loss": mean, "ppl": math.exp(mean)}


class Judge:
    def __init__(self, model_path: str, device: torch.device):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={"": 0}
        ).eval()
        self.device = device

    @torch.no_grad()
    def score(self, reply: str) -> tuple[float, float] | None:
        prompt = (
            "You are grading a chatbot that role-plays Kiyotaka Ayanokoji from Classroom of "
            "the Elite. His voice is calm, blunt, analytical, emotionally restrained and terse. "
            "He rarely exclaims, rarely rambles, and treats social life as strategy.\n\n"
            f"Reply to grade:\n\"{reply}\"\n\n"
            "Give two integers 1-5 separated by a comma: character fidelity, coherence. "
            "Output only the two numbers."
        )
        msgs = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **enc, max_new_tokens=12, do_sample=False, pad_token_id=self.tok.eos_token_id
        )
        raw = self.tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        nums = re.findall(r"[1-5]", raw)
        if len(nums) < 2:
            return None
        return float(nums[0]), float(nums[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base-ckpt", default=None, help="pretrain-only checkpoint for ablation")
    ap.add_argument("--tokenizer", default=str(ROOT / "data/processed/tokenizer.json"))
    ap.add_argument("--test-pairs", default=str(ROOT / "data/processed/sft_test.jsonl"))
    ap.add_argument("--test-tokens", default=str(ROOT / "data/processed/cote_test.bin"))
    ap.add_argument("--n-samples", type=int, default=60)
    ap.add_argument("--judge", default=None, help="local judge model id/path")
    ap.add_argument("--rag-index", default=None, help="optional retrieval index for a sft+rag system")
    ap.add_argument("--out", default=str(ROOT / "reports" / "eval.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = TokenizerWrapper(args.tokenizer)

    pairs = [json.loads(l) for l in open(args.test_pairs, encoding="utf-8")]
    random.shuffle(pairs)
    pairs = pairs[: args.n_samples]
    prompts = [p["user"] for p in pairs]
    refs = [p["assistant"] for p in pairs]

    results: dict = {"n_prompts": len(prompts), "checkpoint": args.ckpt}
    systems = {"sft": args.ckpt}
    if args.base_ckpt:
        systems["base"] = args.base_ckpt
    if args.rag_index:
        systems["sft_rag"] = args.ckpt

    judge = Judge(args.judge, device) if args.judge else None
    retriever = None
    if args.rag_index:
        from src.kiyotaka.rag import Retriever
        retriever = Retriever(args.rag_index, device=str(device))
    for name, ckpt in systems.items():
        model, cfg = load_model(ckpt, device=device)
        ppl = perplexity(model, TokenLoader(args.test_tokens, cfg.block_size, 16, seed=1), 25, device)
        replies = []
        for p in prompts:
            torch.manual_seed(args.seed)
            examples = None
            memory = None
            if name == "sft_rag" and retriever is not None:
                hits = retriever.search(p, k=1, threshold=0.60)
                if hits:
                    row, sim = hits[0]
                    if sim >= 0.72:
                        memory = row["assistant"]
                    else:
                        examples = [
                            {"role": "user", "content": row["user"]},
                            {"role": "assistant", "content": row["assistant"]},
                        ]
            if memory:
                replies.append(memory)
            else:
                replies.append(
                    generate_reply(
                        model, tok, [{"role": "user", "content": p}],
                        max_new_tokens=96, temperature=0.7, top_p=0.9,
                        repetition_penalty=1.15, examples=examples,
                    )
                )
        stats = word_stats(replies)
        entry = {"perplexity_test": ppl, "style": stats, "samples": [
            {"prompt": p, "reply": r, "reference": ref} for p, r, ref in zip(prompts[:15], replies[:15], refs[:15])
        ]}
        if judge:
            scores = [s for r in replies if (s := judge.score(r))]
            entry["judge"] = {
                "n": len(scores),
                "fidelity": float(np.mean([s[0] for s in scores])) if scores else 0.0,
                "coherence": float(np.mean([s[1] for s in scores])) if scores else 0.0,
            }
        results[name] = entry
        print(f"{name}: ppl {ppl['ppl']:.2f} | words {stats['mean_words']:.1f} | "
              f"distinct2 {stats['distinct_2']:.3f}" + (f" | judge {entry.get('judge')}" if judge else ""))

    results["reference_style"] = word_stats(refs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
