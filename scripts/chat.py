#!/usr/bin/env python3
"""Interactive terminal chat with the from-scratch Kiyotaka model.

    python scripts/chat.py --ckpt checkpoints/sft/best.pt
    python scripts/chat.py --ckpt ... --prompt "Who are you?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kiyotaka.generate import generate_reply, load_model  # noqa: E402
from src.kiyotaka.tokenizer import TokenizerWrapper  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "sft" / "best.pt"))
    ap.add_argument("--tokenizer", default=str(ROOT / "data" / "processed" / "tokenizer.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--prompt", default=None, help="one-shot prompt instead of REPL")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--no-rag", action="store_true")
    args = ap.parse_args()

    tok = TokenizerWrapper(args.tokenizer)
    model, cfg = load_model(args.ckpt, device=args.device)
    retriever = None
    index = ROOT / "data" / "processed" / "rag_index.npz"
    if not args.no_rag and index.exists():
        from src.kiyotaka.rag import Retriever
        retriever = Retriever(index, device=args.device)
    print(
        f"Kiyotaka Ayanokoji | {model.num_params():,} params | ctx {cfg.block_size} | "
        f"{args.device}\n(type /reset, /params, /quit)\n",
        file=sys.stderr,
    )

    gen_kwargs = dict(
        max_new_tokens=args.max_new,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stream=not args.no_stream,
    )

    def lookup(text: str):
        """Return (memory_answer, hint_examples)."""
        if retriever is None:
            return None, None
        hits = retriever.search(text, k=1, threshold=0.55)
        if not hits:
            return None, None
        row, sim = hits[0]
        if sim >= 0.72:
            return row["assistant"], None
        return None, [
            {"role": "user", "content": row["user"]},
            {"role": "assistant", "content": row["assistant"]},
        ]

    if args.prompt is not None:
        history = [{"role": "user", "content": args.prompt}]
        memory, examples = lookup(args.prompt)
        print(f"You: {args.prompt}\nKiyotaka: ", end="", flush=True)
        if memory:
            print(memory)
            return 0
        reply = generate_reply(model, tok, history, examples=examples, **gen_kwargs)
        if args.no_stream:
            print(reply)
        return 0

    history: list[dict] = []
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in {"/quit", "/exit"}:
            break
        if user == "/reset":
            history.clear()
            print("[history cleared]")
            continue
        if user == "/params":
            print(
                f"temperature={args.temperature} top_k={args.top_k} top_p={args.top_p} "
                f"repetition_penalty={args.repetition_penalty} max_new={args.max_new}"
            )
            continue

        history.append({"role": "user", "content": user})
        memory, examples = lookup(user)
        if memory:
            print(f"Kiyotaka: {memory}")
            reply = memory
        else:
            print("Kiyotaka: ", end="", flush=True)
            reply = generate_reply(model, tok, history, examples=examples, **gen_kwargs)
            if args.no_stream:
                print(reply)
        history.append({"role": "assistant", "content": reply})
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
