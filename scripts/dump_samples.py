#!/usr/bin/env python3
"""Write a markdown sample sheet from the deployed hybrid (model + memory)."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kiyotaka.generate import generate_reply, load_model  # noqa: E402
from src.kiyotaka.rag import Retriever  # noqa: E402
from src.kiyotaka.tokenizer import TokenizerWrapper  # noqa: E402

PROMPTS = [
    "Who are you?", "What is your name?", "What class are you in?", "Describe yourself.",
    "What do you think about friendship?", "What is trust?", "Do you trust Horikita?",
    "What do you think of Kushida?", "Who is Ichinose?", "What do you think of Ryuen?",
    "What is your goal?", "Why do you hide your ability?", "Are all people equal?",
    "Should I trust my friends?", "How do I get stronger?", "I'm sad.", "Tell me a joke.",
    "Good morning.", "What do you do for fun?", "What is the smartest thing you've done?",
]

def lookup(retriever, text):
    if retriever is None:
        return None, None
    hits = retriever.search(text, k=1, threshold=0.55)
    if not hits:
        return None, None
    row, sim = hits[0]
    if sim >= 0.72:
        return row["assistant"], sim
    return None, None

def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = TokenizerWrapper(ROOT / "data/processed/tokenizer.json")
    model, _ = load_model(ROOT / "checkpoints/sft_56m/best.pt", device=device)
    index = ROOT / "data/processed/rag_index.npz"
    retriever = Retriever(index, device=device) if index.exists() else None
    lines = ["# Sample conversations", "", "Model: `checkpoints/sft_56m/best.pt` (51.95M params). "
             "`memory` means an exact-ish curated intent; `model` means generated.", ""]
    for p in PROMPTS:
        mem, sim = lookup(retriever, p)
        if mem:
            reply, mode = mem, "memory"
        else:
            reply = generate_reply(
                model, tok, [{"role": "user", "content": p}],
                max_new_tokens=90, temperature=0.7, top_p=0.92, repetition_penalty=1.15,
            )
            mode = "model"
        lines += [f"**You:** {p}", "", f"**Kiyotaka:** {reply}", "", f"_{mode}_", ""]
    out = ROOT / "reports/samples.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
