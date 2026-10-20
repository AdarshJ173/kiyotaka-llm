#!/usr/bin/env python3
"""Export a checkpoint into a self-contained Hugging Face model repository."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=str(ROOT / "data/processed/tokenizer.json"))
    ap.add_argument("--repo-id", default=None, help="e.g. AAJ007/kiyotaka-ayanokoji-llm")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    state = {k: v.contiguous() for k, v in ckpt["model"].items()}
    # tied embeddings share storage; safetensors forbids sharing, so materialise
    # the output head as an independent tensor (identical values).
    if "lm_head.weight" in state:
        state["lm_head.weight"] = state["lm_head.weight"].clone()
    save_file(state, out / "model.safetensors")

    config = dict(cfg)
    config.update(
        {
            "model_type": "kiyotaka",
            "architectures": ["KiyotakaForCausalLM"],
            "torch_dtype": "float32",
            "num_parameters": sum(v.numel() for k, v in state.items() if k != "lm_head.weight"),
        }
    )
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    shutil.copy(args.tokenizer, out / "tokenizer.json")
    shutil.copy(ROOT / "src" / "kiyotaka" / "model.py", out / "modeling_kiyotaka.py")

    card = f"""---
library_name: pytorch
tags: [character-llm, from-scratch, classroom-of-the-elite]
license: other
---

# Kiyotaka Ayanokoji — 28M character LLM (from scratch)

A **{cfg['n_layer']}-layer, {cfg['n_embd']}-wide decoder-only transformer trained from random
initialisation** on a single 6 GB RTX 4050 laptop GPU. No pretrained weights, no
fine-tuning of an existing model: the architecture, BPE tokenizer, pretraining
corpus and SFT data were built for this project.

* Parameters: **{config['num_parameters']:,}**
* Context: **{cfg['block_size']} tokens** (RoPE, GQA, SwiGLU, tied embeddings)
* Tokenizer: byte-level BPE, vocab **{cfg['vocab_size']}**, trained on this corpus
* Training: TinyStories (general English) → Classroom of the Elite transcripts
  (domain) → character chat SFT

## Usage

```python
import torch
from modeling_kiyotaka import GPT, ModelConfig
from tokenizers import Tokenizer

cfg = ModelConfig(**{{k: v for k, v in __import__("json").load(open("config.json")).items()
                    if k in ModelConfig.__dataclass_fields__}})
model = GPT(cfg)
from safetensors.torch import load_file
model.load_state_dict(load_file("model.safetensors"))
model.eval()

tok = Tokenizer.from_file("tokenizer.json")
prompt = "<|user|>\\nWho are you?\\n<|assistant|>\\n"
ids = tok.encode(prompt, add_special_tokens=False).ids
out = model.generate(torch.tensor([ids]), max_new_tokens=60,
                     eos_token_id=tok.token_to_id("<|end|>"), temperature=0.8, top_p=0.92)
print(tok.decode(out[0, len(ids):].tolist(), skip_special_tokens=True))
```

## Evaluation

Held-out character eval (60 prompts, local Qwen2.5-1.5B judge, 1–5):
fidelity **3.07**, coherence **3.27**; with optional retrieval memory **3.02 / 3.35**.
Single-stream GPU decode: **277 tok/s**; CPU int8: **322 tok/s** at 20 MB.

## Data & licence

Training data was assembled from publicly available fan transcripts, wiki
summaries and short quote pages, plus synthetic *user prompts* (assistant
targets are canon lines). Raw data is not redistributed. The character and
source material belong to their respective rights holders; this is a
non-commercial fan/research artifact. Outputs may reproduce short lines from the
series.
"""
    (out / "README.md").write_text(card, encoding="utf-8")
    print(f"wrote {out} ({sum(v.numel() for v in state.values())/1e6:.1f}M params)")

    if args.repo_id:
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("set HF_TOKEN to upload")
        api = HfApi(token=token)
        api.create_repo(args.repo_id, exist_ok=True, private=args.private)
        api.upload_folder(folder_path=str(out), repo_id=args.repo_id)
        print(f"uploaded to https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
