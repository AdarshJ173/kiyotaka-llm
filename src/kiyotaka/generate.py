"""Inference helpers shared by the CLI, eval and benchmark scripts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

import torch

from .data import ASSISTANT, END, PERSONA, USER
from .model import GPT, ModelConfig
from .tokenizer import TokenizerWrapper


def load_model(ckpt_path: str | Path, device: str = "cuda", dropout: float = 0.0):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt["config"])
    cfg_dict["dropout"] = dropout
    cfg = ModelConfig(**cfg_dict)
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def build_prompt(history: list[dict]) -> str:
    """history: [{role: user|assistant, content: str}, ...]"""
    parts = [PERSONA]
    for msg in history:
        if msg["role"] == "user":
            parts.append(f"{USER}\n{msg['content'].strip()}\n{ASSISTANT}\n")
        else:
            parts.append(f"{msg['content'].strip()}{END}\n")
    if history and history[-1]["role"] == "user":
        parts.append("")  # assistant tag already emitted
    elif not history or history[-1]["role"] != "user":
        parts.append(f"{USER}\n\n{ASSISTANT}\n")
    return "".join(parts)


@torch.no_grad()
def generate_reply(
    model: GPT,
    tok: TokenizerWrapper,
    history: list[dict],
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    top_p: Optional[float] = 0.92,
    repetition_penalty: float = 1.15,
    stream: bool = False,
    on_text=None,
    examples: list[dict] | None = None,
) -> str:
    if examples:
        history = [*examples, *history]
    prompt = build_prompt(history)
    ids = tok.encode(prompt, add_special_tokens=False)
    # keep the prompt inside the context window, leaving room for the reply
    limit = max(16, model.cfg.block_size - max_new_tokens)
    if len(ids) > limit:
        ids = ids[-limit:]
    idx = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)

    pieces: list[int] = []
    emitted = ""

    def on_token(t):
        nonlocal emitted
        pieces.append(int(t.item()))
        if not (stream or on_text is not None):
            return
        text = tok.decode(pieces, skip_special_tokens=True)
        if text.startswith(emitted):
            delta = text[len(emitted) :]
            emitted = text
        else:
            delta = text
        if delta:
            if on_text is not None:
                on_text(delta)
            else:
                print(delta, end="", flush=True)

    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tok.end_id,
        stream_callback=on_token,
    )
    if stream and on_text is None:
        print()
    new_ids = out[0, idx.shape[1] :].tolist()
    return tok.decode(new_ids, skip_special_tokens=True).strip()


def make_prompt_encoder(tok: TokenizerWrapper):
    return lambda text: tok.encode(text, add_special_tokens=False)
