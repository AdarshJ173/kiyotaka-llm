"""Fast, dependency-free smoke tests for the core model/data invariants."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kiyotaka.data import ChatDataset, PERSONA, format_chat  # noqa: E402
from src.kiyotaka.model import GPT, ModelConfig  # noqa: E402
from src.kiyotaka.tokenizer import TokenizerWrapper  # noqa: E402


def tiny_model() -> GPT:
    return GPT(
        ModelConfig(
            vocab_size=512, block_size=64, n_layer=2, n_head=8, n_kv_head=4,
            n_embd=128, intermediate_size=352, dropout=0.0,
        )
    ).eval()


def test_causality() -> None:
    torch.manual_seed(0)
    m = tiny_model()
    x = torch.randint(0, 512, (1, 32))
    with torch.no_grad():
        a, _, _ = m(x)
        y = x.clone()
        y[0, 20:] = torch.randint(0, 512, (12,))
        b, _, _ = m(y)
    assert torch.allclose(a[:, :20], b[:, :20], atol=0), "future tokens leaked into the past"
    assert not torch.allclose(a[:, 20:], b[:, 20:]), "test is not sensitive"


def test_kv_cache() -> None:
    torch.manual_seed(0)
    m = tiny_model()
    x = torch.randint(0, 512, (1, 24))
    with torch.no_grad():
        full, _, _ = m(x)
        past, outs = None, []
        for t in range(24):
            lg, _, past = m(x[:, t : t + 1], past_kv=past, use_cache=True)
            outs.append(lg[:, -1])
        cached = torch.stack(outs, dim=1)
    assert torch.allclose(full, cached, atol=1e-5), "KV cache diverges from full forward"


def test_loss_alignment() -> None:
    tok = TokenizerWrapper(ROOT / "data/processed/tokenizer.json")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.jsonl"
        p.write_text(
            json.dumps({"user": "Who are you?", "assistant": "Nobody important."}) + "\n",
            encoding="utf-8",
        )
        ds = ChatDataset(p, tok, max_len=64)
        ids, labels = ds[0]
    prompt_len = len(
        tok.encode(PERSONA + "<|user|>\nWho are you?\n<|assistant|>\n", add_special_tokens=False)
    )
    answer = tok.encode("Nobody important.", add_special_tokens=False)
    assert all(l == -100 for l in labels[: prompt_len - 1])
    assert labels[prompt_len - 1 : prompt_len - 1 + len(answer)] == answer, "answer is misaligned"
    assert labels[prompt_len - 1 + len(answer)] == tok.end_id, "end token not supervised"


def test_tokenizer_roundtrip() -> None:
    tok = TokenizerWrapper(ROOT / "data/processed/tokenizer.json")
    text = "Kiyotaka Ayanokōji — \"People are tools.\" 100%"
    assert tok.decode(tok.encode(text)) == text


def test_chat_format() -> None:
    s = format_chat("Hello", "Hi")
    assert s == "<|user|>\nHello\n<|assistant|>\nHi<|end|>\n"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all smoke tests passed")
