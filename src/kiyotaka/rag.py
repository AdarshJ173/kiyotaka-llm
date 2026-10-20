"""Local retrieval over the character Q&A set.

Used as optional in-context grounding: the chat client retrieves the closest
curated exchanges and prepends them, so a 28M model does not have to memorise
every intent. Fully offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).float()
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
    return torch.nn.functional.normalize(pooled, dim=-1)


class Retriever:
    def __init__(self, index_path: str | Path, device: str | None = None, model_name: str = MODEL):
        data = np.load(index_path, allow_pickle=True)
        self.emb = data["emb"].astype(np.float32)
        self.rows = json.loads(str(data["rows"]))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(
                texts[i : i + batch_size], padding=True, truncation=True,
                max_length=128, return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**enc).last_hidden_state
            out.append(mean_pool(hidden, enc["attention_mask"]).cpu().numpy())
        return np.concatenate(out, axis=0)

    def search(self, query: str, k: int = 3, threshold: float = 0.45):
        q = self.embed([query])[0]
        sims = self.emb @ q
        order = np.argsort(-sims)[:k]
        return [(self.rows[i], float(sims[i])) for i in order if sims[i] >= threshold]

    def as_history(self, query: str, k: int = 2, threshold: float = 0.5) -> list[dict]:
        history: list[dict] = []
        for row, _ in self.search(query, k=k, threshold=threshold):
            history.append({"role": "user", "content": row["user"]})
            history.append({"role": "assistant", "content": row["assistant"]})
        return history
