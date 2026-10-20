"""Datasets: memory-mapped pretraining tokens and chat SFT batches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import TokenizerWrapper

USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END = "<|end|>"
PERSONA = "[Kiyotaka Ayanokoji | Class D | calm, blunt, analytical, restrained]\n"


def format_chat(user: str, assistant: str) -> str:
    return f"{USER}\n{user}\n{ASSISTANT}\n{assistant}{END}\n"


class TokenDataset(Dataset):
    """Random fixed-length windows over a uint16 token file."""

    def __init__(self, path: str | Path, block_size: int, length: Optional[int] = None):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.length = length if length is not None else max(1, (len(self.data) - 1) // block_size)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        i = int(torch.randint(0, len(self.data) - self.block_size - 1, (1,)).item())
        x = torch.from_numpy(self.data[i : i + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[i + 1 : i + 1 + self.block_size].astype(np.int64))
        return x, y


class ChatDataset(Dataset):
    """Tokenised single-turn chat examples with prompt loss masking."""

    def __init__(
        self,
        path: str | Path,
        tokenizer: TokenizerWrapper,
        max_len: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples: list[tuple[list[int], list[int]]] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                user = row["user"].strip()
                assistant = row["assistant"].strip()
                if not user or not assistant:
                    continue
                prompt = f"{PERSONA}{USER}\n{user}\n{ASSISTANT}\n"
                full = prompt + assistant + END + "\n"
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                full_ids = tokenizer.encode(full, add_special_tokens=False)
                full_ids = full_ids[: max_len + 1]
                prompt_len = min(len(prompt_ids), len(full_ids))
                input_ids = full_ids[:-1]
                # labels[i] must be the *next* token after input_ids[i]. The
                # last prompt token predicts the first assistant token.
                labels = ([-100] * max(0, prompt_len - 1) + full_ids[prompt_len:])[
                    : len(input_ids)
                ]
                if not any(l != -100 for l in labels):
                    continue
                self.examples.append((input_ids, labels))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int):
        return self.examples[i]


def collate_chat(batch, pad_id: int):
    max_len = max(len(x) for x, _ in batch)
    input_ids, labels = [], []
    for x, y in batch:
        pad = max_len - len(x)
        input_ids.append(x + [pad_id] * pad)
        labels.append(y + [-100] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )
