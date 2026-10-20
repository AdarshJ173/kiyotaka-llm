"""Byte-level BPE tokenizer trained from scratch on our own corpus."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

SPECIAL_TOKENS = ["<|end|>", "<|pad|>", "<|user|>", "<|assistant|>"]


def train_tokenizer(
    texts: Iterable[str],
    vocab_size: int,
    out_path: str | Path,
    min_frequency: int = 2,
) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None, byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    tok.save(str(out_path))
    return tok


class TokenizerWrapper:
    def __init__(self, path: str | Path):
        self.tok = Tokenizer.from_file(str(path))
        self.end_id = self.tok.token_to_id("<|end|>")
        self.pad_id = self.tok.token_to_id("<|pad|>")
        self.user_id = self.tok.token_to_id("<|user|>")
        self.assistant_id = self.tok.token_to_id("<|assistant|>")
        assert None not in (self.end_id, self.pad_id, self.user_id, self.assistant_id)

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self.tok.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=skip_special_tokens)
