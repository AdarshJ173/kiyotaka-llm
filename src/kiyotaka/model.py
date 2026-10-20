"""A small decoder-only transformer, implemented from scratch in PyTorch.

Architecture (GPT-style, modernised):
  * RMSNorm (pre-norm)
  * rotary position embeddings (RoPE)
  * grouped-query causal self-attention (PyTorch SDPA / flash kernel)
  * SwiGLU feed-forward
  * tied input/output embeddings
  * no biases

The model is deliberately dependency-light: no transformers/accelerate needed
for training or inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_kv_head: int = 4
    n_embd: int = 512
    intermediate_size: int = 1408
    dropout: float = 0.0
    rope_theta: float = 10_000.0
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def precompute_rope(head_dim: int, max_seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (T, D/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (T, D)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D); cos/sin: (T, D) -> broadcast over batch/heads
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        assert cfg.n_head % cfg.n_kv_head == 0
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout
        q_dim = cfg.n_head * cfg.head_dim
        kv_dim = cfg.n_kv_head * cfg.head_dim
        self.wq = nn.Linear(cfg.n_embd, q_dim, bias=False)
        self.wk = nn.Linear(cfg.n_embd, kv_dim, bias=False)
        self.wv = nn.Linear(cfg.n_embd, kv_dim, bias=False)
        self.wo = nn.Linear(q_dim, cfg.n_embd, bias=False)
        self.rope_cos, self.rope_sin = precompute_rope(
            cfg.head_dim, cfg.block_size * 2, cfg.rope_theta
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        start = 0 if past_kv is None else past_kv[0].shape[2]
        cos = self.rope_cos[start : start + T].to(x.device)
        sin = self.rope_sin[start : start + T].to(x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_kv is not None:
            k = torch.cat((past_kv[0], k), dim=2)
            v = torch.cat((past_kv[1], v), dim=2)
        present = (k, v) if use_cache else None

        attn_drop = self.dropout if self.training else 0.0
        # is_causal only when the query block is the entire prefix; during
        # cached decoding each query is a single position and may attend to all
        # cached keys.
        is_causal = T > 1
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=attn_drop,
            is_causal=is_causal,
            enable_gqa=self.n_kv_head != self.n_head,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.wo(out), present


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.n_embd, cfg.intermediate_size, bias=False)
        self.w2 = nn.Linear(cfg.intermediate_size, cfg.n_embd, bias=False)
        self.w3 = nn.Linear(cfg.n_embd, cfg.intermediate_size, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, past_kv=None, use_cache=False):
        h, present = self.attn(self.attn_norm(x), past_kv, use_cache)
        x = x + h
        x = x + self.mlp(self.mlp_norm(x))
        return x, present


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        # scaled init for residual projections
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        label_smoothing: float = 0.0,
    ):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        x = self.drop(self.tok_emb(idx))
        presents = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            x, present = block(x, None if past_kv is None else past_kv[i], use_cache)
            if use_cache:
                presents.append(present)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            if loss_mask is None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1),
                    ignore_index=-100, label_smoothing=label_smoothing,
                )
            else:
                flat_logits = logits.view(-1, logits.size(-1))
                flat_targets = targets.view(-1)
                flat_mask = loss_mask.view(-1).bool()
                if flat_mask.any():
                    loss = F.cross_entropy(
                        flat_logits[flat_mask], flat_targets[flat_mask],
                        label_smoothing=label_smoothing,
                    )
                else:
                    loss = logits.sum() * 0.0
        return logits, loss, presents

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.95,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
        stream_callback=None,
    ) -> torch.Tensor:
        self.eval()
        past = None
        generated = idx
        for _ in range(max_new_tokens):
            cur = idx[:, -1:] if past is not None else idx
            logits, _, past = self.forward(cur, past_kv=past, use_cache=True)
            logits = logits[:, -1, :].float()

            if repetition_penalty and repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    used = torch.unique(idx[b])
                    vals = logits[b, used]
                    logits[b, used] = torch.where(
                        vals < 0, vals * repetition_penalty, vals / repetition_penalty
                    )

            logits = logits / max(temperature, 1e-5)
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = -float("inf")
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum = probs.cumsum(dim=-1)
                mask = cum - probs > top_p
                sorted_logits[mask] = -float("inf")
                logits = torch.full_like(logits, -float("inf")).scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_tok), dim=1)
            generated = torch.cat((generated, next_tok), dim=1)
            if stream_callback is not None:
                stream_callback(next_tok)
            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break
        return generated

    def configure_optimizers(self, weight_decay: float, lr: float, betas=(0.9, 0.95)):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2 and "tok_emb" not in name and "lm_head" not in name:
                decay.append(p)
            else:
                no_decay.append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=torch.cuda.is_available())
