#!/usr/bin/env python3
"""Pretraining loop for the from-scratch model.

Single-GPU, bf16 autocast, gradient accumulation, cosine schedule, memmapped
tokens, best/last checkpointing. No trainer framework.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kiyotaka.model import GPT, ModelConfig  # noqa: E402


class TokenLoader:
    def __init__(self, path: str | Path, block_size: int, batch_size: int, seed: int = 0):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def batch(self, device: torch.device):
        max_start = len(self.data) - self.block_size - 1
        starts = self.rng.integers(0, max_start, size=self.batch_size)
        x = np.stack([self.data[s : s + self.block_size] for s in starts]).astype(np.int64)
        y = np.stack([self.data[s + 1 : s + 1 + self.block_size] for s in starts]).astype(np.int64)
        return (
            torch.from_numpy(x).to(device, non_blocking=True),
            torch.from_numpy(y).to(device, non_blocking=True),
        )


def lr_at(step: int, total: int, warmup: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    if step >= total:
        return min_lr
    ratio = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * ratio))


@torch.no_grad()
def evaluate(model: GPT, loader: TokenLoader, iters: int, device: torch.device) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = loader.batch(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args()
    defaults = json.load(open(known.config)) if known.config else {}
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=known.config)
    ap.add_argument("--data")
    ap.add_argument("--val-data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=str(ROOT / "data" / "processed" / "tokenizer.json"))
    ap.add_argument("--init", default=None, help="initial weights (.pt) for continued pretraining")
    ap.add_argument("--resume", default=None, help="full checkpoint to resume")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--n-layer", type=int, default=8)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--n-kv-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=512)
    ap.add_argument("--intermediate-size", type=int, default=1408)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=200)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--save-interval", type=int, default=500)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.set_defaults(**{k: v for k, v in defaults.items() if k != "stage"})
    args = ap.parse_args()
    if not args.data or not args.val_data:
        ap.error("--data and --val-data are required (or set them in --config)")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from tokenizers import Tokenizer

    vocab_size = Tokenizer.from_file(args.tokenizer).get_vocab_size()
    init_ckpt = None
    if args.init:
        init_ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
    arch = dict(init_ckpt.get("config", {})) if init_ckpt else {}
    cfg = ModelConfig(
        vocab_size=arch.get("vocab_size", vocab_size),
        block_size=arch.get("block_size", args.block_size),
        n_layer=arch.get("n_layer", args.n_layer),
        n_head=arch.get("n_head", args.n_head),
        n_kv_head=arch.get("n_kv_head", args.n_kv_head),
        n_embd=arch.get("n_embd", args.n_embd),
        intermediate_size=arch.get("intermediate_size", args.intermediate_size),
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    print(f"model params: {model.num_params():,} (non-embedding {model.num_params(True):,})")
    (out / "model_config.json").write_text(json.dumps(cfg.__dict__, indent=2))

    optimizer = model.configure_optimizers(args.weight_decay, args.lr)
    start_step = 0
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        best_val = ckpt.get("best_val", best_val)
        print(f"resumed from {args.resume} at step {start_step}")
    elif init_ckpt is not None:
        state = init_ckpt.get("model", init_ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"init from {args.init}; missing={len(missing)} unexpected={len(unexpected)}")

    raw_model = model
    if args.compile:
        print("compiling model (first step will be slow)")
        model = torch.compile(model)

    train_loader = TokenLoader(args.data, args.block_size, args.batch_size, seed=args.seed)
    val_loader = TokenLoader(args.val_data, args.block_size, args.batch_size, seed=args.seed + 1)
    tokens_per_step = args.batch_size * args.block_size * args.grad_accum
    total_tokens = train_loader.data.size
    print(
        f"data: {total_tokens/1e6:.1f}M tokens | {tokens_per_step:,} tok/step | "
        f"{args.steps} steps = {args.steps*tokens_per_step/1e6:.1f}M tokens"
    )

    log_path = out / "train_log.jsonl"
    log_fh = open(log_path, "a", encoding="utf-8")
    t0 = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_step, args.steps):
        lr = lr_at(step, args.steps, args.warmup, args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        loss_acc = 0.0
        for _ in range(args.grad_accum):
            x, y = train_loader.batch(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, _ = model(x, targets=y)
            (loss / args.grad_accum).backward()
            loss_acc += loss.item() / args.grad_accum

        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            done = (step - start_step + 1) * tokens_per_step
            tok_s = done / max(elapsed, 1e-6)
            print(
                f"step {step:5d}/{args.steps} | loss {loss_acc:.4f} | lr {lr:.2e} | "
                f"gnorm {grad_norm.item():.2f} | {tok_s/1000:.1f}k tok/s",
                flush=True,
            )
            log_fh.write(
                json.dumps(
                    {
                        "step": step,
                        "loss": loss_acc,
                        "lr": lr,
                        "grad_norm": float(grad_norm),
                        "tok_s": tok_s,
                        "elapsed": elapsed,
                    }
                )
                + "\n"
            )
            log_fh.flush()

        if args.eval_interval and step > 0 and step % args.eval_interval == 0:
            val = evaluate(raw_model, val_loader, args.eval_iters, device)
            print(f"  eval @ {step}: val_loss {val:.4f} (ppl {math.exp(val):.2f})", flush=True)
            log_fh.write(json.dumps({"step": step, "val_loss": val, "val_ppl": math.exp(val)}) + "\n")
            log_fh.flush()
            if val < best_val:
                best_val = val
                torch.save(
                    {"model": raw_model.state_dict(), "step": step, "best_val": best_val,
                     "config": cfg.__dict__, "tokenizer": args.tokenizer},
                    out / "best.pt",
                )
                print(f"  saved best.pt (val {val:.4f})", flush=True)

        if args.save_interval and step > 0 and step % args.save_interval == 0:
            torch.save(
                {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                 "step": step, "best_val": best_val, "config": cfg.__dict__},
                out / "last.pt",
            )

    torch.save(
        {"model": raw_model.state_dict(), "step": args.steps, "best_val": best_val,
         "config": cfg.__dict__, "tokenizer": args.tokenizer},
        out / "final.pt",
    )
    if not (out / "best.pt").exists():
        torch.save(
            {"model": raw_model.state_dict(), "step": args.steps, "best_val": best_val,
             "config": cfg.__dict__, "tokenizer": args.tokenizer},
            out / "best.pt",
        )
    log_fh.close()
    print(f"done in {(time.time()-t0)/60:.1f} min; best val {best_val:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
