#!/usr/bin/env python3
"""Supervised fine-tuning of the from-scratch model on character chat data."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kiyotaka.data import ChatDataset, collate_chat  # noqa: E402
from src.kiyotaka.generate import generate_reply, load_model  # noqa: E402
from src.kiyotaka.tokenizer import TokenizerWrapper  # noqa: E402
from scripts.train import TokenLoader, lr_at  # noqa: E402

PROMPTS = [
    "Who are you?",
    "What do you think about friendship?",
    "Horikita says I should trust her. What should I do?",
    "Are all people equal?",
    "Why do you hide how smart you are?",
    "What is the fastest way to win?",
]


def evaluate(model, loader, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
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
    ap.add_argument("--ckpt")
    ap.add_argument("--train", default=str(ROOT / "data/processed/sft_train.jsonl"))
    ap.add_argument("--val", default=str(ROOT / "data/processed/sft_val.jsonl"))
    ap.add_argument("--tokenizer", default=str(ROOT / "data/processed/tokenizer.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--min-lr", type=float, default=4e-5)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--replay-data", default=str(ROOT / "data/processed/domain_all.bin"))
    ap.add_argument("--replay-interval", type=int, default=10)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    ap.set_defaults(**{k: v for k, v in defaults.items() if k != "stage"})
    args = ap.parse_args()
    if not args.ckpt:
        ap.error("--ckpt is required (or set it in --config)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = TokenizerWrapper(args.tokenizer)
    model, cfg = load_model(args.ckpt, device=device, dropout=args.dropout)
    model.train()
    print(f"params: {model.num_params():,} | dropout {args.dropout}")

    train_ds = ChatDataset(args.train, tok, max_len=args.max_len)
    val_ds = ChatDataset(args.val, tok, max_len=args.max_len)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_chat(b, tok.pad_id), drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_chat(b, tok.pad_id), drop_last=False,
    )
    replay_loader = None
    if args.replay_data and Path(args.replay_data).exists():
        replay_loader = TokenLoader(args.replay_data, cfg.block_size, 8, seed=args.seed + 5)
    print(f"train {len(train_ds)} | val {len(val_ds)} | replay={replay_loader is not None}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    optimizer = model.configure_optimizers(args.weight_decay, args.lr)
    total_steps = args.epochs * math.ceil(len(train_loader) / args.grad_accum)
    print(f"total steps: {total_steps}")

    best_val = float("inf")
    step = 0
    t0 = time.time()
    log = open(out / "sft_log.jsonl", "a", encoding="utf-8")
    for epoch in range(args.epochs):
        agg = 0.0
        nb = 0
        optimizer.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, _ = model(x, targets=y, label_smoothing=args.label_smoothing)
            (loss / args.grad_accum).backward()
            agg += loss.item()
            nb += 1
            if (i + 1) % args.grad_accum == 0:
                lr = lr_at(step, total_steps, args.warmup, args.lr, args.min_lr)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                # small LM replay keeps general language ability alive
                if replay_loader is not None and args.replay_interval and step % args.replay_interval == 0:
                    rx, ry = replay_loader.batch(device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _, rloss, _ = model(rx, targets=ry)
                    (rloss * 0.1).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                if step % 10 == 0:
                    print(
                        f"epoch {epoch+1} step {step}/{total_steps} loss {agg/nb:.4f} lr {lr:.2e} "
                        f"{(time.time()-t0)/60:.1f}m",
                        flush=True,
                    )
                    log.write(json.dumps({"step": step, "loss": agg / nb, "lr": lr}) + "\n")
                    log.flush()
                    agg, nb = 0.0, 0

        val = evaluate(model, val_loader, device)
        print(f"== epoch {epoch+1}: val_loss {val:.4f} ppl {math.exp(val):.2f}", flush=True)
        log.write(json.dumps({"epoch": epoch + 1, "val_loss": val, "val_ppl": math.exp(val)}) + "\n")
        log.flush()
        if val < best_val:
            best_val = val
            torch.save(
                {"model": model.state_dict(), "step": step, "best_val": best_val,
                 "config": cfg.__dict__, "tokenizer": args.tokenizer},
                out / "best.pt",
            )
            print(f"saved {out/'best.pt'}")

        model.eval()
        for p in PROMPTS[:3]:
            reply = generate_reply(model, tok, [{"role": "user", "content": p}], max_new_tokens=48)
            print(f"  [{p}] -> {reply}")
        model.train()

    torch.save(
        {"model": model.state_dict(), "step": step, "best_val": best_val,
         "config": cfg.__dict__, "tokenizer": args.tokenizer},
        out / "final.pt",
    )
    log.close()
    print(f"done in {(time.time()-t0)/60:.1f} min; best val {best_val:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
