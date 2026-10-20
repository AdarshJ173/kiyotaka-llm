#!/usr/bin/env python3
"""Throughput / latency / memory benchmarks for the trained model."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kiyotaka.generate import load_model  # noqa: E402
from src.kiyotaka.tokenizer import TokenizerWrapper  # noqa: E402


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def prefill(model, batch, prompt_len, device, runs=5):
    idx = torch.randint(0, model.cfg.vocab_size, (batch, prompt_len), device=device)
    times = []
    for _ in range(runs):
        sync(device)
        t = time.perf_counter()
        model(idx, use_cache=True)
        sync(device)
        times.append(time.perf_counter() - t)
    return statistics.median(times)


@torch.no_grad()
def decode(model, batch, prompt_len, new_tokens, device, runs=3):
    idx = torch.randint(0, model.cfg.vocab_size, (batch, prompt_len), device=device)
    times = []
    for _ in range(runs):
        sync(device)
        t = time.perf_counter()
        model.generate(idx, max_new_tokens=new_tokens, temperature=0.8, top_k=40, top_p=0.9)
        sync(device)
        times.append(time.perf_counter() - t)
    med = statistics.median(times)
    return med, batch * new_tokens / med


def bench_device(ckpt, device_name, batch_sizes=(1, 4, 8), prompt_len=64, new_tokens=64):
    device = torch.device(device_name)
    model, cfg = load_model(ckpt, device=device)
    results = {"device": device_name, "params": model.num_params(), "runs": {}}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for b in batch_sizes:
        pre = prefill(model, b, prompt_len, device)
        total, tok_s = decode(model, b, prompt_len, new_tokens, device)
        results["runs"][f"batch{b}"] = {
            "prefill_ms": pre * 1000,
            "decode_total_s": total,
            "decode_tokens_per_s": tok_s,
            "ms_per_token": total * 1000 / new_tokens,
        }
        print(f"{device_name} batch {b}: prefill {pre*1000:.1f} ms | {tok_s:.1f} tok/s")
    if device.type == "cuda":
        results["vram_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
        results["vram_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
        print(
            f"VRAM peak: {results['vram_allocated_mb']:.0f} MB allocated / "
            f"{results['vram_reserved_mb']:.0f} MB reserved"
        )
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=str(ROOT / "reports" / "benchmark.json"))
    ap.add_argument("--cpu", action="store_true", help="also benchmark CPU fp32 and int8")
    args = ap.parse_args()

    results = {"checkpoint": args.ckpt, "torch": torch.__version__}
    results["gpu"] = bench_device(args.ckpt, "cuda")
    if args.cpu:
        results["cpu"] = bench_device(args.ckpt, "cpu", batch_sizes=(1,), new_tokens=32)
        model, _ = load_model(args.ckpt, device="cpu")
        q = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        import copy

        idx = torch.randint(0, model.cfg.vocab_size, (1, 32))
        t = time.perf_counter()
        q.generate(idx, max_new_tokens=32, temperature=0.8, top_k=40, top_p=0.9)
        dt = time.perf_counter() - t
        size_mb = sum(p.numel() * p.element_size() for p in q.parameters()) / 2**20
        results["cpu_int8"] = {
            "decode_tokens_per_s": 32 / dt,
            "weight_mb": size_mb,
        }
        print(f"cpu int8: {32/dt:.1f} tok/s | weights {size_mb:.0f} MB")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
