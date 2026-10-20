#!/usr/bin/env python3
"""Assemble the character SFT set from a declarative source config.

Priority order (first occurrence of a user prompt wins):
  1. hand-written persona Q&A (upweighted)
  2. generated persona prompts (relevance-filtered)
  3. generated scene prompts (relevance-filtered)
  4. scene-response pairs built from the preceding line

Canon-derived sources are split by episode: episodes 1-10 train, 11 val, 12
test. Manual data is split deterministically. Everything is driven by
configs/sft_sources.json.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

TEMPLATES = [
    '{speaker}: "{text}"\n\nWhat do you say?',
    '{speaker} just said: "{text}"\n\nHow do you respond?',
    '"{text}" — {speaker}. Your reply?',
    '{speaker} is waiting for an answer: "{text}"',
]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def valid(user: str, assistant: str) -> bool:
    if not (1 <= len(user) <= 240 and 1 <= len(assistant) <= 500):
        return False
    if assistant.strip() in {"...", "…"}:
        return False
    return len(user.split()) >= 2


def split_for(season: int, episode: int) -> str:
    if season <= 2:
        return "train"
    if season == 3:
        return "train" if episode <= 6 else ("val" if episode <= 9 else "test")
    return "test"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "sft_sources.json"))
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())

    buckets: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    seen_users: set[str] = set()

    def add(user: str, assistant: str, source: str, split: str, repeat: int = 1) -> None:
        if not valid(user, assistant):
            return
        key = norm(user)
        if key in seen_users:
            return
        seen_users.add(key)
        row = {"user": user.strip(), "assistant": assistant.strip(), "source": source}
        buckets[split].extend([row] * repeat)

    # 1. manual persona data
    manual: list[tuple[dict, int]] = []
    for spec in cfg.get("manual", []):
        path = ROOT / spec["path"]
        if not path.exists():
            continue
        for line in open(path, encoding="utf-8"):
            manual.append((json.loads(line), int(spec.get("repeat", 1))))
    rng = random.Random(cfg.get("seed", 7))
    rng.shuffle(manual)
    n = len(manual)
    manual_split: dict[str, str] = {}
    val_frac = cfg.get("val_fraction", 0.07)
    test_frac = cfg.get("test_fraction", 0.07)
    for i, (r, repeat) in enumerate(manual):
        split = "train" if i < int(n * (1 - val_frac - test_frac)) else (
            "val" if i < int(n * (1 - test_frac)) else "test"
        )
        manual_split[norm(r["user"])] = split
        add(r["user"], r["assistant"], "manual", split, repeat=repeat)

    # 2-3. generated prompts
    n_gen = 0
    for spec in cfg.get("generated", []):
        path = ROOT / spec["path"]
        if not path.exists():
            continue
        min_words = int(spec.get("min_words", 1))
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if len(r["assistant"].split()) < min_words:
                continue
            split = manual_split.get(norm(r.get("original", "")))
            if split is None and "episode" in r:
                split = split_for(int(r.get("season", 1)), int(r["episode"]))
            if split is None:
                bucket = zlib.crc32(norm(r["user"]).encode()) % 100
                split = "train" if bucket < 86 else ("val" if bucket < 93 else "test")
            add(r["user"], r["assistant"], "generated", split, repeat=int(spec.get("repeat", 1)))
            n_gen += 1

    # 4. scene-response pairs
    n_scene = 0
    if cfg.get("include_scene_pairs", True):
        min_words = int(cfg.get("scene_min_words", 4))
        kiyo = [json.loads(l) for l in open(PROC / "kiyo_lines_v2.jsonl", encoding="utf-8")]
        for i, r in enumerate(kiyo):
            if len(r["text"].split()) < min_words:
                continue
            split = split_for(int(r.get("season", 1)), int(r["episode"]))
            prev = [c for c in r["context"] if "Ayanok" not in c["speaker"] and len(c["text"].split()) >= 2]
            if not prev:
                continue
            c = prev[-1]
            add(
                TEMPLATES[i % len(TEMPLATES)].format(speaker=c["speaker"], text=c["text"]),
                r["text"], "scene", split,
            )
            n_scene += 1

    for split, rows in buckets.items():
        path = PROC / f"sft_{split}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} examples -> {path}")
    print(f"sources: manual={len(manual)} generated={n_gen} scene={n_scene}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
