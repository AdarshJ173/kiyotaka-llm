#!/usr/bin/env python3
"""Scrape SubArchivist (Crunchyroll subtitle transcripts) for S1-S3.

The site exposes per-cue speaker labels, which makes it the best source for
Ayanokoji's actual spoken lines. It is behind Cloudflare, so requests go through
the public r.jina.ai reader; results are cached under data/raw/subarchivist/.

Output: data/raw/subarchivist_cues.jsonl with
  {season, episode, time, speaker, text, type}
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw" / "subarchivist"
CACHE.mkdir(parents=True, exist_ok=True)
BASE = "https://subs.yakuaru.com/books/classroom-of-the-elite"
SEASONS = (1, 2, 3)
CUE = re.compile(r"^\*\*\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.+?)\*\*\s*$")
LINK = re.compile(r"\((https://subs\.yakuaru\.com/books/classroom-of-the-elite/page/e(\d+)-[^)]+)\)")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/130"}


def fetch(url: str, cache: Path, tries: int = 4) -> str | None:
    if cache.exists() and cache.stat().st_size > 2_000:
        return cache.read_text(encoding="utf-8")
    for attempt in range(tries):
        try:
            r = requests.get(f"https://r.jina.ai/{url}", headers=UA, timeout=120)
            if r.status_code == 200 and len(r.text) > 2_000:
                cache.write_text(r.text, encoding="utf-8")
                return r.text
            if r.status_code == 429:
                wait = 20 * (attempt + 1)
                print(f"  429; sleeping {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"  HTTP {r.status_code} for {url}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"  {type(exc).__name__}: {exc}", file=sys.stderr)
        time.sleep(5)
    return None


def strip_md(s: str) -> str:
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = s.replace("**", "").replace("_", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip('"').strip()


def parse_episode(text: str, season: int, episode: int) -> list[dict]:
    body = text.split("Markdown Content:", 1)[-1]
    rows: list[dict] = []
    cur: dict | None = None
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("_**SIGN**_"):
            continue
        m = CUE.match(line)
        if m:
            if cur and cur["parts"]:
                rows.append(cur)
            speaker = re.sub(r"^\((?:Flashback|Monologue|Inner[^)]*)\)\s*", "", m.group(2)).strip()
            cur = {"time": m.group(1), "speaker": speaker, "parts": []}
            continue
        if cur is None:
            continue
        if line.startswith(("Source:", "Translator:", "Editor:", "Timer:", "QC:", "(Please")):
            continue
        cur["parts"].append(strip_md(line))
    if cur and cur["parts"]:
        rows.append(cur)

    out = []
    last_time = ""
    for r in rows:
        text = re.sub(r"\s+", " ", " ".join(r["parts"])).strip()
        text = re.sub(r"\s+of the Elite$", "", text).strip()
        if not text or len(text) < 2:
            continue
        # cues with no new timestamp are continuations of the same line
        last_time = r["time"] or last_time
        out.append(
            {
                "season": season,
                "episode": episode,
                "time": last_time,
                "speaker": r["speaker"],
                "text": text,
                "type": "dialogue" if r["speaker"].lower() != "sign" else "sign",
            }
        )
    return out


def main() -> int:
    all_rows: list[dict] = []
    for season in SEASONS:
        idx_cache = CACHE / f"season{season}_index.md"
        idx = fetch(f"{BASE}/chapter/season-{season}", idx_cache)
        if not idx:
            print(f"WARN: no index for season {season}", file=sys.stderr)
            continue
        links = []
        seen_eps = set()
        for url, ep in LINK.findall(idx):
            ep = int(ep)
            if ep in seen_eps:
                continue
            seen_eps.add(ep)
            links.append((ep, url))
        links.sort()
        print(f"season {season}: {len(links)} episodes")
        for ep, url in links:
            cache = CACHE / f"s{season}e{ep:02d}.md"
            text = fetch(url, cache)
            if not text:
                print(f"  WARN S{season}E{ep} failed", file=sys.stderr)
                continue
            rows = parse_episode(text, season, ep)
            kiyo = sum("ayanok" in r["speaker"].lower() for r in rows)
            print(f"  S{season}E{ep:02d}: {len(rows)} cues, {kiyo} Ayanokoji")
            all_rows.extend(rows)
            time.sleep(1.2)

    out = ROOT / "data" / "raw" / "subarchivist_cues.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    kiyo = [r for r in all_rows if "ayanok" in r["speaker"].lower()]
    print(f"wrote {len(all_rows)} cues ({len(kiyo)} Ayanokoji) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
