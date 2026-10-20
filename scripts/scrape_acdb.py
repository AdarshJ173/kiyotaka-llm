#!/usr/bin/env python3
"""Scrape ACDB series quote pages for Ayanokoji lines (notably Season 4).

ACDB sits behind a JS challenge for normal clients, so pages are fetched via
the public r.jina.ai reader and cached. Each quote carries the episode number in
its audio URL, which is parsed as metadata.

Output: data/raw/acdb_quotes.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw" / "acdb"
CACHE.mkdir(parents=True, exist_ok=True)
SERIES = {1: 105139, 2: 108956, 3: 110082, 4: 111971}
URL = "https://www.animecharactersdatabase.com/quotesbyseries.php"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/130"}
QUOTE = re.compile(
    r"\[([^\]]{1,500})\]\([^)]*line_id=(\d+)\)\s*\n―\s*Kiyotaka Ayanokouji", re.S
)
AUDIO = re.compile(r"audio/\d+/(\d+)/\d+-\d+-\d+\.mp4")


def fetch(season: int, x: int, tries: int = 4) -> str | None:
    cache = CACHE / f"s{season}_x{x}.md"
    if cache.exists() and cache.stat().st_size > 1_000:
        return cache.read_text(encoding="utf-8")
    url = f"https://r.jina.ai/{URL}?pid={SERIES[season]}&x={x}"
    for attempt in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=90)
            if r.status_code == 200 and len(r.text) > 1_000:
                cache.write_text(r.text, encoding="utf-8")
                return r.text
            if r.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
        except Exception as exc:  # noqa: BLE001
            print(f"  {type(exc).__name__}: {exc}", file=sys.stderr)
        time.sleep(4)
    return None


def clean(s: str) -> str:
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = s.replace("_", "").replace("**", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip('"').strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="4")
    ap.add_argument("--max-pages", type=int, default=50)
    args = ap.parse_args()
    seasons = [int(s) for s in args.seasons.split(",")]

    out_rows = []
    seen = set()
    for season in seasons:
        empty = 0
        for page in range(args.max_pages):
            x = page * 10
            text = fetch(season, x)
            if not text:
                print(f"S{season} x={x}: fetch failed", file=sys.stderr)
                empty += 1
                if empty >= 3:
                    break
                continue
            found = 0
            for m in QUOTE.finditer(text):
                line_id = m.group(2)
                if line_id in seen:
                    continue
                seen.add(line_id)
                quote = clean(m.group(1))
                if not quote or len(quote) < 3:
                    continue
                tail = text[m.end() : m.end() + 400]
                ep_match = AUDIO.search(tail)
                episode = int(ep_match.group(1)) if ep_match else 0
                out_rows.append(
                    {"season": season, "episode": episode, "text": quote, "line_id": line_id}
                )
                found += 1
            print(f"S{season} x={x}: {found} new", flush=True)
            if found == 0:
                empty += 1
                if empty >= 3:
                    break
            else:
                empty = 0
            time.sleep(1.0)

    out = ROOT / "data" / "raw" / "acdb_quotes.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out_rows)} quotes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
