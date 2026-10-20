#!/usr/bin/env python3
"""Collect Classroom of the Elite dialogue from public wiki sources.

Sources
-------
1. you-zitsu.fandom.com -- Season 1 episode "Screenplay" pages, which are
   speaker-labelled transcripts (dialogue + stage directions + monologue tags).
2. animetranscript.fandom.com -- independent Season 1 Episode 1 transcript,
   kept as a cross-check / supplement.

Raw wikitext is cached under data/raw/ so the pipeline is reproducible and
polite to the upstream servers. Nothing scraped here is redistributed: the
data/ directory is git-ignored.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
UA = "kiyotaka-research-bot/0.1 (local research; contact: local user)"
API = "https://you-zitsu.fandom.com/api.php"
ALT_API = "https://animetranscript.fandom.com/api.php"
S1_EPISODES = range(1, 13)


def fetch_wikitext(api: str, page: str, cache: Path) -> str:
    if cache.exists() and cache.stat().st_size > 100:
        return cache.read_text(encoding="utf-8")
    r = requests.get(
        api,
        params={
            "action": "parse",
            "page": page,
            "prop": "wikitext",
            "format": "json",
            "formatversion": "2",
        },
        headers={"User-Agent": UA},
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json()
    if "error" in payload:
        raise RuntimeError(f"{page}: {payload['error']}")
    text = payload["parse"]["wikitext"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    time.sleep(0.5)  # be polite
    return text


def strip_markup(s: str) -> str:
    s = s.replace("<nowiki>", "").replace("</nowiki>", "")
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    # [[target|label]] -> label ; [[target]] -> target
    for _ in range(3):
        s = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", s)
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("'''", "").replace("''", "")
    s = s.replace("&nbsp;", " ").replace("&quot;", '"').replace("&amp;", "&")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


SCENE_RE = re.compile(r"^==+\s*(.*?)\s*==+$")
# '''[[Speaker]] (Monologue):''' text    /    '''Speaker:''' text
LINE_RE = re.compile(r"^\s*'''\s*(.+?)\s*:\s*'''\s*(.*)$")


def parse_screenplay(wikitext: str, episode: int) -> list[dict]:
    rows: list[dict] = []
    scene = ""
    for raw_line in wikitext.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = SCENE_RE.match(line)
        if m:
            scene = strip_markup(m.group(1))
            continue
        if line.startswith("{{") or line.startswith("|") or line.startswith("}}"):
            continue
        # Stage direction / scene note (semicolon-prefixed in this wiki)
        if line.startswith(";") or line.startswith(":;"):
            note = strip_markup(line.lstrip(";:"))
            if note:
                rows.append(
                    {"episode": episode, "scene": scene, "type": "stage", "speaker": "", "text": note}
                )
            continue
        m = LINE_RE.match(line)
        if not m:
            # Continuation / unattributed narration
            note = strip_markup(line)
            if note:
                rows.append(
                    {"episode": episode, "scene": scene, "type": "narration", "speaker": "", "text": note}
                )
            continue
        speaker_raw, text_raw = m.group(1), m.group(2)
        speaker = strip_markup(speaker_raw)
        monologue = bool(re.search(r"\(\s*monologue\s*\)", speaker, flags=re.I))
        speaker = re.sub(r"\(.*?\)", "", speaker).strip()
        speaker = speaker.replace("[[", "").replace("]]", "").strip()
        text = strip_markup(text_raw)
        if not text:
            continue
        rows.append(
            {
                "episode": episode,
                "scene": scene,
                "type": "dialogue",
                "speaker": speaker,
                "monologue": monologue,
                "text": text,
            }
        )
    return rows


def main() -> int:
    all_rows: list[dict] = []
    for ep in S1_EPISODES:
        page = f"Episode {ep}/Screenplay"
        cache = RAW / "screenplays" / f"S1E{ep:02d}.wiki"
        try:
            wt = fetch_wikitext(API, page, cache)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"WARN {page}: {exc}", file=sys.stderr)
            continue
        rows = parse_screenplay(wt, ep)
        dialogue = [r for r in rows if r["type"] == "dialogue"]
        print(f"S1E{ep:02d}: {len(rows):4d} rows ({len(dialogue):3d} dialogue)")
        all_rows.extend(rows)

    # Independent transcript for S1E1 (cross-check), parsed with the same rules.
    try:
        alt = fetch_wikitext(
            ALT_API,
            "Classroom of the Elite/Season 1 Episode 1",
            RAW / "animetranscript" / "S1E01.wiki",
        )
        alt_rows = parse_screenplay(alt, 1)
        for r in alt_rows:
            r["source"] = "animetranscript"
        all_rows.extend(alt_rows)
        print(f"ALT S1E01: {len(alt_rows)} rows")
    except Exception as exc:  # noqa: BLE001
        print(f"WARN alt transcript: {exc}", file=sys.stderr)

    out = ROOT / "data" / "raw" / "screenplay_lines.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(all_rows)} rows)")

    kiyo = [r for r in all_rows if r["type"] == "dialogue" and "Ayanok" in r["speaker"]]
    print(f"Ayanokoji dialogue lines: {len(kiyo)}")
    print(f"distinct speakers: {len({r['speaker'] for r in all_rows if r['speaker']})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
