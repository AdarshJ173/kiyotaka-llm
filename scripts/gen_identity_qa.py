#!/usr/bin/env python3
"""Deterministic identity Q&A from wiki infobox + Personality section.

No generation, no guessing: name comes from the page title, class from the
infobox `class` field, description from the first Personality sentence.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
API = "https://you-zitsu.fandom.com/api.php"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/130"}

CHARACTERS = [
    "Kiyotaka Ayanokōji", "Suzune Horikita", "Kikyō Kushida", "Honami Ichinose",
    "Kei Karuizawa", "Arisu Sakayanagi", "Kakeru Ryūen", "Yōsuke Hirata",
    "Ken Sudō", "Rokusuke Kōenji", "Sae Chabashira", "Manabu Horikita",
    "Airi Sakura", "Haruka Hasebe", "Miyabi Nagumo",
]

TEACHER = {"Sae Chabashira"}


def clean(s: str) -> str:
    s = re.sub(r"thumb\|[^|]*\|", "", s)
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
    s = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("'''", "").replace("''", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main() -> int:
    rows = []
    for page in CHARACTERS:
        r = requests.get(
            API,
            params={"action": "parse", "page": page, "prop": "wikitext", "format": "json", "formatversion": "2"},
            headers=UA, timeout=40,
        ).json()
        wt = r.get("parse", {}).get("wikitext", "")
        m = re.search(r"\|\s*class\s*=\s*([^\n|]+)", wt)
        raw_cls = m.group(1).strip() if m else ""
        if page in TEACHER:
            cls = "Class 1-D's homeroom teacher"
        elif raw_cls:
            cls = "Class " + raw_cls
        else:
            cls = "a student at this school"

        first = page.split()[0]
        sec = re.search(r"==\s*Personality\s*==(.*?)(?:\n==|\Z)", wt, re.S)
        body = clean(sec.group(1) if sec else wt[:1500])
        body = re.sub(r"^" + re.escape(page.split()[0]) + r"\s+", "", body)
        sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", body) if len(x.split()) >= 5]
        desc = sentences[0] if sentences else "A student at this school."
        if desc[0].islower():
            desc = f"{first} {desc}"
        desc = re.sub(r"^If there was one sentence to describe \w+, it would be that ", "", desc)
        desc = re.sub(r"^At the start of the series, ", "", desc)
        if desc and desc[0].islower():
            desc = desc[0].upper() + desc[1:]
        desc = desc.strip()
        if desc and not desc.endswith((".", "!", "?")):
            desc += "."
        identity = f"{page}. {cls}. {desc}"

        for q in (
            f"Who is {first}?", f"Who is {page}?", f"Tell me about {first}.",
            f"Describe {first}.", f"What do you know about {first}?",
        ):
            rows.append({"user": q, "assistant": identity, "source": f"identity:{page}"})
        for q in (f"What class is {first} in?", f"What class is {page} in?"):
            rows.append({"user": q, "assistant": cls + ".", "source": f"identity:{page}"})
        print(f"{page} | {cls} | {desc[:90]}")
        time.sleep(0.2)

    out = PROC / "identity_qa.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} identity pairs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
