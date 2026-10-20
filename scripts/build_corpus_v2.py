#!/usr/bin/env python3
"""Merge every collected source into one all-season corpus.

Sources
  * data/raw/subarchivist_cues.jsonl   S1-S3 per-cue speaker labels
  * data/raw/screenplay_lines.jsonl    S1 wiki screenplays (richer monologues)
  * data/raw/acdb_quotes.jsonl         S4 quotes + extras (optional)
  * data/processed/domain_extra.txt    wiki prose

Outputs
  data/processed/corpus_v2.txt
  data/processed/corpus_v2_{train,val,test}.txt
  data/processed/kiyo_lines_v2.jsonl   Ayanokoji lines with dialogue context
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

ALIASES = {
    "aya": "Ayanokoji", "koji": "Ayanokoji", "kiyo": "Ayanokoji", "kiyotaka": "Ayanokoji",
    "hori": "Horikita", "kushi": "Kushida", "cha": "Chabashira", "hira": "Hirata",
    "ike": "Ike", "ichi": "Ichinose", "sak": "Sakura", "ryu": "Ryuen", "sudo": "Sudo",
    "yama": "Yamauchi", "karu": "Karuizawa", "saka": "Sakayanagi", "ibu": "Ibuki",
    "koen": "Koenji", "shino": "Shinohara", "soto": "Sotomura", "ishi": "Ishizaki",
}
DROP = {"sign", "---", "?", "??", "ma", "man", "mana", "kat", "chi", "kan", "sen"}
KIYO = re.compile(r"ayanok|kiyotaka|kiyo", re.I)


def clean(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+of the Elite$", "", text)
    text = re.sub(r"^[A-Za-z ,.'-]{0,30}? of the Elite\s*$", "", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)  # cue markers
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip('"').strip("“”").strip()
    return text


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def normalise_speaker(speaker: str, season: int) -> str:
    s = re.sub(r"^\((?:Flashback|Monologue|Inner[^)]*)\)\s*", "", speaker).strip()
    low = s.lower()
    if season == 1 and low in ALIASES:
        return ALIASES[low]
    if low in {"ayanokoji", "ayanokouji", "ayanokōji"}:
        return "Ayanokoji"
    return s


def is_kiyo(speaker: str) -> bool:
    return bool(KIYO.search(speaker))


def main() -> int:
    # --- SubArchivist S1-S3 ------------------------------------------------
    cues = [json.loads(l) for l in open(RAW / "subarchivist_cues.jsonl", encoding="utf-8")]
    by_ep: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for c in cues:
        spk = normalise_speaker(c["speaker"], c["season"])
        if spk.lower() in DROP or spk.lower() == "sign":
            continue
        text = clean(c["text"])
        if len(text) < 2 or norm(text) in {"", "sign"}:
            continue
        by_ep[(c["season"], c["episode"])].append(
            {
                "season": c["season"], "episode": c["episode"], "time": c.get("time", ""),
                "speaker": spk, "text": text, "source": f"subarchivist_s{c['season']}",
            }
        )

    # --- S1 wiki screenplays (prefer for monologues) -----------------------
    sc = [json.loads(l) for l in open(RAW / "screenplay_lines.jsonl", encoding="utf-8")]
    sc = [r for r in sc if r.get("source") != "animetranscript"]
    for r in sc:
        if r["type"] != "dialogue":
            continue
        text = clean(r["text"])
        if len(text) < 2:
            continue
        by_ep[(1, r["episode"])].append(
            {
                "season": 1, "episode": r["episode"], "time": "",
                "speaker": normalise_speaker(r["speaker"], 1), "text": text,
                "source": "screenplay_s1", "monologue": bool(r.get("monologue")),
            }
        )

    # --- ACDB S4 quotes and extras ----------------------------------------
    acdb_path = RAW / "acdb_quotes.jsonl"
    if acdb_path.exists():
        for line in open(acdb_path, encoding="utf-8"):
            r = json.loads(line)
            text = clean(r["text"])
            if len(text) < 3:
                continue
            by_ep[(r["season"], r.get("episode", 0))].append(
                {
                    "season": r["season"], "episode": r.get("episode", 0), "time": "",
                    "speaker": "Ayanokoji", "text": text, "source": "acdb",
                }
            )

    # stable order, dedupe exact (speaker, text) repeats inside an episode
    for key, rows in by_ep.items():
        seen = set()
        out = []
        for r in rows:
            k = (r["speaker"], norm(r["text"]))
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
        by_ep[key] = out

    # --- corpus text + Ayanokoji lines with context ------------------------
    corpus_parts: list[str] = []
    kiyo_lines: list[dict] = []
    seen_kiyo: set[str] = set()
    for season in (1, 2, 3, 4):
        eps = sorted(e for (s, e) in by_ep if s == season)
        for ep in eps:
            rows = by_ep[(season, ep)]
            corpus_parts.append(f"\n## Classroom of the Elite — Season {season}, Episode {ep}\n")
            recent: list[dict] = []
            for r in rows:
                stamp = f"[{r['time']}] " if r["time"] else ""
                corpus_parts.append(f"{stamp}{r['speaker']}: {r['text']}\n")
                if is_kiyo(r["speaker"]):
                    key = norm(r["text"])
                    if key not in seen_kiyo and len(r["text"].split()) >= 2:
                        seen_kiyo.add(key)
                        kiyo_lines.append(
                            {
                                "season": season,
                                "episode": ep,
                                "source": r["source"],
                                "monologue": bool(r.get("monologue")),
                                "text": r["text"],
                                "context": [
                                    {"speaker": c["speaker"], "text": c["text"]} for c in recent[-3:]
                                ],
                            }
                        )
                    recent = []
                else:
                    recent.append(r)
                    if len(recent) > 4:
                        recent.pop(0)

    corpus = "".join(corpus_parts)
    (PROC / "corpus_v2.txt").write_text(corpus, encoding="utf-8")
    with (PROC / "kiyo_lines_v2.jsonl").open("w", encoding="utf-8") as fh:
        for r in kiyo_lines:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- splits -----------------------------------------------------------
    def split_for(season: int, ep: int) -> str:
        if season <= 2:
            return "train"
        if season == 3:
            return "train" if ep <= 6 else ("val" if ep <= 9 else "test")
        return "test"

    split_text: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for season in (1, 2, 3, 4):
        for ep in sorted(e for (s, e) in by_ep if s == season):
            target = split_for(season, ep)
            split_text[target].append(f"\n## Season {season}, Episode {ep}\n")
            for r in by_ep[(season, ep)]:
                split_text[target].append(f"{r['speaker']}: {r['text']}\n")
    for name, parts in split_text.items():
        (PROC / f"corpus_v2_{name}.txt").write_text("".join(parts), encoding="utf-8")

    print(f"corpus_v2: {len(corpus)/1e6:.2f} MB, {sum(len(v) for v in by_ep.values())} cues")
    print(f"kiyo_lines_v2: {len(kiyo_lines)} unique Ayanokoji lines")
    for name, parts in split_text.items():
        print(f"  {name}: {len(''.join(parts))/1e3:.0f} KB")
    import collections

    print(collections.Counter(r["season"] for r in kiyo_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
