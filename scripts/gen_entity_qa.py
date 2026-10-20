#!/usr/bin/env python3
"""Generate entity-grounded Q&A from wiki character pages.

For each main character we fetch the wiki intro, then ask a local instruct model
for short first-person Q&A pairs in Ayanokoji's voice, grounded only in that
text. This is the data that teaches the tiny model *who is who*.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
API = "https://you-zitsu.fandom.com/api.php"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/130"}

CHARACTERS = [
    "Kiyotaka Ayanokōji", "Suzune Horikita", "Kikyō Kushida", "Honami Ichinose",
    "Kei Karuizawa", "Arisu Sakayanagi", "Kakeru Ryūen", "Yōsuke Hirata",
    "Ken Sudō", "Rokusuke Kōenji", "Sae Chabashira", "Manabu Horikita",
    "Airi Sakura", "Haruka Hasebe", "Ichika Amasawa", "Miyabi Nagumo",
    "Takuya Yagami", "Yūsuke Hirata",
]

SYSTEM = (
    "You write short Q&A pairs for a roleplay chatbot of Kiyotaka Ayanokoji. "
    "Use ONLY the supplied character description. Questions are what a classmate "
    "might ask. Answers are Kiyotaka's: first person, calm, blunt, analytical, "
    "1-2 short sentences, never cheerful, never mention being an AI. Output each "
    "pair as exactly two lines:\nQ: <question>\nA: <answer>\n"
    "Write 4 pairs."
)


def fetch(page: str) -> str:
    cache = RAW / "characters" / (re.sub(r"[^A-Za-z0-9]+", "_", page) + ".txt")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and cache.stat().st_size > 200:
        return cache.read_text(encoding="utf-8")
    r = requests.get(
        API,
        params={"action": "parse", "page": page, "prop": "wikitext", "format": "json", "formatversion": "2"},
        headers=UA,
        timeout=40,
    )
    wt = r.json().get("parse", {}).get("wikitext", "")
    wt = re.sub(r"<!--.*?-->", "", wt, flags=re.S)
    wt = re.sub(r"\{\{[^{}]*\}\}", "", wt)
    wt = re.sub(r"<ref[^>]*>.*?</ref>", "", wt, flags=re.S)
    wt = re.sub(r"<[^>]+>", "", wt)
    wt = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", wt)
    wt = wt.replace("'''", "").replace("''", "")
    wt = re.sub(r"^[|!{}=#*:;-].*$", "", wt, flags=re.M)
    wt = re.sub(r"\n{2,}", "\n", wt)
    wt = re.sub(r"[ \t]+", " ", wt).strip()
    cache.write_text(wt, encoding="utf-8")
    time.sleep(0.4)
    return wt


def clean(s: str) -> str:
    s = s.strip().strip('"').strip("'").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default=str(PROC / "entity_qa.jsonl"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0}
    ).eval()
    tpl = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "{prompt}"}],
        tokenize=False,
        add_generation_prompt=True,
    )

    rows = []
    seen_q = set()
    for page in CHARACTERS:
        text = fetch(page)
        desc = text[:1800]
        prompt = f"Character: {page}\nDescription:\n{desc}\n\nWrite 4 Q&A pairs:"
        enc = tok(tpl.replace("{prompt}", prompt), return_tensors="pt", truncation=True, max_length=2048).to(0)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=320, do_sample=True, temperature=0.8,
                top_p=0.95, repetition_penalty=1.05,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        raw = tok.decode(gen[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        q = None
        kept = 0
        for line in raw.splitlines():
            line = line.strip()
            m = re.match(r"^(?:Q|Question)\s*:\s*(.+)$", line, re.I)
            if m:
                q = clean(m.group(1))
                continue
            m = re.match(r"^(?:A|Answer)\s*:\s*(.+)$", line, re.I)
            if m and q:
                a = clean(m.group(1))
                if (
                    3 <= len(q.split()) <= 22
                    and 4 <= len(a.split()) <= 50
                    and q.lower() not in seen_q
                    and "as an ai" not in a.lower()
                    and not a.endswith("?")
                ):
                    seen_q.add(q.lower())
                    rows.append({"user": q, "assistant": a, "source": f"entity:{page}"})
                    kept += 1
                q = None
        print(f"{page}: {kept} pairs")
        time.sleep(0.2)

    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} entity pairs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
