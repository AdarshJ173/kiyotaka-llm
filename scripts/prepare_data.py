#!/usr/bin/env python3
"""Build the pretraining corpora, tokenizer and binary token files.

Outputs (all under data/processed/):
  cote_corpus.txt / cote_{train,val,test}.txt
  kiyo_lines.jsonl            Ayanokoji lines with dialogue context
  domain_extra.txt            CoTE wiki prose (world/character knowledge)
  tokenizer.json
  general_{train,val}.bin     TinyStories tokens
  cote_{train,val,test}.bin   in-domain tokens
  meta.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kiyotaka.tokenizer import TokenizerWrapper, train_tokenizer  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "kiyotaka-research-bot/0.1"}
VOCAB_SIZE = 8192
TRAIN_EPS = range(1, 11)
VAL_EP = 11
TEST_EP = 12
KIYO_NAMES = ("Kiyotaka Ayanok", "Ayanokoji", "Ayanokouji")


def clean_text(s: str) -> str:
    s = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("'''", "").replace("''", "").replace("&nbsp;", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_kiyo(speaker: str) -> bool:
    return any(n in speaker for n in KIYO_NAMES)


def load_rows() -> list[dict]:
    rows = [json.loads(l) for l in open(RAW / "screenplay_lines.jsonl", encoding="utf-8")]
    # The alt transcript duplicates S1E1; keep it only for cross-checking.
    rows = [r for r in rows if r.get("source") != "animetranscript"]
    clean = []
    seen = set()
    for r in rows:
        r["text"] = clean_text(r["text"])
        r["speaker"] = clean_text(r.get("speaker", ""))
        if not r["text"] or re.search(r"\[\[|\]\]|\{\{|\}\}|<nowiki>", r["text"]):
            continue
        key = (r["episode"], r["scene"], r["type"], r["speaker"], r["text"])
        if key in seen:
            continue
        seen.add(key)
        clean.append(r)
    return clean


def render_episode(rows: list[dict], episode: int) -> str:
    ep_rows = [r for r in rows if r["episode"] == episode]
    out: list[str] = [f"## Classroom of the Elite - Episode {episode}\n"]
    scene = None
    for r in ep_rows:
        if r["scene"] != scene:
            scene = r["scene"]
            out.append(f"\n### Scene {scene}\n")
        if r["type"] == "stage":
            out.append(f"* {r['text']}\n")
        elif r["type"] == "narration":
            out.append(f"{r['text']}\n")
        else:
            tag = " (monologue)" if r.get("monologue") else ""
            out.append(f"{r['speaker']}{tag}: {r['text']}\n")
    return "".join(out)


def build_kiyo_lines(rows: list[dict]) -> list[dict]:
    dialogue = [r for r in rows if r["type"] == "dialogue"]
    out = []
    for i, r in enumerate(dialogue):
        if not is_kiyo(r["speaker"]):
            continue
        prev = dialogue[max(0, i - 3) : i]
        out.append(
            {
                "episode": r["episode"],
                "scene": r["scene"],
                "speaker": r["speaker"],
                "monologue": bool(r.get("monologue")),
                "text": r["text"],
                "context": [
                    {"speaker": p["speaker"], "text": p["text"], "monologue": bool(p.get("monologue"))}
                    for p in prev
                ],
            }
        )
    return out


WIKI_API = "https://you-zitsu.fandom.com/api.php"


def wiki_pages() -> list[str]:
    pages = [
        "Yōkoso Jitsuryoku Shijō Shugi no Kyōshitsu e",
        "Kiyotaka Ayanokōji",
        "Kiyotaka Ayanokōji/Relationships",
        "Suzune Horikita",
        "Honami Ichinose",
        "Kei Karuizawa",
        "Arisu Sakayanagi",
        "Kikyō Kushida",
        "Kakeru Ryūen",
        "Yōsuke Hirata",
        "Sae Chabashira",
        "Manabu Horikita",
    ]
    for prefix in ("Light Novel",):
        r = requests.get(
            WIKI_API,
            params={
                "action": "query",
                "list": "allpages",
                "apprefix": prefix,
                "aplimit": 500,
                "format": "json",
                "formatversion": "2",
            },
            headers=UA,
            timeout=30,
        ).json()
        for p in r.get("query", {}).get("allpages", []):
            if p["title"].endswith("/Summary"):
                pages.append(p["title"])
    return pages


def scrape_domain() -> str:
    cache = RAW / "wiki_domain.txt"
    if cache.exists() and cache.stat().st_size > 10_000:
        return cache.read_text(encoding="utf-8")
    chunks: list[str] = []
    pages = wiki_pages()
    print(f"scraping {len(pages)} wiki pages")
    for i, page in enumerate(pages):
        try:
            r = requests.get(
                WIKI_API,
                params={
                    "action": "parse",
                    "page": page,
                    "prop": "wikitext",
                    "format": "json",
                    "formatversion": "2",
                },
                headers=UA,
                timeout=30,
            )
            wt = r.json().get("parse", {}).get("wikitext", "")
        except Exception as exc:  # noqa: BLE001
            print(f"  WARN {page}: {exc}", file=sys.stderr)
            continue
        if len(wt) < 500:
            continue
        # Drop templates, tables, files, refs and headings; keep prose.
        wt = re.sub(r"<!--.*?-->", "", wt, flags=re.S)
        wt = re.sub(r"\{\{[^{}]*\}\}", "", wt)
        wt = re.sub(r"\[\[File:[^\]]*\]\]", "", wt)
        wt = re.sub(r"<ref[^>]*>.*?</ref>", "", wt, flags=re.S)
        wt = re.sub(r"<[^>]+>", "", wt)
        wt = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", wt)
        wt = wt.replace("'''", "").replace("''", "")
        wt = re.sub(r"^[|!{}=#*:;-].*$", "", wt, flags=re.M)
        wt = re.sub(r"\n{3,}", "\n\n", wt)
        wt = re.sub(r"[ \t]+", " ", wt).strip()
        if len(wt) > 400:
            chunks.append(f"### {page}\n{wt}\n")
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(pages)}")
    text = "\n\n".join(chunks)
    cache.write_text(text, encoding="utf-8")
    return text


def tinystories_texts(parquet_path: Path):
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=4096, columns=["text"]):
        for t in batch.column("text").to_pylist():
            yield t


def encode_file(texts, tok: TokenizerWrapper, out_path: Path, eos_id: int) -> int:
    total = 0
    with open(out_path, "wb") as fh:
        for t in texts:
            ids = tok.encode(t, add_special_tokens=False) + [eos_id]
            if not ids:
                continue
            fh.write(np.asarray(ids, dtype=np.uint16).tobytes())
            total += len(ids)
    return total


def main() -> int:
    rows = load_rows()
    print(f"clean rows: {len(rows)}")
    episodes = {ep: render_episode(rows, ep) for ep in range(1, 13)}
    corpus = "\n".join(episodes[ep] for ep in range(1, 13))
    (OUT / "cote_corpus.txt").write_text(corpus, encoding="utf-8")
    for name, eps in (("train", TRAIN_EPS), ("val", [VAL_EP]), ("test", [TEST_EP])):
        (OUT / f"cote_{name}.txt").write_text("\n".join(episodes[e] for e in eps), encoding="utf-8")

    kiyo = build_kiyo_lines(rows)
    with (OUT / "kiyo_lines.jsonl").open("w", encoding="utf-8") as fh:
        for r in kiyo:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Ayanokoji lines: {len(kiyo)} ({sum(r['monologue'] for r in kiyo)} monologues)")

    domain = scrape_domain()
    (OUT / "domain_extra.txt").write_text(domain, encoding="utf-8")
    print(f"domain prose: {len(domain)/1e6:.2f} MB")

    ts_files = sorted((RAW / "tinystories").rglob("train-*.parquet"))
    val_files = sorted((RAW / "tinystories").rglob("validation-*.parquet"))
    if not ts_files:
        raise SystemExit("TinyStories parquet missing")
    print(f"TinyStories shards: train={len(ts_files)} val={len(val_files)}")

    # --- tokenizer -------------------------------------------------------
    sample = []
    for p in ts_files:
        for i, t in enumerate(tinystories_texts(p)):
            sample.append(t)
            if i >= 12_000:
                break
    sample.append(corpus)
    sample.append(domain)
    tok_path = OUT / "tokenizer.json"
    print(f"training BPE vocab={VOCAB_SIZE} on {sum(map(len, sample))/1e6:.1f} MB")
    train_tokenizer(iter(sample), VOCAB_SIZE, tok_path)
    tok = TokenizerWrapper(tok_path)
    print(f"tokenizer vocab: {tok.vocab_size}")

    # --- tokenise --------------------------------------------------------
    eos = tok.end_id
    n = encode_file(
        (t for p in ts_files for t in tinystories_texts(p)), tok, OUT / "general_train.bin", eos
    )
    print(f"general_train tokens: {n/1e6:.1f}M")
    if val_files:
        nv = encode_file(
            (t for p in val_files for t in tinystories_texts(p)), tok, OUT / "general_val.bin", eos
        )
        print(f"general_val tokens: {nv/1e6:.1f}M")

    for name in ("train", "val", "test"):
        text = (OUT / f"cote_{name}.txt").read_text(encoding="utf-8")
        n = encode_file([text], tok, OUT / f"cote_{name}.bin", eos)
        print(f"cote_{name} tokens: {n:,}")

    dtext = domain + "\n" + corpus
    nd = encode_file([dtext], tok, OUT / "domain_all.bin", eos)
    print(f"domain_all tokens: {nd:,}")

    stats = {
        "vocab_size": tok.vocab_size,
        "cote_lines": len(rows),
        "kiyo_lines": len(kiyo),
        "kiyo_words": sum(len(r["text"].split()) for r in kiyo),
        "general_train_tokens": int(n),
        "domain_chars": len(domain),
        "train_eps": list(TRAIN_EPS),
        "val_ep": VAL_EP,
        "test_ep": TEST_EP,
    }
    (OUT / "meta.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
