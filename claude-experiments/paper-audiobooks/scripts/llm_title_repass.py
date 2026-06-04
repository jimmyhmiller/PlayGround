"""Targeted LLM title/author re-pass over existing corpus records.

For records whose title is missing or looks bad (and that weren't resolved by
online enrichment), ask the local llama.cpp server to read the real title and
author from the document's opening text. Updates the record in place; sets
title_source='llm' when the LLM supplies a title.

Reads the document text straight from the record's own `chapters` (full text is
stored there) — no re-download, no re-extraction. Idempotent: re-running only
touches records that still have a bad title.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paper_audiobooks.metadata_lookup import (
    llm_extract_title_author,
    looks_like_bad_title,
)
from paper_audiobooks.server import ensure_server

OUT_DIR = Path("output")
CORPUS_DIR = OUT_DIR / "corpus"
LLM_BASE = "http://127.0.0.1:8080"


def doc_head_text(record: dict, limit: int = 6000) -> str:
    """Reconstruct the opening text of the doc from its stored chapters."""
    parts = []
    total = 0
    for ch in record.get("chapters", []):
        body = ch.get("body", "")
        parts.append(body)
        total += len(body)
        if total >= limit:
            break
    return "\n".join(parts)[:limit]


def candidates() -> list[Path]:
    out = []
    for jf in CORPUS_DIR.rglob("*.json"):
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue
        if "title_source" not in d:
            continue
        if d.get("title_source") == "enriched":
            continue
        if not looks_like_bad_title(d.get("title")):
            continue
        out.append(jf)
    return out


def main() -> int:
    cands = candidates()
    print(f"[repass] {len(cands)} records with bad/None titles to re-check via LLM", flush=True)
    if not cands:
        return 0

    fixed = 0
    unchanged = 0
    failed = 0
    with ensure_server(LLM_BASE, log_path=OUT_DIR / "llama-server.log", startup_timeout=600.0) as started:
        print(f"[repass] llama-server {'started' if started else 'reused'}", flush=True)
        for i, jf in enumerate(cands, 1):
            try:
                d = json.loads(jf.read_text())
            except Exception:
                failed += 1
                continue
            head = doc_head_text(d)
            if not head.strip():
                unchanged += 1
                continue
            title, authors = llm_extract_title_author(head, base_url=LLM_BASE)
            if title:
                d["title"] = title
                d["title_source"] = "llm"
                if authors and not d.get("authors"):
                    d["authors"] = authors
                tmp = jf.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False))
                tmp.replace(jf)
                fixed += 1
                if i % 10 == 0 or fixed <= 20:
                    print(f"[repass] {i}/{len(cands)} {jf.stem[:40]}: -> {title[:60]!r}", flush=True)
            else:
                unchanged += 1
    print(f"[repass] done: {fixed} titles fixed, {unchanged} left unchanged, {failed} errors", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
