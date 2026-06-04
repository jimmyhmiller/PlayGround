#!/usr/bin/env python
"""Retry the 2 corpus docs that failed during the S3 build.

Both failed because their S3 keys have ~200-char filenames; the build script
named the local temp download by the slugified key, which exceeded the
filesystem's 255-byte filename limit, so `aws s3 cp` couldn't create the file.

Fix: download each to a SHORT temp name, then extract + build the record under
the correct slug path (so it lands in the corpus exactly where it belongs).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paper_audiobooks import corpus as C
from paper_audiobooks.fast_extract import can_fast_extract, extract_structured_pdf
from paper_audiobooks.pipeline import PipelinePaths, stage_extract_subproc

BUCKET = "jimmyhmiller-bucket"
KEYS = [
    "pdfs/philosophy-unsorted/Philosophy/Faith and Philosophy/Inquiring About God_ Selected Essays, Volume, 1 by Nicholas Wolterstorff, edited by Terence Cuneo; and Practices of Belief_ Selected Essays, Volume 2, by Nicholas Wolterstorff, edited by Terence Cuneo.pdf",
    "pdfs/philosophy-unsorted/Philosophy/Faith and Philosophy/The God of Metaphysics_ Being a Study of the Metaphysics and Religious Doctrines of Spinoza, Hegel, Kierkegaard, T. H. Green, Bernard Bosanquet, Josiah Royce, A. N. Whitehead, Charles Hartshorne, and Concluding with a Defence of Pantheistic Idealism.pdf",
]
OUT_DIR = Path("output")


def slug_key(key: str) -> str:
    rel = Path(key).with_suffix("")
    parts = [re.sub(r"[^A-Za-z0-9._-]+", "_", p).strip("_") or "_" for p in rel.parts]
    return str(Path(*parts))


def record_path_for(key: str) -> Path:
    return C.corpus_dir(OUT_DIR) / (slug_key(key) + ".json")


def md_path_for(key: str) -> Path:
    # Use a short hash-based md name to also avoid the long-name problem here.
    import hashlib
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return OUT_DIR / "corpus_md" / "fixed" / (h + ".md")


for key in KEYS:
    rec_path = record_path_for(key)
    if rec_path.exists():
        print(f"[skip] already have record: {rec_path.name}")
        continue
    print(f"[fix] {key.split('/')[-1][:70]}...")
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / "doc.pdf"  # SHORT name — the whole point
        r = subprocess.run(
            ["aws", "s3", "cp", f"s3://{BUCKET}/{key}", str(local), "--quiet"],
            capture_output=True, text=True, env={**__import__("os").environ, "AWS_REGION": "us-east-2"},
        )
        if r.returncode != 0:
            print(f"  download FAILED: {r.stderr.strip()}")
            continue
        md_path = md_path_for(key)
        # Fast path if it has a text layer; else marker OCR.
        if can_fast_extract(local):
            markdown = extract_structured_pdf(local)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(markdown)
            print("  [fast] structured text extract")
        else:
            paths = PipelinePaths(source=local, md=md_path, chapters=md_path, audio=md_path, log=md_path)
            markdown = stage_extract_subproc(paths)
            print("  [marker] OCR extract")

        record = C.build_record(
            local, markdown,
            content_chapters_only=True, do_enrich=True, enrich_timeout=10.0,
            llm_title_fallback=False,
        )
        record.source_path = f"s3://{BUCKET}/{key}"
        record.source_name = Path(key).name
        record.stem = slug_key(key)
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = rec_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(record), indent=2, ensure_ascii=False))
        tmp.replace(rec_path)
        print(f"  wrote {rec_path.name}: title={record.title!r} ch={record.n_chapters} chars={record.total_chars}")

print("done")
