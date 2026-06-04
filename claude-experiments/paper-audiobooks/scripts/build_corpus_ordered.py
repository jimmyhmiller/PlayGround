#!/usr/bin/env python
"""Build the corpus, processing NO-OCR (text-layer) documents FIRST and
deferring scanned (OCR-needed) documents to a backlog that's drained last.

Why: ~most PDFs carry an embedded text layer and extract in ~seconds (no GPU).
A minority are scanned and need marker OCR — minutes to hours each. Processing
in bucket order interleaves them, so the slow scans block the fast bulk. This
builder front-loads all the fast docs so the bulk of the corpus lands quickly,
then grinds the scanned backlog at the end.

Flow (the simple streaming approach):
  Phase 1 — stream every doc key:
    download → probe text layer →
      has text layer  → extract fast (no GPU) + build record now
      scanned / EPUB / DJVU / no-text → append key to the OCR backlog, move on
  Phase 2 — drain the backlog:
    for each deferred key → download → marker (persistent server) → build record

Independent of the old `build_corpus_from_s3.py` and its `s3_manifest.json`
(which only snapshotted the `pdfs/` prefix). This one lists the WHOLE bucket
fresh each run so nothing is missed, and writes the SAME record/.md layout, so
the 230 docs already done are detected as fresh and skipped on resume.

⚠️  Records contain the ENTIRE text of each doc. Under output/corpus/
(git-ignored). Never publish.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paper_audiobooks import corpus as C
from paper_audiobooks.extract_client import ExtractClient, ExtractServerDead
from paper_audiobooks.fast_extract import (
    can_fast_extract,
    extract_structured_pdf,
)
from paper_audiobooks.pipeline import PipelinePaths, stage_extract_subproc
from paper_audiobooks.server import ensure_server

SUPPORTED_EXTS = {".pdf", ".epub", ".djvu"}
BACKLOG_NAME = "ocr_backlog.json"


def slug_key(key: str) -> str:
    rel = Path(key).with_suffix("")
    parts = [re.sub(r"[^A-Za-z0-9._-]+", "_", p).strip("_") or "_" for p in rel.parts]
    return str(Path(*parts))


def md_path_for(out_dir: Path, key: str) -> Path:
    return out_dir / "corpus_md" / (slug_key(key) + ".md")


def record_path_for(out_dir: Path, key: str) -> Path:
    return C.corpus_dir(out_dir) / (slug_key(key) + ".json")


def log(logf, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if logf:
        logf.write(line + "\n"); logf.flush()


def list_bucket_docs(bucket: str, prefix: str = "") -> list[str]:
    """Every PDF/EPUB/DJVU key in the bucket (optionally under a prefix)."""
    keys: list[str] = []
    token = None
    while True:
        cmd = ["aws", "s3api", "list-objects-v2", "--bucket", bucket,
               "--output", "json"]
        if prefix:
            cmd += ["--prefix", prefix]
        if token:
            cmd += ["--starting-token", token]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"list-objects-v2 failed: {res.stderr.strip()}")
        data = json.loads(res.stdout or "{}")
        for obj in data.get("Contents", []) or []:
            k = obj["Key"]
            if Path(k).suffix.lower() in SUPPORTED_EXTS:
                keys.append(k)
        token = data.get("NextToken")
        if not token:
            break
    return keys


def download(bucket: str, key: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        ["aws", "s3", "cp", f"s3://{bucket}/{key}", str(dest), "--quiet"],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"download failed: {res.stderr.strip()}")


def write_record_at(path: Path, record: C.CorpusRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(record), indent=2, ensure_ascii=False))
    tmp.replace(path)


def already_done(out_dir: Path, key: str) -> bool:
    """A record exists for this key (any schema) → skip. Resume is cheap; we
    don't re-verify content hash here since the build is idempotent per key."""
    return record_path_for(out_dir, key).exists()


def build_and_write(out_dir: Path, bucket: str, key: str, markdown: str,
                    local: Path, args, logf) -> C.CorpusRecord:
    record = C.build_record(
        local, markdown,
        content_chapters_only=not args.keep_all_sections,
        do_enrich=not args.no_enrich,
        enrich_timeout=args.enrich_timeout,
        llm_title_fallback=args.llm_title_fallback,
        llm_base_url=args.llm_base_url,
    )
    record.source_path = f"s3://{bucket}/{key}"
    record.source_name = Path(key).name
    record.stem = slug_key(key)
    write_record_at(record_path_for(out_dir, key), record)
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", default="jimmyhmiller-bucket")
    ap.add_argument("--prefix", default="", help="Limit to a prefix (default: whole bucket).")
    ap.add_argument("--out-dir", type=Path, default=Path("output"))
    ap.add_argument("--no-enrich", action="store_true")
    ap.add_argument("--keep-all-sections", action="store_true")
    ap.add_argument("--enrich-timeout", type=float, default=10.0)
    ap.add_argument("--llm-title-fallback", action="store_true")
    ap.add_argument("--llm-base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--no-auto-llm", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N this run.")
    ap.add_argument("--skip-backlog", action="store_true",
                    help="Phase 1 only (fast docs); leave the OCR backlog for a later run.")
    ap.add_argument("--backlog-only", action="store_true",
                    help="Skip phase 1; just drain the existing OCR backlog.")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    C.corpus_dir(out_dir).mkdir(parents=True, exist_ok=True)
    logf = open(C.corpus_dir(out_dir) / "s3_build.log", "a", buffering=1)
    backlog_path = C.corpus_dir(out_dir) / BACKLOG_NAME

    # llama-server only if LLM title fallback requested.
    llm_ctx = (
        ensure_server(args.llm_base_url, log_path=out_dir / "llama-server.log",
                      startup_timeout=600.0)
        if args.llm_title_fallback and not args.no_auto_llm
        else nullcontext(False)
    )

    # Persistent marker server is only needed for the backlog (scanned) phase.
    extract_log = open(C.corpus_dir(out_dir) / "s3_extract.log", "a", buffering=1)

    n_fast = n_deferred = n_skip = n_fail = 0
    processed = 0
    backlog: list[str] = []
    if backlog_path.exists():
        try:
            backlog = json.loads(backlog_path.read_text())
        except Exception:
            backlog = []

    def save_backlog():
        backlog_path.write_text(json.dumps(sorted(set(backlog)), indent=2))

    with llm_ctx as started:
        if args.llm_title_fallback:
            log(logf, f"llm-title-fallback on ({'started' if started else 'reusing'} server)")

        # ---------------- Phase 1: fast (no-OCR) docs first ----------------
        if not args.backlog_only:
            log(logf, f"[phase1] listing s3://{args.bucket}/{args.prefix or ''} ...")
            keys = list_bucket_docs(args.bucket, args.prefix)
            log(logf, f"[phase1] {len(keys)} document keys; streaming fast docs, deferring scans")

            for key in keys:
                if already_done(out_dir, key):
                    n_skip += 1
                    continue
                if args.limit is not None and processed >= args.limit:
                    log(logf, f"[phase1] hit --limit {args.limit}"); break
                try:
                    with tempfile.TemporaryDirectory() as td:
                        local = Path(td) / (slug_key(key).replace("/", "__") + Path(key).suffix)
                        download(args.bucket, key, local)
                        # Decide: fast path now, or defer to OCR backlog.
                        if can_fast_extract(local):
                            md_path = md_path_for(out_dir, key)
                            markdown = extract_structured_pdf(local)
                            if not markdown or not markdown.strip():
                                # Fast extract produced nothing — treat as scan.
                                backlog.append(key); save_backlog(); n_deferred += 1
                                log(logf, f"[defer] {key} (empty fast extract)")
                                continue
                            md_path.parent.mkdir(parents=True, exist_ok=True)
                            md_path.write_text(markdown)
                            rec = build_and_write(out_dir, args.bucket, key, markdown, local, args, logf)
                            n_fast += 1; processed += 1
                            log(logf, f"[fast] {record_path_for(out_dir,key).name}: "
                                      f"title={rec.title!r} ch={rec.n_chapters} chars={rec.total_chars}")
                        else:
                            backlog.append(key); save_backlog(); n_deferred += 1
                            log(logf, f"[defer] {key} (needs OCR)")
                except KeyboardInterrupt:
                    log(logf, "[phase1] interrupted; progress saved"); break
                except Exception as exc:
                    n_fail += 1
                    log(logf, f"[phase1] FAILED {key}: {type(exc).__name__}: {exc}")
                    continue

            save_backlog()
            log(logf, f"[phase1] done: {n_fast} fast, {n_deferred} deferred to OCR, "
                      f"{n_skip} already-done, {n_fail} failed. Backlog: {len(set(backlog))}")

        # ---------------- Phase 2: drain the OCR backlog ----------------
        if not args.skip_backlog:
            todo = [k for k in sorted(set(backlog)) if not already_done(out_dir, k)]
            log(logf, f"[phase2] draining OCR backlog: {len(todo)} scanned docs via marker")
            extractor = ExtractClient(ready_timeout=900, log_file=extract_log) if todo else None
            try:
                for key in todo:
                    if args.limit is not None and processed >= args.limit:
                        log(logf, f"[phase2] hit --limit {args.limit}"); break
                    try:
                        with tempfile.TemporaryDirectory() as td:
                            local = Path(td) / (slug_key(key).replace("/", "__") + Path(key).suffix)
                            download(args.bucket, key, local)
                            md_path = md_path_for(out_dir, key)
                            if md_path.exists():
                                markdown = md_path.read_text()
                            else:
                                try:
                                    markdown = extractor.extract(local, md_path)
                                except ExtractServerDead:
                                    extractor = ExtractClient(ready_timeout=900, log_file=extract_log)
                                    markdown = extractor.extract(local, md_path)
                            rec = build_and_write(out_dir, args.bucket, key, markdown, local, args, logf)
                            n_fast += 0; processed += 1
                            log(logf, f"[ocr] {record_path_for(out_dir,key).name}: "
                                      f"title={rec.title!r} ch={rec.n_chapters} chars={rec.total_chars}")
                    except KeyboardInterrupt:
                        log(logf, "[phase2] interrupted; progress saved"); break
                    except Exception as exc:
                        n_fail += 1
                        log(logf, f"[phase2] FAILED {key}: {type(exc).__name__}: {exc}")
                        continue
            finally:
                if extractor is not None:
                    extractor.close()

    extract_log.close()
    log(logf, f"DONE this run: {n_fast} fast, {n_deferred} newly-deferred, "
              f"{processed} processed total, {n_fail} failed.")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
