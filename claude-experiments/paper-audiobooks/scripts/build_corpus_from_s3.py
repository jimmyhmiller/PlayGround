#!/usr/bin/env python
"""Build the full-text corpus from every document in an S3 prefix.

Downloads each PDF/EPUB/DJVU under an S3 prefix, runs the paper-audiobooks
extraction + chaptering + metadata-enrichment pipeline (NO audio), and writes
one full-content JSON record per document under `output/corpus/`, mirroring the
S3 key structure (so `pdfs/computer-philosophy/<hash>.pdf` ->
`output/corpus/pdfs/computer-philosophy/<hash>.json`).

Designed for a multi-day, strictly-serial, RESUMABLE run:
  - Lists the prefix once into a manifest (cached on disk).
  - Skips any key whose record is already fresh (same content + schema version).
  - Downloads to a temp file, extracts (marker GPU OCR), enriches, writes record,
    deletes the temp file. The extracted `.md` is cached under a key-mirrored
    path so re-runs skip re-extraction.
  - Runs ONE llama-server for the whole batch (for --llm-title-fallback).
  - Logs progress + failures to output/corpus/s3_build.log; a failed doc is
    recorded and skipped, never aborts the run.

⚠️  PRIVACY / COPYRIGHT: records contain the ENTIRE text of each document.
    They live under `output/corpus/` (git-ignored) and MUST NOT be published.

Usage:
    uv run python scripts/build_corpus_from_s3.py \
        --bucket jimmyhmiller-bucket --prefix pdfs/ \
        --llm-title-fallback

    # Resume is automatic — just re-run the same command.
    # Dry run (list what WOULD be processed, no downloads/GPU):
    uv run python scripts/build_corpus_from_s3.py --bucket ... --prefix ... --dry-run
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

# Make the package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paper_audiobooks import corpus as C
from paper_audiobooks.extract_client import ExtractClient, ExtractServerDead
from paper_audiobooks.pipeline import PipelinePaths, stage_extract_subproc
from paper_audiobooks.server import ensure_server


# A single long-lived marker extraction server, loaded once and reused for the
# whole run (amortizes the multi-minute model load that the per-doc subprocess
# paid every time). Restarted on death; falls back to per-doc extraction if it
# can't stay up.
class _PersistentExtractor:
    def __init__(self, log_file=None):
        self._log_file = log_file
        self._client: ExtractClient | None = None
        self.disabled = False  # set True after repeated failures -> use subprocess

    def _ensure(self):
        if self._client is None or not self._client.alive():
            self._client = ExtractClient(ready_timeout=900, log_file=self._log_file)

    def extract(self, local: Path, md_path: Path, max_pages: int | None) -> str:
        page_cap = _clamp_pages(local, max_pages)
        # The persistent server reuses md_path if it already exists? No — it
        # always re-extracts. The caller handles md caching, so only call this
        # on a cache miss.
        self._ensure()
        try:
            return self._client.extract(local, md_path, page_range=_page_range(page_cap))
        except ExtractServerDead:
            # Server died mid-doc. Try one restart for THIS doc.
            try:
                self._client = ExtractClient(ready_timeout=900, log_file=self._log_file)
                return self._client.extract(local, md_path, page_range=_page_range(page_cap))
            except Exception:
                # Persistent path is unreliable — disable and fall back.
                self.disabled = True
                raise

    def close(self):
        if self._client is not None:
            self._client.close()


def _clamp_pages(local: Path, max_pages: int | None) -> int | None:
    page_cap = max_pages
    if page_cap is not None and local.suffix.lower() == ".pdf":
        n = _pdf_page_count(local)
        if n is not None:
            page_cap = min(page_cap, n)
    return page_cap


def _page_range(page_cap: int | None) -> list[int] | None:
    return list(range(page_cap)) if page_cap else None


def _extract_one(local, md_path, args, extractor, logf):
    """Route one document to the cheapest extractor that produces good output:

      1. FAST (no GPU): text-layer PDFs that have an outline OR are short — font
         metadata recovers structure in milliseconds (no OCR). ~95% of docs.
      2. marker via the persistent server: scanned PDFs and long outline-less
         books that need the real layout/OCR model.
      3. per-doc marker subprocess: fallback if the persistent server died.

    `--no-fast-extract` forces everything to marker.
    """
    from paper_audiobooks.fast_extract import can_fast_extract, extract_structured_pdf

    page_cap = _clamp_pages(local, args.max_pages)

    if not args.no_fast_extract and can_fast_extract(local):
        try:
            md = extract_structured_pdf(local, page_range=_page_range(page_cap))
            if md and md.strip():
                md_path.parent.mkdir(parents=True, exist_ok=True)
                md_path.write_text(md)
                log(logf, "  [fast] structured text extract (no GPU)")
                return md
            # Empty result — fall through to marker.
            log(logf, "  [fast] empty result; falling back to marker")
        except Exception as exc:
            log(logf, f"  [fast] failed ({exc}); falling back to marker")

    # marker path (persistent server, then subprocess).
    if extractor is not None and not extractor.disabled:
        try:
            return extractor.extract(local, md_path, args.max_pages)
        except Exception as exc:
            log(logf, f"  persistent extractor failed ({exc}); using subprocess")
    return extract_via_subprocess(local, md_path, args.max_pages)


def _pdf_page_count(path: Path) -> int | None:
    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(str(path))
        try:
            return len(doc)
        finally:
            doc.close()
    except Exception:
        return None


def extract_via_subprocess(local: Path, md_path: Path, max_pages: int | None) -> str:
    """Extract using the same crash-isolated subprocess worker the audio
    pipeline uses (marker is unstable in-process on AMD ROCm). Clamps
    max_pages to the document's real length so a short PDF doesn't assert."""
    page_cap = max_pages
    if page_cap is not None and local.suffix.lower() == ".pdf":
        n = _pdf_page_count(local)
        if n is not None:
            page_cap = min(page_cap, n)
    paths = PipelinePaths(
        source=local, md=md_path, chapters=md_path, audio=md_path, log=md_path,
    )
    return stage_extract_subproc(paths, max_pages=page_cap)

# Formats our extractor understands. Everything else in the bucket
# (.sketch, .db, .doc, .docx, .html, ...) is skipped — extract.py only handles
# these three.
SUPPORTED_EXTS = {".pdf", ".epub", ".djvu"}


def slug_key(key: str) -> str:
    """Turn an S3 key into a filesystem-safe relative path that preserves the
    folder structure. `pdfs/ai/abc.pdf` -> `pdfs/ai/abc` (extension dropped;
    record/`.md` get their own suffixes)."""
    # Drop the extension; keep the directory structure.
    p = Path(key)
    rel = p.with_suffix("")
    # Sanitize each path component (S3 keys can contain spaces, etc.).
    parts = [re.sub(r"[^A-Za-z0-9._-]+", "_", part).strip("_") or "_" for part in rel.parts]
    return str(Path(*parts))


def md_path_for(out_dir: Path, key: str) -> Path:
    return out_dir / "corpus_md" / (slug_key(key) + ".md")


def record_path_for(out_dir: Path, key: str) -> Path:
    return C.corpus_dir(out_dir) / (slug_key(key) + ".json")


def list_prefix(bucket: str, prefix: str) -> list[tuple[str, int]]:
    """Return [(key, size), ...] for every object under the prefix, via the AWS CLI."""
    out: list[tuple[str, int]] = []
    token = None
    while True:
        cmd = [
            "aws", "s3api", "list-objects-v2",
            "--bucket", bucket, "--prefix", prefix,
            "--output", "json",
        ]
        if token:
            cmd += ["--starting-token", token]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"list-objects-v2 failed: {res.stderr.strip()}")
        data = json.loads(res.stdout or "{}")
        for obj in data.get("Contents", []) or []:
            out.append((obj["Key"], obj.get("Size", 0)))
        token = data.get("NextToken")
        if not token:
            break
    return out


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


def record_is_fresh(path: Path, markdown: str) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False
    if data.get("schema_version") != C.CORPUS_SCHEMA_VERSION:
        return False
    return data.get("content_sha256") == C._sha256(markdown)


def log(logf, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if logf:
        logf.write(line + "\n")
        logf.flush()


def rebuild_index(out_dir: Path) -> Path:
    """Regenerate index.jsonl over the nested corpus tree (key + record fields)."""
    cdir = C.corpus_dir(out_dir)
    idx = C.index_path(out_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    n = 0
    with idx.open("w") as f:
        for jf in sorted(cdir.rglob("*.json")):
            try:
                rec = C.CorpusRecord(**json.loads(jf.read_text()))
            except Exception:
                continue
            entry = rec.to_index_entry()
            # record relative path so the corpus is navigable by key.
            entry["record_path"] = str(jf.relative_to(cdir))
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            n += 1
    return idx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", default="jimmyhmiller-bucket")
    ap.add_argument("--prefix", default="pdfs/")
    ap.add_argument("--out-dir", type=Path, default=Path("output"))
    ap.add_argument("--llm-title-fallback", action="store_true",
                    help="Use the local llama.cpp server to read title/author "
                         "when the scrape looks bad and online lookup fails.")
    ap.add_argument("--llm-base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--no-auto-llm", action="store_true",
                    help="Don't auto-start llama-server (assume one is running).")
    ap.add_argument("--no-enrich", action="store_true",
                    help="Skip online metadata lookups.")
    ap.add_argument("--keep-all-sections", action="store_true")
    ap.add_argument("--no-persistent", action="store_true",
                    help="Disable the persistent marker server; use a fresh "
                         "subprocess per doc (slower, but maximal crash isolation).")
    ap.add_argument("--no-fast-extract", action="store_true",
                    help="Disable the fast no-GPU text-layer extractor; send "
                         "every PDF through marker (slower, max layout fidelity).")
    ap.add_argument("--enrich-timeout", type=float, default=10.0)
    ap.add_argument("--max-pages", type=int, default=None,
                    help="Cap marker extraction per doc (useful for a fast pilot).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N documents this run (for piloting).")
    ap.add_argument("--refresh-manifest", action="store_true",
                    help="Re-list S3 even if a cached manifest exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be processed; no downloads, no GPU.")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    C.corpus_dir(out_dir).mkdir(parents=True, exist_ok=True)

    # 1. Manifest (cached so we don't re-list a huge bucket each resume).
    manifest_path = C.corpus_dir(out_dir) / "s3_manifest.json"
    if manifest_path.exists() and not args.refresh_manifest:
        manifest = json.loads(manifest_path.read_text())
        print(f"[manifest] reusing {manifest_path} ({len(manifest)} objects)")
    else:
        print(f"[manifest] listing s3://{args.bucket}/{args.prefix} ...")
        objs = list_prefix(args.bucket, args.prefix)
        manifest = [{"key": k, "size": s} for k, s in objs]
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[manifest] {len(manifest)} objects -> {manifest_path}")

    # 2. Filter to supported document formats.
    docs = [m for m in manifest if Path(m["key"]).suffix.lower() in SUPPORTED_EXTS]
    skipped_fmt = len(manifest) - len(docs)
    print(f"[scope] {len(docs)} documents ({skipped_fmt} non-document objects skipped)")

    if args.dry_run:
        todo = [m for m in docs if not record_path_for(out_dir, m["key"]).exists()]
        print(f"[dry-run] {len(todo)} need processing, "
              f"{len(docs) - len(todo)} already have a record.")
        for m in todo[:30]:
            print(f"  TODO {m['key']}")
        if len(todo) > 30:
            print(f"  ... and {len(todo) - 30} more")
        return 0

    # 3. Optional llama-server for LLM title fallback.
    llm_ctx = (
        ensure_server(args.llm_base_url, log_path=out_dir / "llama-server.log",
                      startup_timeout=600.0)
        if args.llm_title_fallback and not args.no_auto_llm
        else nullcontext(False)
    )

    logf = open(C.corpus_dir(out_dir) / "s3_build.log", "a", buffering=1)
    n_done = n_skip = n_fail = 0
    t_start = time.monotonic()

    # Persistent marker server: load models once, reuse across all docs. This
    # is the big speedup vs. the per-doc subprocess (which reloaded marker every
    # time). Falls back to per-doc subprocess automatically if it can't stay up.
    extract_log = open(C.corpus_dir(out_dir) / "s3_extract.log", "a", buffering=1)
    extractor = None if args.no_persistent else _PersistentExtractor(log_file=extract_log)
    if extractor is not None:
        log(logf, "using persistent marker extraction server (models load once)")

    with llm_ctx as started:
        if args.llm_title_fallback:
            log(logf, f"llm-title-fallback on ({'started' if started else 'reusing'} "
                      f"server at {args.llm_base_url})")

        processed = 0
        for i, m in enumerate(docs, 1):
            key = m["key"]
            rec_path = record_path_for(out_dir, key)
            md_path = md_path_for(out_dir, key)

            # Fast skip: record already exists AND we have its .md to verify
            # freshness. If the record exists but no .md (record carried over),
            # trust the record's existence as "done".
            if rec_path.exists() and not md_path.exists():
                n_skip += 1
                continue

            if args.limit is not None and processed >= args.limit:
                log(logf, f"hit --limit {args.limit}; stopping for this run")
                break

            log(logf, f"=== {i}/{len(docs)} {key} ===")
            try:
                # The downloaded file must stay on disk through build_record:
                # PDF chaptering re-opens the source to read its outline
                # (split_by_pdf_toc). So keep the temp file alive for the whole
                # extract+build. Extraction runs in a crash-isolated subprocess
                # (marker is unstable in-process on AMD ROCm).
                with tempfile.TemporaryDirectory() as td:
                    local = Path(td) / (slug_key(key).replace("/", "__") + Path(key).suffix)
                    download(args.bucket, key, local)
                    # Resume: reuse cached .md if present (skip re-extraction).
                    if md_path.exists():
                        markdown = md_path.read_text()
                    else:
                        markdown = _extract_one(
                            local, md_path, args, extractor, logf,
                        )

                    if record_is_fresh(rec_path, markdown):
                        n_skip += 1
                        continue

                    record = C.build_record(
                        local,  # live file → PDF-outline chaptering works
                        markdown,
                        content_chapters_only=not args.keep_all_sections,
                        do_enrich=not args.no_enrich,
                        enrich_timeout=args.enrich_timeout,
                        llm_title_fallback=args.llm_title_fallback,
                        llm_base_url=args.llm_base_url,
                    )

                # Stamp S3 provenance (overwrites the temp-file paths).
                record.source_path = f"s3://{args.bucket}/{key}"
                record.source_name = Path(key).name
                record.stem = slug_key(key)
                write_record_at(rec_path, record)

                processed += 1
                n_done += 1
                rate = (time.monotonic() - t_start) / max(processed, 1)
                log(logf, f"wrote {rec_path.name}: title={record.title!r} "
                          f"[{record.title_source}] chapters={record.n_chapters} "
                          f"chars={record.total_chars} subjects={len(record.subjects)} "
                          f"| ~{rate:.0f}s/doc")
            except KeyboardInterrupt:
                log(logf, "interrupted by user; progress saved, safe to resume")
                break
            except Exception as exc:
                n_fail += 1
                log(logf, f"FAILED {key}: {type(exc).__name__}: {exc}")
                continue

    if extractor is not None:
        extractor.close()
    extract_log.close()
    idx = rebuild_index(out_dir)
    log(logf, f"DONE this run: {n_done} written, {n_skip} skipped, {n_fail} failed. "
              f"Index: {idx}")
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
