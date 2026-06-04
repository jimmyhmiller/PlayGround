"""Persistent (long-lived) marker extraction worker.

The per-document subprocess worker (`extract_worker.py`) reloads marker's full
model stack — `create_model_dict()` — on every invocation. For small papers
that startup dominates wall time (minutes of model load to OCR a 10-page PDF).

This server loads the model dict ONCE, then processes many documents over a
simple line-delimited JSON protocol on stdin/stdout, so the expensive load is
amortized across the whole corpus run.

Protocol (one JSON object per line, request and response):
  request:  {"id": <int>, "source_path": "...", "out_path": "...",
             "page_range": [0,1,...] | null}
  response: {"id": <int>, "ok": true,  "out_path": "..."}
        or  {"id": <int>, "ok": false, "error": "..."}

A line {"cmd": "shutdown"} exits cleanly.

WHY a long-lived process is OK here when in-process marker is banned:
  The project rule is "no marker in the *audio pipeline* process" and "no
  CONCURRENT marker instances" — those crash on AMD ROCm. A SINGLE, dedicated,
  long-lived marker process that does nothing but extract, one doc at a time, is
  a different shape. If it proves unstable over a long run, the caller falls
  back to the per-doc subprocess worker. The caller also restarts this server
  if it dies, so a crash costs one doc, not the run.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

# Same HIP allocator tuning as the per-doc worker — Strix Halo unified memory
# fragments badly otherwise.
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _make_response(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False)


def main() -> int:
    # Load the heavy model dict ONCE. Everything else reuses it.
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
        artifact_dict = create_model_dict()
    except Exception:
        # Couldn't even start — report on stderr and exit non-zero so the
        # caller falls back to per-doc extraction.
        traceback.print_exc()
        return 1

    # Signal readiness so the caller knows the (slow) model load is done.
    print(_make_response({"ready": True}), flush=True)

    def extract_pdf(path: Path, page_range: list[int] | None) -> str:
        # Fresh converter per doc (config/page_range is per-doc) but the model
        # dict — the expensive part — is shared.
        config = {"page_range": page_range} if page_range else None
        converter = PdfConverter(artifact_dict=artifact_dict, config=config)
        rendered = converter(str(path))
        text, _, _ = text_from_rendered(rendered)
        return text

    def extract_any(source: Path, page_range: list[int] | None) -> str:
        suffix = source.suffix.lower()
        if suffix == ".pdf":
            return extract_pdf(source, page_range)
        if suffix == ".epub":
            # EPUB/DJVU don't use the marker model dict; defer to the normal
            # extract path (cheap, no model reload concern).
            from .extract import _extract_epub
            return _extract_epub(source, chapter_range=page_range)
        if suffix == ".djvu":
            from .extract import _extract_djvu
            return _extract_djvu(source, page_range=page_range)
        raise ValueError(f"unsupported source format: {source.suffix}")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            print(_make_response({"ok": False, "error": "bad json"}), flush=True)
            continue

        if req.get("cmd") == "shutdown":
            return 0

        rid = req.get("id")
        try:
            source = Path(req["source_path"])
            out = Path(req["out_path"])
            page_range = req.get("page_range")
            markdown = extract_any(source, page_range)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(markdown)
            print(_make_response({"id": rid, "ok": True, "out_path": str(out)}),
                  flush=True)
        except Exception as exc:
            # Print the traceback to stderr (visible/logged) and a compact error
            # back to the caller. Do NOT exit — keep serving the next doc.
            traceback.print_exc()
            print(_make_response({"id": rid, "ok": False,
                                  "error": f"{type(exc).__name__}: {exc}"}),
                  flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
