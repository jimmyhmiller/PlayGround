"""Subprocess worker for marker PDF extraction.

Reads JSON {"pdf_path": str, "out_path": str} on stdin, writes the extracted
markdown to out_path, prints "ok" on success.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def main() -> int:
    try:
        req = json.loads(sys.stdin.read())
        from paper_audiobooks.extract import extract_markdown

        markdown = extract_markdown(Path(req["source_path"]), page_range=req.get("page_range"))
        out = Path(req["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown)
        print("ok")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
