#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Find manifest entries whose `local_path` points to a non-PDF (usually HTML
saved with a .pdf extension), mark them as broken (clear local_path, set
error), and delete the bad files. Run before re-running wayback_recover.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MANIFEST = Path.home() / "Documents" / "pel-papers" / "manifest.json"


def is_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"%PDF"
    except Exception:
        return False


def main() -> int:
    entries = json.loads(MANIFEST.read_text())
    bad = 0
    for e in entries:
        lp = e.get("local_path")
        if not lp:
            continue
        p = Path(lp)
        if not p.exists():
            # Manifest references a missing file — mark broken so wayback retries.
            e["local_path"] = None
            e["bytes"] = None
            e["sha256"] = None
            e["error"] = "file missing on disk"
            bad += 1
            continue
        if not is_pdf(p):
            print(f"NOT A PDF: {lp} ({p.stat().st_size} bytes)")
            p.unlink()
            e["local_path"] = None
            e["bytes"] = None
            e["sha256"] = None
            e["error"] = "not a PDF (HTML or other content saved as .pdf)"
            # Clear any prior wayback attempt — let it retry from scratch.
            e.pop("wayback_timestamp", None)
            e.pop("wayback_error", None)
            bad += 1
    print(f"\nmarked {bad} entries as broken")
    MANIFEST.write_text(json.dumps(entries, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
