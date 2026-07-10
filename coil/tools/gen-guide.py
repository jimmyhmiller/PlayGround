#!/usr/bin/env python3
"""Generate selfhost/src/guide.coil from docs/LANGUAGE_GUIDE.md.

`coil guide` prints an embedded copy of the language guide. That copy lives in
selfhost/src/guide.coil as a string constant so the compiled binary is
self-contained (works from the global install, no repo needed). This script
keeps it in sync with the markdown — the markdown is the source of truth.

Run from the repo root after editing docs/LANGUAGE_GUIDE.md:
    python3 tools/gen-guide.py
then rebuild the compiler (selfhost/rebootstrap.sh) — and because main.coil is in
the gate corpus, regenerate the snapshot first:
    COIL_REF_BIN=./target/release/coil ./selfhost/oracle/snapshot-full.sh
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
md = open(os.path.join(ROOT, "docs/LANGUAGE_GUIDE.md")).read()
# Coil string literals need \ and " escaped; literal newlines are kept verbatim.
esc = md.replace("\\", "\\\\").replace('"', '\\"')
out = (
    "; selfhost/src/guide.coil — GENERATED from docs/LANGUAGE_GUIDE.md.\n"
    "; Do not edit by hand; regenerate with: python3 tools/gen-guide.py\n"
    "(module guide)\n\n"
    "(defn guide-text [] (-> (slice u8))\n  \"" + esc + "\")\n"
)
open(os.path.join(ROOT, "selfhost/src/guide.coil"), "w").write(out)
print(f"wrote selfhost/src/guide.coil ({len(out)} bytes) from docs/LANGUAGE_GUIDE.md")
