#!/usr/bin/env python3
"""Generate src/compiler/guide.coil from docs/reference/LANGUAGE_GUIDE.md.

`coil guide` prints an embedded copy of the language guide. That copy lives in
src/compiler/guide.coil as a string constant so the compiled binary is
self-contained (works from the global install, no repo needed). This script
keeps it in sync with the markdown — the markdown is the source of truth.

Run from the repo root after editing docs/reference/LANGUAGE_GUIDE.md:
    python3 scripts/docs/gen-guide.py
then rebuild the compiler (scripts/compiler/rebootstrap.sh) — and because main.coil is in
the gate corpus, regenerate the snapshot first:
    python3 scripts/oracle.py snapshot full --compiler build/bin/coil
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
md = open(os.path.join(ROOT, "docs/reference/LANGUAGE_GUIDE.md")).read()
# Coil string literals need \ and " escaped; literal newlines are kept verbatim.
esc = md.replace("\\", "\\\\").replace('"', '\\"')
out = (
    "; src/compiler/guide.coil — GENERATED from docs/reference/LANGUAGE_GUIDE.md.\n"
    "; Do not edit by hand; regenerate with: python3 scripts/docs/gen-guide.py\n"
    "(module guide)\n\n"
    "(defn guide-text [] (-> (slice u8))\n  \"" + esc + "\")\n"
)
open(os.path.join(ROOT, "src/compiler/guide.coil"), "w").write(out)
print(f"wrote src/compiler/guide.coil ({len(out)} bytes) from docs/reference/LANGUAGE_GUIDE.md")
