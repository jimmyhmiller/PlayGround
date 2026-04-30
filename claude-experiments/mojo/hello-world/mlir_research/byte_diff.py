#!/usr/bin/env python3
"""Byte-level diff of two MLIR bytecode files using difflib (proper LCS).

Shows the localized insert/replace edits with section context."""
import sys
from difflib import SequenceMatcher
sys.path.insert(0, ".")
from parse_mlir_bc import parse_file

def section_at(info, off):
    for s in info["sections"]:
        if s["payload_off"] <= off < s["payload_off"] + s["length"]:
            return s["name"], off - s["payload_off"]
    if off >= info["trailer_off"]:
        return "trailer", off - info["trailer_off"]
    return "header/gap", off

def main(a_path, b_path):
    a = parse_file(a_path); b = parse_file(b_path)
    A = open(a_path, "rb").read(); B = open(b_path, "rb").read()
    print(f"A: {a_path} ({len(A)} bytes)")
    print(f"B: {b_path} ({len(B)} bytes)")
    print(f"size diff: {len(B) - len(A):+d}")
    print()
    sm = SequenceMatcher(a=A, b=B, autojunk=False)
    edits = [(tag, i1, i2, j1, j2) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]
    print(f"=== {len(edits)} non-equal edits ===")
    print(f"  {'tag':<7}  {'a_off':>6}  {'section':<18}  {'sec_off':>7}  {'a_len':>5} {'b_len':>5}  context")
    for tag, i1, i2, j1, j2 in edits:
        sec, sec_off = section_at(a, i1)
        a_chunk = A[i1:i2]; b_chunk = B[j1:j2]
        a_hex = a_chunk.hex() if len(a_chunk) <= 16 else a_chunk[:16].hex() + "..."
        b_hex = b_chunk.hex() if len(b_chunk) <= 16 else b_chunk[:16].hex() + "..."
        print(f"  {tag:<7}  {i1:>6}  {sec:<18}  {sec_off:>7}  {i2-i1:>5} {j2-j1:>5}  A={a_hex}  B={b_hex}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
