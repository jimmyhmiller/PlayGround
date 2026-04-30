#!/usr/bin/env python3
"""Extract the embedded AMDGCN ELF kernel from a Mojo cache file's textual MLIR.

The transform-stage MLIR contains a `kgen.param.constant: string = "..."` whose
value is the entire fully-compiled gfx1151 ELF binary. We pull it out so it
can be disassembled with rocdl-objdump or analyzed as a real ELF.
"""
import re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dump_mlir import parse_bytecode_to_mlir

def unescape_mlir_string(s):
    """Decode an MLIR string literal: handles \\xx hex escapes and \\\\ etc."""
    result = bytearray()
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            c = s[i+1]
            # MLIR uses \xx hex escapes for non-printable bytes — check this FIRST
            if len(s) >= i + 3 and all(d in "0123456789abcdefABCDEF" for d in s[i+1:i+3]):
                result.append(int(s[i+1:i+3], 16)); i += 3
            elif c == "\\": result.append(0x5c); i += 2
            elif c == '"': result.append(0x22); i += 2
            elif c == "n": result.append(0x0a); i += 2
            elif c == "t": result.append(0x09); i += 2
            elif c == "r": result.append(0x0d); i += 2
            else: result.append(ord(s[i])); i += 1
        else:
            result.append(ord(s[i])); i += 1
    return bytes(result)

def extract_elf(path):
    text, _, _ = parse_bytecode_to_mlir(path)
    # find string constants whose first 4 bytes are the ELF magic
    # MLIR encodes \x7F as \7F
    pattern = re.compile(r'kgen\.param\.constant:\s*string\s*=\s*<"((?:[^"\\]|\\.)*?)">')
    found = []
    for m in pattern.finditer(text):
        raw = m.group(1)
        decoded = unescape_mlir_string(raw)
        if decoded.startswith(b"\x7fELF"):
            found.append(decoded)
    return found

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    for path in sys.argv[1:]:
        elfs = extract_elf(path)
        print(f"# {path}: found {len(elfs)} ELF blob(s)")
        for i, elf in enumerate(elfs):
            out = Path(path).with_suffix(f".kernel{i}.elf")
            out.write_bytes(elf)
            print(f"  → {out} ({len(elf)} bytes)")
