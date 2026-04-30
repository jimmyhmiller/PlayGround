#!/usr/bin/env python3
"""Parser for Mojo's `.mojo_cache` bytecode files.

These are MLIR bytecode (producer "MLIR23.0.0git", version 6) wrapped in a
Modular-specific envelope:

    [ MLIR bytecode (sections) ]
    [ Modular diagnostics trailer ]
    [ 16-byte content hash ]

See FINDINGS.md for the format details we've reverse-engineered so far.

Usage:
    parse_mlir_bc.py <file>                   # show top-level structure
    parse_mlir_bc.py <file> --strings         # + dump string section
    parse_mlir_bc.py <file> --ir              # + hex-dump op tree (section 8)
    parse_mlir_bc.py <file> --decode-ir       # + heuristic op-name decode of section 8
    parse_mlir_bc.py <a> <b> --diff           # section-size diff between two files
"""
import sys, os, struct
from pathlib import Path

# ---------------------------------------------------------------- magic bytes
MAGIC = b"ML\xefR"

SECTION_NAMES = {
    0: "String",
    1: "Dialect",          # format diverges from upstream — see FINDINGS.md
    2: "AttrTypeOffsets",
    3: "Attribute",
    4: "Type",
    5: "Resource",
    6: "IR (vestigial)",   # 1 byte in MLIR23.0.0git — ops moved to id=8
    8: "OpTree",           # Modular-specific section ID
}

# ---------------------------------------------------------------- varint
def read_varint(buf, off):
    """MLIR's 1-9 byte varint. byte_count = trailing_zeros(first_byte)+1, capped 9."""
    first = buf[off]
    if first == 0:
        return int.from_bytes(buf[off+1:off+9], "little"), off + 9
    extra = 0
    while not (first >> extra) & 1:
        extra += 1
    n = extra + 1
    raw = int.from_bytes(buf[off:off+n], "little")
    return raw >> n, off + n

def read_cstring(buf, off):
    end = buf.index(0, off)
    return buf[off:end].decode("utf-8", errors="replace"), end + 1

# ---------------------------------------------------------------- header + sections
def parse_header(buf):
    if buf[:4] != MAGIC:
        raise ValueError("not MLIR bytecode")
    off = 4
    version, off = read_varint(buf, off)
    producer, off = read_cstring(buf, off)
    return version, producer, off

def parse_sections(buf, sections_start):
    """Walk sections starting at `sections_start`. Returns (sections, end_offset)
    where end_offset is where the Modular trailer begins."""
    sections = []
    seen = set()
    off = sections_start
    while off < len(buf):
        if off + 1 >= len(buf):
            break
        section_byte = buf[off]
        sid = section_byte & 0x7F
        has_align = bool(section_byte & 0x80)
        try:
            length, after_len = read_varint(buf, off + 1)
        except IndexError:
            break
        if has_align:
            try:
                _, after_len = read_varint(buf, after_len)
            except IndexError:
                break
        if length > len(buf) - after_len:
            break
        if sid in seen:
            # Likely a phantom — section IDs should appear at most once
            break
        seen.add(sid)
        sections.append({
            "id": sid,
            "name": SECTION_NAMES.get(sid, f"?({sid})"),
            "header_off": off,
            "payload_off": after_len,
            "length": length,
            "has_align": has_align,
            "payload": buf[after_len:after_len+length],
        })
        off = after_len + length
    return sections, off

# ---------------------------------------------------------------- string section
def parse_strings(payload):
    """numStrings, then numStrings varint lengths, then string bytes packed
    at the END of the section (read backward, so strings[0] is the LAST byte
    range and strings[N-1] is at the start of the data block)."""
    p = 0
    n, p = read_varint(payload, p)
    if n > 100000:
        raise ValueError(f"implausible string count {n}")
    lengths = []
    for _ in range(n):
        ln, p = read_varint(payload, p)
        lengths.append(ln)
    end = len(payload)
    strings = []
    for ln in lengths:
        end -= ln
        s = payload[end:end+ln]
        strings.append(s.rstrip(b"\x00").decode("utf-8", errors="replace"))
    return strings

# ---------------------------------------------------------------- pretty print
def hexdump(data, indent="  "):
    for off in range(0, len(data), 16):
        chunk = data[off:off+16]
        h = " ".join(f"{b:02x}" for b in chunk).ljust(48)
        a = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        print(f"{indent}{off:4x}  {h}  {a}")

# ---------------------------------------------------------------- categorization
KGEN_OP_NAMES = set()
def kgen_op_names():
    global KGEN_OP_NAMES
    if KGEN_OP_NAMES:
        return KGEN_OP_NAMES
    pyi = Path(__file__).parent / "dialects" / "kgen.pyi"
    if not pyi.exists():
        return set()
    import re
    out = set()
    for m in re.finditer(r"^class ([A-Z][A-Za-z]+)Op\b", pyi.read_text(), re.MULTILINE):
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", m.group(1)).lower()
        out.add(snake)
    KGEN_OP_NAMES = out
    return out

OP_NAME_PLAIN = {
    "main", "func", "return", "load", "store", "constant", "cast",
    "add", "sub", "mul", "div", "floordiv",
    "cmp", "and", "or", "xor", "neg", "abs", "select",
    "fence", "yield", "if", "loop", "break", "continue",
    "shrs", "shrl", "shl", "maxs", "maxu", "mins", "minu",
    "external_call", "create_closure", "stack_allocation",
    "aligned_free", "offset",
}
OP_NAME_PREFIXES = ("simd.", "atomic.", "stack_alloc.", "pointer.", "union.",
                    "string.", "try", "kgen.", "M.", "cast_", "pointer_to_index")

def categorize_string(s):
    if not s: return "empty"
    if s.startswith("KGEN_CompilerRT_"): return "rt-symbol"
    if s.startswith("std::"): return "mangled-stdlib"
    if s.startswith("oss/modular/") or s.endswith(".mojo"): return "source-path"
    if (s.startswith("apple-") or s.startswith("arm64-") or s.startswith("metal:")
        or "+aes," in s or s.startswith("e-m:") or s == "pic"):
        return "target-meta"
    if s.startswith("__"): return "compiler-flag"
    if s.startswith("_loop_") or s.startswith("try") or s.startswith("main_closure_"):
        return "synth-label"
    if s in OP_NAME_PLAIN: return "op-name"
    if any(s.startswith(p) for p in OP_NAME_PREFIXES): return "op-name"
    if s in kgen_op_names(): return "op-name"
    if len(s) <= 32: return "short"
    return "other"

# ---------------------------------------------------------------- top-level
def parse_file(path):
    buf = open(path, "rb").read()
    version, producer, sections_start = parse_header(buf)
    sections, end_of_sections = parse_sections(buf, sections_start)
    return {
        "path": path,
        "size": len(buf),
        "version": version,
        "producer": producer,
        "sections": sections,
        "trailer_off": end_of_sections,
        "trailer": buf[end_of_sections:],
    }

def print_structure(info):
    print(f"file: {info['path']} ({info['size']} bytes)")
    print(f"  magic    : ML\\xefR")
    print(f"  version  : {info['version']}")
    print(f"  producer : {info['producer']!r}")
    print()
    print(f"  {'off':>6}  {'id':>2}  {'name':<18}  {'len':>8}  align?")
    for s in info["sections"]:
        align = "yes" if s["has_align"] else "no"
        print(f"  {s['header_off']:>6}  {s['id']:>2}  {s['name']:<18}  {s['length']:>8}  {align}")
    if info["trailer"]:
        print(f"  {info['trailer_off']:>6}  --  Modular trailer    {len(info['trailer']):>8}  --")

def print_strings(info):
    by_id = {s["id"]: s for s in info["sections"]}
    if 0 not in by_id:
        print("  (no String section)"); return
    strings = parse_strings(by_id[0]["payload"])
    print(f"\n=== String section: {len(strings)} entries ===")
    cats = {}
    for i, s in enumerate(strings):
        cats.setdefault(categorize_string(s), []).append((i, s))
    for cat in sorted(cats, key=lambda c: -len(cats[c])):
        print(f"\n  -- {cat} ({len(cats[cat])}) --")
        for idx, s in cats[cat][:30]:
            display = s if len(s) < 120 else s[:117] + "..."
            print(f"    [{idx:>4}] {display!r}")
        if len(cats[cat]) > 30:
            print(f"    ... ({len(cats[cat]) - 30} more)")

def print_ir_hex(info):
    by_id = {s["id"]: s for s in info["sections"]}
    if 8 not in by_id:
        print("  (no OpTree section)"); return
    print(f"\n=== OpTree (id=8): {by_id[8]['length']} bytes ===")
    hexdump(by_id[8]["payload"])

def decode_ir_heuristic(info):
    """Scan section 8 byte-by-byte; for every position, try interpreting it as
    a 1- or 2-byte varint and flag the ones that index a string we've
    classified as an op-name."""
    by_id = {s["id"]: s for s in info["sections"]}
    if 0 not in by_id or 8 not in by_id:
        print("  (need both String and OpTree sections)"); return
    strings = parse_strings(by_id[0]["payload"])
    payload = by_id[8]["payload"]
    print(f"\n=== Heuristic op-name scan of OpTree (id=8) ===")
    print(f"  {'offset':>6}  {'value':>5}  {'op name':<28}  {'context':<24}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*28}  {'-'*24}")
    matches = 0
    p = 0
    while p < len(payload):
        try:
            v, _ = read_varint(payload, p)
        except Exception:
            p += 1; continue
        if 0 <= v < len(strings):
            s = strings[v]
            if categorize_string(s) == "op-name":
                a = max(0, p - 3); b = min(len(payload), p + 5)
                bytes_ctx = list(payload[a:b])
                rel = p - a
                ctx = " ".join((f"[{x:02x}]" if i == rel else f"{x:02x}")
                                for i, x in enumerate(bytes_ctx))
                print(f"  {p:>6}  {v:>5}  {s:<28}  {ctx}")
                matches += 1
        p += 1
    print(f"\n  matched {matches} op-name references")

# ---------------------------------------------------------------- diff
def diff_files(info_a, info_b):
    print(f"\n=== diff ===")
    print(f"  {os.path.basename(info_a['path']):>30}: {info_a['size']:>8} bytes")
    print(f"  {os.path.basename(info_b['path']):>30}: {info_b['size']:>8} bytes")
    print(f"  {'delta':>30}: {info_b['size'] - info_a['size']:>+8} bytes")
    print()
    a_by_id = {s["id"]: s for s in info_a["sections"]}
    b_by_id = {s["id"]: s for s in info_b["sections"]}
    print(f"  {'section':<18}  {'A len':>8}  {'B len':>8}  {'delta':>6}")
    for sid in sorted(set(a_by_id) | set(b_by_id)):
        a_len = a_by_id.get(sid, {}).get("length", 0)
        b_len = b_by_id.get(sid, {}).get("length", 0)
        d = b_len - a_len
        print(f"  {SECTION_NAMES.get(sid, f'?({sid})'):<18}  {a_len:>8}  {b_len:>8}  {d:+d}" if d else
              f"  {SECTION_NAMES.get(sid, f'?({sid})'):<18}  {a_len:>8}  {b_len:>8}  {'.':>6}")
    at = len(info_a["trailer"]); bt = len(info_b["trailer"])
    d = bt - at
    print(f"  {'trailer':<18}  {at:>8}  {bt:>8}  {d:+d}" if d else
          f"  {'trailer':<18}  {at:>8}  {bt:>8}  {'.':>6}")

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    args = sys.argv[1:]
    flags = {a for a in args if a.startswith("--")}
    paths = [a for a in args if not a.startswith("--")]

    if "--diff" in flags:
        if len(paths) != 2:
            print("--diff needs exactly two paths"); sys.exit(2)
        diff_files(parse_file(paths[0]), parse_file(paths[1]))
    else:
        for path in paths:
            info = parse_file(path)
            print_structure(info)
            if "--strings" in flags: print_strings(info)
            if "--ir" in flags: print_ir_hex(info)
            if "--decode-ir" in flags: decode_ir_heuristic(info)
            print()
