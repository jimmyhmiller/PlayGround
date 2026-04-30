#!/usr/bin/env python3
"""Dump a Mojo `.mojo_cache` bytecode file as textual MLIR.

Uses Modular's bundled MLIR Python bindings (max._mlir + max._core), which
include their fork's `BytecodeReader` and all the kgen/pop/m/mosh/mo/rmo
dialect parsers. We strip the Modular trailer first (the part the upstream
reader can't handle), then hand the MLIR bytecode to `Module.parse()`.

Run with:  pixi run python dump_mlir.py samples/tiny.mlirbc
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse_mlir_bc import parse_file

def parse_bytecode_to_mlir(path):
    """Returns (textual_mlir, mlir_bytecode_size, trailer_size)."""
    info = parse_file(path)
    raw = open(path, "rb").read()
    mlir_bytes = raw[:info["trailer_off"]]
    trailer = raw[info["trailer_off"]:]

    from max._mlir import ir
    from max._core import graph
    reg = ir.DialectRegistry()
    graph.load_modular_dialects(reg)
    ctx = ir.Context()
    ctx.append_dialect_registry(reg)
    ctx.load_all_available_dialects()
    mod = ir.Module.parse(mlir_bytes, ctx)
    return str(mod), len(mlir_bytes), len(trailer)

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    for path in sys.argv[1:]:
        if path.startswith("--"): continue
        print(f"# === {path} ===")
        try:
            text, mlir_n, trailer_n = parse_bytecode_to_mlir(path)
        except Exception as e:
            print(f"# FAILED: {e}")
            continue
        print(f"# MLIR bytecode: {mlir_n} bytes, Modular trailer: {trailer_n} bytes")
        print(f"# decoded to {len(text)} chars of textual MLIR")
        print()
        print(text)
        print()

if __name__ == "__main__":
    main()
