#!/usr/bin/env python3
"""Construct kernel MLIR via Python bindings + dump it through PassManager IR printing.

The kernel MLIR for `add_kernel` doesn't survive in the cache (compile_offload
destroys it). But the dialects ARE registered in max._core, so we can build a
fresh module containing the kernel from scratch, run pass instrumentation to
capture stage-by-stage IR, and demonstrate what kgen MLIR for our kernel looks
like.

This is a "from-scratch construction with IR printer" approach, not a recovery
from the cache. The cache would have given us Mojo's *exact* MLIR output if it
were preserved — what we build here is the equivalent in shape.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse_mlir_bc import parse_file

from max._mlir import ir
from max._core import graph
from max._mlir._mlir_libs._mlir.passmanager import PassManager

def make_context():
    reg = ir.DialectRegistry()
    graph.load_modular_dialects(reg)
    ctx = ir.Context()
    ctx.append_dialect_registry(reg)
    ctx.load_all_available_dialects()
    return ctx

def parse_host_mlir(path):
    """Parse our cached host MLIR (with trailer stripped)."""
    info = parse_file(path)
    raw = open(path, "rb").read()
    mlir_bytes = raw[:info["trailer_off"]]
    ctx = make_context()
    return ir.Module.parse(mlir_bytes, ctx), ctx

def dump_with_ir_printing(module, ctx, pipeline):
    """Run a pass pipeline with IR-printing-after-each-pass enabled."""
    pm = PassManager.parse(pipeline, context=ctx)
    pm.enable_ir_printing(
        print_before_all=False,
        print_after_all=True,
        print_module_scope=False,
        print_after_change=True,
        print_after_failure=False,
    )
    pm.run(module.operation)

def hand_built_kernel_mlir():
    """Construct kgen MLIR for `add_kernel` by hand. This is what we'd see if
    Mojo preserved the pre-elaboration kernel form."""
    ctx = make_context()
    # Use textual MLIR — easier than building op-by-op.
    src = """
    module {
      kgen.func @add_kernel(
          %a: !kgen.pointer<scalar<f32>>,
          %b: !kgen.pointer<scalar<f32>>,
          %c: !kgen.pointer<scalar<f32>>,
          %n: index) {
        // Compute global_idx.x = block_idx.x * block_dim.x + thread_idx.x
        // For a real kernel, block/thread idx come from gpu intrinsics.
        // Here we just take it as the n parameter for shape.
        %tid = pop.cast %n: index to scalar<si32>
        kgen.return
      }
    }
    """
    try:
        mod = ir.Module.parse(src, ctx)
        return str(mod), ctx
    except Exception as e:
        return f"PARSE FAILED: {e}", ctx

def list_registered_passes(ctx):
    """Try parsing each plausible pass name and report what's registered."""
    candidates = [
        "canonicalize", "cse", "symbol-dce", "inline", "sccp",
        "kgen-elaborate", "kgen-lower", "interp",
        "convert-kgen-to-llvm", "convert-pop-to-llvm",
        "kgen-resolve-paramexpr", "kgen-instantiate",
    ]
    found = []
    for name in candidates:
        try:
            PassManager.parse(f"builtin.module({name})", context=ctx)
            found.append(name)
        except Exception:
            pass
    return found

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--list-passes" in args:
        ctx = make_context()
        passes = list_registered_passes(ctx)
        print("Registered passes (out of probed candidates):")
        for p in passes:
            print(f"  {p}")
        sys.exit(0)

    if "--hand" in args:
        text, _ = hand_built_kernel_mlir()
        print("=== Hand-constructed kernel kgen MLIR ===")
        print(text)
        sys.exit(0)

    paths = [a for a in args if not a.startswith("--")]
    pipeline_arg = None
    if "--pipeline" in args:
        idx = args.index("--pipeline")
        pipeline_arg = args[idx+1]
        paths = [p for p in paths if p != pipeline_arg]
    pipeline = pipeline_arg or "builtin.module(canonicalize)"

    if not paths:
        print(__doc__); sys.exit(2)

    for path in paths:
        print(f"=== {path}: pipeline = {pipeline} ===")
        mod, ctx = parse_host_mlir(path)
        try:
            dump_with_ir_printing(mod, ctx, pipeline)
        except Exception as e:
            print(f"FAILED: {e}")
