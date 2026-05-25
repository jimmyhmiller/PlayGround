# The six-layer funnel

The bug lives at one of these layers, in increasing depth. Each layer's
plan starts from the previous layer's output and produces a new
artifact.

## Layer 1 — PyTorch op level
**Question:** which single `aten::*` call inside `HiFTGenerator.decode`
produces wrong output?
**Output:** a tuple `(input_tensor, weight_tensor, op_kwargs)` such that
`F.conv*d(input, weight, **kwargs)` reproduces the divergence on GPU vs
CPU.
**Plan:** `PLAN_LAYER1.md`.

## Layer 2 — MIOpen solver level
**Question:** which named MIOpen solver does the broken op resolve to?
Is the bug solver-specific (different solver fixes it)?
**Output:** the solver name, plus a table mapping `MIOPEN_DEBUG_CONV_*`
toggles to correctness.
**How:** run the Layer 1 minimal repro under
`MIOPEN_ENABLE_LOGGING=1 MIOPEN_ENABLE_LOGGING_CMD=1 MIOPEN_LOG_LEVEL=6`.
Disable solver families one at a time
(`MIOPEN_DEBUG_CONV_WINOGRAD=0`, `MIOPEN_DEBUG_CONV_DIRECT=0`,
`MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0`). The toggle that flips output to
correct names the broken solver family.

## Layer 3 — MIOpen kernel source
**Question:** what does the broken solver actually compute? Is the bug
in the C++/HIP source, or only in what the source compiles to?
**Output:** annotated reading of `MIOpen/src/solver/<solver>.cpp` and
the kernel(s) it dispatches under `MIOpen/src/kernels/`. A note on
which tile config the Layer 1 shape resolves to.
**How:** clone MIOpen source matching the installed version (1.0.70202
≈ ROCm 7.2.x). Trace from solver's `IsApplicable` → `GetSolution` →
kernel template → emitted kernel binary in `~/.cache/miopen/`.

## Layer 4 — Compiled kernel binary
**Question:** does the kernel binary in `~/.cache/miopen/` match what
the source says it should compute?
**Output:** disassembly of the cached `.co` for the broken solver,
diffed against the same solver compiled for gfx1100 (RDNA 3, known
working) where applicable.
**How:** `/opt/rocm/llvm/bin/llvm-objdump -d --triple=amdgcn` on the
cached object. Compare against expected ISA for the source.

## Layer 5 — LLVM AMDGPU codegen
**Question:** is the gfx1151 ISA wrong because the source asked for
something wrong, or because the AMDGPU backend miscompiled correct
source?
**Output:** a minimal HIP-source → ISA repro that shows correct source
producing wrong ISA on gfx1151 only.
**How:** if Layer 4 finds a suspect instruction sequence, recompile the
relevant kernel template with `-S -emit-llvm` and `-S` for both
gfx1100 and gfx1151. Diff the LLVM IR, then the ISA. If IR matches but
ISA diverges, that's a backend miscompile.

## Layer 6 — Hardware / microcode
**Question:** does correct, well-formed gfx1151 ISA produce the wrong
result on this silicon?
**Output:** a tiny standalone HIP program that issues the suspect
instruction sequence on synthetic data and gets wrong results. ~30
lines. The kind of repro AMD can act on.
**How:** if Layer 5 shows ISA looks correct but execution is wrong,
write a HIP microbenchmark for the suspect instruction (likely a
`v_dual_*` or `v_wmma_*` op new to RDNA 3.5) on a known input/output
pair from Layer 1. Compare against scalar reference.

## Stop condition

Each layer can falsify the hypothesis that the bug lives below it. If
Layer 2 shows that *every* solver family produces the same wrong output
on gfx1151, it's almost certainly compiler/silicon, not solver
selection — and we'd skip to Layer 5. If Layer 4 shows the cached binary
exactly matches a known-correct reference, the bug is in something
Layer 4 doesn't see (driver, queue, command processor) and we're at
the point of needing AMD's help.

The whole funnel terminates with either:
- A pinpointed buggy line of MIOpen source (file a fix-it patch upstream).
- A pinpointed AMDGPU backend miscompile (file a fix-it patch on LLVM).
- A pinpointed silicon/microcode bug (file a hardware bug with AMD,
  with the Layer 6 repro attached).
