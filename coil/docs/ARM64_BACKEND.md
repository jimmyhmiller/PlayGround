# The arm64 backend (selfhost, zero-deps)

A second backend for the **self-hosted** Coil compiler (`selfhost/src/`), written
in Coil, that lowers the monomorphized `Program` directly to AArch64 machine code
in a Mach-O object — no LLVM anywhere in the path. It is the **debug backend**:
DWARF (line tables + subprograms + frame variables) is always on, generated-code
quality is secondary, and compile speed is primary (target: ≥10× faster than the
LLVM backend end-to-end).

## Files
- `selfhost/src/a64.coil` — AArch64 instruction encoder. Pure: appends u32
  instructions to a code buffer; label/fixup management for branches.
- `selfhost/src/macho.coil` — Mach-O `MH_OBJECT` writer: sections, relocations
  (`BRANCH26`, `PAGE21`, `PAGEOFF12`, `UNSIGNED`), symbol table, string table.
- `selfhost/src/dwarf.coil` — DWARF v4 emitters: `__debug_line` (per-expression
  rows), `__debug_abbrev`/`__debug_info` (CU, subprograms, params/locals),
  `__debug_str`.
- `selfhost/src/codegen_a64.coil` — the lowering: mono `Program` → code+data+
  relocs+DWARF. Mirrors `codegen.coil`'s `emit-expr` structure and semantics
  (the LLVM backend is the behavioral spec) but emits machine code.
- `selfhost/src/main.coil` — `build <file> -o out --backend arm64` plumbs the
  same pipeline into the new backend; link step stays `cc <out>.o -o <out>`.

## Code generation model (deliberately naive — it's a debug backend)
- **Everything lives in fp-relative stack slots.** Each expression result is
  materialized into a fresh entry-frame slot; each `let` binding gets a stable
  named slot for its whole scope. No register allocation, no liveness. This is
  what makes precise DWARF frame-variables trivial: every binding has one
  fp-relative location for its lifetime.
- Scratch regs: x8–x15 (int/ptr), d0–d7/v16.. (float/vec) — used only within a
  single expression's emission, never across statements.
- Frame: `stp x29,x30,[sp,#-16]!; mov x29,sp; sub sp,sp,#frame` (frame size
  back-patched after the function body is emitted). fp chain intact everywhere
  → lldb backtraces work without eh_frame.
- Aggregates are slot-resident by value; copies are inline `ldr/str` loops
  (small) or `memcpy` calls (large).
- **One calling convention: AAPCS64 (Apple)** for both internal functions and
  externs — reuses the `abi-classify-aapcs64` classifier already ported in
  `codegen.coil`. Small structs ≤16B in x-pairs (HFA in v-regs), >16B indirect
  via x8 (sret) / caller copy. Variadic calls: Apple rule — anonymous args go
  on the stack, 8-byte aligned. `(vec f32 4)` internally passes in q-regs.
- Sum construct/match, slices, EIf/ELoop divergence discipline, entry-alloca
  discipline: all mirror `codegen.coil` exactly (see the lowering spec that
  file embodies; tag = i32 variant index at offset 0, payload = `[words x i64]`).

## The `llvm-ir` escape hatch
The corpus' inline-IR surface is finite and straight-line (no control flow, no
phi). The backend contains a small IR-subset lowerer: after `$N/$tN/$ret`
substitution it parses each line and lowers structurally:
- `insertvalue`/`extractvalue` on `{ptr,i64}`-shaped aggregates (slices) and
  cmpxchg results; `undef`; `ret`.
- atomics (`atomicrmw add/sub/xchg`, `cmpxchg`, `load/store atomic seq_cst`) →
  LSE (`ldaddal/swpal/casal/ldar/stlr`).
- `inttoptr`, `trunc`, `load/store volatile`.
- vector ops (`insertelement`, `shufflevector` splat, `fadd/fsub/fmul <4 x float>`,
  `@llvm.fma.v4f32`, `@llvm.vector.reduce.fadd.v4f32`) → NEON.
Anything outside the subset is a **hard error** naming the unsupported line.
(`call void asm sideeffect` is freestanding/ELF-only → hard error on this
Mach-O backend.)

## DWARF (always on)
DWARF v4, mirroring the Rust `-g` reference: CU (language C, not-optimized),
one `DW_TAG_subprogram` per function at its `defn` line with linkage name,
per-expression line rows from Expr spans (skip spans from other sources),
`DW_TAG_formal_parameter`/`DW_TAG_variable` with `DW_OP_fbreg` locations
(frame base = x29), base/pointer/struct/array/slice DI types. DWARF lives in
the `.o`; the linker's debug map makes lldb find it (same as `clang -g` without
dsymutil).

## Gates (behavioral — runtime equality, not IR equality)
`selfhost/oracle/arm64/`:
- `snapshot-run.sh <bin>` — builds the corpus with the **LLVM** backend and
  records stdout+exit per program.
- `gate-run.sh <bin>` — builds the same corpus with `--backend arm64`, runs,
  diffs stdout+exit byte-for-byte. Teeth: a miscompiled corpus program fails.
- lldb gate: breakpoint-by-line, step, `frame variable` on scalars/structs —
  mirrors `tests/debuginfo.rs` assertions.
- The finale: the arm64-built self-host compiler must itself pass the existing
  oracle gates (it is a working compiler), and `bench` compile-time comparison
  must show ≥10× vs the LLVM backend on both a small program and
  `selfhost/src/main.coil`.

## Baseline numbers (2026-07-01, M-series host)
- stage1 (`coil-self`, LLVM backend) builds `selfhost/src/main.coil`: **5.24 s**
  (frontend ≈0.4 s of that; `emit-ir` incl. LLVM module build = 0.46 s).
- builds `examples/fib.coil`: **0.57 s** (fixed LLVM+O3 overhead dominates; the
  frontend is 15 ms, bare `cc` link of a big object is ~40 ms).
- 10× targets: main.coil ≤ **0.52 s**, fib ≤ **57 ms**.

## Status / progress log
- [x] Architecture survey, lowering spec, llvm-ir + extern + DWARF inventories.
- [x] Behavioral snapshot harness (`snapshot-run.sh`).
- [x] a64 encoder (llvm-mc-verified) + spike: fib → .o → link → runs (exit 55).
- [x] Mach-O writer (BRANCH26/PAGE21/PAGEOFF12/UNSIGNED relocs, symbol permutation).
- [x] Full Expr lowering: **42/42 behavioral corpus gate** (sums/match, slices,
      strings, closures, hashmaps, json, lisp, dyn dispatch, fnptr/callptr,
      casts, arbitrary-width ints incl. u2, floats, bitfield structs, explicit
      layouts, AAPCS64 struct-by-value + variadic + sret, atomics via LSE
      (ldaddal/swpal/casal/ldar/stlr), vectors scalarized per-lane (exact IEEE),
      threads through pthread fnptrs). Gate teeth proven: caught 12 segfaults
      (copy-helper reg clobber + inverted tail conditions) and a u2-width
      miscompile before passing.
- [ ] DWARF line/info + lldb gate.
- [ ] Self-host-via-arm64 bootstrap + oracle gates.
- [ ] 10× bench report.
- [ ] (stretch) direct executable emission + ad-hoc codesign, no `cc`.
