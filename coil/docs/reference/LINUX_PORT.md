# The Linux x86-64 port

Coil self-hosts on Linux x86-64. From a clean checkout on an Ubuntu-ish box with
LLVM 21 (apt.llvm.org) and a C compiler:

    selfhost/rebootstrap-linux.sh          # seed -> stage1 -> stage2 -> stage3
                                           # fixpoint + linux gate-full + linux
                                           # gate-run + gate-cli, installs ./coil-linux

`./coil` and `./coil-nollvm` stay the committed **macOS arm64** binaries; the Linux
compiler installs as `./coil-linux` and its seed is
`selfhost/seed/coil-seed-linux-x86_64` (ELF, dynamic libLLVM 21). If your libLLVM
doesn't match, bootstrap a stage0 from the shipped IR — `selfhost/seed/linux-ir/NOTES.md`.

## How the port went in (July 2026)

One-shot IR cross-emission from macOS (`--target x86_64-unknown-linux-gnu`, the
driver's existing cross support + one `musttail` fix), linked on the Linux box, then
self-hosted natively and all remaining issues fixed in `.coil` source on Linux. The
port surface was small because the LLVM backend was already triple-parametric
(`arch 0 = x86_64` predates this) and x86_64-linux shares the SysV ABI with the
x86_64-apple lowering that already existed.

What changed in the compiler (all host-guarded; macOS emission byte-unchanged
except where noted):

- **`metaengine.coil`** — Darwin GCD semaphores (`dispatch_semaphore_*`) replaced by
  a portable pthread mutex+condvar counting semaphore (`me-sem-*`); this one is a
  behavior-neutral swap on macOS too. Host probe `me-host-linux?`
  (`access("/proc/self/maps")`) picks the dylib link line: `cc -shared` + `.so` on
  ELF vs `cc -dynamiclib` + `.dylib` + `-Wl,-undefined,dynamic_lookup` on Mach-O.
- **`comptime_eval.coil`** — same host-aware link line for compiled-comptime dylibs.
- **`jit.coil`** — `sys_icache_invalidate` is now `dlsym`'d at runtime (found on
  Darwin, absent-and-unneeded on x86 ELF) so the ELF link has no Darwin extern. The
  MObj JIT itself emits arm64 and never runs on this host (see gaps).
- **`main.coil`** — the in-memory MObj comptime fast path is registered only when
  the HOST is arm64 (`host-arm64?`); on Linux, comptime sites the interpreter can't
  fold take `comptime_eval`'s dylib route (`cc -shared` + `dlopen`), which passes
  the full gate-cli comptime battery including aggregate/sum/string readbacks.
- **`driver.coil`** — `run`/`test` prepend `exec` to the `system()` command so a
  signal death reaches `wait-report` unmasked on shells that fork (Linux dash);
  `-arch` (a Mach-O toolchain flag) is not passed on ELF hosts; `dsymutil` is
  skipped on ELF (DWARF lives in the executable).
- **`codegen.coil`** — **real x86-64 bug fix**: the SysV classifier's `abi-flatten`
  re-derived field offsets by natural alignment, mis-placing every field of a
  `:layout packed`/`:layout explicit` struct (packed `{u8 i64}` put `b` at offset 8
  in a 9-byte struct and aborted the build). Flattening now walks the struct's REAL
  layout (`abi-flatten-cs`: `LLVMOffsetOfElement` for LC/packed bodies, declared
  `:at` offsets for explicit ones). Invisible on arm64 (AAPCS64 never flattens
  int-field structs); on x86 it unblocked by-value packed/explicit structs entirely.

## Verification (the Linux oracle)

- `selfhost/oracle/linux/gate-run.sh` — builds the shared behavioral corpus with the
  LLVM backend and diffs stdout+exit against the SAME reference snapshots the arm64
  gate uses (56/56; the two arm64-register-convention examples must fail with the
  per-arch diagnostic, asserted).
- `selfhost/oracle/linux/gate-full.sh` + `snapshot-full.sh` — byte-exact `emit-ir`
  against a Linux-blessed IR snapshot (`full-reference/`, 58 refs + the 2 asserted
  per-arch errors). The macOS snapshot can't serve here (triple/datalayout/musttail
  legitimately differ).
- `gate-cli.sh` — shared, now host-aware: Mach-O cross-link, arm64-object execution,
  and dSYM checks run on Darwin only; the gen-10 perf probe skips when the committed
  (Mach-O) seed can't exec on the host; everything else — including the full
  compiled-comptime readback battery — runs on both.
- `metaprog-poc/compile-and-run/parity.sh` — 114/116 byte-identical between the
  interpreter and compiled metaprogram engines on Linux. The 2 divergences are
  pre-existing engine differences, not port issues (see INTERP_DELETION.md).

## Known gaps (by design or deferred)

- `--backend arm64`, the nollvm build (`main_a64`, Mach-O writer), and the MObj
  in-memory JIT are macOS/arm64-only. `emit-obj --backend arm64` still works on
  Linux (codegen is host-independent); you just can't run the result here.
- Cross-LINKING from Linux (e.g. `--target` an Apple triple) is not orchestrated by
  the driver (no `-arch` on ELF; you'd need a cross toolchain).
- `-g` on Linux: DWARF is emitted into the object/executable but the lldb/dsymutil
  packaging checks are Darwin-only; Linux debugger UX is unverified.
- `COIL_LLVM_LINK=static` and `llvm-link-flags.sh` remain macOS-shaped; the Linux
  script discovers the libdir itself (override: `COIL_LLVM_LIBDIR`).
- The x86 `musttail` downgrade (`codegen.coil::emit-tail`): 25 aggregate-returning
  self-tail-calls are `tail` (best-effort) instead of guaranteed TCO on x86 —
  bounded recursions; a stack overflow inside `comptime.*` on pathological input
  would be this.

- `gate-diag` on Linux: 31/33 — the two failures are platform-text refs
  (`02-link-fail` bakes macOS ld64 wording incl. the `_main` underscore;
  `14-shim-bad-gpr` bakes the arm64-register diagnostic). Pre-existing, not in the
  Linux authoritative set (`rebootstrap-linux.sh` runs gate-full/gate-run/gate-cli).
