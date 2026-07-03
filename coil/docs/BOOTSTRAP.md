# Bootstrapping Coil without the Rust toolchain

The Coil compiler is self-hosted (`selfhost/src/*.coil`). Historically every build
rooted at `cargo build` of the Rust reference compiler (`src/*.rs`), which needs
Rust, `inkwell`, and the LLVM 21 headers. This document describes the **seed
bootstrap**, which lets a fresh checkout rebuild a fully verified compiler with no
Rust toolchain — in two flavors:

| Path | Command | Needs | Compiler it builds |
|------|---------|-------|--------------------|
| **LLVM-free** (recommended) | `selfhost/rebootstrap-nollvm.sh` | just `cc` | arm64 backend only |
| Full | `selfhost/rebootstrap.sh` | `cc` + `libLLVM.dylib` | LLVM + arm64 backends |

## LLVM-free: zero external dependencies

```sh
selfhost/rebootstrap-nollvm.sh   # builds + verifies + installs ./coil-nollvm
```

Uses the committed seed `selfhost/seed/coil-seed-nollvm` (a ~2.1 MB self-host
compiler that links **only libSystem** — no libLLVM). The whole toolchain a fresh
machine needs is a C compiler. The produced compiler is built from
`selfhost/src/main_a64.coil`, which omits the LLVM backend: it compiles programs via
the native **arm64** backend (`build --backend arm64`, the default for this binary)
and emits Mach-O objects directly. Commands that require the LLVM backend
(`emit-ir`, `dump-ir`, `__normalize` — textual LLVM IR) fail loudly with a clear
diagnostic instead of doing nothing.

## Full: LLVM + arm64 backends

```sh
selfhost/rebootstrap.sh          # builds + verifies + installs ./coil
```

Uses the committed seed `selfhost/seed/coil-seed`. This is the complete compiler
(both backends, plus `emit-ir`/`dump-ir`), so its binary links `libLLVM` even when
the arm64 backend does the codegen — the compiler *embeds* an LLVM backend
(`codegen.coil` FFIs into the LLVM-C API). **Requirements:** `libLLVM.dylib`
(`brew install llvm`) + `cc`. Both paths prefer `./target/debug/coil` if you happen
to have a Rust build; force a stage0 with `STAGE0=/path/to/coil`.

## How the two builds share one codebase

The CLI dispatch and the whole compile pipeline live in the backend-agnostic
`selfhost/src/driver.coil`, which never imports the LLVM backend. The two LLVM entry
points (`build` via LLVM, `emit-ir`) and `__normalize` are injected into
`driver-main` as **function pointers**. The two top files differ only in what they
inject:

- `selfhost/src/main.coil` imports `codegen.coil`/`normalize.coil` and injects the
  real LLVM entry points → full compiler, links libLLVM.
- `selfhost/src/main_a64.coil` imports neither and injects hard-error stubs → no
  reference to any LLVM symbol → links no libLLVM.

There is no code duplication between them, and the gate-full corpus includes both
top files so the Rust oracle keeps them from drifting apart.

## The seeds

Two prebuilt, committed self-host compilers:

- `selfhost/seed/coil-seed` — full (LLVM + arm64), ~2.4 MB, links libLLVM.
  Provenance in `selfhost/seed/SEED_VERSION`.
- `selfhost/seed/coil-seed-nollvm` — LLVM-free (arm64 only), ~2.1 MB, links only
  libSystem. Provenance in `selfhost/seed/SEED_VERSION_NOLLVM`.

Neither seed is **trusted blindly.** Each rebootstrap re-derives the compiler from
source on every run and proves the result faithful independently, so a stale or
tampered seed cannot slip through:

1. **Fixpoint** — `stage0 → stage1 → stage2 → stage3`, then `stage2.o` must be
   byte-identical to `stage3.o`. The native arm64 backend is fully deterministic, so
   a faithful compiler reproduces its own object exactly. (stage1 is lowered by
   stage0's default backend; stage2/stage3 use `--backend arm64`. Only the
   stage2==stage3 fixpoint is required.)
2. **Gates** — the LLVM-free path runs `oracle/arm64/gate-run.sh` (built programs
   produce identical stdout + exit code vs the LLVM reference) and asserts the binary
   links no libLLVM; the full path additionally runs `oracle/gate-full.sh` (emitted
   IR byte-exact vs the reference snapshot across the corpus). The LLVM-free build has
   no `emit-ir`, so gate-full does not apply to it.

This is the standard trusting-trust mitigation: the binary blob is validated against
source on every use, and you can always re-anchor to Rust with
`STAGE0=./target/debug/coil selfhost/rebootstrap-nollvm.sh` (or `rebootstrap.sh`).

## Refreshing the seeds

When you change `selfhost/src` in a way that touches the language the **compiler
itself** is written in (new syntax/semantics the current seed can't parse), the old
seed may no longer compile the new source. Refresh it in the same commit:

```sh
selfhost/refresh-seed.sh              # refresh BOTH seeds (rebuild + verify each)
# or: selfhost/refresh-seed.sh nollvm   /   selfhost/refresh-seed.sh full
git add selfhost/seed/ && git commit -m 'refresh self-host seeds'
```

`refresh-seed.sh` refuses to update a seed unless its fixpoint + gates pass, so a
broken seed can never be committed. If you ever forget and land a source change a
seed can't build, the escape hatch is a one-time `cargo build` to re-seed.

## Relationship to the other bootstrap scripts

- `selfhost/bootstrap.sh` — classic LLVM-backend fixpoint (stage2==stage3 as
  executables, with UUID canonicalization). Proves the LLVM path self-reproduces.
- `selfhost/bootstrap-arm64.sh` — native arm64-backend fixpoint. `rebootstrap.sh` is
  this same proof, generalized to start from the seed and to install + gate the
  result. These remain the canonical from-Rust proofs; `rebootstrap.sh` is the
  everyday Rust-free path.
