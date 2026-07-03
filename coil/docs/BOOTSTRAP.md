# Bootstrapping Coil without the Rust toolchain

The Coil compiler is self-hosted (`selfhost/src/*.coil`). Historically every build
rooted at `cargo build` of the Rust reference compiler (`src/*.rs`), which needs
Rust, `inkwell`, and the LLVM 21 headers. This document describes the **seed
bootstrap**, which lets a fresh checkout rebuild a fully verified compiler with no
Rust toolchain at all.

## The one command

```sh
selfhost/rebootstrap.sh          # builds + verifies + installs ./coil
```

On a fresh checkout this uses the committed seed (`selfhost/seed/coil-seed`) as
stage0. If you happen to have a Rust build (`./target/debug/coil`) it prefers that;
you can also force a specific stage0 with `STAGE0=/path/to/coil`.

**Requirements:** `libLLVM.dylib` (`brew install llvm`) and a C compiler (`cc`).
Not required: cargo, rustc, inkwell, LLVM headers, or the 180 MB Rust debug binary.

Why libLLVM is still needed: the compiler *embeds* an LLVM backend (`codegen.coil`
FFIs into the LLVM-C API), so the compiler binary links `libLLVM` even when its
native **arm64** backend does the codegen. The seed removes the Rust *build*
toolchain, not the libLLVM runtime dependency.

## The seed

`selfhost/seed/coil-seed` is a prebuilt, committed arm64 self-host compiler (~2.4 MB).
`selfhost/seed/SEED_VERSION` records the commit + timestamp it was built from.

The seed is **never trusted blindly.** `rebootstrap.sh` re-derives the compiler from
source on every run and proves the result faithful two independent ways, so a stale
or tampered seed cannot slip through:

1. **Fixpoint** — `stage0 → stage1 → stage2 → stage3`, then `stage2.o` must be
   byte-identical to `stage3.o`. The native arm64 backend is fully deterministic, so
   a faithful compiler reproduces its own object exactly. (stage1 is lowered by
   stage0's default LLVM backend; stage2/stage3 use `--backend arm64`. Only the
   stage2==stage3 fixpoint is required.)
2. **Gates** — `oracle/gate-full.sh` (emitted IR byte-exact vs the reference snapshot
   across the whole corpus) and `oracle/arm64/gate-run.sh` (built programs produce
   identical stdout + exit code).

This is the standard trusting-trust mitigation: the binary blob is validated against
source on every use, and you can always re-anchor to Rust with
`STAGE0=./target/debug/coil selfhost/rebootstrap.sh`.

## Refreshing the seed

When you change `selfhost/src` in a way that touches the language the **compiler
itself** is written in (new syntax/semantics the current seed can't parse), the old
seed may no longer compile the new source. Refresh it in the same commit:

```sh
selfhost/refresh-seed.sh         # rebuild + verify, then update selfhost/seed/
git add selfhost/seed/coil-seed selfhost/seed/SEED_VERSION
git commit -m 'refresh self-host seed'
```

`refresh-seed.sh` refuses to update the seed unless the full fixpoint + gates pass,
so a broken seed can never be committed. If you ever forget and land a source change
the seed can't build, the escape hatch is a one-time `cargo build` to re-seed.

## Relationship to the other bootstrap scripts

- `selfhost/bootstrap.sh` — classic LLVM-backend fixpoint (stage2==stage3 as
  executables, with UUID canonicalization). Proves the LLVM path self-reproduces.
- `selfhost/bootstrap-arm64.sh` — native arm64-backend fixpoint. `rebootstrap.sh` is
  this same proof, generalized to start from the seed and to install + gate the
  result. These remain the canonical from-Rust proofs; `rebootstrap.sh` is the
  everyday Rust-free path.
