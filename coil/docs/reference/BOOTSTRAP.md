# Bootstrapping Coil

The Coil compiler is self-hosted (`src/compiler/*.coil`) and bootstraps from committed
seeds, so a fresh checkout rebuilds a fully verified compiler with nothing but a C
compiler (and, for the LLVM backend, libLLVM) — in three flavors:

| Path | Command | Needs | Compiler it builds |
|------|---------|-------|--------------------|
| **LLVM-free** (recommended) | `python3 scripts/dev.py build nollvm` | just `cc` | arm64 backend only |
| Full | `python3 scripts/dev.py build full` | `cc` + `libLLVM.dylib` | LLVM + arm64 backends |
| Linux x86-64 | `python3 scripts/dev.py build linux` | `cc` + libLLVM 21 | LLVM backend, ELF ([LINUX_PORT.md](LINUX_PORT.md)) |

## LLVM-free: zero external dependencies

```sh
python3 scripts/dev.py build nollvm   # builds + verifies + installs build/bin/coil-nollvm
```

Uses the committed seed `bootstrap/seeds/native/coil-seed-nollvm` (a ~2.1 MB self-host
compiler that links **only libSystem** — no libLLVM). The whole toolchain a fresh
machine needs is a C compiler. The produced compiler is built from
`src/compiler/main_a64.coil`, which omits the LLVM backend: it compiles programs via
the native **arm64** backend (`build --backend arm64`, the default for this binary)
and emits Mach-O objects directly. Commands that require the LLVM backend
(`emit-ir`, `dump-ir`, `__normalize` — textual LLVM IR) fail loudly with a clear
diagnostic instead of doing nothing.

## Full: LLVM + arm64 backends

```sh
python3 scripts/dev.py build full          # builds + verifies + installs build/bin/coil
```

Uses the committed seed `bootstrap/seeds/native/coil-seed`. This is the complete compiler
(both backends, plus `emit-ir`/`dump-ir`), so its binary links `libLLVM` even when
the arm64 backend does the codegen — the compiler *embeds* an LLVM backend
(`codegen.coil` FFIs into the LLVM-C API). **Requirements:** `libLLVM.dylib`
(`brew install llvm`) + `cc`. Force a specific stage0 with `STAGE0=/path/to/coil`.

## How the two builds share one codebase

The CLI dispatch and the whole compile pipeline live in the backend-agnostic
`src/compiler/driver.coil`, which never imports the LLVM backend. The two LLVM entry
points (`build` via LLVM, `emit-ir`) and `__normalize` are injected into
`driver-main` as **function pointers**. The two top files differ only in what they
inject:

- `src/compiler/main.coil` imports `codegen.coil`/`normalize.coil` and injects the
  real LLVM entry points → full compiler, links libLLVM.
- `src/compiler/main_a64.coil` imports neither and injects hard-error stubs → no
  reference to any LLVM symbol → links no libLLVM.

There is no code duplication between them, and the gate-full corpus includes both
top files so the snapshot oracle keeps them from drifting apart.

## The seeds

Two prebuilt, committed self-host compilers:

- `bootstrap/seeds/native/coil-seed` — full (LLVM + arm64), ~2.4 MB, links libLLVM.
  Provenance in `bootstrap/seeds/native/SEED_VERSION`.
- `bootstrap/seeds/native/coil-seed-nollvm` — LLVM-free (arm64 only), ~2.1 MB, links only
  libSystem. Provenance in `bootstrap/seeds/native/SEED_VERSION_NOLLVM`.

Neither seed is **trusted blindly.** Each rebootstrap re-derives the compiler from
source on every run and proves the result faithful independently, so a stale or
tampered seed cannot slip through:

1. **Fixpoint** — `stage0 → stage1 → stage2 → stage3`, then `stage2.o` must be
   byte-identical to `stage3.o`. The native arm64 backend is fully deterministic, so
   a faithful compiler reproduces its own object exactly. (stage1 is lowered by
   stage0's default backend; stage2/stage3 use `--backend arm64`. Only the
   stage2==stage3 fixpoint is required.)
2. **Gates** — the LLVM-free path runs `python3 scripts/oracle.py runtime gate arm64` (built programs
   produce identical stdout + exit code vs the LLVM reference) and asserts the binary
   links no libLLVM; the full path additionally runs `python3 scripts/oracle.py gate full` (emitted
   IR byte-exact vs the reference snapshot across the corpus). The LLVM-free build has
   no `emit-ir`, so gate-full does not apply to it.

This is the standard trusting-trust mitigation: the binary blob is validated against
source on every use, and you can always re-anchor to a different stage0 with
`STAGE0=/path/to/coil python3 scripts/dev.py build nollvm` (or the `full` variant).

## Refreshing the seeds

When you change `src/compiler` in a way that touches the language the **compiler
itself** is written in (new syntax/semantics the current seed can't parse), the old
seed may no longer compile the new source. Refresh it in the same commit:

```sh
scripts/compiler/refresh-seed.sh              # refresh BOTH seeds (rebuild + verify each)
# or: scripts/compiler/refresh-seed.sh nollvm   /   scripts/compiler/refresh-seed.sh full
git add bootstrap/seeds/native/ && git commit -m 'refresh self-host seeds'
```

`refresh-seed.sh` refuses to update a seed unless its fixpoint + gates pass, so a
broken seed can never be committed.

⚠ **If you forget, the LLVM-free seed is the one that strands** — it is its own only
stage0, so a language change it cannot parse means it can never build its own
replacement (this really happened: `isize`/`usize` landed in the compiler source and
`rebootstrap-nollvm.sh` died at stage1 with `unknown type 'isize'` while the full
build stayed green). The escape hatch is to **bridge with the other compiler**, not to
re-seed from anything external:

```sh
build/bin/coil build src/compiler/main_a64.coil -o /tmp/nl --backend arm64
STAGE0=/tmp/nl ./scripts/compiler/refresh-seed.sh nollvm    # re-verifies, then updates the seed
python3 scripts/dev.py build nollvm                      # confirm it self-sustains with no override
```

The rebootstrap commands create ignored binaries under `build/bin/`. Before the
first rebuild, invoke a committed seed under `bootstrap/seeds/native/` directly.

The committed artifact is the arm64 fixpoint stage2, not the bridged stage1, so the
bridge washes out (the arm64 backend is deterministic). Whenever you change the
language the compiler itself is written in, run `refresh-seed.sh` for **both** seeds.

## Relationship to the other bootstrap scripts

- `python3 scripts/dev.py build full` — full LLVM + arm64 fixpoints and all compiler gates.
- `python3 scripts/dev.py build nollvm` — LLVM-free native arm64 fixpoint and runtime gates.
