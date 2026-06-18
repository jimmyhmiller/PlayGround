# `tallyc` — the λ-Tally compiler (Rust + inkwell/LLVM)

The real-compiler successor to the Python POC (`../poc/`). Same idea, native
codegen: the L3 address/permission split, a linear/permission checker (the novel
part), and an **erasing** lowering to LLVM IR — so permissions/regions/ghosts
cost nothing at runtime.

```
src/lexer  → parser → ast → check        (frontend: pure Rust, no deps)
                              │ erase
                              ▼
                            codegen (inkwell, behind `--features llvm`)
```

## Status (v0)

- **Frontend:** lexer, parser, and the v0 **linear/permission checker** — the L3
  core (`alloc` / read / write / `free`, `Addr` vs linear `Perm`). Rejects
  double-free, use-after-free, use-after-move, leaks, and dereferencing a bare
  aliased pointer. The discipline is the one proved sound in
  `../agda/CombinedSound.agda`.
- **Backend:** `codegen.rs` lowers a checked program to `tally_main() -> i64`
  and JITs it; cells become real `malloc`'d blocks, `free` is libc `free`. The
  end-to-end test compiles `alloc{val:0}; a.val=42; let r=a.val; free a; r;` to
  native code and gets `42`.

Next: port the v1 region/cursor checker (intrusive doubly-linked list, O(1)
remove) and emit object files / a `main`.

## Build & test

Frontend only (no LLVM needed):

```
cargo test          # lexer/parser/checker tests
cargo run -- check <file.tal>
```

With the LLVM backend:

```
cargo test --features llvm
```

### LLVM backend prerequisites

Needs **LLVM 18** dev libraries. On this Ubuntu sandbox they are installed with:

```
apt-get install -y llvm-18-dev libpolly-18-dev libzstd-dev
export LLVM_SYS_181_PREFIX=$(llvm-config-18 --prefix)
```

The cloud sandbox ships `libLLVM.so` + `llvm-config-18` but not the `-dev`
headers/static libs, so add the three `apt` packages above (e.g. in a repo
setup script) before `cargo build --features llvm`. `inkwell` is pinned to the
`llvm18-0` feature.
