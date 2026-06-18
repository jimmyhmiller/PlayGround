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

## Status (v0.2 — type-directed)

- **Type system:** `struct` declarations and the L3 split *as types* —
  `Own<S>` (a LINEAR owning capability, must be used exactly once) vs `Ptr<S>`
  (an UNRESTRICTED, copyable bare address that **cannot** be dereferenced). The
  checker is type-directed and **structurally linear**; memory safety falls out
  of the type discipline. Key typing rules:
    - a struct field's type must be **copyable** (never `Own`) — you cannot
      fabricate a capability by reading memory;
    - `alloc S {..} : Own<S>`; `addr(x)` borrows `x:Own<S>` to a copyable
      `Ptr<S>`; `e.f` requires the base to be `Own<S>` (a `Ptr<S>` base is
      rejected — no capability);
    - an `Own` must be consumed once (free / move); dropping, re-using, or
      leaving it live at end of scope is an error.
  Rejects double-free, use-after-free, use-after-move, leaks, deref of a bare
  `Ptr`, type mismatches, missing/unknown fields, and linear struct fields. The
  discipline matches `../agda/CombinedSound.agda`.
- **Backend:** `codegen.rs` lowers a *checked* program to `tally_main() -> i64`
  and JITs it (types are erased: cells are `malloc`'d blocks, `free` is libc
  `free`). End-to-end test compiles
  `struct C{val:Int} let a=alloc C{val:0}; a.val=42; let r=a.val; free a; r;`
  to native code and gets `42`.
- **CLI / examples:** `tally check examples/twonodes.tal` (accept),
  `examples/bad_deref.tal` (reject, with a precise error).

Next: **functions with multiplicity-annotated parameters** (the usage-context
algebra, threading linear capabilities across calls), then the region/cursor
discipline for the intrusive doubly-linked list with O(1) remove.

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
