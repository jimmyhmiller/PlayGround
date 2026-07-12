# Metaprogram PoC — the four kinds, on today's machinery

A metaprogram is a Coil function that runs at compile time and operates on the
program. This shows the **Checker** kind (reject power) working now; the
**Transformer** kind is in `../gc-dialect-poc` (a GC as a metaprogram).

## checker.coil — a use-after-free checker that VETOES compilation

`check-uaf` is a macro `[Code…] -> Code` that scans a sequence for a `(free X)`
followed by a later use of `X`, and calls `error` (aborting the build) if it
finds one — otherwise it returns the body unchanged. Uses only Code builtins.

```sh
../coil run ok.coil    # clean → compiles & runs
../coil run bad.coil   # use-after-free → REJECTED: "check-uaf: use after free …"
```

Limitation: scans the straight-line top-level sequence handed to it (wrap a region
in `(check-uaf …)`). Making it *automatic* over the whole program — no wrapping —
is the compiler-level `(checker f)` hook in `docs/METAPROGRAMS.md` (Phase 1.1).
