# C-interop dominance (§6) — Phase 1 findings

Built per the approved scope (docs/CINTEROP_SCOPE.md): `coil cimport` (auto-import C
headers) + a `--link-flag` passthrough (link C libraries/objects). Jimmy's vision —
make Coil a drop-in for C codebases — realized for the common case, end-to-end.

## What works (Phase 1)

- **`coil cimport <header.h>`** → a `.coil` bindings module (externs + defstructs),
  produced by walking **clang's JSON AST** (`clang -Xclang -ast-dump=json`). clang does
  the real parse — real C grammar, includes, typedef desugaring — Coil never hand-rolls
  a C parser (that would be the hack). `src/cimport.rs`.
- **ABI-faithful type mapping** from clang's reported types (not assumptions):
  `int`→`i32`, `long`→`i64`, `double`→`f64`, `char*`→`(ptr i8)`, `T*`→`(ptr T)`,
  `struct X`→`(defstruct X …)`, `void` return→`void`, varargs→`...`.
- **`--link-flag <arg>` / `-l<lib>`** on `build`/`run` → passed straight to the `cc`
  link line, so a Coil program links against any C library or object
  (`build_executable_linked`). A generic passthrough, NOT a baked-in C-lib mode.
- **End-to-end, verified by CALLING real C** (ABI-correctness, not just plausible
  syntax): `sqrt(1764.0)=42` + `labs(-7)=7` through cimported libc bindings; and the
  full custom-library loop — a hand-written `triple_it` in C, compiled to a `.o`,
  cimported, linked via `--link-flag`, and called from Coil → `triple_it(14)=42`.
  tests/cimport.rs (gated on clang).

## The cardinal: REFUSE unmappable — never a silent-wrong binding

A wrong width/layout silently corrupts the ABI — the whole risk of doing this by hand,
and what cimport must never reproduce. So cimport REFUSES what it can't map correctly,
loudly (stderr warning + a `; SKIPPED …` note in the output), rather than guessing:

- A **union** is refused (overlapping layout — emitting it as a sequential `defstruct`
  would be the silent-wrong failure; caught by a test). A struct *containing* a union is
  refused too (its field is unmappable).
- Unknown/unsupported types (typedefs like `size_t`, enums, function-pointer params,
  `long double`, …) → the declaration is skipped with a clear reason, never mis-bound.
- De-dup: clang lists builtin libc functions twice (`sqrt`/`labs`); cimport emits one.

## Deferred (honestly noted, Phase 2+)

typedefs (resolve via AST desugar), enums (→ `const i32`), `#define` constant macros
(preprocessor — not in the AST; need a `clang -dM` pass), unions/bitfields,
anonymous/nested structs, function-pointer typedefs, complex struct layouts with
non-natural alignment (Phase-1 simple scalar structs match Coil's layout; clang's
explicit field offsets would be needed for the general case). Each is *refused*, not
faked — the bindings you get are correct, and what's missing is visible.

## A later ergonomic (noted)

A `(cimport "header.h")` MACRO (the macro invoking clang at expansion, importing
seamlessly) is a natural next layer on the same extractor. CLI-first was chosen so the
generated bindings are INSPECTABLE — you can audit the externs/defstructs for
ABI-correctness, which matters precisely because ABI corruption is the risk.

## Verdict
Coil imports C headers (real clang parse, ABI-faithful, refuse-unmappable) and links C
libraries — a Coil program calls both libc and a custom C library, verified by running
them. The "drop-in for C" play works for the common case; the gaps are deferred and
honest. Built as a CLI subcommand + a generic link passthrough — minimal core, no
baked-in C-interop "mode".
