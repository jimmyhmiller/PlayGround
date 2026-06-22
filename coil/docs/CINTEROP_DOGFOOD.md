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
- A struct with a **bitfield** is refused (`int a:3` packs into bit ranges — emitting a
  full-width `(a i32)` would corrupt the layout; clang marks the field `isBitfield`).
  This was a real silent-wrong binding the red-team test caught before it shipped.
- **`_Bool`** maps to `u8` (a 1-byte C-ABI value), NOT Coil `bool` (i1, whose C-ABI
  width would be a guess) — ABI-safe by construction.
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

---

# Phase 2 — typedefs, enums, #defines (DONE)

Built per docs/CINTEROP_PHASE2_SCOPE.md, on the SAME clang-AST extractor, same cardinal
(ABI-faithful + refuse-unmappable). One scope correction surfaced while building and was
resolved with the user: the scope's plan to emit enum/`#define` values as `(def …)` is
WRONG — `def` is a *compile-time* (macro-environment) binding, so `(def RED 0)` leaves
`RED` unbound in ordinary runtime code. C constants need a *runtime* value.

## The enabler: a `const` primitive (a real, minimal language addition)

`(const NAME VALUE)` / `(const NAME TYPE VALUE)` — a named scalar constant. A reference
elaborates to the literal **inline** (resolved at the checker's `Expr::Var` fallback, so
zero runtime cost and codegen is untouched): an *untyped* const re-enters integer-width
inference exactly like writing the literal would (so `RED` slots into an `i32`/`u8`/`i64`
context unchanged), a *typed* const pins the width and fit-checks at definition. Consts
live in a flat global namespace (referenced bare, never module-renamed — the same rule
`extern` uses), which matches how C enum constants and `#define`s behave, and sidesteps
the resolver's "a bare `Var` is always a local" assumption. Locals shadow consts (locals
resolve first), so a parameter named the same as a const is unaffected. src/ast.rs (Const
/ ConstLit), src/parse.rs, src/check.rs (const table + Var resolution); tests/const.rs.

This was the right call over (a) overloading `def` to be sometimes-runtime — which puts a
type-dependent hole in the clean compile-time/runtime wall — and (b) a zero-arg macro
`(RED)` — which loses the bare-name C drop-in feel.

## What Phase 2 adds

- **Typedefs** resolve through a `name → underlying qualType` table built from the whole
  translation unit (so a typedef used from the target header resolves even when defined in
  a system include): `size_t`→`u64`, a typedef-of-typedef (`mysize`→`size_t`→`u64`), a
  typedef'd struct (`typedef struct Point Point;`)→the struct. Recursive with a depth
  guard. An underlying type that's itself unmappable still reaches the same refusal.
- **Enums**: an `enum X` *type* lowers to its integer width (C enums are `int`→`i32`; a
  fixed underlying type overrides) — so an enum parameter is no longer refused. The
  *constants* become `const` defs with C's value rules (explicit initializer sets the
  value; implicit continues the running counter): `enum Color { RED, GREEN=5, BLUE }` →
  `(const RED 0) (const GREEN 5) (const BLUE 6)`. A non-integer enumerator → the whole
  enum is refused (no guessed value).
- **`#define` object-like constant macros**: a separate `clang -dM -E` pass (they never
  reach the AST), diffed against an empty-TU baseline to drop builtins/system macros. A
  simple line parser keeps object-like int/float-literal bodies (`#define MAX_LEN 4096` →
  `(const MAX_LEN 4096)`; hex/octal/`-`/`U`/`L` suffixes handled) and SKIPS function-like
  macros (`#define M(x) …`), string bodies, and expression bodies — refuse-not-guess.

## Verified end-to-end (running, not just plausible text)

A custom C library with an **enum-typed** function (`int color_code(enum Color)`) and a
**typedef'd** one (`size_t add_sizes(size_t,size_t)`) is compiled, its header cimported,
linked via `--link-flag`, and called from Coil using the generated enum constant as a
bare name — `(color_code BLUE)` with `BLUE`=6 returns 60, ABI-correct. tests/cimport.rs
(`cimport_resolves_typedefs_abi_faithfully`, `cimport_enum_constants_and_defines_are_usable`).
The Phase-1 red-team carries over: bitfields, unions, `long double`, and function-pointer
params are still refused with a clear note; `_Bool`→`u8`.

## Still deferred (honestly)
unions/bitfields (overlapping/packed layout), anonymous/nested structs, function-pointer
typedefs, `long double`, and complex non-natural-alignment struct layouts — each *refused*,
not faked. The `(cimport "h.h")` MACRO (seamless compile-time import on this same
extractor) remains the natural ergonomic capstone.
