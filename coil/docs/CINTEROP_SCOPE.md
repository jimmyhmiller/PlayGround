# C-interop dominance (§6) — SCOPE (for review before building)

Jimmy's vision: make Coil a **drop-in for C codebases** — `cimport` (auto-import C
headers) + `coil cc`-style linking (pull in C libraries seamlessly). Manual FFI already
works (cinterop proved it — externs, struct-by-value, callbacks, the C ABI); this makes
it EFFORTLESS. Scoped freestanding/concurrency-style: evidence-backed, anti-shortcut
flagged, decisions surfaced.

## Finding 1 — the gap

Manual FFI is correct but TEDIOUS + error-prone: every C function needs a hand-written
`(extern name :cc c [param-types…] (-> ret))` whose types must EXACTLY match the C
header (get a width wrong → silent ABI corruption). For a real C library (dozens–
hundreds of functions + structs) that's prohibitive. `cimport` removes it.

## Finding 2 — a REAL header import is feasible (de-risked)

The anti-shortcut mandate: `cimport` must be a REAL header parse, not a hand-maintained
binding hack. PROTOTYPED: `clang -Xclang -ast-dump=json -fsyntax-only -x c <header>`
emits the full machine-readable AST; walking it extracts exactly what's needed —
verified on a sample header:
- `int add(int a, long b)`  → `FN add :: int (int, long)`
- `double scale(double x)`  → `FN scale :: double (double)`
- `char *greet(const char*)`→ `FN greet :: char *(const char *)`
- `struct Point { int x; int y; }` → fields `[(x,int),(y,int)]`

So `cimport` = invoke clang → walk the AST → map C types → emit Coil `extern`s (+
`defstruct`s for records). clang does the real parsing (handles the actual C grammar,
includes, typedef desugaring); Coil never hand-rolls a C parser. Anti-shortcut satisfied.

## Design

- **`coil cimport <header.h> [names…]`** — a CLI subcommand: runs clang's JSON AST,
  maps C types → Coil types, emits a `.coil` bindings file (`extern`s with `:cc c` +
  `defstruct`s). Then `(import "header.coil")`. (A CLI tool FIRST: simplest, composable,
  no macro-system surgery. A `(cimport "header.h")` MACRO that shells out at expansion
  is a later ergonomic, built on the same extractor.)
- **Type mapping** (C → Coil): `int`→`i32`, `long`/`long long`→`i64`, `short`→`i16`,
  `char`→`i8`, `unsigned …`→`u…`, `float`→`f32`, `double`→`f64`, `void`→`void` (the new
  return type!), `T*`→`(ptr <T>)`, `char*`→`(ptr i8)`, `struct X`→`(defstruct X …)`
  ABI-compatible, function pointers→`(fnptr c …)`. `const`/`restrict` qualifiers
  dropped (no Coil equivalent; harmless for ABI).
- **Linking C libraries** (`coil cc`): extend the build to pass `-l<lib>` / C object
  files to the existing `cc` link step (a generic `--link-flag`/`-l` passthrough, the
  same minimal mechanism considered for freestanding). Optionally compile a `.c` file
  via clang and link it. So a Coil program can call into any installed C library.

## Anti-shortcut + minimal-first scope (honest coverage)

Phase 1 = the 80%: **functions + scalar/pointer types + simple structs** (most C APIs).
Deferred + HONESTLY NOTED (not faked, not silently mishandled — emit a clear
"unsupported: <construct>" rather than a wrong binding):
- typedefs → resolvable via the AST's desugared type (likely Phase 1/2).
- enums → `const i32` values (Phase 2).
- `#define` constant macros → NOT in the AST (preprocessor); need a separate clang pass
  (`-dM`) or out-of-scope v1 — noted.
- unions, bitfields, varargs (`...`), anonymous/nested structs, `long double`,
  `_Complex` → Phase 2+ / noted-unsupported. cimport must REFUSE what it can't map
  correctly (no silent wrong binding — the ABI-corruption risk is the whole point of
  doing this right).

The cardinal for the eventual bar: a REAL clang-parsed header import (not hand bindings)
+ correct type mapping (a generated extern calls the real C function correctly, verified)
+ honest coverage (refuses/【clearly-notes】 unmappable constructs, never emits a wrong
binding).

## De-risk status + first build increment
DONE: clang JSON-AST extraction (functions + structs) — the parse is real + tractable.
NEXT de-risk (first build step): generate a Coil `extern` for a real libc function from
the clang AST, compile + CALL it, verify it runs (end-to-end: clang → mapped extern →
working call) — e.g. `sqrt` from `math.h` → `(extern sqrt :cc c [f64] (-> f64))` →
`sqrt(1764.0)=42`. Proves the generated binding is ABI-correct, not just syntactically
plausible.

## Decisions for the steer
1. **`cimport` as a CLI tool** (`coil cimport h.h → .coil`) vs a macro builtin (shells
   out at expansion)? Lean: CLI tool first (simplest, composable); macro later.
2. **clang JSON-AST subprocess** vs **libclang** (link the C API)? Lean: JSON-AST
   subprocess — no libclang linking dependency; clang is already the toolchain.
3. **Phase-1 coverage**: functions + scalars + pointers + simple structs (typedefs if
   cheap); enums/#defines/unions/varargs deferred-and-noted. Right cut?
4. **Linking**: a generic `-l`/`--link-flag` build passthrough (+ maybe a `(link-lib
   "m")` source directive later)? Lean: build-flag passthrough first.
