# C-interop §6 Phase 2 — SCOPE (for review; build held for the Phase-1 verdict)

Phase 1 (cimport: functions + scalars + pointers + simple structs + refuse-unmappable +
`--link-flag`) is at the bar. Phase 2 extends COVERAGE, cheapest-high-value first, on the
SAME clang-AST extractor — same cardinal: ABI-faithful + REFUSE what it can't map (never
a silent-wrong binding). Build is HELD until Phase 1 clears (foundation-first).

## De-risk (read-only, from clang's AST)

- **Typedefs** — a `TypedefDecl` gives `name → underlying qualType`: `size_t →
  "unsigned long"`, `Point → "struct Point"`. So: build a `HashMap<name, underlying>`
  from the header's TypedefDecls, and in `map_type` resolve an unknown name through it
  (recursively, with a cycle guard). Then `size_t`→`u64`, a typedef'd struct → its
  struct, a typedef'd primitive → the primitive. Cheap + high-value (real headers are
  typedef-heavy: `size_t`, `uint32_t`, `FILE`, …).
- **Enums** — an `EnumDecl` gives the constants (`RED, GREEN=5, BLUE`) and
  `fixedUnderlyingType` (None = default `int`). So: `enum X` as a type → `i32` (or the
  fixed underlying); AND emit the enum CONSTANTS as Coil defs (`(def RED 0) (def GREEN
  5) (def BLUE 6)`) — extract each `EnumConstantDecl`'s value (explicit or sequential).
- **`#define` constant macros** — NOT in the AST (preprocessor). A SEPARATE pass `clang
  -dM -E -x c <header>` dumps every macro; filter to simple object-like constant macros
  (`#define NAME <int/float literal>`) → `(def NAME value)`. Skip function-like macros
  (`#define M(x) …`) and non-literal bodies (refuse-not-guess).

## Design (on the Phase-1 extractor)

1. **Typedef resolution**: collect TypedefDecls (header-scoped) into a table BEFORE the
   emit loop; `map_type` consults it for unknown names (recursive resolve + cycle guard).
   A typedef whose underlying is itself unmappable → still refused (the resolution just
   reaches the same refusal). ABI-faithful (clang's desugared widths).
2. **Enums**: an `EnumDecl` → emit `(def <CONST> <value>)` per constant; `enum X` as a
   value type → `i32` (or `fixedUnderlyingType`). Constant values from the AST (refuse a
   non-integer enumerator).
3. **`#define`s**: a second `clang -dM` invocation; a tiny line parser for `#define NAME
   <int-or-float-literal>` → `(def NAME v)`. Everything else (function-like, string,
   expression bodies) skipped with a clear note. (Object-like int/float constants are the
   80% that matter — feature flags, sizes, error codes.)

## Cardinal (unchanged, load-bearing)
Still: ABI-faithful + REFUSE-unmappable, never silent-wrong. A typedef to an unmappable
type, a non-integer enumerator, a function-like/expression `#define` → refused/skipped
with a clear note, NEVER guessed. The Phase-1 red-team (unions, bitfields, `_Bool`→u8)
discipline carries over.

## Decisions for the steer
1. **Order**: typedefs + enums first (same AST pass, cheap, high-value), then `#define`s
   (a second `-dM` pass)? Lean: yes — typedefs unblock most real headers immediately.
2. **Enum representation**: `enum X` type → `i32` + constants as `(def …)` — right, vs a
   Coil enum/sum? Lean: `i32` + const defs (matches the C ABI exactly; a sum would
   change the representation). 
3. **`#define` scope**: object-like int/float-literal constants only (skip
   function-like + expression macros)? Lean: yes — refuse the rest, don't evaluate C
   expressions.
4. **Then the `(cimport "h.h")` MACRO** (seamless compile-time import on this extractor)
   as the ergonomic capstone — after Phase-2 coverage, or before? Lean: after (coverage
   first; the macro is sugar over the same extractor).

## Status
De-risked read-only (typedef table, enum constants, `-dM` for #defines — all available
from clang). Build HELD for the Phase-1 verdict (Phase 2 stacks on the extractor; fix any
Phase-1 flaw before building on it).
