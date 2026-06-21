# Design: comptime type reflection for macros

DESIGN ONLY — for Leader + jimmyhmiller review before any implementation.

## The goal (why this is THE goal-critical work)

Coil's thesis is **expressiveness ONLY through macros** — language features are
libraries over a tiny core, never baked in. That holds for everything *syntactic*
(control flow, `defer`, iterators, `fmt`, the whole stdlib). It BREAKS for anything
**type-directed**, because the macro Lisp can emit code but cannot *introspect a
type*. Today, to write `==`/`hash`/`KeyOps`/`Show`/serialization for a struct you
must either:

- **hand-write per-type ops** (the `KeyOps` pattern, friction D10) — not expressive,
  the boilerplate the goal is supposed to eliminate; or
- **bake `derive` into the compiler core** — violates the prime directive.

Neither is acceptable for the stated goal. The fix is to **let macros reflect over a
type's structure at expansion time**, so `derive(Eq/Hash)`, generic `==`, and
`KeyOps` become *macro-generated* — expressiveness through macros, made true for the
type-directed half of the language.

This is the single highest-leverage feature for Coil's actual goal: it is the
difference between "macros are powerful for syntax" and "macros are powerful, period."

## What's possible — and the timing constraint

Macros expand at the **`expand` phase** (pass 2 of `macros.rs`), *before*
resolve → check → mono → codegen. At that point:

- type **definitions** (`defstruct`/`defsum`) are present and already scanned in
  pass 1 (`scan_defs` records their names today);
- types are **not** resolved/qualified, checked, or monomorphized — so no sizes,
  offsets, or concrete layouts exist yet.

**Therefore the reflection API exposes a type's *syntactic structure*** — its fields
(name + written type) and a sum's variants — *not* its memory layout. That is
exactly what the target use cases need: `derive(Eq/Hash)` iterates fields by **name
and type** to recurse structurally; it never needs a size or an offset. (`sizeof`/
`alignof`/`offsetof` already exist as core *value* builtins for the codegen-time
needs; this is the orthogonal *compile-time-metaprogramming* half.)

So: **expand-time syntactic reflection** is the right scope. It is small, composes
with the existing two-pass expander + hygiene, and delivers the whole win.

## Recommended design

### 1. A type-shape table in the macro context
In pass 1, alongside the existing `defs: DefNames`, capture the **structure** of each
`defstruct`/`defsum` into a `types` table on `MacroCtx`:

```
TypeShape {
  kind:        Struct | Sum,
  type_params: Vec<String>,          // generic params, e.g. [T] for (Pair T T)
  fields:      Vec<(name, type_sexp)>,        // struct: each field's name + written type
  variants:    Vec<(name, Vec<(name, type_sexp)>)>,  // sum: variant name + its fields
}
```

`type_sexp` is the field's type **as a `Sexp`** (e.g. `i64`, `(ptr T)`, `(slice u8)`,
`Point`), so a macro can splice it straight into generated code. Population reuses the
existing parser's `parse_defstruct`/`parse_defsum` (run early on the raw forms) rather
than re-parsing Sexps by hand — one source of truth for "what a struct is."

Like `defs`, macro-generated type definitions are folded in via `macro_note_defs`, so
a macro-generated `defstruct` is reflectable by a *later* macro (incremental, matching
the proper-Lisp namespace rule already in place).

### 2. Comptime reflection builtins
New builtins in the macro evaluator (`eval`'s builtin dispatch), each taking a type
**name symbol** and returning ordinary macro `Value`s (lists/symbols) the macro
already knows how to fold over with `map`/`first`/`rest`/quasiquote:

- `(struct-fields T)` → list of `(field-name field-type)` pairs.
  `(struct-fields Point)` ⇒ `((x i64) (y i64))`.
- `(sum-variants T)` → list of `(variant-name (field…))`.
  `(sum-variants Shape)` ⇒ `((Circle ((r i64))) (Rect ((w i64) (h i64))))`.
- `(type-kind T)` → `:struct` | `:sum` | `:scalar` | `:unknown` — lets a macro branch
  (e.g. generic `==` recurses on aggregates, `icmp-eq`s scalars).
- `(type-params T)` → `(T …)` for a generic type (for generic derives), `()` otherwise.

The type name resolves against the **current expansion module + imports**, reusing the
`macro_qualify` path — so `(struct-fields Point)` finds `app.Point` the same way a
template symbol reference would. The returned `field-type` Sexps are spliced into the
macro's output and resolve normally through existing hygiene.

### 3. How a `derive` becomes a library macro (the payoff)
```clojure
; lib/derive.coil — PURE LIBRARY, no compiler support
(defmacro derive-eq [tname]
  (let [fields (struct-fields tname)]
    `(defn ~(symbol "eq-" tname) [(a ~tname) (b ~tname)] (-> bool)
       ~(eq-fold fields (quote a) (quote b)))))      ; (and (== a.f b.f) …)
```
`eq-fold` is a comptime helper (an ordinary `def` lambda) that folds the field list
into `(and (field-eq f1) (field-eq f2) …)`, emitting `(icmp-eq (load (field a f))
(load (field b f)))` for a scalar field and `(eq-FieldType …)` for an aggregate one
(recursion via the field type's own derived `eq`). Validated against today's
primitives: `field`/`load`/`icmp-eq` + the `and` macro already produce correct code
for `(a (ref Struct))` params (#4 made aggregate params by-ref, so `(field a f)` works).

`KeyOps` for a struct key, generic `==`, `Show`/print, and (later) serialization all
follow the same shape — each a library macro over `struct-fields`/`sum-variants`.

## Alternatives considered

- **Full Zig-style comptime** (comptime evaluation of Coil *functions*, comptime
  values flowing into types, `Array(T, comptime_len)`): the eventual destination
  (FUTURE_WORK §8), but a much larger project touching the evaluator, the type system,
  and generics. The reflection bridge is the **subset that delivers derive/KeyOps now**
  without that machinery. Recommend the bridge first; full comptime later.
- **A `derive` keyword in the compiler core**: violates the prime directive — the whole
  point is that derive is macro-able. Rejected.
- **Post-check reflection (with layouts/sizes/offsets)**: more information, but the
  wrong phase — macros run before check. It would require a *second* macro-expansion
  pass after type-checking, a large architectural change, for information the target
  use cases don't need. Rejected for v1.
- **Reflection builtins that re-parse Sexps on demand** (no table): simpler to add but
  duplicates the parser's notion of a struct and misses macro-generated types.
  Rejected in favor of the pass-1 table reusing `parse_defstruct`/`parse_defsum`.

## Decisions needed (for the review)

1. **API surface** — the four builtins above (`struct-fields`, `sum-variants`,
   `type-kind`, `type-params`) and the exact shape of the returned descriptors
   (`(name type)` pairs as lists — OK?).
2. **Type naming at the call site** — a bare symbol `(struct-fields Point)` resolved
   via the macro's module (recommended, matches template hygiene), vs. an explicit
   quote.
3. **Recursion for aggregate fields** — generate a call to the field type's derived op
   (require the user derive those too), vs. a generic-`==`/protocol dispatch. (Lean:
   call the field type's op; simplest, explicit.)
4. **Scope** — expand-time *syntactic* reflection now (recommended), with full comptime
   deferred to FUTURE_WORK §8.
5. **Incremental visibility** — should reflection see macro-generated types (via
   `macro_note_defs`), or only source-written ones in v1? (Lean: include
   macro-generated, it's nearly free and matches the existing def rule.)
6. **Error semantics** — `(struct-fields NotAStruct)` / unknown type / a `(type-kind)`
   of `:scalar` used where fields are expected → hard `error` at expansion (no silent
   empty list), per the no-silent-wrong rule.

## Why this is the right next milestone
Perf is at `cc -O3` parity, control is beyond C, and the prime directive has held
through Phase-1/2 — the one gap that *defines* the goal is that type-directed features
still can't be macros. This closes it. After the bridge lands, `lib/derive.coil`
(eq/hash/show) + a string/struct `KeyOps` derive become the proof, and the friction
D10 ("no generic ==/hashing/derive") is answered the Coil way: as a library.
