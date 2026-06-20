# Phase-1 dogfood — friction report

Written after building the full Phase-1 stdlib (slice / arraylist / mem / hashmap /
fmt) and a working dogfood program (`examples/calc.coil`, a tiny expression-language
interpreter: lexer → ArrayList, recursive-descent parser → AST sum, eval → HashMap
env, fmt output). Brutally honest: every rough edge I hit, so it can set the
Phase-2 agenda (per FUTURE_WORK §13). Ordered by how much it hurt.

## A. Aggregates & references — the biggest day-to-day friction cluster

The mental model "a struct/sum passed/bound by value is actually an immutable
reference" leaks everywhere and is the #1 source of confusing errors:

1. **Can't pass an rvalue to a by-value aggregate param.** `(hm-new a (scalar-keyops K))`
   failed ("expects a reference … got KeyOps"); had to `(let [o (scalar-keyops K)] (hm-new a o))`.
   A by-value param wants a *place* to borrow, so a function-call result isn't accepted.
2. **Returning a `(mut)`-bound aggregate needs explicit `(load x)`.** The binding is a
   place, not the value; `lex` ended in `toks` and failed until `(load toks)`. Easy to miss.
3. **Whole-aggregate `store!` is INCONSISTENT.** Storing a by-value **struct** param
   through a pointer failed (`(store! (field m ops) ops)` → "(ref KeyOps) vs KeyOps"),
   but storing a by-value **sum** param worked (`(store! p x)` for an `Ast` in `node`).
   Same syntactic shape, different result — surprising. Worked around by copying the
   struct field-by-field, or holding it behind a `(ptr T)` vtable.
4. **`(mut x)` does not coerce to `(ptr T)`** at call sites — *neither* for `call-ptr`
   *nor* ordinary calls. Forces `(alloc-stack T)` to obtain a real pointer wherever a
   function wants `(ptr T)`. Hit repeatedly (call-ptr args; the parser cursor `P`;
   hashmap key spill).
5. **Re-passing a `(mut)` param needs explicit `(mut x)`** — can't pass the param bare
   to another `(mut)` param.
6. **No `(ref T)` syntax** for an immutable-reference param; the by-value-is-a-ref rule
   is implicit. A spelled `(ref T)` (and consistent value vs ref semantics) would help.

*Highest-leverage fix:* make aggregate value/ref/store semantics **uniform and
explicit** (structs == sums; one rule for store!; an immutable `(ref T)`; auto-borrow
a place to `(ptr T)` where unambiguous).

## B. Control flow

7. **`break`/`continue`/`return-from` type as `i64`, not a bottom/`Never`.** So
   `(if c (do …non-i64…) (break))` is a branch-type mismatch — I added trailing `0`s in
   the parser loops to force both branches to `i64`. A `Never` type that unifies with
   anything removes this cleanly and is a tidy Phase-2 item.

## C. Result/Option ergonomics

8. **No `?`/`try`.** Every fallible call (io/alloc/hashmap all return `Result`/`Option`)
   is a `match` pyramid. A `try`/`?` macro would be a huge ergonomic win — and it's
   **macro-able right now** (early-exit via `return-from`/`break`, exactly like `block`).
   Strong candidate for the next macro.

## D. Stdlib gaps (what I wanted and didn't have)

9. **No real STRING type** — strings are C-`(ptr i8)`. Multi-char identifiers / string
   map keys need a manual content-hash `KeyOps` + `strlen`/`strcmp` externs. The calc
   uses single-char vars (`i64` keys) to dodge this. A `(slice u8)`/string type +
   string-keyed HashMap convenience is the **biggest** stdlib gap.
10. **No generic `==` / hashing / ordering** as language facilities — every generic
    container needs explicit ops (the `KeyOps` pattern). Philosophically fine, but a
    derive/interface story (or comptime reflection) would cut real boilerplate.
11. **No unsigned div/rem readily** (`idiv`/`irem` are signed) — hex/uint printing
    assumes non-negative.

## E. Inference

12. **Nested generic-call inference gap.** `(subslice (slice-of a 5) 1 4)` fails to infer
    `T`; bind the base first. Generic `T` doesn't flow through a nested generic call's
    result type.

## F. What worked well (so we don't regress it)

- `defer`/`scope`, `for-in`, the iterator protocol, `fmt` format strings, HashMap with
  explicit `KeyOps`, and recursive AST sums all composed cleanly once the idioms were known.
- **extern-dedup** (fixed this phase) made the stdlib genuinely composable.
- The macro system carried the entire phase: while/for/defer/early-return/iterators/fmt
  are all userland over a tiny core.

## Suggested Phase-2 ordering (by leverage)

1. Aggregate value/ref/store **consistency** + auto-borrow to `(ptr T)` (cluster A) —
   the most pervasive friction.
2. `Never` type for divergent expressions (B7) — small, removes a recurring papercut.
3. `?`/`try` macro for Result/Option (C8) — macro-able now, big ergonomic payoff.
4. Real string type + string-keyed HashMap (D9) — biggest capability gap.
5. Nested-generic inference (E12); unsigned arithmetic surface (D11).
