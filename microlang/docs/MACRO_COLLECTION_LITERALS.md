# Collection literals are code, not data — the macro-argument problem

## The symptom

A macro that inspects a collection-literal argument sees construction *code*, not
a collection:

```clojure
(defmacro vq [v] (list 'quote (list (vector? v) (type-of v))))
(vq [1 2])
;; => (false Vector)      ; real Clojure: (true clojure.lang.PersistentVector)
```

Inside a macro body, `(vector? [1 2])` is `false`, `(map? {:a 1})` is `false`,
`(first [1 2])` fails, and `(type-of [1 2])` is the reader tag `Vector`. In real
Clojure a macro receives an actual `PersistentVector` whose elements are the raw
(unevaluated) forms, so `vector?`, `first`, `nth`, `seq`, and destructuring all
work on it.

This is the single blocker keeping `clojure/core.match` from running. `match`
loads and expands, but it introspects its pattern literals with
`vector?`/destructuring; given a literal pattern arg it can't see the vector, so
it emits a bogus atomic compare (`(= ocr-6 [1 2])`) instead of destructuring the
row. Fed *real* collections (via a quoted `macroexpand-1`) it produces correct
code — proving the algorithm is fine and only the argument representation is
wrong.

Any macro that reflects on literal-collection arguments hits this: core.match,
core.logic, spec-style macros, most data-driven DSLs.

## Why this happens — the two representations

There is a deliberate split between how a collection literal is *read* and how it
exists at *runtime*.

**Reader** (`clojure-stub/src/reader.rs:241`): `[a b]` reads to a tagged record,
not a vector:

```rust
Tok::Open('[') => {
    let items = self.until(rt, ']');
    let lst = rt.vec_to_list(&items);      // a LIST of the element forms
    record(rt, VECTOR, vec![lst])          // Record{ type: 'Vector, fields: [(a b)] }
}
```

So `[a b]` is `Record{'Vector, [(a b)]}` — field 0 is a *list of unevaluated
element forms*. `{...}` and `#{...}` read the same way to `'Map` / `'Set`
records. These records are **code**: a description of a literal, not the literal.

**Compiler** (`clojure-stub/src/lib.rs:471`, in `expand`): a `Vector` record in
*expression position* is rewritten to a constructor call so its elements
evaluate at runtime:

```rust
} else if let Some(elems) = binding_items(rt, f) {
    let lst = rt.vec_to_list(&elems);
    build_call(rt, cs, macros, comp, "vector", lst)   // (vector a b)
```

Evaluating `(vector a b)` produces the actual runtime vector — a `PVec` (the
load-time persistent vector) or a `PersistentVector` (the cljs-ported type, once
`cljs_types` has loaded). **Runtime `vector?`/`first`/`nth`/`seq`/`count`/`get`
recognize only `PVec`/`PersistentVector`, never the reader `Vector` record.**

**Macro invocation** (`clojure-stub/src/lib.rs:365`): a macro is called with its
argument forms *raw*:

```rust
let args = rt.list_to_vec(f);
let mut margs = vec![f, nilv];         // &form, &env
margs.extend_from_slice(&args[1..]);   // the raw argument forms
let result = cs.invoke(cs, rt, mfn, &margs);
```

No expansion, no data conversion. So a literal-vector argument arrives inside the
macro as a `Record{'Vector, ...}`, and every runtime data operation on it fails.

**Why `quote`/`macroexpand-1` work.** `build_quote` (`lib.rs:1620`) converts
reader `Vector`/`Map`/`Set` records into runtime constructor calls with quoted
leaves. So `'[x]` and `macroexpand-1 '(match [x] ...)` hand the macro *real*
runtime collections, and everything works. That is exactly the gap: normal macro
calls skip this conversion.

## Why the split is load-bearing

The reader `Vector` record is not dead weight — our own macros consume it.
`defmacro`, `fn`, `let`, `loop`, and destructuring pull params out of the reader
record via `record_field0(_, VECTOR)` (6 sites in `lib.rs`) and `binding_items`
(20 sites). Importantly, **`binding_items` already accepts both** the reader
record and a runtime `PVec`/`PersistentVector`:

```rust
fn binding_items(rt, form) -> Option<Vec<u64>> {
    if let Some(lst) = record_field0(rt, form, reader::VECTOR) { return ... } // reader record
    if record_field0(rt, form, "PVec").is_some()             { return ... } // load-time runtime
    if record_field0(rt, form, "PersistentVector").is_some() { return ... } // user-time runtime
    None
}
```

So the codebase is *partially* unified already. The remaining asymmetry is that
(a) macro arguments are never converted to data, and (b) the runtime collection
predicates/accessors don't know about the reader record.

## Clojure's model, for reference

Clojure has one representation. `[1 2]` reads as an actual `PersistentVector`
holding the element forms (symbols, numbers, nested forms). Macros receive real
vectors. The compiler special-cases a literal vector *in evaluation position* by
emitting code that builds a fresh vector with each element evaluated. Code and
data are the same objects; there is no reader-only record type.

## Fix options

### Option A — teach runtime collection ops to recognize reader records

Make `vector?`/`map?`/`set?`/`seq?`, `first`/`nth`/`count`/`seq`/`get`, and `=`
treat a reader `Vector`/`Map`/`Set` record as its collection (peering into field
0's element list).

- **Pro:** smallest change to reader/compiler; unblocks macro introspection
  without touching macro invocation.
- **Con:** semantically muddy — a reader record's field 0 is a *list*, not the
  elements, so every accessor needs a special case; equality and hashing between
  a reader record and a runtime `PVec` get inconsistent; reader "code" objects
  would now answer data queries in non-macro contexts too. Fragile.
- **Verdict:** stopgap only. Do not ship as the real fix.

### Option B — convert macro arguments to data at the call site (recommended)

At the macro-invocation site (`lib.rs:365`), recursively replace reader
`Vector`/`Map`/`Set` records in each argument form (and in `&form`) with the
corresponding *runtime* collection, then invoke the macro. This matches Clojure
exactly: macros receive data.

- The macro then sees real vectors/maps/sets; `vector?`, `first`, `nth`,
  destructuring all work.
- The macro's **output** may now contain runtime collections. The compiler
  already lowers a runtime `PVec`/`PersistentVector` in expression position
  (via `binding_items` at `lib.rs:471`), so expansions still compile.
- **Risk, and it is contained:** our own macros currently call
  `record_field0(_, VECTOR)` directly on a param arg that would now be a runtime
  vector. Route those 6 sites through `binding_items` (already handles both).
- **Verdict:** correct semantics, bounded blast radius. Recommended first move.

The transform — call it `form_to_data` — is *not* `build_quote`:

| position          | `build_quote` (for `'x`)      | `form_to_data` (for macro args)          |
| ----------------- | ----------------------------- | ---------------------------------------- |
| reader Vector rec | `(vector 'a 'b)` (code)       | a runtime vector of `form_to_data(elem)` |
| reader Map/Set    | `(hash-map …)` (code)         | a runtime map/set of converted entries   |
| a list `(f a)`    | `(list 'f 'a)` (quoted data)  | keep as the list `(f a)`, recurse into elems |
| symbol / atom     | `'sym` / self                 | itself, untouched                        |

The key difference: a **list stays a list** (a macro treats `(foo 1)` as a seq of
code), while every **vector/map/set literal becomes the runtime collection**. We
recurse through lists so nested vector literals inside a call form are still
converted.

Subtlety — *which* runtime vector type to build. `binding_items` and `vector?`
must recognize whatever `form_to_data` constructs. Before `cljs_types` loads the
runtime type is `PVec`; after, user code expects `PersistentVector`. Build via
the same phase-aware path the compiler's expression rewrite uses (or pick the
type by whether `PersistentVector` is defined yet). This is the same phase issue
that already bit `binding_items`.

### Option C — full unification (eventual end state)

Make the reader emit a runtime persistent vector directly for `[...]` (no reader
`Vector` record). The compiler recognizes a literal persistent vector in
evaluation position and emits element-wise evaluation. Collapse the `VECTOR`
branch of `build_quote`, and convert all `record_field0(_, VECTOR)` sites to
`binding_items`. One representation for code and data, exactly like Clojure.

- **Pro:** removes the dual representation entirely; simplest end state; kills
  this whole class of bug.
- **Con:** largest blast radius — reader, compiler literal handling, every
  destructuring site, `quote`, quasiquote, `#?@` reader-conditional splicing
  (`reader.rs:519`), equality, and printing all touch it. Highest risk to the
  passing suites.
- **Verdict:** the right long-term target; do it deliberately with the full test
  suite as the gate, after Option B proves the data-shape works.

## Recommendation

Do **Option B now** to unblock core.match with correct semantics and contained
risk. Treat **Option C** as the eventual cleanup once B has shaken out the
data-shape and phase-type questions.

## Concrete plan for Option B

1. Add `form_to_data(rt, form) -> u64` per the table above (reader collection
   records → runtime collections; lists stay lists but recurse; leaves
   untouched). Resolve the `PVec` vs `PersistentVector` phase question up front.
2. At `lib.rs:365`, apply `form_to_data` to `f` (the `&form` slot) and to each
   `args[1..]` before `cs.invoke`. Leave `&env` nil.
3. Audit the 6 `record_field0(_, VECTOR)` sites; switch any that read a
   macro-supplied param/binding vector to `binding_items`.
4. Confirm macro *output* containing runtime collections still lowers correctly
   (it should, via `lib.rs:471`); add a case if a gap appears.

## Test gate

Everything green now must stay green: **conformance 64/64**, **JIT 11/11**,
medley smoke. Add these, which must flip from broken to correct:

```clojure
;; the representation regression test — currently (false Vector)
(defmacro vq [v] (list 'quote (vector? v)))
(vq [1 2])                                             ; => true

;; core.match end-to-end (currently returns bogus code as data)
(require '[clojure.core.match :refer [match]])
(let [x 1] (match [x] [1] :a :else :b))               ; => :a
(let [x 5] (match [x] [1] :a :else :b))               ; => :b
(let [x 1 y 2] (match [x y] [1 2] :A [_ 2] :B :else :D)) ; => :A
(let [x 9 y 2] (match [x y] [1 2] :A [_ 2] :B :else :D)) ; => :B
(let [x 9 y 9] (match [x y] [1 2] :A [_ 2] :B :else :D)) ; => :D
```

Run both tiers (tree-walk and `--features jit`) for each.

## Key code references

- `clojure-stub/src/reader.rs:241` — `[ ]` / `{ }` / `#{ }` read to tagged records
- `clojure-stub/src/lib.rs:365` — macro invocation passes raw arg forms (the fix site)
- `clojure-stub/src/lib.rs:471` — expression-position lowering of a collection to a constructor call
- `clojure-stub/src/lib.rs:1620` — `build_quote` (why `quote`/`macroexpand-1` work)
- `clojure-stub/src/lib.rs:1652` — `binding_items` (already dual-aware; the migration target)
- `clojure-stub/src/reader.rs:21` — `VECTOR`/`MAP`/`SET` tag constants
