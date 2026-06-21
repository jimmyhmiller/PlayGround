# Phase-2 dogfood #2 — friction report (JSON parser)

Written after building `examples/json.coil` — a JSON parser + recursive value model
exercising the Phase-2 features *at scale*: `(slice u8)` strings + `str-keyops`, a
recursive sum (`Json`) with collection pointers, generic containers
(`ArrayList<Json>`, `HashMap<(slice u8) Json>`), recursion, references, and the
control-flow macros. It works (`coil run examples/json.coil` ⇒ 42). As with the
Phase-1 dogfood, this records every rough edge so real usage — not speculation —
sets the next agenda.

## Headline: compile-speed-at-scale is NOT the bottleneck (the flagged risk)

The goal-status flagged whole-program monomorphization + the tree-walking macro
interpreter as the one untested compile-speed risk. Measured on this dogfood (84
monomorphized functions — every `ArrayList<Json>`/`HashMap<…>`/`Option<…>`/`create`
instantiation):

- **macro expand** (the tree-walking interp): **0.02s** user CPU.
- **front-end + mono + LLVM codegen** (`emit-ir`): **0.02s** user CPU.
- full `build`: ~4.4s *real*, but only ~0.2s user — the rest is **fixed overhead**:
  LLVM library init (~2s) + the `cc` link against the LLVM static libs (~2.5s),
  neither of which scales with program size.

So at realistic program sizes the mono/macro-interp cost is negligible; the
user-visible "compile time" is the **fixed LLVM-init + link overhead**. That's a
known, separate item (FUTURE_WORK §11: a fast non-LLVM debug backend / incremental
relink), NOT the whole-program-mono risk. The mono risk is real only at 100k+ LOC
(§12) — unreachable with one dogfood. **Verdict: don't prioritize compile-speed
yet; the front-end is fast.**

## Friction (ordered by how much it hurt)

1. **No wildcard / default `match` arm (the #1 friction).** `match` must list EVERY
   variant. The `Json` sum has 6 variants, so every query (`as-num`, `arr-get`,
   `obj-get`) spells out all 6 even when 5 return the same default — and `(_ [] …)`
   is rejected (`_` is read as a variant name). For sum-heavy code (interpreters,
   ASTs, JSON) this is pervasive boilerplate. A `_`/`else` catch-all arm (covering
   the remaining variants; the switch `default` block instead of `unreachable`) is
   the highest-leverage next item. Small, checker + parser + codegen.

2. **Recursive-sum + heap-collection dance is verbose.** A `JArr`/`JObj` holds a
   `(ptr (ArrayList Json))` for recursion, built with
   `(unwrap-ptr (create [(ArrayList Json)] a))` + `store!` + the pointer. Correct,
   but a `heap-init`-style helper (`(box a expr)` → allocate + store + ptr) would
   remove the ceremony. Library macro candidate.

3. **No HashMap iteration.** There's `hm-get`/`hm-len` but no `hm-for`/entries, so
   an object can be QUERIED but not serialized/walked. Stdlib gap — an `hm-for`
   over occupied slots (like `al-for`) is straightforward library.

4. **`and`/`or` are 2-ary only.** Multi-way conditions nest:
   `(or a (or b (or c d)))` (e.g. the whitespace test). A variadic `and`/`or`
   macro (folding to the 2-ary core form) would read better. Small macro.

## What worked well (don't regress)

- `(slice u8)` strings as views: parsing string values as `subslice` of the source
  (zero-copy) was clean; `str-keyops` made `HashMap<(slice u8) Json>` object keys
  just work (content-keyed).
- Recursive sums + `match` + the generic containers composed correctly under heavy
  nesting (arrays of objects of arrays).
- The reference model carried it: aggregates pass cleanly, `(field p …)` on the
  cursor, rvalue Json values spill into containers (#4a), `(ptr …)` collections
  satisfy `(mut)` params (#4e).
- 84 monomorphized functions with no inference hand-holding (the #6 nested-inference
  fix paid off — nested generic container calls inferred without bind-first).

## Suggested next agenda (friction-driven)

1. **Wildcard `match` arm** (friction 1) — pervasive, small, highest leverage.
2. `hm-for` iteration + a `box` heap-init macro (frictions 2–3) — stdlib/library.
3. Variadic `and`/`or` (friction 4) — small macro.
4. Compile-speed: deprioritized (front-end is fast; the cost is fixed LLVM/link
   overhead — revisit only with a fast debug backend, §11, if iteration speed bites).
