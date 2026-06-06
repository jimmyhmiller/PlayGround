# Plan: binding-time separation (keep program structure static across data loops)

This is the design for the **one capability** that unblocks both major goals of
this project at once:

- **simple.js deobfuscation** — the obfuscator's VM stays a runtime interpreter
  because its byte array / program counter go dynamic when the data-dependent VM
  loop residualizes.
- **self-application** (Futamura projection 2) — a partial evaluator written in
  the subset can't specialize a *looping* interpreter, because the interpreter's
  program structure (pc / AST node) goes dynamic when a data-dependent loop
  residualizes.

Empirically (see `docs/self-application-blockers.md` update + the `/tmp/se*.js`
experiments) **everything else already works**: the engine's worklist + memo
loop-tie folds, residual op-lists build and apply to dynamic input, tree-walk
interpreters fold for static-count loops. The single missing piece is below.

This is the classic **"the trick"** of partial-evaluation-of-interpreters
(Jones–Gomard–Sestoft, *binding-time improvement* / *polyvariant
specialization*). Well-charted in the literature; a real new capability here.

---

## 0. The precise problem (with ground truth)

When a **data-dependent loop residualizes**, the PE turns values that are
*static in binding-time terms* (the program counter, the AST node, simple.js's
byte array) into **dynamic** residual variables. Once the program structure is
dynamic, dispatch on it (`blocks[pc]`, `node.op`, `tape[ptr]`) can no longer
fold, and the interpreter never compiles away.

**Two confirmed mechanisms** (same disease):

1. **Dynamic if-join merge** (the flat dispatch-loop case, `/tmp/se4.js`).
   A jump `if (acc !== 0) { pc = target } else { pc = pc + 1 }` assigns the slot
   `pc` two *different but static* values in its arms. At the join the PE
   *materializes* `pc` to a dynamic loop-carried variable (`if_join` →
   `pending_joins` → `materialize`). The residual then has
   `v1004 = v1005[1]` / `v1004 = v1004 + 1` merged into one dynamic `v1004`, and
   `blocks[pc]` becomes `v1000[v1004]` — unfoldable. **Code:** `if_join` build at
   `src/js.rs:1481`, the merge in `Instr::JmpIfFalsy` (`materialize` of the
   join slots), `fn materialize` ~`src/js.rs:4166`.

2. **Loop / heap materialization** (the tree-walk recursion case, `/tmp/bfdyn.js`).
   When `while (tape[ptr] !== 0)` residualizes, the loop-carried heap is
   materialized/escaped, dragging the static AST `node` to dynamic; the recursive
   `exec(node.seq, ...)` then has an all-dynamic argument list and trips the
   `is_recursive && args.all(dynamic)` guard (`do_call` ~`src/js.rs:3568`),
   panicking "dynamic-depth recursion". (Exact escape path for `node` needs the
   step-0 diagnosis below — it is NOT simply `generalize`, which only materializes
   *differing* slots.)

**The invariant we want:** a value that takes a *bounded set of static values*
and *drives dispatch* must stay static — the control flow **splits** per static
value (polyvariance) and **reconverges via the memo** — instead of being merged
to a single dynamic variable. Only genuinely-dynamic *data* residualizes.

---

## 1. What "already works" that we build on (do not redo)

- **The memo loop-tie** ties a residual loop when a `(point, state)` repeats
  (`engine.rs` `create_or_get` / `memo`). This is the reconvergence mechanism
  polyvariance needs — splitting is safe *because* the memo re-ties matching
  states.
- **The whistle** (`Js::whistle` = `dynamically_controlled`) generalizes a loop
  whose control is genuinely dynamic. Static bounded loops already unroll; truly
  dynamic loops already residualize. We do **not** change when a loop
  residualizes — only what stays static *inside* it.
- **Partial-static modeling** already exists: `bf.rs`'s `dynamic_base +
  static_offset` pointer, scalar-replaced objects, static-key objects, `in`,
  `pop`/`shift`. The pc/node is the same idea applied to a *dispatch* value.

So this is a **refinement of the merge/materialize policy**, not a new engine.

---

## 2. Approach: online polyvariance (recommended), with offline BTA as the escalation

Two classical routes:

- **(A) Offline binding-time analysis (BTA).** A pre-pass classifies every
  variable/expression `static` or `dynamic` and specialization is driven by the
  classification (the program is static, the data is dynamic, fixed up front).
  Most principled; biggest change (a whole analysis + making specialization
  BTA-aware). This is "the textbook mix".

- **(B) Online polyvariance.** Keep the PE online (binding-times discovered while
  specializing), and fix the one place that loses static-ness: **don't merge a
  dispatch-driving static value at a join; keep the paths split, bounded by the
  memo (reconverge) and the whistle (generalize if it explodes).**

**Recommendation: start with (B).** It fits the existing online architecture,
touches one policy (the join merge + the loop-materialize), and the
reconvergence/termination machinery (memo + whistle) is already there. Keep (A)
in reserve if the online split-vs-merge decision proves too imprecise.

---

## 3. The core change (Approach B)

### 3.1 The split-vs-merge decision — the crux

At a dynamic if-join (and the loop-header materialize), for each slot whose two
arm-values differ:

- **Merge to dynamic (current behavior)** if the slot is plain *data* — used only
  in arithmetic / stored / returned. Avoids code duplication.
- **Keep split (new)** if the slot is *dispatch-driving* — used downstream as an
  array index (`a[slot]`), an object key (`o[slot]`), or a branch/switch
  discriminant. Splitting lets the dispatch fold on each path; the memo
  reconverges matching states.

**How to decide "dispatch-driving" — a lightweight static use-analysis** (not a
full BTA). Per function, per slot, compute a boolean `drives_dispatch[slot]`:
the slot's value flows (directly, or through static copies) into:
- the index position of a `GetIndex` / `SetIndexOp` (`a[slot]`), or
- the key of a `GetProp`/`SetPropOp` via a computed access, or
- a `JmpIfFalsy` / switch discriminant.

This is a backward/forward dataflow pass on the `Instr` stream (cheap, runs once
in `Js::new`, same place `if_join` / `loop_modified` are computed). Conservative:
when unsure, mark `drives_dispatch` (favor folding) — over-splitting only costs
code size, never correctness, and the whistle bounds it.

Then the merge policy: at a join, partition the differing slots into
`drives_dispatch` (keep split) vs not (merge as today).

### 3.2 "Keep split" mechanically

Today a dynamic `if` with differing slots schedules a join (`pending_joins`) and
`materialize`s the merged slots at the join, then both arms jump to one merged
block. To keep a slot split:

- **Do not materialize that slot at the join.** Let each arm carry its own static
  value past the join. The arms therefore reach the continuation with *different
  states* (different static `pc`).
- **Do not force a single merged successor block.** Each arm creates/looks-up its
  own successor via `create_or_get` (which already memoizes on the full state).
  Two arms with the *same* static dispatch value AND same data-shape memo-hit to
  one block (reconverge); different values get different blocks (the desired
  per-pc residual blocks). This is exactly how the engine already specializes its
  *own* clients polyvariantly — we're letting a JS-level dispatch value participate
  in the same mechanism instead of being merged away first.

The data slots still merge as today, so the dynamic data (tape, acc) residualizes
into shared loop-carried variables; only the dispatch slot multiplies the blocks.

### 3.3 The heap version (program structure as a heap object)

The pc case is a primitive slot; the AST-`node` / byte-array case is a **heap
object** that must stay static across a residualizing loop. The analogous rule:

- A heap object that is **loop-invariant** (not mutated in the loop body) and
  **read only for dispatch** (its fields/elements feed `node.op`, `blocks[pc]`,
  decode reads) must **not be escaped/materialized** when the loop residualizes.
- The loop/`residualize`-time materialization must escape only the **mutated /
  genuinely-dynamic** objects (the tape, the cursor), leaving static program
  objects in the abstract heap so reads against them keep folding.

Step 0 must pin down *why* `node` currently escapes (it is not `generalize`, which
skips unchanged slots — suspect the recursion's fresh return-value arrays growing
the heap, or an escape triggered by a dynamic-index/opaque-call path). The fix is
the same shape: **don't escape loop-invariant static program objects.**

---

## 4. Termination & soundness

- **Termination.** Splitting a *bounded* static set (the pcs of a fixed program /
  the nodes of a fixed AST) terminates: finitely many `(pc, data-shape)` states →
  the memo ties. Splitting an *unbounded* set (a static counter `0,1,2,…`) must
  not explode — but an unbounded static counter only arises in a loop the whistle
  *already* generalizes when the count is dynamic; when the count is genuinely
  static the loop legitimately unrolls. Backstop: the existing
  `SPEC_WEIGHT_BUDGET` and a (new, if needed) per-point split cap that falls back
  to merge. Never silently truncate — `log!` if a split is capped.

- **Soundness.** Polyvariance is observationally transparent: splitting control
  flow by a static value and reconverging identical states cannot change
  behavior (it is the same computation, specialized per path). The oracle is
  unchanged: the **fuzzer** (a wrong merge/split shows as a value/throw
  divergence) and **difftrace** on simple.js. Gate every step on both. This is a
  policy change to a soundness-sensitive area (merge/escape), so fuzz **broadly**.

---

## UPDATE 2026-06-04 (2) — milestone 1 FULLY LANDED (static accumulators too)

The static-accumulator gap below is now closed. Added the **inductor/accumulator
distinction**:

- `loop_guards[loop_head]` — the slots in a loop's HEADER condition (the `Load`s
  up to its exit `JmpIfFalsy`), i.e. induction-variable candidates.
- `loop_diverges(seen, cand)` replaces the old `state_embeds`: re-entering a loop
  point diverges (must residualize) iff some `Dyn` local grows as a superterm AND
  every other CHANGED local is a non-`Dyn`, non-`Ref`, **non-guard** value (an
  *accumulator* like `sum`, which is generalized). A changed **guard** slot is an
  *inductor* (`i` in `i<10`) ⇒ the loop is bounded ⇒ return false (keep
  unrolling). A changed **Ref** distinguishes real heap structure ⇒ return false
  (not the same iteration — this is what keeps the static program `blocks` array
  from escaping).

**Result:** `/tmp/se4.js` (flat dispatch loop, **static** accumulator `sum=0`
growing `0,2,4,…`) now folds to a clean residual loop, dispatch gone — same as the
dynamic-accumulator case. `flat_dispatch_loop_folds_the_trick` updated to the
static case. Static-count loops (`for i<10`) still unroll (the guard `i` blocks
the fire). Sound (fuzz 0 div), correct (simple.js 13/13), 121 tests. simple.js
~69s. **Milestone 1 is complete.** Next: milestone 2 (the tree-walk / recursion
heap version) — see §5.

## UPDATE 2026-06-04 (1) — milestone 1 LANDED (for the common case)

Implemented Approach B (online polyvariance + growth whistle) and it works,
soundly:

- **`drives_dispatch[func][slot]`** use-analysis (a `Load(slot)` immediately
  before a `GetIndex` = the slot is an array index) computed in `Js::new`.
- **Join filter**: in `jump_to`, a `drives_dispatch` slot that is STATIC in this
  arm is kept SPLIT (not materialized) at the pending-join merge — the pc stays
  static, the arms reconverge via the memo. (`src/js.rs` pending-join loop.)
- **Growth whistle**: `whistle = dynamically_controlled || state_embeds`, where
  `state_embeds`/`rexpr_subterm` detect a loop-carried `Dyn` value growing as a
  superterm (`acc -> acc-1 -> (acc-1)-1`). This residualizes a data loop whose
  surface condition is static (the dispatch `pc`), stabilizing the data while the
  pc stays static. The `heap ==` is checked LAST (after the cheap per-slot checks)
  so it doesn't dominate spec time.

**Result:** `flat_dispatch_loop_folds_the_trick` — a flat
`while(pc>=0){ dispatch on blocks[pc]; … }` interpreter with a data-dependent loop
and a **dynamic** accumulator compiles to a clean residual loop with the **opcode
dispatch gone**. Sound (fuzz 0 divergences), correct (simple.js difftrace 13/13),
and FAST (simple.js 47s, ≤ baseline). 121 tests green.

**Remaining gap (the deep sub-problem):** a **static** accumulator (`sum` starting
at `0`, growing `0,2,4,…` as a static `Num`) in a dynamic-count loop does NOT fold
(`/tmp/se4.js` budget-exceeds). It is locally indistinguishable from a static
induction variable — both are differing `Num`s at the loop header — so the growth
whistle can't fire on it without also residualizing legitimate static-count loops
(a regression). This is the genuine whistle-precision / partly-undecidable core.
A real fix needs to tell an *accumulator* (generalize) from an *inductor* (unroll)
— e.g. "a static value NOT used in the loop-controlling condition that grows
monotonically." Left for the next pass; the common dynamic-accumulator case (the
typical interpreter shape, and what self-application needs) is done.

## 5. Milestones (each independently verifiable)

1. **se4 folds.** `/tmp/se4.js` (flat pc-loop, data-dependent loop) must
   specialize to a residual loop with the **dispatch folded** — no `blocks` array
   in the output, no `v1000[v1004]`, just the per-op residual arithmetic inside a
   residual `while`. Smallest proof of the if-join polyvariance.
2. **bfdyn folds.** `/tmp/bfdyn.js` (tree-walk, data-dependent bf loop) must
   specialize without the recursion panic, to a residual loop over the tape with
   the opcode dispatch and the AST gone. Proves the heap version.
3. **simple.js advances.** The VM dispatch loop folds; `grep -c 'switch(__pc)'`
   and `.value[` counts drop sharply; difftrace stays 13/13. (May expose further
   layers — the CPS trampoline is still higher-order; this is necessary, maybe
   not sufficient, for simple.js. It IS sufficient for plain interpreter VMs.)
4. **Self-application step.** Hand-write the `bf` client + minimal engine in the
   subset, specialize it on a static bf program *with a data-dependent loop*, and
   get a compiled bf program (residual tape loop, no engine, no interpreter
   dispatch). This is projection-1-through-the-ported-PE with loops — the gate to
   projection 2.

Do them in order; each gates the next.

## 6. Verification plan (every milestone)

- `cargo test -p partial --release` (12) and `-p js-frontend --release` (now 120)
  stay green; add a regression test per milestone (`assert_node_equiv`).
- **Fuzzer** broadly across fresh bases (the merge/escape policy is global) —
  `./target/release/fuzz` must stay **0 divergences**. Remember the stale-`fuzz`
  binary gotcha (`rm -f target/release/deps/fuzz-*` then rebuild; confirm findings
  against `js-frontend --js`).
- **difftrace** on simple.js stays 13/13.
- Watch `SPEC_WEIGHT_BUDGET` for split-explosion (a finite-but-huge residual is the
  new failure mode; the split cap is the backstop, and it must `log!`).

## 7. Prior art (read before building)

- Jones, Gomard & Sestoft, *Partial Evaluation and Automatic Program Generation*
  (1993) — binding-time analysis, "the trick", polyvariant specialization. The
  canonical source for exactly this.
- Sørensen, Glück & Jones, *A Positive Supercompiler* — online driving +
  generalization; our whistle is this. Polyvariance falls out of driving when you
  don't prematurely merge.
- Graal *Partial Escape Analysis* (Stadler et al., CGO 2014) — the heap version:
  keep objects virtual (static) and materialize minimally, at merges, only when
  they escape. §3.3 is this applied to "program" heap objects.
- Futamura projections / Truffle `@ExplodeLoop` — the engineering precedent for
  "fold the dispatch loop, keep the program structure static."

## 8. Out of scope / do not do

- Do **not** change `src/engine.rs` (polyvariance is a JS-client merge-policy
  concern; the engine already memoizes/whistles correctly).
- Do **not** start with full offline BTA — try the targeted use-analysis +
  online split first; escalate only if it is too imprecise.
- Do **not** split *every* differing slot (that explodes and bloats every
  residual); split only `drives_dispatch` slots.
- Do **not** special-case simple.js or self-application — this is a general
  capability the fuzzer must endorse.

---

## 9. First concrete step

Diagnose + land milestone 1 (se4). Specifically:
1. Add the `drives_dispatch[func][slot]` use-analysis in `Js::new` (alongside
   `if_join`/`loop_modified`).
2. At the `JmpIfFalsy` dynamic-branch join and in `materialize`, partition the
   merge slots: keep `drives_dispatch` slots **unmaterialized** (split); merge the
   rest as today. Let `create_or_get` reconverge via the memo.
3. Gate on se4 folding + the full fuzzer + difftrace.
4. Only then move to the heap version (milestone 2), starting with the step-0
   diagnosis of why `node` escapes.

---

## 10. UPDATE — milestone 2 landed (tree-walk interpreter / heap version)

The data-dependent tree-walk interpreter `/tmp/bfdyn.js` now folds: its static
AST `node` stays STATIC across the residualizing data loop, so `switch(node.op)`
and the `exec(node.seq, …)` recursive descent compile away to a flat bytecode
loop that is node-equivalent to the source. (Before: `node` was materialized at
the dynamic loop head → became `Abs::Dyn` → `exec(node.seq,…)` hit the
"dynamic-depth recursion" panic.)

The fix is a precise, SOUND keep-static gate at the loop-head materialize in
`Js::jump_to`. A heap-ref slot live across a dynamic loop is normally
materialized; it is instead kept STATIC iff ALL THREE hold:

1. **Loop-invariant read-only** — never reassigned (`!loop_modified`) and never
   the base of a mutation in the loop body (`!bases[slot]`). Computed by
   `analyze_loop_mutations` (abstract-stack base attribution; bails → no
   invariants → original materialize-everything behavior on any loop it can't
   attribute precisely).
2. **Drives dispatch** — a value derived from the slot through ≥1 field/index
   access reaches a branch condition OR a call argument/callee. This is the
   tree-walk `node`: `switch(node.op)` (branch) and `exec(node.seq,…)` /
   `exec(node.body[i],…)` (recursive descent). Computed by
   `analyze_dispatch_drivers` (abstract-stack taint: each entry tracks its bare
   `slot` and the set of slots it is field-derived `ff` from; consuming ops fold
   `slot` into `ff`; `JmpIfFalsy` and `Call` mark the popped operands' `ff`).
   **This is the gate that keeps simple.js from bloating** — its loop-invariant
   *data* objects (config/env, read for values but never branched on) are NOT
   drivers, so they are materialized once to a variable instead of having their
   literal duplicated at every use. Without this gate simple.js was 8022 lines;
   with it, 4113 (≈ the 4097 materialize-everything baseline).
3. **Transitive heap closure disjoint from mutated objects** — at materialize
   time, `heap_touches(heap, slot_ref, mutated_ids)` walks the abstract heap from
   the slot; if it reaches any object mutated in the loop (`mutated_ids` = the
   ids the `bases` slots currently point to) the slot is materialized after all.
   This is the SOUNDNESS guard for aliasing: a closure that captured a
   loop-mutated array (regression test
   `closure_captured_array_aliases_direct_mutation`) must NOT stay static, else
   the closure reads the frozen pre-loop array (stale read).

Gate verified: bfdyn folds + node-equiv; closure-alias test passes; simple.js
13/13 difftrace and 4113 lines; se4/se4b/bf.js/memocore fold; 121 unit tests; 0
fuzz divergences over 4000 programs. `engine.rs` unchanged. Env toggles for A/B:
`DISABLE_INV` (force materialize-everything), `INV_DEBUG` (log kept-static slots).

Next: milestone 3 (simple.js VM dispatch — does the bytecode `switch(program[pc])`
advance further once the dispatch object can stay static?) and milestone 4
(self-application: the engine + a bf interpreter both in the JS subset, on a
looping program).
