# Handoff: fix simple.js effect #4 (the byte-array duplication)

> **STATUS: FIXED (2026-06-03).** Implemented the §5 recommendation (escape at
> creation, driven by the §5.4(A) duplication profiler). `difftrace` now reports
> all 13 effects identical; simple.js specializes cleanly (exit 0) and the residual
> is smaller (~4k lines). Changes are in `src/js.rs`: `Js::specialize_program`
> (profile -> flag -> re-specialize to a fixpoint), `dup_counts`/`creation_pc`
> tracking, eager `escape` at flagged `NewArray`/`NewObject`, and a
> `FORCE_ESCAPE_ALL` transparency mode with the `escape_at_creation_is_transparent`
> regression test. The normal fuzzer adds 0 new divergences/panics (the try/catch
> panic + throw-mismatch it finds are pre-existing, verified on the pre-fix binary).
> See the resolution note in `docs/HANDOFF.md` for the full summary. The rest of
> this document is the original (still-accurate) diagnosis and plan.

---


This is a self-contained handoff for ONE task: make the residual of `simple.js`
observationally equivalent to the original past `Date.now` effect #4. Read
`docs/HANDOFF.md` for the broader project state; this doc is everything you need for
this fix and is written to favor the **most correct** change, not the fastest.

The bug has been fully diagnosed, root-caused, and an obvious-but-wrong fix has been
tried and ruled out with evidence. What remains is a real, principled change to the
partial evaluator's object-identity model. Budget it as a focused multi-day effort,
not a patch.

---

## 0. TL;DR

- **Symptom:** `simple.js`'s residual throws / decodes wrong starting at the 4th
  `Date.now` effect. `tools/difftrace.js` reports `FIRST DIVERGENCE at effect #4`.
- **Proven root cause:** the partial evaluator materializes ONE abstract heap object
  (the obfuscator's byte array) into **multiple distinct runtime objects**. The
  loader decrypts one copy; the decoder reads a different copy; the decrypt is
  therefore invisible to the read ("off by `^143`").
- **Where:** `materialize` in `src/js.rs` re-constructs the same abstract array
  (instrumented: slot 6, addr 44, `Array[1197]`) **4 times** at different
  control-flow merge points, each emitting a fresh `NewArray`.
- **The invariant being violated:** one abstract heap object must correspond to at
  most one runtime object. Today `materialize` can emit a fresh construction of the
  same abstract object at every loop-header/branch-join it is carried through.
- **Why it is not a one-liner:** the only sound way to share a single runtime
  instance is to construct it ONCE at a point that **dominates** every use. A flat
  "materialize once, reuse the var" memo was implemented and **fails**: the reused
  var is referenced in blocks its construction does not dominate (proven: it throws
  `Cannot read properties of undefined (reading 'value')`). See §4.
- **Recommended fix:** escape such objects **at their creation site** (`NewArray` /
  `NewObject`), which is the one program point guaranteed to dominate all uses. Drive
  it with a precise trigger (a profiling pre-pass; §5). This preserves identity by
  construction.

---

## 1. How to reproduce and observe the bug

```bash
# specialize simple.js -> residual (large; ~14k lines)
SPEC_WEIGHT_BUDGET=999999999 ./target/release/js-frontend \
  --js /Users/jimmyhmiller/Documents/Code/deob/simple.js > /tmp/r.js

# the oracle: differential external-effect trace (Date.now / console)
node tools/difftrace.js /Users/jimmyhmiller/Documents/Code/deob/simple.js
#   -> "original effects: 13, residual effects: 5; FIRST DIVERGENCE at effect #4"
```

The fix is done when difftrace reports **no divergence** (the residual reproduces all
13 effects) and `simple.js` still specializes cleanly (`exit 0`).

### Runtime instrumentation that PROVED it is duplication (not ordering)

An earlier handoff guessed "effect ordering, same object." That is **stale/wrong**
for the current code. The decisive experiment tags the byte arrays with a WeakMap id
and logs every read/write at the diverging index (1131). Reproduce with the
`/tmp/instr3.js` pattern (recreate it; it is short):

- strip the `--- residual ... ---` banner line(s) from `/tmp/r.js` (else it parses as
  `--` `-` and throws "Invalid left-hand side in prefix operation");
- replace the decoder read line
  `v1262 = v900192.value[v900193.value][v600151];`
  with a logging wrapper that records `["R", id(innerArray), index, value]`;
- replace the two decrypt writes
  `v1646[v1647] = (v1646[v1647] ^ 143);` and `… ^ v1643);`
  with wrappers that record `["W", id(innerArray), index, before, after]`;
- run `main(0)` in a `vm` context whose globals are the reals
  (`Date, String, Uint8Array, TextDecoder, …`), then filter the trace for index 1131.

Observed result (this is the ground truth to preserve):

```
events at index 1131: [["W",1,1131,152,23],["W",2,1131,94,209],["R",3,1131,152]]
```

Three distinct object ids. The decrypt writes obj#1 (`152 ^ 143 = 23`) and obj#2; the
read hits obj#3, which still holds the un-decrypted `152`. The original reads `23`.
That is the divergence, and it is an **identity** bug.

---

## 2. The program being specialized (what simple.js does)

`/Users/jimmyhmiller/Documents/Code/deob/simple.js` is a multi-layer obfuscator:

- A byte array `v5` (an array of chunk arrays) is built ONCE
  (`simple.js:259`, inside `while (v10 >= 0) switch (v10 & 1) { case 1: v5 = [...] }`,
  an init loop that runs effectively once).
- A **decoder** reads it one byte at a time (`simple.js:35`):
  ```js
  function v1() { v3 = v5[v6][v7++]; v4 = v3 ^ v8; v8 = (v8 + v4 + 1) & 255; return v4; }
  ```
- A **loader** decrypts ranges of it in place (`simple.js:1583`,
  `v11[v9 + v25] ^= v3;` where `v11 = v5[v6]`).
- Decode and decrypt run interleaved inside a bytecode VM loop.

The chunk index `v6` is static at the hot accesses; the byte index (`v7`, `v9+v25`) is
dynamic. So the OUTER array stays partial-static (folds, gets re-materialized), while
the INNER chunk is what is read/written dynamically and what gets duplicated.

The decoder is residualized as `__rf3` (a runtime closure that receives the array box
as a parameter), so the read is already a runtime operation:
`v900192.value[v900193.value][v600151]`. The decrypt is a runtime `SetIndex`. Both are
runtime; the bug is purely that they operate on **different runtime objects**.

---

## 3. The exact mechanism (where the duplication is emitted)

`materialize` (`src/js.rs`, the loop/branch materializer called from `generalize` and
the join logic) walks the loop-carried / merged slots and, for each `Abs::Ref(addr)`
slot, emits a construction (`NewArray` / `NewObject` / nested) and sets the slot to
`Var(stable_id)`.

A debug probe (a temporary `eprintln!` over large Ref slots at the top of
`materialize`, gated on an env var) showed:

```
[MAT] slot=6 addr=44 Array[1197]   (printed 4 times)
```

Same abstract addr (44), four separate `materialize` calls at four different merge
points, four fresh constructions → four runtime arrays. There is **no cross-call
memory** that "addr 44 already lives in residual var X", so each merge point rebuilds
it. (Re-add this probe to confirm in your environment; remove before committing.)

A related, DIFFERENT bug was found and FIXED this session: *within* a single
`materialize` call, an object reachable from two slots (held directly AND captured by
a closure) was materialized twice because each Ref slot used its own `seen` map. That
is fixed (one shared `seen`, plus ordering objects before closures); see
`materialize_into_seen` and `docs/effect4-minimal-repro.js`. THAT fix does not touch
simple.js (dup count unchanged at 52, still effect #4). Do not confuse the two: this
task is the **across-`materialize`** (cross-merge-point) duplication.

---

## 4. The hard constraint: dominance (and the fix that was tried and reverted)

The naive fix is "materialize each abstract object at most once: keep a global
`addr -> residual_var` memo, emit the construction the first time, and on later
materializations just reference the var." This was fully implemented (memo keyed by
`addr`, value `(var, content-snapshot)`; reuse only when the snapshot is unchanged so
a genuinely-mutated object still re-materializes; the memo cleared/restored around
residual-function specialization since those vars are out of scope).

Result, measured:

- simple.js array-duplication dropped **52 → 5** (the memo does dedup), AND
- it then threw `TypeError: Cannot read properties of undefined (reading 'value')` at
  effect #0 — i.e. it got WORSE.

Why: the construction was emitted at whichever block first materialized the object,
and later blocks referenced that var. But that block does **not dominate** the later
blocks (they are on different control-flow paths through the trampoline). At runtime
a block can reach a reference to the var before the block that constructs it has run →
`undefined`.

**Conclusion (do not relitigate this): a flat materialize-once memo is unsound.**
Sharing one runtime instance is correct ONLY IF its single construction dominates
every reference. The fix must guarantee dominance.

---

## 5. Recommended fix: escape at the creation site (the universal dominator)

### 5.1 The invariant to enforce

> Each abstract heap object is realized as **at most one** runtime object, and once an
> abstract object is "escaped" (committed to a runtime instance) it is **never
> re-materialized as a fresh literal** — every reference resolves to its single
> canonical residual variable.

### 5.2 Why "at creation"

The construction must dominate all uses (§4). The set of program points that dominate
*every* use of an object is exactly the points at or before its creation. The
**creation site** (`NewArray` / `NewObject`) trivially dominates all uses — you cannot
use an object before you create it — and it is a single, well-defined point. No other
candidate (first write, first read, loop header) is guaranteed to dominate (a
loop-header construction fails for pre-loop uses; a first-write construction fails for
the first iteration's read). So: when an object needs a stable runtime identity,
**construct it once, as a runtime variable, at its creation, and mark its abstract
form as already-escaped.**

This is the existing partial-escape machinery (`escape` / `escape_var(addr)`), but
applied eagerly at creation instead of lazily at first residual use.

### 5.3 What "escape at creation" means concretely

At the `NewArray` / `NewObject` step for an object that needs identity:

1. Allocate the canonical residual var `Js::escape_var(addr)` (already stable per
   addr).
2. Emit its construction there: `escape_var(addr) = [ ... ]` (folding any
   statically-known elements; non-static elements escape recursively).
3. In the abstract heap, represent the object as **already escaped** — i.e. push
   `Abs::Dyn(RExpr::Var(escape_var(addr)))` (a runtime reference) rather than
   `Abs::Ref(addr)` into the operand stack, AND ensure nothing later treats addr as a
   foldable static object. The cleanest representation is a new heap state "escaped"
   (see §6.1) so that any subsequent `GetIndex` / `SetIndexOp` / `GetProp` /
   `SetPropOp` on it residualizes against the var (which they already do for a
   `Dyn` base), and `materialize` never reconstructs it.

After this, all reads and writes to the object are runtime `Index` / `SetIndex` /
`Get` / `SetProp` against `escape_var(addr)` — one instance, correct identity. The
construction at creation dominates the loop, so the loop simply carries the var.

For simple.js this means: the byte array (and its chunks) is built once at the top of
`main`, every decrypt is a runtime `SetIndex` on it, and the decoder closure reads the
same array. Identity preserved; effect #4 fixed. Little folding is lost because the
decode/decrypt are already runtime.

### 5.4 The trigger: which creation sites escape eagerly?

Escaping EVERY object at creation would gut the partial-static optimization (e.g. a
static lookup table indexed by a constant would stop folding). So escape eagerly only
when the object genuinely needs runtime identity: it is **mutated and/or read through
residualized (runtime) operations while live across a loop** — i.e. exactly the
objects that would otherwise duplicate.

Pick ONE of these triggers (ranked by correctness/precision; all are sound — over-
escaping only costs folding, never correctness):

**(A) Profiling pre-pass (recommended — precise and direct).** Specialize twice:

1. *Phase 1* — specialize normally, but instrument `materialize` to record, per
   abstract `addr`, how many times it was constructed. Track each heap object's
   **creation pc** (store it in the `HeapObj`, or in a side map `addr -> creation_pc`
   maintained by `alloc`/the `NewArray`/`NewObject` steps). Any addr constructed more
   than once → its creation pc is a "must escape eagerly" site.
2. *Phase 2* — re-specialize with those creation pcs flagged; at a flagged
   `NewArray`/`NewObject`, escape at creation (§5.3).

This is precise (only the objects that actually duplicated are escaped), and correct
by construction (creation dominates). Cost: a second specialization pass. For full
robustness, iterate phases to a fixpoint (phase 2 could in principle expose a new
duplication); in practice one extra phase is expected to suffice — gate on difftrace +
the fuzzer.

**(B) Static analysis pre-pass (one specialization).** Before specializing, analyze
the bytecode: a creation site escapes eagerly if its object can be (i) live across a
loop back-edge AND (ii) the base of a dynamic-index access or flow into opaque
code/closure capture. This is a standard reachability + escape analysis on the
`Instr` stream. It needs no second pass but is more conservative (it cannot always
tell a static index from a dynamic one pre-specialization, so it over-escapes more).

Recommendation: implement **(A)**. It targets exactly the failing objects, keeps
folding for everything else, and its correctness argument is the cleanest ("we escape
precisely the objects that were proven to duplicate, at the one point that dominates
their uses").

---

## 6. Implementation guidance

All changes are in the JS client `src/js.rs`. The generic engine (`src/engine.rs`)
must NOT be changed — object identity is a JS-semantics concern.

### 6.1 Represent "escaped" objects so they are never re-materialized

The current heap is `Heap = im::OrdMap<usize, HeapObj>` with
`HeapObj = Object | Array | Closure | Builtin`. An escaped object needs to stop being
a foldable static object. Two viable representations:

- *Preferred:* add `HeapObj::Escaped(usize)` (the residual var id). Reads/writes on an
  `Abs::Ref(addr)` whose heap entry is `Escaped(var)` residualize against
  `RExpr::Var(var)` (mirror the existing `Abs::Dyn` base paths in `GetIndex` /
  `SetIndexOp` / `GetProp` / `SetPropOp`). `materialize` and `mat_inline*` treat
  `Escaped(var)` as `RExpr::Var(var)` (no construction). `escape()` already replaces
  references with `Dyn(Var(escape_var(addr)))`; align with that.
- *Alternative:* don't keep a heap entry at all — at creation push
  `Abs::Dyn(RExpr::Var(escape_var(addr)))` directly. Simpler, but you lose the ability
  to know the var came from this addr (matters if multiple slots must alias — the
  `Dyn(Var)` already carries the var, so this is usually fine).

Either way the key property: **once escaped, the object is a `Var` everywhere; no code
path can turn it back into a static `Ref` that gets re-materialized.** This is exactly
what the flat memo failed to guarantee (its abstract object stayed a static `Ref` at
memoized loop headers).

### 6.2 Escape at creation

`Instr::NewArray` / `Instr::NewObject` handlers (`src/js.rs`, ~line 2020): if the
creation site is flagged (§5.4), after building the object, immediately escape it:
materialize it into `escape_var(addr)` (folding static elements), record it as escaped
(§6.1), and push the escaped value. Reuse `materialize_value` / `mat_inline` for the
construction (they already handle nested static elements and the two-phase cyclic
case).

Subtlety — **simultaneous nested escape:** if a flagged array contains other flagged
(or escapable) objects, escape them too, sharing one `seen` map, so the nested
identities are also single instances (the `materialize_into_seen` shared-`seen` work
from this session is the tool).

### 6.3 Creation-provenance tracking (for trigger (A))

`State::alloc` (`src/js.rs` ~line 486) hands out monotonic addresses. To map an addr
back to its creation pc, either add a `created_at: usize` field to each `HeapObj`
variant (heavier) or maintain a side map on the client
(`RefCell<HashMap<addr, pc>>`), written at each `NewArray`/`NewObject` step and read in
phase 1's duplication accounting. Addresses are never reused, so this is unambiguous.

### 6.4 Do not break what works

- The within-`materialize` shared-`seen` fix (this session) stays; build on it.
- Keep folding for objects NOT flagged — that is the whole point of the precise
  trigger.
- Residual functions (`__rf`) are specialized in nested `specialize` calls sharing the
  same `Js` (interior mutability). Any per-specialization state you add (e.g. the
  duplication counters, the flagged-sites set if it is mutable) must be scoped/cleared
  around `residual_fn_for` the way `probing`/`halt_at`/`eager_stores` are (see the
  save/clear/restore block in `residual_fn_for`, ~line 1495). A var from the enclosing
  function is out of scope inside an `__rf` body.

---

## 7. Soundness argument to preserve

The fix is correct iff, for every object given a single runtime instance:

1. **One construction.** The object is constructed exactly once (at its creation
   site).
2. **Dominance.** That construction dominates every read/write/reference of the
   object. Guaranteed because the construction is at creation and you cannot reference
   an object before creating it.
3. **No silent divergence of contents.** Because the object is `Var` everywhere after
   escape, every write is a runtime `SetIndex`/`SetProp` and every read a runtime
   `Index`/`Get` on the same var — there is no abstract copy that could drift. (This is
   the property the duplication broke.)

Conversely, objects that are NOT escaped at creation must continue to be fully static
(every operation folds) — if any operation on them residualizes, that path's existing
`escape()` still applies, and the profiling trigger (A) would have flagged them if it
caused a duplication. The fuzzer is the backstop for any case the trigger misses
(it would surface as a value/throw divergence).

---

## 8. Verification plan (gate every step)

1. **Build a true minimal repro FIRST.** None exists yet for the cross-`materialize`
   case — every hand-crafted attempt so far (see the list below) collapses to a single
   shared instance because `escape()` escapes the whole graph from the root. You need
   a program where an object stays partial-static AND is re-materialized at multiple
   merge points. The instrumentation in §3 (count constructions per addr) is the way
   to confirm a candidate reproduces (dup count > 1 for one addr). Likely ingredients,
   from the simple.js structure: a nested array created in an OUTER loop; the OUTER
   array read with a static chunk index but the inner chunk written with a DYNAMIC
   byte index; carried through a dynamically-controlled inner loop; the array also
   captured by a residualized closure. Tried-and-folded variants (do not just retry
   these): array in a dynamic loop; nested arrays; static-outer/dynamic-inner index;
   boxed + closure-captured; array created in an init loop; all combined. A correct
   repro must defeat `escape()`'s whole-graph escape — study why those fold (the array
   becomes a single `Var`) and construct a case where it does NOT.
2. **Unit tests (keep green):** `cargo test -p partial --release` (12) and
   `cargo test -p js-frontend --release` (currently 112). Add a regression test for
   the new minimal repro via `assert_node_equiv`.
3. **The fuzzer (soundness oracle):** `./target/release/fuzz --seed 1 --count 2000
   --batch 250` and across fresh bases (10000, 20000, …). It sets
   `SPEC_WEIGHT_BUDGET=100000` itself. A correctness regression shows up as a
   value/throw divergence; the flat-memo attempt would NOT have passed this if run on
   the right program, so run it broadly. Current state: clean across ~14k programs.
4. **difftrace on simple.js:** must reach `effect #13` with no divergence.
5. **simple.js still specializes:** `SPEC_WEIGHT_BUDGET=999999999 ./target/release/
   js-frontend --js <simple.js>` exits 0; residual size stays sane (a precise trigger
   should not blow it up; if it grows a lot, the trigger is over-escaping).

Note on the gate's strength: the flat-memo bug broke simple.js at effect #0
immediately, so a wrong dominance story is caught fast by difftrace. Subtle content
divergences are caught by the fuzzer. Use both at every iteration.

---

## 9. Pointers / artifacts

- Residual: regenerate `/tmp/r.js` with the `--js` command in §1.
- The diverging byte: index **1131**, three runtime objects, off by `^143`.
- Instrumented-array experiment: recreate the `/tmp/instr3.js` pattern (§1, §3).
- `materialize` / `materialize_into_seen` / `mat_inline*`: `src/js.rs` (~line 3850+),
  the construction site to change behavior at.
- `Instr::NewArray` / `Instr::NewObject`: `src/js.rs` (~line 2020), the creation site
  to escape at.
- `escape` / `escape_var` / partial-escape machinery: `src/js.rs` (~line 3470+).
- `residual_fn_for` save/clear/restore pattern (for any per-specialization state):
  `src/js.rs` (~line 1495).
- `docs/effect4-minimal-repro.js`: the WITHIN-`materialize` aliasing repro (already
  fixed) — useful as a template for writing the across-`materialize` repro, but it is
  NOT this bug.
- `docs/HANDOFF.md` "Remaining issue #2": the running diagnosis log (this doc
  supersedes it for the fix plan).

---

## 10. Explicitly out of scope / do not do

- Do not re-attempt the flat `addr -> var` materialize memo (§4): proven unsound.
- Do not "fix" it by emitting all constructions in block 0 unconditionally — that is
  just a worse hoist and breaks objects whose contents are genuinely loop-variant.
- Do not change `src/engine.rs`; identity is JS-client semantics.
- Do not lower simple.js's `SPEC_WEIGHT_BUDGET` or special-case simple.js — the fix
  must be a general soundness improvement that the fuzzer endorses.
