# Self-Application Blockers

Goal: port this partial evaluator to JavaScript **written in the subset the PE
itself can specialize**, so it can eventually self-apply (Futamura projection 2:
`spec(spec, interp) = compiler`).

This document enumerates everything that currently blocks that, with concrete JS
examples we'd need the PE to support, prioritized so we can attack them in order.

---

## UPDATE 2026-06-04 — empirically re-scoped; the core memo loop-tie now WORKS

Two of this doc's assumptions turned out to be wrong when measured against the
actual PE, and the keystone is **much smaller than feared**:

1. **P1 (loop-carried growing collections) is ALREADY handled.** The `panic!`/
   "heap divergence out of scope" claims below are STALE — a loop-carried growing
   array (`while (i<n) w.push(i)`) and a loop-carried dynamic-key map now
   **residualize cleanly** (a residual loop over a residual array/object), no
   panic. That hole was closed since this doc was written.

2. **The P0 keystone was NOT "a large partially-static map abstraction."** For the
   real milestone — specialize a worklist memoizer over a *static* subject — the
   memo keys are **static** (structural fingerprints of states the static
   interpreter visits; dynamic data lives only in the leaves/values). So the only
   actual gaps were two small, sound modeling additions:
   - **`key in obj` on a static object folds** to a boolean (own field OR inherited
     `Object.prototype` member). `Instr::OpaqueOp` in `src/js.rs`.
   - **`arr.pop()` / `arr.shift()` on a static array are modeled** (mutate the
     abstract array, return the removed element), so a static worklist stays static
     instead of escaping. `GetProp` emits an `Array.<m>` bound-method marker;
     `do_call` folds it.

**Milestone ACHIEVED.** The smallest end-to-end proof from the bottom of this doc
now passes: a worklist-driven memoizing graph traversal over a static 3-cycle
**fully folds** — the memo ties the cycle at spec time and the whole loop reduces
to `return 3 + input`. Regression test `memo_driven_traversal_loop_ties`
(+ `static_in_folds`, `array_pop_shift_modeled`). So the engine's termination
mechanism (memo/seen loop-tie) survives the subset for static subjects.

**What's left for full self-application** (genuinely dynamic subjects, where the
memo keys *do* carry dynamic structure) is the harder P0 partially-static-key map
— still real, but no longer on the critical path for the first proof. Re-prioritize
from here: the next concrete step is to hand-write the `bf` client + a minimal
engine driver in the subset and run them, finding the *next* real blocker
empirically (the same measure-don't-assume approach that found these two).

The rest of this document is the original analysis (accurate on P0 maps for the
dynamic-subject case; stale on P1 panics).

## TL;DR

The engine (`src/engine.rs`, ~200 lines) is **iterative** — a `while
work.pop()` worklist, not recursion — so dynamic-depth recursion is *not* the
main problem. The problem is that the engine and every client are built on four
data-structure patterns, and the JS subset (`src/js.rs`) hard-errors on all
four:

| # | Pattern | Where it lives | Subset status |
|---|---------|----------------|---------------|
| P0 | **Map with a runtime-derived key** | `memo: HashMap<Key,_>`, `seen: HashMap<Point,_>`, `fn_memo`, abstract `heap` | hard error: dynamic property key / dynamic index |
| P0 | **Array indexed by a runtime value** | `blocks[bid.0]`, `locals[slot]`, `ops` patching | hard error: dynamic array index |
| P1 | **Loop-carried / runtime-sized growing collection** | `blocks: Vec`, `work: Vec`, operand stacks, the `im::OrdMap` heap | hard error: loop-carried heap object, "heap divergence out of scope" |
| P1 | **Structural hash/eq over a runtime-sized value** | `State`'s derived `Hash`/`Eq` (walks heap + frames) | follows from P0/P1 |
| P2 | **f64 vs i64**, **string-keyed dispatch**, **closures stored in data** | scattered | partial / semantic mismatch |

Everything else the PE needs — `if`/`while`/`for`/`switch`/`try`, static
objects/arrays/closures, static-depth recursion over the source AST — is
**already supported** and folds away. See "What already works" at the bottom so
we don't over-scope.

Why these bite during self-application: in projection 2 the *interpreter source*
is static but the *subject program's input* is dynamic, so every `State` the
engine memoizes carries dynamic values (`RExpr`s over the dynamic input).
Therefore the memo **keys are dynamic**, and a `Map`/array keyed by a dynamic
value is precisely what the subset refuses to fold.

---

## P0 — Maps keyed by a runtime-derived value

This is the heart of the engine. Memoization is the entire termination story:

```rust
// src/engine.rs
memo: HashMap<C::Key, BlockId>,          // line 92
seen: HashMap<C::Point, Vec<C::State>>,  // line 94
// create_or_get:
let k = client.key(&st);
if let Some(&b) = cx.memo.get(&k) { return b; }   // lines 145-146
cx.memo.insert(k, id);                            // line 154
```

and the JS client adds its own:

```rust
// src/js.rs
fn_memo: std::cell::RefCell<HashMap<usize, usize>>,  // line 1205
type Heap = im::OrdMap<usize, HeapObj>;              // line 434
```

In JS this becomes `memo.get(key)` / `memo.set(key, id)` where `key` is a
canonical encoding of a `State` that contains dynamic data. The subset rejects
exactly this — from `src/js.rs:21`: *"a dynamic property key or array index"* is
out of scope, and the read path bails:

```rust
// src/js.rs ~2333
// dynamic index, or a type-mismatched key ... the container and the index
// escape and the read residualizes.
```

### Minimal example we must support

```js
// A map whose KEY is computed from the dynamic input must fold statically,
// so that the memo lookup decides control flow at specialization time.
function memoLookup(table, key) {
  if (key in table) return table[key];   // dynamic key `in`/read
  table[key] = freshBlockId();           // dynamic key write
  return table[key];
}
```

Today `table[key]` with a non-constant `key` escapes the whole `table` to a
runtime object and residualizes the access; the engine's `if memo.get(k)`
branch can no longer be decided at spec time, so the loop-tying that makes
specialization *terminate* never happens.

**What "support" means here:** the PE needs an abstract model of an associative
container with **partially-static keys** — a key that is `dynamic_tag +
static_structure` foldable to a decision, analogous to how `bf.rs` already
models a partially-static *pointer* (`dynamic_base + static_offset`,
`src/bf.rs:9-15`). This is the single biggest piece of new modeling.

**Size:** large. New abstract domain in `js.rs` (a "partially-static map/dict"
heap object), plus key-canonicalization that the engine can reason about.

---

## P0 — Arrays indexed by a runtime value

The engine stores blocks in a vector and indexes it by a `BlockId` that came out
of the (runtime) memo table:

```rust
// src/engine.rs
cx.blocks[bid.0].ops = ops;        // line 131  — write at dynamic index
cx.blocks[bid.0].term = term;      // line 132
// and clients:
locals[slot]                       // frame slot read by dynamic slot in js.rs
```

The subset folds `arr[0]`, `arr[i]` only when `i` is static; a dynamic index
escapes:

```rust
// src/js.rs:2599-2603
// dynamic index, or a type-mismatched key such as `arr["x"]` / `obj[5]`):
// the caller escapes the container ... unescaped static container with an
// unservable index => unreachable! (a bug)
```

### Minimal example we must support

```js
function setBlock(blocks, bid, ops) {
  blocks[bid].ops = ops;   // bid is dynamic -> blocks escapes today
}
```

This is the array twin of P0-maps. If we build the map abstraction (above) as a
dense integer-keyed structure, this likely shares the same solution: a
**partially-static array** whose element *reads/writes at a dynamic-but-bounded
index* stay analyzable instead of escaping the whole array.

**Size:** medium, if it rides on the map work. Standalone: large.

---

## P1 — Loop-carried, runtime-sized, growing collections

The engine grows three collections whose size depends on runtime control flow:

```rust
// src/engine.rs
blocks: Vec<Block<..>>,   // grows per specialization context
work:   Vec<(BlockId, State)>,
cx.work.push((id, st));   // line 156  — push inside the worklist loop
cx.blocks.push(Block{..});// line 150
```

and each `State` carries growing operand stacks and an `im::OrdMap` heap
(`src/js.rs:441,458`) that is mutated across loop back-edges.

The subset explicitly refuses loop-carried heap objects:

```rust
// src/js.rs:392-394
Abs::Ref(_) => panic!(
  "... a loop-carried object); unsupported in this subset")
// src/js.rs:2508
// Heap divergence is out of scope.
```

### Minimal example we must support

```js
function drive(work) {
  while (work.length > 0) {
    const item = work.pop();     // mutate a loop-carried array
    const block = { ops: [] };
    blocks.push(block);          // grow a loop-carried array
    work.push(successorOf(item));
  }
}
```

A growing worklist is fundamental and probably **can't be fully static** — and
that's fine: in projection 2 the worklist's *length* tracks the static source's
structure, so the loop may legitimately specialize to a residual loop. The
requirement is weaker than P0: the subset must **tolerate** a residual
array/queue that is pushed/popped across a residual `while`, instead of
panicking on the loop-carried `Ref`. Compare `src/js.rs:2443-2456` (arrays
mutated in branches must escape) — we need the *loop* analogue to residualize
cleanly rather than error.

**Size:** medium. Mostly: turn the `panic!`/`unreachable!` loop-carried-object
paths into clean escape-to-residual-array paths (no new domain, just close the
hole the subset documents as out of scope).

---

## P1 — Structural hash / equality over runtime-sized values

`State` is the memo key, and its identity is a structural walk over the heap and
frames:

```rust
// src/js.rs:470  (custom PartialEq, deliberately excludes next_addr)
fn eq(&self, o) -> bool { self.frames == o.frames && self.heap == o.heap && ... }
type Key = State;          // src/js.rs:1815
type Point = Vec<usize>;   // src/js.rs:1816  (a vector of pcs)
```

In JS this is `canonicalize(state) -> string` (or a deep-equal), i.e. a
recursive walk producing a value that is then used as a **map key (P0)**. So it
inherits P0's blocker, plus it walks a runtime-sized structure.

### Minimal example we must support

```js
function keyOf(state) {
  let s = "";
  for (const f of state.frames) s += f.pc + ":" + hashLocals(f.locals) + "|";
  return s + hashHeap(state.heap);   // build a dynamic string, use as map key
}
```

The string is built from dynamic contents and then used to index `memo`. Even if
P0 makes dynamic-key maps work, the **key-construction loop over a runtime-sized
`frames`/`heap`** must specialize. Tied to P1-collections above.

**Size:** small-to-medium once P0 + P1-collections land (it's a consumer of
them, not new machinery).

---

## P2 — Smaller semantic gaps

These don't block the *shape* of self-application but will cause wrong results
or unnecessary escapes.

### P2a — Number model: i64 vs f64
`src/js.rs` uses `i64` throughout (`Abs::Num(i64)`, `RExpr::Num(i64)`,
`src/js.rs:18`). Real JS — including the PE's own arithmetic on addresses,
counts, and indices — is f64 with `ToInt32`/`ToUint32` only for bitwise ops.
Self-application runs the PE's *own* integer arithmetic through the abstract
evaluator, so divergence here is a correctness bug, not just an escape.

```js
const id = blocks.length;     // f64 in real JS; i64 in the model
const half = total / 2;       // `/` is Opaque today (src/js.rs:105) — won't fold
```

**Size:** medium (introduce a float-aware `Num`, fold `/`, `%`).

### P2b — Closures stored in data and dynamically dispatched
The engine calls `client.step(...)`, `client.key(...)` etc. — fine if the client
is a single static object (it inlines, `src/js.rs:3172`). But idiomatic JS would
store handler closures in tables (`{ Add: fn, Sub: fn }[op]`) and dispatch by
dynamic key — which is P0 again plus a closure escape (`src/js.rs:3492` "a
closure that escapes becomes a generated residual function"). Keep the PE port
written as static dispatch (`switch` on tag) to avoid this.

**Size:** none if we follow a coding discipline; otherwise large.

### P2c — Strings as dynamic associative keys
Codegen builds output strings (fine — that's the residual). But any place that
*branches on* or *keys a map by* a dynamically-built string hits P0. Audit the
port for `obj[someBuiltString]`.

---

## What already works (don't re-scope these)

So the plan stays focused, these PE-needed features are already supported and
fold at spec time:

- **Iterative worklist** — the engine is `while let Some = work.pop()`, *not*
  recursion (`src/engine.rs:108`). No dynamic-depth recursion needed in the core.
- **Static-depth recursion over the source AST** — lowering, codegen, and
  inlining recurse over static structure; the subset inlines these (bounded by
  `src/js.rs:3232` inline-depth guard).
- **Control flow:** `if`/`while`/`for`/`switch`/`continue`/`break`/`try`/`catch`
  all modeled (`src/js.rs:171-194`). `throw` is modeled as control flow.
- **Static objects/arrays/closures** with static keys/indices fold away
  entirely (partial escape analysis) — the abstract heap (`src/js.rs:434`).
- **`&& || ?: ++ --`**, bitwise/shift with 32-bit semantics (`src/js.rs:52-58`),
  comparisons — all fold on static operands.
- **`arguments`**, `this`, globals, opaque pass-through ops — modeled.

---

## Suggested priority order

1. **P0 maps with partially-static keys** — the keystone. Without it the memo
   table can't decide control flow at spec time and nothing terminates. Model it
   the way `bf.rs` models the partially-static pointer.
2. **P0 dynamic-index arrays** — fold into the P0 work (dense-integer-keyed case).
3. **P1 loop-carried collections** — convert the `panic!`/`unreachable!`
   loop-carried-object paths into clean escape-to-residual-array (close the
   documented "heap divergence out of scope" hole).
4. **P1 structural key-construction** — consumer of 1–3; mostly falls out.
5. **P2a float model** — correctness for the PE's own arithmetic.
6. **P2b/c** — coding discipline + a small audit.

Milestone check after P0+P1: can the engine + `bf` client, hand-written in the
JS subset, specialize a *static BF program* and have the **memo loop-tie fire**
(producing a residual loop, not an infinite unroll)? That's the smallest
end-to-end proof that self-application's core mechanism survives the subset.
