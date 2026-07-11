# Resumable native execution

The production execution unit is not a conventional native stack frame. It is a
heap-resident `Frame` containing a function-version ID, program counter, typed
register slots, and return destination. This makes the exact paused computation
GC-visible and durable across recompilation.

The compiler lowers source to typed, register-based continuation IR. Every
potentially trapping operation ends a basic block. Initially the interpreter
executes one instruction at a time. LLVM later compiles groups of non-trapping
instructions into functions with this ABI:

```text
step(frame*, runtime*) -> { continue, call, return, condition, yield }
```

LLVM therefore accelerates execution without owning continuation semantics.
Native temporaries never survive a step boundary; all live references are in
typed frame slots, giving the precise GC a complete root map.

## Updating code

Published function versions are immutable. Existing frames pin their version.
New calls resolve the current entry. A schema edit re-verifies affected current
functions and publishes broken entries for invalid ones while retaining old code
for pinned frames. Entering a broken function raises a condition before any of
its instructions execute.

Old code can be reclaimed when no frame identifies its version. Later optimized
direct calls must remain within one immutable code-version group; every call
that may cross a live boundary goes through an entry slot.

## Updating data

References name stable object handles. Each handle owns a body tagged with a
schema version. Field access is a migration barrier. A migration builds and
validates a replacement body before swapping it into the handle, preserving
aliases and preventing partial layouts from becoming observable. Missing plans
raise conditions without advancing the frame.

The prototype uses a map as the stable-handle table and implements precise
mark/sweep collection from frame registers. A native runtime can use moving
bodies behind non-moving handles, or add a handle-resolution read barrier. The
nominal type ID and schema version must not be conflated with a compact GC layout
table index.

## Effects

A frame advances past `Emit` only after the effect is committed. A later pause
resumes at its exact PC, so earlier effects are not replayed. For compound
external operations, actors will use a transactional outbox: stage messages and
state changes, commit them together, and assign stable effect IDs for downstream
deduplication.

## What “resume” means

Repair resumes the suspended continuation, not an arbitrary machine instruction
and not the beginning of a tick. Values already computed remain in registers;
completed effects remain completed. The repair may satisfy the condition by
installing a valid function version or a validated migration. The trapping
instruction is then retried.

This model supports loops, branches, and suspension once they are added to the
IR: their continuation is simply a PC plus registers. It does not require stack
copying, native deoptimization, or replay of an entire callback.

## Implementation status

Both executors exist and are proven equivalent:

- **Interpreter** (`src/runtime.rs`) — the reference executor, one instruction
  per step, over heap-resident frames.
- **LLVM `step` backend** (`src/jit.rs`) — each Ready function version is
  JIT-compiled (inkwell / LLVM 21) to a native
  `step(RawFrame*, Runtime*) -> outcome` over a C-ABI frame. Native code runs
  the pure ops (`Const`/`SubI64`/`LtI64`) and the non-pausing runtime calls
  (`New`/`GetField`/`Emit`, via the `lt_*` externs), and hands control back to
  the Rust driver for the boundaries that own continuation semantics —
  `Call` (push a frame), `Return` (pop), a migration barrier (`condition`), and
  `Yield` (recurring safe point). It resumes at an exact PC via a
  `switch(frame->pc)` dispatch. Crucially, **no SSA value crosses a basic
  block**: every register read/write goes through `frame->regs[i]`, so loops and
  branches need no phi nodes and every live reference is a typed frame slot the
  GC roots from directly (`JitActor::roots`).

The IR now includes control flow (`LtI64`, `Jump`, `Branch`, `Yield`), verified
by a CFG-worklist type checker (`src/verify.rs`) that joins register
environments at merge points.

### The soundness invariant is enforced, not asserted

The §4 invariant — *no running code ever observes an ill-typed value* — is now a
runtime check (`Runtime::value_ok` / `expect_value`) applied at every boundary
where a value would be observed or published:

- **Call arguments** vs the callee's parameter types,
- **Return values** vs the function version's result type,
- **New / migration** object bodies vs the schema's field types (`value_ok`),
- **operand tags** for `SubI64`/`LtI64`/`Branch` (the interpreter matches by
  value; the JIT emits an explicit tag guard, `Codegen::guard_tags`).

Because references match at the *nominal* type, a migrated object still satisfies
a `Ref(T)`-typed use; the check fires only on a genuine representation confusion.
This closes the con-freeness corner (T2): a frame paused at a `Yield` inside an
**old pinned function** that resumes after a migration changed a field's
representation now **traps** (quarantining the frame) instead of proceeding — and
the JIT traps *identically* to the interpreter rather than silently reading a
`Ref`'s object id as an integer.

### Con-freeness traps are repairable

A type trap is not a dead end — the frozen frame is a one-shot delimited
continuation, and repair resumes it, optionally supplying the value the trapping
instruction should have produced (`Runtime::resume_with` / `jit::resume_with`,
shaped by `resume_shape`). A subtraction that froze on a `Ref` resumes with the
`Int` it should have yielded; a `Return` that froze resumes with the function's
result; a `Branch` with the `Bool` to take. The offering is checked against the
trap's expected type (`pause_expected` reports it), so **repair can never
reintroduce an ill-typed value** — a wrong-typed offering is rejected and the
frame stays quarantined. `tests/repair.rs` proves the arithmetic and return
cases resume identically on both executors and that an ill-typed offering is
refused without consuming the trap.

### Migrations are auto-derived where trivial

Installing a new schema version auto-derives the previous-version migration
(`Runtime::derive_migration`): a field is **copied** when it survives unchanged,
**default-initialized** when it is new (or retyped) and has a default, and
otherwise is a **gap** that abandons derivation and leaves a developer to supply
the transformer (a `MissingMigration` trap on first cross). Derived migrations
are copy/default only, so they are type-sound by construction; an explicit
`install_migration` for the same step overrides them. Adding a defaulted field
therefore needs no hand-written migration, while a genuine representation change
(e.g. `Int → Ref`, no default) still traps for one.

### Invalidation is demand-driven

A schema change no longer re-verifies every function. Each Ready function stores
the set of nominal types it references (`World.function_deps`, produced by
`verify_function`); `invalidate_functions` re-verifies only the functions whose
set contains the changed type, then propagates through the call graph — when a
function newly breaks, its callers are enqueued, because a call to a broken
function no longer type-checks. This reaches a fixpoint over exactly the affected
functions (and is *more* complete than the old single map-order pass, which could
miss a caller re-checked before its callee broke). `tests/invalidation.rs`
asserts both directions: an unrelated function is never re-versioned, and
brokenness propagates to a transitive caller.

`tests/jit_matches_interpreter.rs` drives both executors through the same hot
updates on the Box/Wrapper migration, the Account/Money break-fix-migrate story,
a loop with a mid-loop update at a `Yield`, two con-freeness traps (arithmetic
and return), and auto-derived vs abstained migrations — asserting identical
effects, pause conditions, migrations, heaps, and results at every boundary. A
further test collects garbage precisely from native frame slots.

### Concurrency: quarantine holds over a shared heap

The design's highest-value open question (§7 corner 1) was whether the soundness
invariant survives when several computations **share a heap and one migrates a
value mid-flight while another holds it**. It does — and no world-freeze or
ownership rule was needed. `run_interleaved` schedules multiple actors over the
one `Runtime` (shared heap, world, effects), round-robin at `Yield` granularity.
`tests/concurrent.rs` builds the exact race: actor X reads a shared object with
new code, migrating it `v1 → v2` in place; actor Y — a *pinned old reader* of the
same object — then resumes and reads the migrated value. Y traps at the use
(quarantined, never observing the value as the wrong type) while X completes
cleanly. The reason it composes for free: soundness is enforced at each value
*use* (`value_ok`), independent of which actor caused the migration, so a
migrated value is caught the moment any old-typed frame tries to consume it.

This is *semantic* concurrency — deterministic interleaving over a shared heap,
which is precisely the setting the soundness question is about.

### Real OS threads over a shared heap

The object model is now a **non-moving handle over an atomically-swappable
body** (`Object` = a stable `ObjectId` plus an `Arc<Body>`; a migration builds a
new `Body` and swaps the pointer, old bodies reclaimed by refcount — no hazard
pointers). On that, `src/mt.rs` adds a thread-safe runtime tier: setup runs
through the ordinary single-threaded `Runtime` and is frozen via `into_parts`,
then `Shared` drives actors on **real `std::thread`s** over one shared heap
(per-object `Mutex<Arc<Body>>`, a locked handle table, atomic id counter; the
world is immutable during a run since updates land at quiescent points).
Concurrent migration is race-free: a migrator builds the next body without
holding the object lock, then swaps under it with a double-check, so two threads
migrating the same object never tear and the loser's work is discarded as
garbage. `tests/mt_threads.rs` drives **8 threads racing to migrate one shared
object over 200 iterations** and they all agree.

The GC is a **preemptive stop-the-world collector** (`request_gc`): any thread
can request a collection, every running actor parks at its next safepoint
(checked each instruction) after publishing its live roots, and once all have
parked the collector sweeps from the union of roots and releases them — the way
a real STW runtime collects *while mutators run*, not only at quiescence.
`tests/mt_threads.rs` runs four actors churning allocations while a separate
thread hammers `request_gc`, over 20 iterations: every actor finishes correctly
despite being repeatedly paused and having its garbage swept mid-run, and
nothing leaks. Actors are **multi-frame** (`Call`/`Return` push and pop a real
call stack) and communicate by **message passing**: `Send { target, value }`
drops a value into another actor's mailbox and `Recv { dst, ty }` blocks until
one arrives — checking it has type `ty`, so a wrong-typed message traps like any
other con-freeness violation. Messages may carry `Ref`s, so actors share heap
objects *and* pass messages, over the same shared heap. (`Send`/`Recv` are
concurrent-tier only; the interpreter and JIT reject them with a clear error.)

The runtime is split into two crates so the concurrency can be race-checked: the
LLVM-free **`livetype-core`** (IR, verifier, interpreter, `mt`) and **`livetype`**
(which adds the inkwell JIT and depends on core). Because `livetype-core` links
no LLVM, its thread tests run under **Miri's data-race detector**
(`cargo +nightly miri test -p livetype-core --test mt_threads`) — all pass,
including across varied scheduler interleavings via `-Zmiri-many-seeds`, so the
shared-heap migration and the STW-GC coordination are *machine-verified*
race-free, not just race-free by construction. (ThreadSanitizer's own runtime
SIGSEGVs on `aarch64-apple-darwin` regardless of the code — confirmed with a
non-threaded, LLVM-free test — so Miri is the gate on this platform.)

### Migration chains cross many versions at once

A type may change repeatedly; an object first touched only after several updates
migrates across the whole chain in a single `GetField` (`migrate` loops until the
object reaches the current schema). `tests/migration_chain.rs` covers a
three-version auto-derived chain and a mixed chain (an auto-derived step followed
by an explicit retype), on both executors — closing the design's "a type can
change more than once" concern. Every change is a forward version with a
per-step migration, so a "revert" is just another forward version.

### Differential fuzzing

`tests/differential_fuzz.rs` generates random *valid* programs (type-directed, so
every instruction is well-typed by construction) plus random schema evolutions
(auto-derivable additions and no-migration retypes), then runs the same scenario
— up to a `Yield`, apply updates, resume — on both executors and asserts they
agree on effects, status, and heap. Over 600 seeds it reaches normal completion,
migration-gap traps, and con-freeness traps (~526 / 22 / 52 by outcome); the JIT
and interpreter agree on every one. A divergence would pin a bug in codegen, the
verifier, migration, or the soundness checks.

### Still open

All nine core design decisions (D1–D9) are realized; the shared-heap concurrency
question is answered; multi-version migration chains work; con-freeness traps are
repairable. What remains, in the owner's chosen direction:

- **A properly thread-safe runtime** (JVM-shaped): built so far — a
  non-moving-handle / atomic-body object model, a `Shared` tier running real OS
  threads over one heap with race-free concurrent migration, and a **preemptive
  stop-the-world collector** that parks running threads at safepoints and sweeps
  mid-run, multi-frame actors, and message passing over the shared heap
  (`crates/livetype-core/src/mt.rs`), split into an LLVM-free crate and
  **race-checked under Miri**. Remaining: the JIT under threads; a concurrent
  (non-STW) collector; and a blocking (rather than polling) `Recv`.
  *(Largely there.)*
- **Name-based instant propagation** is the identity model (not content
  addressing): callers already resolve a callee's current version by name, so a
  signature-compatible republish is visible instantly. The one gap is eager
  re-verification of callers on a signature *change* (today they are caught
  dynamically at the call boundary instead). Small follow-up.
- **Update timing as a mode**: prefer static knowledge of where an update is
  safe, fall back to the runtime trap where it isn't. Needs the con-free
  liveness analysis; the trap stays as the floor.

