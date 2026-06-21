# Concurrency / atomics dogfood — findings

Built per the scope (docs/CONCURRENCY_SCOPE.md): `lib/atomic.coil` + `lib/thread.coil`
+ two dogfoods. The breadth axis the earlier dogfoods didn't stress (a language "as low
as Zig/C" needs concurrency), validated end-to-end.

## Headline: the ZERO-CORE-CHANGE discipline extends to concurrency

Both halves are PURE LIBRARIES — `grep -ni atomic|pthread src/` is empty:
- **Atomics** (`lib/atomic.coil`) — `atomic-add/sub/load/store/xchg/cas` on `(ptr i64)`,
  each a one-line `llvm-ir` wrapper (`atomicrmw`/`cmpxchg`/atomic load+store, `seq_cst`).
  Exactly like `lib/mmio.coil`: a systems primitive expressed through the escape hatch,
  no compiler support.
- **Threads** (`lib/thread.coil`) — a thin `Thread{handle}` over `pthread_create`/
  `pthread_join` externs, with a Coil function as the thread body via `fnptr-of` — the
  same C-interop callback pattern cinterop proved with qsort. Hosted (links libpthread).

So the macro/llvm-ir/C-interop philosophy holds at the concurrency axis too: no feature
was hacked into the core to get threads + atomics.

## Stage 1 — shared atomic counter (race-free N·M)

`examples/threads.coil`: 4 threads × 100 000 `atomic-add` on a SHARED counter → exactly
400 000, race-free and DETERMINISTIC (verified across many runs). The test asserts the
exact N·M. The non-atomic contrast (plain load+add+store) races and loses updates — but
a data race is NON-DETERMINISTIC, so that's a demonstration, never an asserted test (no
flaky tests). This is the clean proof the atomics genuinely synchronize.

## Stage 2 — genuinely lock-free stack (Treiber, CAS), ABA-honest

`examples/lockfree.coil`: a lock-free stack via `atomic-cas` — NO mutex. 4 threads
concurrently push 1000 disjoint pre-allocated nodes each (contending on the head pointer
via a CAS retry loop); draining finds all 4000 (deterministic across runs). Genuinely
lock-free (CAS only). **ABA handled honestly, not faked**: the hazard needs a node freed
+ reused while a stale pointer is held; this dogfood sidesteps it by construction —
nodes are pre-allocated once, never freed, each thread pushes a disjoint range, and the
drain is single-threaded (no concurrent push/pop reuse). A *production* lock-free stack
with concurrent push/pop + reclamation needs tagged pointers / hazard pointers / epoch
reclamation — explicitly out of scope; this is a bounded demonstration, not that.

## Friction surfaced (minor, no core change needed)

- **Pointer-CAS via int↔ptr.** `atomic-cas` is on `(ptr i64)`, so the lock-free stack
  treats node addresses as `i64` (`(cast i64 (cast (ptr i8) (index pool k)))`). Works,
  but a generic `atomic-cas` over `(ptr (ptr T))` would be tidier — a friction-driven
  library addition later, not a core change.
- **Atomics are `(ptr i64)`-only** (not generic over width / `(ptr T)`). Sufficient
  here; a generic/over-widths version is a later library lean-in if a program needs it.
- The reference / `(mut)` place model needed NOTHING special for shared memory: a shared
  counter is just a `(ptr i64)` passed to each thread; threading didn't perturb it.

## Verdict
Concurrency + atomics work as pure libraries (zero core change) — race-free atomics
(deterministic N·M) and a genuinely lock-free, ABA-honest CAS stack. The "as low as
Zig/C, expressive only through macros/library" thesis holds on the concurrency axis too.
Memory orderings (relaxed/acquire/release) and generic/pointer atomics remain
friction-driven follow-ups, only if a real program needs them.
