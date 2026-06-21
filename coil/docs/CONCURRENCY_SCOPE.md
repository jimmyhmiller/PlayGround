# Concurrency / atomics breadth dogfood — SCOPE (for review before building)

The systems-language axis the dogfoods (calc / JSON / nl / cinterop / bare-metal)
haven't stressed: a language "as low as Zig/C" needs concurrency + atomics. Scoped
freestanding-style (evidence-backed; decisions flagged). Building waits for this
review + the freestanding-runtime bar.

## Finding 1 — greenfield, but the building blocks already work

Coil has NO atomic or thread support today (`grep -ni atomic|thread|pthread src/ lib/`
finds only Rust-internal gensym + the `|>` threading macros — unrelated). But nothing
needs to be hacked into the core:
- **Threads** = C interop (already proven by qsort's callback): `pthread_create`/
  `pthread_join` externs, with a Coil function as the thread body via `fnptr-of` (the
  exact callback pattern cinterop validated).
- **Atomics** = the `llvm-ir` escape hatch: `atomicrmw` / `cmpxchg` / `fence` are just
  IR. A `lib/atomic.coil` wraps them as ordinary functions — a PURE LIBRARY, like
  lib/mmio.coil. No core primitive needed.

PROTOTYPED + VERIFIED (de-risked, like the freestanding qemu/ld.lld check):
- 2 threads × 100k `atomicrmw add` on a shared counter → exactly **200000**, race-free.
- `cmpxchg` (compare-and-swap) via llvm-ir → works (`cas(x,41,42)` → 42).
- The non-atomic version (plain load+add+store) → **non-deterministic** (races; some
  runs lose updates, some don't). This is the key testability fact below.

## Finding 2 — testability: atomic = deterministic, race = demonstrative

The atomic counter's result (N threads × M increments = **N·M**) is DETERMINISTIC and
race-free → a real CI assertion. The non-atomic contrast (lost updates) is
NON-DETERMINISTIC by nature (a data race) → it's a demonstration, not a test (asserting
"< N·M" would be flaky). So the dogfood TESTS the atomic version's exact N·M; the race
contrast is shown/explained, not asserted (honest-labeling).

## Design

- **`lib/atomic.coil`** (pure library over `llvm-ir`): `atomic-add` / `atomic-sub` /
  `atomic-load` / `atomic-store` / `atomic-xchg` / `atomic-cas` on `(ptr i64)` (and
  maybe `(ptr i32)`), `seq_cst` to start. Memory-ordering variants (relaxed/acquire/
  release) added only if friction calls for them (start minimal).
- **Threads**: `pthread_create`/`pthread_join` externs + a Coil thread-body fn via
  `fnptr-of`. Optionally a thin `lib/thread.coil` (spawn/join helpers over the raw
  externs) if it reduces friction — else use the externs directly in the dogfood.
- **Dogfood** (`examples/threads.coil` or similar): N worker threads each doing M
  atomic increments on a shared counter → asserts N·M (race-free). Hosted (links libc/
  pthread — concurrency is a hosted concern; bare-metal concurrency is a separate axis).
  Second stage: a genuinely lock-free stack via `atomic-cas` (stresses CAS + a real
  lock-free structure), with **ABA handled honestly** (the mandate's "don't fake
  lock-free"): a CAS push/pop hits the classic ABA hazard if a node is freed + reused.
  PLAN: a bounded / no-free dogfood-stack (pre-allocated nodes, never freed) — which
  SIDESTEPS ABA by construction — and a clear NOTE that a production lock-free stack
  needs tagged-pointers / hazard-pointers / epoch reclamation. NO hidden mutex; do not
  overclaim "production lock-free" without ABA mitigation. The bar checks: genuinely
  lock-free (CAS only) + ABA-handled-or-honestly-noted.

## The cardinal (for the eventual bar)
Atomics are a PURE LIBRARY (`lib/atomic.coil` over `llvm-ir`), ZERO core change —
same discipline as `:bits`/derive. The prototype already shows this works; the bar
confirms grep-src-empty + the library genuinely synchronizes (the N·M result is
race-free under real thread contention, not luck).

## Decisions for the steer
1. **Dogfood shape**: the shared-atomic-counter (minimal, clear, deterministic N·M) —
   plus a lock-free CAS stack as a second stage? Lean: counter first (clean
   demonstration + CI-testable), add the CAS stack if it strengthens the axis.
2. **Atomics as library** (confirmed feasible by the prototype) vs a minimal primitive
   — lean library (the macro/llvm-ir philosophy holds; no core change).
3. **Thread API**: raw pthread externs in the dogfood, or a thin `lib/thread.coil`
   wrapper? Lean: a thin wrapper (spawn/join) — cleaner + reusable, still pure library.
4. **Memory ordering**: `seq_cst`-only to start, or expose orderings now? Lean:
   `seq_cst` first; add orderings friction-driven.

## Proposed first build
`lib/atomic.coil` (seq_cst atomic-add/load/store/cas over llvm-ir) + a thin thread
spawn/join + a dogfood: N threads × M atomic increments → N·M, with a CI test asserting
the exact race-free total. Then iterate on friction (orderings, a CAS lock-free stack).
