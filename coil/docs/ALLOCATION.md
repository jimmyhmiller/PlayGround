# Allocation in the frontend, and the arena that is already there

Where the compiler's remaining time goes, why allocation is the biggest single item, and
what routing the frontend through `lib/alloc.coil`'s existing `Arena` would and would not
buy. Written as a handoff after the 2026-07-28 speed work took the whole-compiler
frontend from 10.16s to 0.51s.

## The measurement

`coil build selfhost/src/main.coil -o /dev/null` — compile-check only, no codegen, over a
51,183-line import closure. 0.51s. Sampled at 1 ms, bucketed by leaf symbol:

| | share | of 510 ms |
| --- | --- | --- |
| actual compiler work | 30.7% | ~157 ms |
| **allocation** | **24.2%** | **~124 ms** |
| process/dyld startup | 18.3% | ~93 ms |
| string hash/compare | 14.6% | ~75 ms |
| file I/O | 7.9% | ~40 ms |
| memory copy | 4.2% | ~22 ms |

Allocation is `_xzm_xzone_*` / `_malloc_zone_malloc` under `alloc.ma-alloc` (26 samples)
and `alloc.ma-resize` (15). Attributed to callers:

```
26  alloc.ma-alloc          15  alloc.ma-resize
10  parser.parse-list-expr   6  reader.parse-one
 5  check.synth-inner        4  resolve.qualify-expr
 2  parser.parse-type        2  mono.resolve-expr-list
```

That is AST construction. Every `Sexp`, every `Expr`, every `ArrayList` backing them goes
through `malloc`.

**Set expectations before starting.** Allocation is 24%. Eliminating *all* of it lands at
~0.39s. Eliminating all of it *and* all string hashing lands at ~0.30s. A 100 ms
whole-compiler frontend is not reachable by allocator work alone — it needs the 157 ms of
"actual compiler work" to come down too, which means fewer intermediate representations,
not a faster `malloc`.

## What already exists

`lib/alloc.coil` already has a bump allocator. This work is **routing, not building**.

```coil
(defstruct Arena [(base (ptr i8)) (off :i64) (cap :i64)])
(defn ar-alloc  ...)   ; bump: align up, check cap, return base+off
(defn ar-resize ...)   ; ALWAYS returns (None) — no in-place resize
(defn ar-free   ...)   ; no-op — bulk-freed
(defn arena-over-buffer [(ar …) (a …) (buf (ptr i8)) (cap :i64)] …)  ; freestanding-capable
(defn arena-allocator [(cap :i64)] …)                                ; malloc-backed wrapper
```

Note the comment already in that file about `alloc-static` giving one cell per *call
site* rather than per *call* — a previous version of `arena-over-buffer` made every arena
in the program share one `Arena`. Instance state cannot come from a static. That trap is
already documented there; do not re-introduce it.

## Why the frontend fits a bump allocator

AST nodes are allocated during a compile and live until the process exits. Nothing is
individually freed; `ma-free` is called almost never on them. That is precisely the
lifetime a bump allocator is for: pointer increment to allocate, and either no free at
all or one bulk reset.

## The catch that has to be handled first

`ar-resize` returns `(None)` unconditionally. Look at what `al-reserve!` does with that
(`lib/arraylist.coil:46`): it falls back to `alloc-slice` + element-by-element copy, and
**never frees the old block on an arena** (`ar-free` is a no-op). So an `ArrayList` that
grows by doubling on an arena consumes `4 + 8 + 16 + … + n ≈ 2n` and strands all of it.
Growable lists on an arena are quadratic in *space*.

Two ways out, and the first is worth doing regardless:

1. **Give `ar-resize` the standard bump-allocator in-place path.** If the block being
   resized is the most recent allocation — `p + old == base + off` — just move `off` and
   return `p`. That makes append-heavy `ArrayList` growth free on an arena and is about
   six lines. Without it, routing lists through an arena is a memory regression.
2. **Split the policy**: arena for immutable nodes (`Sexp`, `Expr`, `Type`), keep
   `malloc-allocator` for the growable tables (`ArrayList`s in `Cx`, `LS`, the def table).
   Coarser, but avoids the issue entirely.

## Exhaustion is currently undiagnosable

`ar-alloc` returns `(None)` when full, `alloc-slice` propagates it, and `al-reserve!`
finishes with `(oom)` — which is `(defn oom [] (abort) 0)`, a **silent** `abort()`. A
compiler that runs out of arena dies with exit 134 and no message.

**Make `oom` print before aborting** as the first commit of this work. It costs nothing
and it is the difference between a five-minute diagnosis and an afternoon. There is
already one open bug whose entire difficulty is that this abort is silent — see
[OPEN-BUG-wasm64-reserve-abort.md](OPEN-BUG-wasm64-reserve-abort.md), which is very
likely an allocation-exhaustion abort and which this work will touch directly.

## Threading

The pipeline already passes `a (ptr Allocator)` nearly everywhere; the sites that bypass
it call `malloc-allocator` directly:

```
resolve.coil 38   comptime.coil 24   codelib.coil 9   loader.coil 7
metahost.coil 6   ast.coil 6         expander.coil 3  driver.coil 3   reader.coil 1
```

Those 97 call sites are the actual work item. Most are convenience — a helper that needs
*an* allocator and grabs the global one rather than taking a parameter. Each needs to
either take the caller's `a` or deliberately keep `malloc` (a process-lifetime cache like
the interner or a memo index legitimately wants malloc, not a per-compile arena).

## Suggested staging

Each step is independently verifiable; do not batch them.

1. `oom` prints before aborting. Commit alone.
2. `ar-resize` grows the last allocation in place. Add a unit test that an `ArrayList`
   appending 100k items to an arena uses O(n) arena space, not O(n²).
3. Convert one leaf module's `malloc-allocator` sites to take `a` — `reader.coil` (1
   site) then `ast.coil` (6) are the smallest. Gates after each.
4. Give the driver a per-compile arena and pass it as `a` for the AST. Measure peak RSS
   as well as wall time; a bump allocator trades memory for speed and 51k lines of AST in
   an arena that is never reset is a real number worth knowing.
5. Only then look at `resolve.coil`'s 38 sites, which are the bulk.

## How to verify

`gate-full` is the safety net: it asserts the emitted IR is byte-identical to the
reference across a 60-program corpus, so any allocation change that alters behaviour
fails loudly. Run the ten stage gates too — they say *which* stage moved when something
does. `selfhost/rebootstrap.sh` runs all of it plus the fixpoint check.

Benchmark honestly: min-of-3, back-to-back against the previous binary in the same
session. Single measurements at this scale move ±20% with machine load — during this work
a "0.31 → 0.49s regression" turned out to be pure noise and the two binaries were
identical when re-run.

And check which backend built the binary you are timing. `rebootstrap.sh` used to install
the arm64-backend build, which is ~11x slower than the LLVM one; a long stretch of
profiling went into the wrong binary before that was noticed. It now installs the LLVM
build — see `5752aea3f`.
