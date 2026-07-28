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

## The catch that had to be handled first — DONE

`ar-resize` used to return `(None)` unconditionally. Look at what `al-reserve!` does with
that (`lib/arraylist.coil`): it falls back to `alloc-slice` + element-by-element copy, and
**never frees the old block on an arena** (`ar-free` is a no-op). So an `ArrayList` that
grew by doubling on an arena consumed `4 + 8 + 16 + … + n ≈ 2n` and stranded all of it.
Growable lists on an arena were quadratic in *space*.

`ar-resize` now has the standard bump-allocator in-place path: if the block being resized
is the most recent allocation — `p + old == base + off` — it moves `off` and returns `p`.
Any other block has live allocations stacked on top of it and still returns `(None)`, so
the copy fallback is unchanged for that case.

Measured, appending 100,000 `i64` to an arena (`examples/arena-growth.coil`):

| | arena bytes consumed |
| --- | --- |
| before (`ar-resize` always `(None)`) | 2,097,120 |
| after (in-place growth of the top block) | 1,048,576 |

Exactly 2n → n, as predicted: the final 131,072-element buffer is now all the arena ever
holds. The test asserts that number exactly rather than as a bound, because every growth
after the first is a pure bump — and it fails on the old `ar-resize`, which is the only
reason it is worth having.

The alternative that was considered and is no longer needed — splitting the policy so
arenas hold only immutable nodes (`Sexp`, `Expr`, `Type`) while growable tables stay on
`malloc-allocator` — remains available if a future measurement wants it, but it is now a
choice rather than a workaround.

## Exhaustion is diagnosable — DONE

`ar-alloc` returns `(None)` when full, `alloc-slice` propagates it, and `al-reserve!`
finished with `(oom)` — which was `(defn oom [] (abort) 0)`, a **silent** `abort()`. A
compiler that ran out of arena died with exit 134 and no message.

`oom` now reports before aborting, through `oom-at (site, bytes)`:

```
out of memory: arraylist al-reserve! failed to allocate 2048 bytes; the allocator is exhausted
```

The three call sites (`al-reserve!`, `hm-grow!`, `str-key-copy`) each name themselves and
pass the request size. `oom` with no site or size is still there for callers that have
neither, but nothing in the tree uses it. Printing goes through a local `write(2)` and a
hand-rolled decimal formatter rather than `io.coil`, so the reporting path adds no module
dependency to `alloc.coil` and stays usable from anywhere the allocator is.

**It does not cost nothing** — see [What it cost](#what-it-cost). The earlier claim in
this document that it would was wrong.

The bug that motivated this — a silent exit-134 wasm64 build — turned out not to
reproduce, and the allocation hypothesis behind it was disproved rather than confirmed:
the compiler has no arena anywhere, so `al-reserve!` never leaves its `realloc` path. The
investigation is recorded in [wasm64-reserve-abort.md](wasm64-reserve-abort.md), and the
`read-file` defect it was blocking is fixed.

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

## Staging

Each step is independently verifiable; do not batch them.

1. ~~`oom` prints before aborting.~~ **Done.**
2. ~~`ar-resize` grows the last allocation in place, with a test that an `ArrayList`
   appending 100k items to an arena uses O(n) arena space, not O(n²).~~ **Done** —
   `examples/arena-growth.coil`, which the reader/load/expand corpora pick up
   automatically because they glob `examples/*.coil`.
3. Convert one leaf module's `malloc-allocator` sites to take `a` — `reader.coil` (1
   site) then `ast.coil` (6) are the smallest. Gates after each. **Next.**
4. Give the driver a per-compile arena and pass it as `a` for the AST. Measure peak RSS
   as well as wall time; a bump allocator trades memory for speed and 51k lines of AST in
   an arena that is never reset is a real number worth knowing.
5. Only then look at `resolve.coil`'s 38 sites, which are the bulk.

A bulk-append primitive landed alongside step 2 because `loader.read-file` needed it (see
[wasm64-reserve-abort.md](wasm64-reserve-abort.md)):

```coil
(defn al-extend! [T] [(l (mut (ArrayList T))) (src (ptr T)) (n i64)] (-> i64)
```

It reserves the whole run up front and `mem-copy`s it, instead of a call plus a capacity
test per element.

## What it cost

Steps 1 and 2 are not free, and this document previously asserted step 1 would be. The
measured price of the whole batch — diagnostic `oom`, in-place `ar-resize`, `al-extend!`,
and the `read-file` rewrite:

| | binary | vs HEAD |
| --- | --- | --- |
| HEAD (`77a5d8032`) | 2,004,864 B | |
| + `ar-resize`, `al-extend!`, `read-file` | 2,031,440 B | +26.6 KB |
| + the `oom` message | 2,054,176 B | **+49.3 KB (+2.5%)** |

Wall time, `coil build selfhost/src/main.coil -o /dev/null`, 15 **paired** runs
alternating the two binaries so machine drift cancels (block-at-a-time rounds of this
measurement wandered between 0.47s and 0.58s for the *same* binary, so pairing is not
optional here):

| | min | median |
| --- | --- | --- |
| HEAD | 0.460s | 0.470s |
| after | 0.470s | 0.480s |

Paired delta **+20 ms median, +14 ms mean, slower in 12 of 15 pairs**. About 3%.

It is a size cost, not a hot-path cost: rebuilding with the `al-reserve!` call site back
to the old argument-free `(oom)` produced a binary that timed identically to the full one,
so the extra code is not slowing the allocator — there is simply ~49 KB more of it in
every module that imports `alloc.coil`, which is every module. Whether 3% of the frontend
is worth permanent OOM diagnosability is a judgement call; it is recorded here so it is a
judgement call made with a number rather than a guess.

The `read-file` rewrite is worth about 10 ms of that back (2,031,440-byte build measured
between the other two), consistent with the ~14 ms the earlier estimate predicted and, as
predicted, not separable from noise in a single measurement.

## How to verify

`gate-full` is the safety net: it asserts the emitted IR is byte-identical to the
reference across a 60-program corpus, so any allocation change that alters behaviour
fails loudly. Run the ten stage gates too — they say *which* stage moved when something
does. `selfhost/rebootstrap.sh` runs all of it plus the fixpoint check.

Benchmark honestly: min-of-3, back-to-back against the previous binary in the same
session. Single measurements at this scale move ±20% with machine load — during this work
a "0.31 → 0.49s regression" turned out to be pure noise and the two binaries were
identical when re-run.

Two mechanical traps this work hit, both worth knowing before touching `lib/`:

* **A new `lib/` function used by `selfhost/src` breaks the bootstrap.** `stage0` is the
  committed seed, and it resolves `(import "arraylist.coil")` to its own *embedded* copy,
  not the one on disk — so the seed cannot compile a `loader.coil` that calls a function
  added to `lib/` in the same change. Build a bridge stage0 first
  (`COIL_STDLIB_DIR=$PWD ./coil build selfhost/src/main.coil -o /tmp/coil-bridge`, which
  bakes the new `lib/` in via `include-str`), run `STAGE0=/tmp/coil-bridge
  selfhost/rebootstrap.sh`, then `STAGE0=./coil selfhost/refresh-seed.sh both` so a plain
  rebootstrap works again. Verify by running `selfhost/rebootstrap.sh` with no overrides.
* **`COIL_STDLIB_DIR=$PWD` is what makes a `lib/` edit visible** to an already-built
  compiler. Without it you are testing the embedded stdlib and your edit does nothing —
  silently.

And check which backend built the binary you are timing. `rebootstrap.sh` used to install
the arm64-backend build, which is ~11x slower than the LLVM one; a long stretch of
profiling went into the wrong binary before that was noticed. It now installs the LLVM
build — see `5752aea3f`.
