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

**How that prediction held up.** The arena landed 20-30 ms of the ~124 ms, not the whole
budget. The profile attributes time to `ma-alloc`/`ma-resize`, but only the part that is
malloc's own bookkeeping goes away — the call, the vtable indirection through `call-ptr`,
and the `Option` return around every allocation all remain, and they are most of it.
"Allocation is 24%" was right; "a bump allocator removes 24%" was never what it said, and
is not what happened.

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

It costs ~23 KB of binary in every program that imports `alloc.coil`, and no measurable
wall time — see [What it cost](#what-it-cost).

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

Most are convenience — a helper that needs *an* allocator and grabs the global one rather
than taking a parameter. Each needs to either take the caller's `a` or deliberately keep
`malloc` (a process-lifetime cache like the interner or a memo index legitimately wants
malloc, not a per-compile arena).

⚠ **This section called those 97 sites "the actual work item". That was wrong**, and it is
the main thing to unlearn from the original draft. The one site that mattered was
`driver.coil:3369` — the root binding every other allocation is threaded from. Changing
it is what moved the number; the other 96 are cleanup of the paths that bypass `a`. See
[What the arena actually bought](#what-the-arena-actually-bought). `reader.coil`'s single
site is now converted, and `driver.coil`'s count is 2 rather than 3.

## Staging

Each step is independently verifiable; do not batch them.

1. ~~`oom` prints before aborting.~~ **Done.**
2. ~~`ar-resize` grows the last allocation in place, with a test that an `ArrayList`
   appending 100k items to an arena uses O(n) arena space, not O(n²).~~ **Done** —
   `examples/arena-growth.coil`, which the reader/load/expand corpora pick up
   automatically because they glob `examples/*.coil`.
3. ~~Convert one leaf module's `malloc-allocator` sites to take `a`.~~ `reader.coil`'s
   single site (a per-token `slice->cstr` scratch buffer for `strtod`) **done** — `atom`
   takes `a`, which was already in scope at its one call site. `ast.coil`'s 6 are
   **deliberately still malloc**; see [The sites that should stay on
   malloc](#the-sites-that-should-stay-on-malloc).
4. ~~Give the driver an arena and pass it as `a`.~~ **Done, and it is the step that
   mattered** — see [What the arena actually bought](#what-the-arena-actually-bought).
5. `resolve.coil`'s 38 sites, which are the bulk. **Next**, and now worth much less than
   it looked; read step 4's result first.

A bulk-append primitive landed alongside step 2 because `loader.read-file` needed it (see
[wasm64-reserve-abort.md](wasm64-reserve-abort.md)):

```coil
(defn al-extend! [T] [(l (mut (ArrayList T))) (src (ptr T)) (n i64)] (-> i64)
```

It reserves the whole run up front and `mem-copy`s it, instead of a call plus a capacity
test per element.

## What the arena actually bought

The 97 `malloc-allocator` call sites are **not** where the time was. The pipeline already
threads `a` nearly everywhere, and `a` was `malloc-allocator` only because
`driver-main` bound it that way at `driver.coil:3369`. Changing that one binding puts the
whole AST on a bump allocator. Steps 3 and 5 are cleanup of the sites that *bypass* the
threaded `a`; step 4 is the change.

`build selfhost/src/main.coil -o /dev/null`, 15 paired runs alternating binaries, input
pinned with `COIL_STDLIB_DIR` (see [How to measure this
honestly](#how-to-measure-this-honestly) — without it these numbers are wrong):

| | median | paired vs malloc root | faster |
| --- | --- | --- | --- |
| `malloc-allocator` root (was) | 0.550s | — | — |
| overflow arena root, 1 GiB | 0.540s | **-30 ms** | 12/15 |

**Peak RSS drops too: 1044 MiB → 1021 MiB.** This was the open question in step 4 —
whether a never-reset arena would trade unacceptable memory for speed. It does the
opposite, because the compiler was already retaining essentially everything it allocated;
malloc's `free` was reclaiming almost nothing, so dropping it costs no memory and malloc's
per-block bookkeeping is what goes away.

Region size was chosen by measurement. Comparing the capacities against each other (these
binaries share an embedded stdlib, so they are directly comparable):

| region | median | note |
| --- | --- | --- |
| 256 MiB | 0.575s | **slower than malloc** — see below |
| 1 GiB | 0.540s | shipped |
| 2 GiB | 0.540s | no better |
| 4 GiB, strict | 0.540s | no better |

The win **scales with how much of the workload the region covers**. A self-build allocates
about 1 GB; at 256 MiB most allocations overflow, and an overflowing allocation pays the
failed bump *and* the malloc, so the change becomes a regression. Past 1 GiB there is
nothing left to win, so 1 GiB is where it lands.

The reservation does not tax small invocations. `check` on a two-line file: 44.3 MiB peak
RSS and a 20 ms median against 45.2 MiB and 20 ms for the malloc build (paired delta
+0.0 ms median over 25 pairs). It is lazily backed, so it costs one `mmap` and no resident
pages.

### Why it is an *overflow* arena

A fixed region is the wrong contract for a **root** allocator, and this is measurable
rather than theoretical. `repl-cmd` reuses one allocator across every evaluation and
retains about 17 MiB per eval — 200 evals reach 3.5 GiB of RSS. That retention is *not*
new; the malloc build grows identically (3566 MiB vs 3482 MiB at 200 evals). But a strict
4 GiB arena turns it into a hard abort at roughly 240 evaluations, where malloc simply
keeps going.

So `overflow-arena-allocator` bumps while its region lasts and falls back to malloc
afterwards: the fast path stays a pointer increment and the worst case degrades to
exactly the behaviour it replaced. 500 REPL evaluations run clean on it.

It is a separate vtable (`aro-alloc`/`aro-resize`/`aro-free`) rather than a flag on
`Arena`, because `arena-over-buffer` must stay reachable from freestanding code and a
branch to `malloc` on the live path keeps the symbol referenced even when it can never be
taken — `--gc-sections` can only drop what nothing mentions.

Two traps that cost real time and are guarded by tests in `examples/arena-growth.coil`:

* **`resize` with a null pointer.** A collection's first growth calls `resize(NULL, 0, n)`.
  Answering it with `realloc(NULL, n)` is correct C and completely wrong here: it routes
  *every* list's initial buffer to malloc and leaves the arena serving nothing. It must
  return `(None)` so the caller reaches `alloc-slice`, which bumps.
* **Interpreting a foreign pointer as an offset.** `ar-resize` computes `p - base`; with
  overflow blocks in play, `p` may not be the arena's at all. It now range-checks before
  treating the difference as an offset.

### The sites that should stay on malloc

`ast.coil`'s 6 sites back `type-res-box` and `src-mod-box` — `alloc-static` singletons
holding the resolver's side tables, reached from `type-res-record`/`src-mod-record`, which
take no allocator and are called from deep inside `resolve.coil`. Threading `a` to them
means threading it through the whole resolver, which is step 5's job and not a leaf
conversion. They are also process-lifetime by construction, which is the category this
document already carved out as legitimately malloc. Left alone deliberately, not missed.

(Both `-reset` functions overwrite their list with a fresh `al-new`, leaking the previous
backing buffer. Harmless today — reset runs once per compile and the process exits — and
noted here because an arena would make it free.)

### Where the gigabyte goes

Compile-checking the compiler costs ~1 GB of peak RSS, which is ~20 KB per source line.
That number deserved an explanation, so here is the measurement.

**It is linear, not quadratic.** Synthetic files of N trivial one-line functions,
`coil check`:

| defns | peak RSS |
| --- | --- |
| 1,000 | 103.9 MiB |
| 2,000 | 163.5 MiB |
| 4,000 | 282.0 MiB |
| 8,000 | 511.2 MiB |
| 16,000 | 971.9 MiB |

A dead-straight line: ~59 KB per trivial function on a ~46 MiB baseline. There is no
runaway; the constant is simply enormous.

Separating the two variables — 4,000 defns × 1 expression vs 4,000 × 10 vs 1 × 40,000 —
gives roughly **36 KB per definition plus 10 KB per expression**. For scale, `sizeof Sexp`
is 64 bytes and `sizeof Expr` is 168. An `(iadd x 5)` is a few hundred bytes of actual
data and costs about 10 KB, so we spend on the order of 100× the size of the thing being
represented.

Peak RSS by stage, same input (4,000 defns × 10 expressions), against an empty module so
the fixed prelude cost is visible:

| stage | empty | 4,000 exprs | 40,000 exprs | per expr |
| --- | --- | --- | --- | --- |
| `dump-read` | 21 MiB | 32 MiB | 56 MiB | ~0.9 KB |
| `dump-expand` | 41 MiB | 181 MiB | 440 MiB | **~10 KB** |
| `dump-ir` | 48 MiB | 250 MiB | 674 MiB | ~16 KB |

**Macro expansion is the multiplier: it costs about 11× what parsing the same program
costs.** The parsed `Sexp` tree is comparatively cheap. Note the test input barely uses
macros — the bodies are `do` and `iadd`, both core — so this is not the price of a
macro-heavy program, it is what the expander costs on a program that has almost nothing
to expand.

And nothing is ever released: no stage frees its predecessor's tree, so peak RSS is
simply *total bytes ever allocated*. That is why moving to an arena did not raise RSS —
`free` was already reclaiming nothing.

This is the same conclusion this document reached from the profile at the top, arrived at
from the memory side: the remaining wins are **fewer intermediate representations**, not a
faster allocator. A single unnecessary whole-tree copy in the expander is worth more than
everything in this document put together. Worth checking first: whether the expander
re-materializes the tree on every pass even when a pass expands nothing.

### The first parsimony win: initial list capacity

Histogramming every allocation by *exact* size (a throwaway build counting requests inside
`aro-alloc`) says where the bytes are far better than any guess. `check
selfhost/src/main.coil`, 5.8M allocations, 891 MiB:

| size | count | MiB | what it is |
| --- | --- | --- | --- |
| 672 | 360,022 | 230.7 | `(ArrayList Expr)`, capacity 4 (4 × 168) |
| 256 | 917,016 | 223.9 | `(ArrayList Sexp)`, capacity 4 (4 × 64) |
| ≥4096 | 3,000 | 114.5 | read buffers and other large one-offs |
| 168 | 673,612 | 107.9 | a single heap `Expr` |
| 32 | 2,102,219 | 64.2 | the `ArrayList` struct itself |
| 1344 | 11,036 | 14.1 | `(ArrayList Expr)`, capacity 8 |

The two biggest line items are *first* blocks, and the capacity-8 blocks are rarer than
the capacity-4 blocks by more than 20×. Almost nothing was outgrowing its first
allocation — the first allocation was simply too big. `al-push!` started every list at 4
elements, which for a 168-byte `Expr` is 672 bytes handed to a list that usually holds one
or two.

Changing that one literal from 4 to 2:

| initial capacity | peak RSS | median wall |
| --- | --- | --- |
| 4 (was) | 1022 MiB | 0.530s |
| **2 (now)** | **841 MiB** | **0.495s** |
| 1 | 811 MiB | 0.500s |

**-18% memory and -30 ms**, from one token. It is faster *and* smaller because less memory
allocated is less memory to fault in and miss on — at this scale the two are the same
lever, not a trade. 1 saves a little more memory but the extra growth step stops paying,
so 2 is where it lands.

`examples/arena-growth.coil` caught this change immediately (it asserts exact arena
offsets), which is the argument for asserting exact numbers rather than bounds.

The next candidate, not yet done: **`Expr` is 168 bytes because `ExprKind` is padded to
its largest variant, `EDynDispatch`, whose payload is exactly 120 bytes.** Every
expression in every program pays for a `dyn`-dispatch variant that almost none of them
are. Boxing it behind a pointer should take `Expr` to ~112 bytes and save roughly another
70 MiB, at the cost of touching its ~17 match sites across 13 files.

### What is left

Step 5's 38 `resolve.coil` sites now look much less valuable than they did. Step 4 already
moved every allocation that flows through the threaded `a`; what remains at those sites is
whatever explicitly bypasses it. The honest expectation is single-digit milliseconds, not
the 124 ms the profile attributes to allocation overall — that budget was mostly claimed
by one binding.

The larger remaining idea is the one the original staging hinted at with "per-compile":
this arena is per-*process* and never reset. A driver that reset it between compiles would
make the REPL's 17 MiB-per-eval retention vanish outright, which is a far bigger number
than 40 ms. It needs the long-lived REPL state (`defs`, the interner) separated from
per-compile state first, so it is genuinely a next piece of work rather than a tweak.

## What it cost

Binary size, which is a real cost and the only one:

| | binary | vs HEAD |
| --- | --- | --- |
| HEAD (`77a5d8032`) | 2,004,864 B | |
| + `ar-resize`, `al-extend!`, `read-file` | 2,031,440 B | +26.6 KB |
| + the `oom` message | 2,054,176 B | **+49.3 KB (+2.5%)** |

The `oom` message adds ~23 KB to every program that imports `alloc.coil`, which is every
program. That is the price of never seeing a silent exit 134 again.

Wall time costs nothing — it is a small win. **An earlier revision of this document
reported step 1 as a 3% regression. That was a measurement artifact**, described in the
next section; the corrected numbers are in [How to measure this
honestly](#how-to-measure-this-honestly).

## How to measure this honestly

⚠ **Two compiler binaries compiling `selfhost/src/main.coil` are not compiling the same
input.** Each binary bakes its own copy of `lib/*.coil` in via `include-str`, and
`main.coil`'s import closure resolves to *that embedded copy*, not to the disk. So a build
whose only change is 60 added lines in `lib/alloc.coil` is compiling 60 more lines than
the binary you are comparing it against — and it looks slower and hungrier for reasons
that have nothing to do with the change.

Pin the input with `COIL_STDLIB_DIR=$PWD` on **both** sides. Everything below does.

This artifact produced two wrong conclusions before it was caught, both of which had been
written into this document as findings:

* Step 1 measured as **+20 ms (a 3% regression)**. Controlled, it is **-20 ms**, slower in
  only 2 of 15 pairs. There was never a regression.
* The arena measured as **+50 MiB of peak RSS**. Controlled, it is **-23 MiB**. Three
  separate builds were made chasing that phantom — testing the `ar-resize` range guard,
  then the `reader.coil` conversion, then strict-vs-overflow — and all three came back
  identical, which is the clue that the variable was the input rather than the code.

Corrected, `build selfhost/src/main.coil -o /dev/null`, 15 paired runs each, input pinned:

| | median | paired vs previous | peak RSS |
| --- | --- | --- | --- |
| HEAD (`77a5d8032`), malloc root | 0.590s | — | 1045 MiB |
| + `oom`, `ar-resize`, `al-extend!`, `read-file` | 0.570s | **-20 ms** (slower in 2/15) | 1044 MiB |
| + arena root, `reader.coil` (this change) | 0.540s | **-30 ms** (faster in 12/15) | 1021 MiB |

Cumulative against HEAD: **-40 ms median, -56 ms mean, faster in 11 of 15 pairs**, and
**23 MiB less peak RSS**.

Medians still drift between rounds with machine load — HEAD measured 0.590s here and
0.550s an hour earlier — which is why every row is a *paired* delta against a binary run
back-to-back with it, not a comparison of numbers from different rounds.

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
