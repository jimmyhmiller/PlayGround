# The wasm64 `al-reserve!` abort: not reproducible, never root-caused

Status: **the blocked fix has landed; the abort itself does not reproduce.** Reported
2026-07-28 while fixing a defect in `loader.read-file`; re-investigated the same day with
a diagnostic `oom` in place. The `read-file` fix is now at HEAD and `gate-wasm` is green
with it. No root cause was ever established, so this record is kept rather than deleted —
if the abort comes back, start from [If it returns](#if-it-returns).

## What was originally reported

Patching `src/compiler/loader.coil` so `read-file` sized its output list up front instead
of pushing one byte at a time, rebuilding the compiler, then:

```
coil build src/compiler/main_wasm.coil --target wasm64-unknown-unknown --wasm-stack-size=64 -o /tmp/w.wasm
```

made the **host** compiler die with exit 134 (SIGABRT), writing nothing to stdout or
stderr. Every other gate passed with the same binary; only the wasm64 target aborted. The
bisect recorded at the time:

| variant | wasm64 build |
| --- | --- |
| helper + per-byte `al-push!`, no reserve | works |
| helper + `al-reserve!(oldlen + 4)` | works |
| helper + `al-reserve!(oldlen + 65536)` | **aborts** |
| `al-reserve!(oldlen + 65536)` + per-byte `al-push!` (no `mem-copy`) | **aborts** |

## What happens now

Both "aborts" rows were reconstructed and rebuilt. Neither aborts.

| variant rebuilt | wasm64 build |
| --- | --- |
| `al-reserve!(oldlen + 65536)` + per-byte `al-push!` (row 4) | builds; 10/10 consecutive runs, exit 0 |
| `al-reserve!(oldlen + n)` + `mem-copy` (row 3, i.e. the real fix) | builds; `gate-wasm` PASS |
| `al-reserve!(oldlen + 4194304)`, far past the reported threshold | builds |

Each was tried twice: against `src/stdlib/` as modified by the allocation work, and against
`src/stdlib/` restored to its state at `77a5d8032` with the stock `build/bin/coil`. Same result. The
wasm64 output passes `wasm-tools validate --features=memory64`, is a single static
module, and self-checks the compiler source under Node, so these are real wasm64 builds
and not a target-string fallback.

## The stated hypothesis is disproved

The original write-up reasoned that a silent `abort()` could only be `oom`
(`src/stdlib/alloc.coil`), since every other abort in the tree prints first — and therefore that
something on the wasm64 path was running under an allocator that could not satisfy a
~64 KB request, with `Arena` the obvious suspect.

The first half held: `oom` was the only silent abort. It no longer is.

The second half does not. **The compiler never constructs an arena.** `arena-allocator`
and `arena-over-buffer` appear nowhere under `src/compiler` except inside `guide.coil`'s
documentation text. Every `ArrayList` in the compiler, on every target, is backed by
`malloc-allocator`, whose `ma-resize` is `realloc` — so `al-reserve!` never reaches its
`alloc-slice` fallback at all, and a 64 KB request cannot come back `(None)` short of
libc genuinely failing.

That also dissolves the contradiction the original write-up flagged as the thing to chase
first: incremental doubling *should* consume more arena than one large reserve, and the
observation was the other way round. There was no arena. The reasoning was sound; the
premise was false.

## What the abort was, then

Unknown. The most likely explanation is that the tree at the time carried other
uncommitted work — the report itself begins "found while fixing an unrelated defect in
`loader.read-file`" — and that the reverted fix was not the only thing reverted. That is
a guess and is labelled as one. What can be said with evidence is only that the recorded
repro does not reproduce at `77a5d8032` or at HEAD.

## The defect it blocked — fixed

`read-file` and `read-file-opt` copied their 64 KB read buffer into an `ArrayList` one
byte at a time: a function call and a capacity test per byte, roughly two million of each
for the ~2 MB the compiler reads of its own tree, plus a realloc-and-copy at every
doubling.

Both now share one `drain-fd!` helper that appends each read with a single `al-extend!`
(`src/stdlib/arraylist.coil`), which reserves the run up front and `mem-copy`s it. Measured
impact is about 10 ms of a ~470 ms whole-compiler frontend run — real, consistent with
the ~14 ms originally predicted, and, as predicted, not separable from noise in any
single measurement. It is a quality fix, not a performance one. See
[ALLOCATION.md](../reference/ALLOCATION.md).

## Why this class of bug cannot recur silently

`oom` prints before aborting, naming the caller and the request size:

```
out of memory: arraylist al-reserve! failed to allocate 2048 bytes; the allocator is exhausted
```

A silent exit 134 with an empty log is no longer a possible outcome of an allocation
failure. It cost 49 KB of binary and about 3% of frontend wall time; that trade is
measured and argued in [ALLOCATION.md](../reference/ALLOCATION.md).

## If it returns

1. Read the abort message. It now exists, and it names the collection and the byte count.
2. If there is still no message it is **not** `oom` — every other `abort()` in the tree
   prints first, so look for a libc-raised `SIGABRT` (`malloc` heap corruption, a failed
   `__chk` guard) rather than an allocation failure. `lldb -- <compiler> build …` with
   `break set -n abort` gets the frame in a minute.
3. Confirm which allocator is actually in use before theorising about it. `grep -rn
   'arena-allocator\|arena-over-buffer\|malloc-allocator' src/compiler` is the whole
   answer and takes ten seconds; skipping it is what sent the first investigation after
   an arena that was never there.
4. Record whether the working tree was clean. `git status` at the moment of the
   observation would have settled this one.
