# OPEN BUG: a large `al-reserve!` in the loader aborts the wasm64 target build

Status: **open, not fixed.** Found 2026-07-28 while fixing an unrelated defect in
`loader.read-file`. The fix was reverted; `read-file` is still wrong at HEAD (see
[The defect this blocks](#the-defect-this-blocks)).

## Reproduction

Patch `selfhost/src/loader.coil` so `read-file` sizes its output list up front instead of
pushing one byte at a time — any form of `al-reserve!` with a large capacity will do —
rebuild the compiler, then:

```
coil build selfhost/src/main_wasm.coil --target wasm64-unknown-unknown --wasm-stack-size=64 -o /tmp/w.wasm
```

The **host** compiler dies with exit 134 (SIGABRT) and writes **nothing** to stdout or
stderr. `gate-wasm` reports it as `FAIL: cannot build the wasm compiler from
main_wasm.coil`.

Every other gate passes with the same binary: `gate-full`, all ten stage gates,
`gate-cli`, `gate-meta-engines`. Only the wasm64 target aborts.

## What was bisected

Each row is a separate build of the compiler, differing only in the body of the
`push-bytes!` helper called from `read-file`:

| variant | wasm64 build |
| --- | --- |
| helper + per-byte `al-push!`, no reserve | **works** |
| helper + `al-reserve!(oldlen + 4)` | **works** |
| helper + `al-reserve!(oldlen + 65536)` | **aborts** |
| `al-reserve!(oldlen + 65536)` + per-byte `al-push!` (no `mem-copy`) | **aborts** |

So:

* `mem-copy` is **not** involved — reserve alone reproduces it.
* The helper function itself is **not** involved — with the reserve removed it works.
* It is **size-dependent**. A 4-byte reserve is a no-op (`al-reserve!` returns early when
  `newcap <= oldcap`), so the abort tracks *actually performing a large growth*.
* It is not codegen selection: the *same host binary* compiles every other target fine
  and only aborts for wasm64.

## The most likely cause

`abort()` is called from exactly one place in the allocation path, and it is silent:

```coil
; lib/alloc.coil:25
(defn oom [] (-> :i64) (abort) 0)
```

`lib/assert.coil` also aborts but prints first, and our capture (`> log 2>&1`) was zero
bytes — so this is almost certainly `oom`, i.e. **allocation failure**, not memory
corruption. That reframes the bug: something in the wasm64 path is running under an
allocator that cannot satisfy a ~64 KB request.

The path to `oom` from a reserve is:

```
al-reserve!  ->  raw-resize            ; returns None on an allocator with no in-place resize
             ->  alloc-slice           ; the fallback: allocate new + copy
             ->  None                  ; allocator exhausted
             ->  (oom)                 ; abort(), no message
```

`lib/alloc.coil`'s `Arena` is exactly such an allocator — `ar-resize` unconditionally
returns `(None)` and `ar-alloc` returns `(None)` once `off + size > cap`.

**This is a hypothesis, not a confirmed diagnosis.** What is *not* yet established: which
allocator the wasm64 path actually uses for this list, and why a 64 KB request exhausts
it when incremental doubling to the same total does not. A bump allocator never reuses
freed space, so doubling (4 + 8 + … + 64K ≈ 128K) should consume *more* arena than one
64 KB reserve, which is the opposite of what is observed. That contradiction is the thing
to chase first.

## Where to look next

1. Confirm the abort site. Put a distinguishing `write` before `(abort)` in `oom`, or run
   the failing build under lldb and get the stack. This is ten minutes and turns the
   hypothesis into a fact.
2. Find which allocator backs the `ArrayList` inside `read-file` on the wasm64 path.
   `read-file` takes `a` from its caller; trace who passes what when `--target wasm64` is
   set.
3. Check whether the wasm64 path reads *different or larger* files than the native path
   (the wasm runtime shims, `main_wasm.coil`'s imports), so the reserve is much larger
   than for a native build.

## Why it was not worked around

The abort disappears if the reserve is capped below some threshold. Shipping that would
mean tuning a magic number around an undiagnosed memory bug and burying a real defect
under a green gate. `gate-wasm` caught something worth keeping caught.

## The defect this blocks

`loader.read-file` (and `read-file-opt`) copy their 64 KB read buffer into an
`ArrayList` **one byte at a time**:

```coil
(loop (if (icmp-ge (load i) (cast i64 n))
          (break)
          (do (al-push! (mut bytes) (cast u8 (load (index buf (load i)))))
              (store! i (iadd (load i) 1)))))
```

That is a function call and a capacity test per byte — roughly two million of each for
the ~2 MB the compiler reads of its own tree — plus a realloc-and-copy every time the
list doubles. `lib/mem.coil` already has `mem-copy`; the fix is three lines once the
reserve can be made safely.

Measured impact is small (~14 ms of a 510 ms whole-compiler frontend run, inside noise on
a single measurement), so this is a correctness/quality fix, not a performance one. See
[ALLOCATION.md](ALLOCATION.md) for where the frontend's time actually goes.
