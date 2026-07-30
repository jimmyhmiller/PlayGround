# meta — the SAME Scheme source run by our Coil and by Chez

`evalcore.scm` is a first-order metacircular Scheme evaluator (seval/sapply, assoc-list
env, closures/prims as tagged data, mutable global env). The exact same source is:
- native-compiled by **Chez** (`chez --script evalcore.scm`), and
- compiled to native Coil by **us**: `scm2coil.py` maps the Scheme subset onto the Val
  runtime (`sval.coil`) so the transparent-GC metaprogram (`gcauto2.coil`) inserts GC.

Same interpreter, same target (fib 30), different host — a compiler-quality comparison.

## The interning bug (answering "does the boxed path have it too?")

Yes — and worse than the unboxed `proper/` version. `scm2coil.py`'s `cquote` lowered a
special-form literal like `(quote if)` to `(mk-sym (intern "if"))` **inline at every use
site**, so each `seval`/`sapply`/`doprim` dispatch re-scanned the symbol table (`intern`
is a linear `str-eq` walk) AND allocated a fresh boxed symbol (`mk-sym` calls `alloc-val`),
up to four times per expression.

Fix: intern each special-form name **once** (lazy, cached in a static i64 cell) and compare
`(car e)` by its interned **id** (`hid` = symp?→sym-id else -1) instead of re-boxing. Ids
are plain i64s, so there is no GC-root problem (a cached *boxed* symbol would be collected —
it isn't a root). Effect at fib(30):

| metric              | before  | after   |
|---------------------|---------|---------|
| fib(30) wall-clock  | 9.45 s  | **5.17 s** (1.83× faster) |
| vs Chez (295 ms)    | 32.0×   | **17.5×** |
| allocations         | 377 M   | **237 M** (−37%) |
| GC collections      | 19 047  | **11 972** (−37%) |

(Measured with hyperfine at load ~3; the pre-fix binary rebuilt from git for an
apples-to-apples run. The wall-clock win exceeds the alloc drop because removing the
per-dispatch `intern` also removes a linear symbol-table scan, not just an allocation.)

## Then: the GC runtime, not the representation, was the rest of the gap

I first assumed the residual 17.5× was the value representation (boxing every integer).
Profiling proved that wrong: `malloc`/`free` per object dominated, and ints were only ~4 M
of 233 M allocations — **cons cells** (arg lists + assoc-list env frames) are the bulk. So
the fix was the *metaprogram's collector* (`sval.coil`), leaving the Scheme source and the
transform untouched:

| runtime / transform change                                  | fib(30) | vs Chez |
|-------------------------------------------------------------|---------|---------|
| intern fix (above)                                          | 5.17 s  | 17.5×   |
| + pool/slab allocator (no malloc/free per object)           | 1.68 s  |         |
| + gc-threshold 20 000 → 500 000 (11 972 → 465 collections)  | 1.68 s  |         |
| + fixnum tagging (immediate ints, GC skips them)            | 1.66 s  |         |
| + no live-list: linear-slab sweep, alloc writes only the object | 1.357 s | 4.6× |
| + nil & booleans immediate too (233 M → 21.5 M allocs)      | 0.541 s | ~1.8×   |
| + elide GC frames for non-allocating fns (`gcauto2`)        | **~0.33 s** | **~1.15×** |

From 32× down to **~1.15× Chez** (0.295 s), all in the metaprogram — the GC runtime
(`sval.coil`) plus one transform change (`gcauto2.coil`), with the Scheme interpreter
source untouched. (Late measurements min-of-20 under machine load; ratios are stable, wall
-clock inflated.) The last two steps mattered most: the env lookups allocated **two boxed
booleans per iteration** (`s-null` + `s-eq`), which immediates removed; and the transform
was wrapping every function — including the non-allocating lookups — in a root-frame, which
an allocation-effect analysis now elides. GC soundness stress-tested at threshold 500
(59 919 collections, still 832040) and against Chez on a map/closure program (65).

What remains is the eval/apply core and the assoc-list lookups themselves — the same
algorithm Chez compiles, now within codegen noise of it. The unboxed `proper/` version
(arena, no GC) sits at the same ~parity point from the other direction.

Note: `native/schemenative.coil` (hand-written boxed) and `scheme.coil` (stdin tree-walker,
boxed) still contain the `(mk-sym (intern …))`-per-dispatch pattern; they are demos, not the
benchmarked path, and can be fixed the same way. Both benefit from the `sval.coil` GC rewrite.

    python3 scm2coil.py evalcore.scm > evalcore.coil
    ../../coil build evalcore.coil -o /tmp/evalcore && /tmp/evalcore
