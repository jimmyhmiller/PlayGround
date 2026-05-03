# Toolkit Improvements — Lessons From the Beagle Port

These docs capture friction points found while porting the beagle frontend
(`crates/beagle`) to the dynamic-toolkit. Each describes a pattern beagle had
to hand-roll that should be a library primitive, with a problem statement, a
proposed API, and an implementation plan.

## Status

Nine of ten docs are implemented; doc 09 remains as a plan (it's a pure
beagle-side AST-walker refactor, not a toolkit primitive). Each implemented
doc starts with an **Implementation status** header pointing to the actual
code paths and tests.

| # | Title | Touches | Status |
|---|-------|---------|--------|
| [01](01-property-ic.md) | Property-access inline cache | dynlang, dynsym | ✅ Implemented |
| [02](02-gc-rooted-temps.md) | GC-rooted temporaries across alloc | dynlang | ✅ Implemented |
| [03](03-forwarding-pointer-helper.md) | Forwarding-pointer chase helper | dynalloc | ✅ Implemented |
| [04](04-extern-registration.md) | Extern + slow-path registration | dynlang | ✅ Implemented (defaults only) |
| [05](05-host-context.md) | Host context for JIT extern thunks | dynlang | ✅ Implemented |
| [06](06-nanbox-helpers.md) | NanBox embedder helpers | dynvalue | ✅ Implemented (constants/encoders only) |
| [07](07-indexed-seq.md) | Built-in indexed sequence type | dynlang | ✅ Implemented (CopyOnPush only) |
| [08](08-typed-binops.md) | Type-specialized binop selection | dynlang | ✅ Implemented |
| [09](09-ast-visitor.md) | AST visitor boilerplate | beagle | 📋 Plan only |
| [10](10-misc-helpers.md) | Small `DynFunc` helpers | dynlang | ✅ Implemented |

## Cumulative impact

Beagle's `lower.rs` + `main.rs` shrank by ~755 net lines. The toolkit gained
~370 lines of reusable primitives across `dynlang`, `dynalloc`, and
`dynvalue`. Every primitive carries unit tests; both beagle benchmarks
(`binary_trees.bg` checksum 524272, `ray_cast_bench.bg` checksum
789443528.5380735) verify the migrations end-to-end.

Each doc is self-contained — read in any order.
