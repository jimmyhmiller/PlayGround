# 01 — Property-Access Inline Cache

## Implementation status: ✅ Implemented

- **Code**: [`crates/dynlang/src/ic.rs`](../../crates/dynlang/src/ic.rs) — `PropertyIc` builder, `PropertyIcRuntime` with `install_thread`, `prop_slow_thunk` auto-bound by `DynGcRuntime::compile_jit` ([gc.rs](../../crates/dynlang/src/gc.rs)).
- **Tests**: 3 unit tests in `ic::tests` (build/finalize, register_type, install/restore).
- **Migration**: beagle's `IcContext`, `count_property_accesses_with_inlining`, `emit_ic_property_load`, and `ext_prop_slow` deleted; replaced with `PropertyIc` calls in [`crates/beagle/src/lower.rs`](../../crates/beagle/src/lower.rs).
- **Differs from plan**: indirection-cell strategy chosen over IR-mutation. Polymorphic IC strategy not yet implemented (monomorphic only).

## Problem

Property access (`obj.field`) is the canonical hot path in dynamic languages.
The toolkit ships `dynsym::InlineCacheArray`, `InlineCacheEntry`, and
`DispatchTable` — the right *primitives* — but every embedder has to assemble
the same scaffolding around them. In beagle, that scaffolding is:

- **Class-key encoding trick.** `IcContext` stores per-type tables keyed by
  `u16 type_id + 1` because `0` is reserved by `InlineCacheEntry::EMPTY`
  ([`lower.rs:75-85`](../../crates/beagle/src/lower.rs#L75)). Every embedder
  has to rediscover this and document the `+1`.
- **Stable base address embedded as IR const.** The IC array's base pointer is
  iconst'd into every IC site
  ([`lower.rs:387-391`](../../crates/beagle/src/lower.rs#L387)). The array
  must be sized up-front and never reallocated — a fragile invariant in code
  comments only.
- **Pre-counting walker.** A 180-line
  `count_property_accesses_with_inlining`
  ([`lower.rs:2038-2221`](../../crates/beagle/src/lower.rs#L2038)) walks the
  whole program *with simulated inlining* just to size the array.
- **Cache-id allocator.** A `&mut u32` counter threaded through `Lowerer`
  with a post-lowering `assert_eq!(next_cache_id, num_ic_sites)`
  ([`lower.rs:450-453`](../../crates/beagle/src/lower.rs#L450)) papering
  over the count/consume mismatch.
- **60-line IC emit.** `emit_ic_property_load`
  ([`lower.rs:1143-1195`](../../crates/beagle/src/lower.rs#L1143)) emits the
  guard / fast-load / slow-call / merge-block IR shape every dynamic language
  wants.
- **Slow-path extern.** `ext_prop_slow`
  ([`main.rs:153-206`](../../crates/beagle/src/main.rs#L153)) reads the
  object header, walks any forwarding pointer, looks up the field offset
  in the per-type table, fills the IC entry, returns the value. ~50 lines
  of code that's identical across embedders.

This is the single biggest piece of plumbing in the port — easily ~250 lines
and most of `Lowered`'s public surface (`ic`, `array_type_id_u16`,
`array_len_offset`).

## Proposed API

A high-level `PropertyIc` builder owned by `dynlang`, sitting on top of
`dynsym`'s primitives:

```rust
use dynlang::ic::{PropertyIc, IcStrategy};

// Create during module setup. Captures &mut DynModule so it can declare
// the slow-path extern itself — embedders never see `beagle_prop_slow`.
let mut ic = PropertyIc::new(&mut dm, IcStrategy::Monomorphic);

// Register dispatch tables as struct types are declared. The +1 class-key
// dance is internal.
for s in structs {
    ic.register_type(s.type_id, s.field_offsets.iter().map(|(n, o)| (n.as_str(), *o)));
}

// During lowering, one call per property-access site. Returns the loaded
// value. Mints + tracks cache slot ids internally.
let v = ic.emit_load(f, obj, "field_name");

// After lowering, finalize. Allocates the IcArray to fit observed sites,
// patches the embedded base pointer, returns the live runtime context.
let ic_runtime = ic.finalize();
```

`finalize` removes the pre-counting walker entirely.

### Resolving the "stable base address" problem without pre-counting

Two acceptable shapes:

1. **Indirection slot.** Emit an iconst'd pointer to a single
   `*const InlineCacheEntry` cell that the runtime fills at finalize time.
   IR loads through it. One extra load per IC site — measurably cheap.
2. **Patch list.** `emit_load` returns an `IrPatch` token. `finalize`
   allocates the array and rewrites every recorded iconst in the IR. No
   runtime indirection but requires an IR mutation API on `dynir`.

(1) is simpler and ports trivially across backends; (2) is faster but
intrusive. Default to (1) and offer (2) as an opt-in for benchmark-driven
embedders.

## Implementation plan

1. **`dynlang::ic` module.** Owns:
   - `PropertyIc` struct with internal `SymbolTable`, `DispatchTable` map,
     site counter, slow-path `FuncRef`, base-indirection cell.
   - `register_type(ObjTypeId, &[(name, offset)])` — does the `+1` keying.
   - `emit_load(&mut DynFunc, obj: Value, field: &str) -> Value` — emits the
     guard / fast / slow shape. Interns the field name internally.
   - `finalize() -> PropertyIcRuntime` — allocates `InlineCacheArray`,
     writes the indirection cell, returns the runtime handle. The handle
     owns the array and the per-type tables; embedder stashes it in a
     long-lived place (thread-local, `Lowered`, etc.).

2. **Built-in slow path.** `dynlang::ic` registers an internal extern
   (`__dynlang_prop_slow__`) and binds it via the GcRuntime's auto-bind
   mechanism (same as `__gc_alloc__`). Forwarding-pointer walking goes
   through [03](03-forwarding-pointer-helper.md).

3. **Polymorphic strategy (later).** `IcStrategy::Polymorphic { slots: 4 }`
   emits a small linear search before the slow path. Same registration
   surface, different `emit_load` body. Out of scope for the first cut.

4. **Beagle migration.** Delete `IcContext`, `count_property_accesses_with_inlining`,
   `emit_ic_property_load`, `ext_prop_slow`, the `next_cache_id` plumbing,
   and the `assert_eq!`. Replace with three `PropertyIc` calls. Estimated
   delta: −250 LOC.

## Open questions / risks

- **`SymbolTable` ownership.** Beagle's `Lowered::ic.symbols` is exposed
  for the slow path's error-message lookups. The new `PropertyIcRuntime`
  should expose `try_name(sym)` so embedders can still produce friendly
  errors from their own externs.
- **Cross-module IC arrays.** If a future embedder JIT-compiles multiple
  modules sharing a runtime, the IC array can't be per-module. `PropertyIc`
  may need a `from_existing(runtime)` constructor. Not blocking.
- **Inlining double-count.** Beagle's pre-counter had to simulate inlining
  (each call site to an inlinable callee gets fresh slots). `emit_load`
  inherently sees every emit, so this falls out for free — no special case.
