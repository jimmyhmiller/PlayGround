# 07 — Built-in Indexed Sequence Type

## Implementation status: ✅ Implemented (CopyOnPush only)

- **Code**: [`crates/dynlang/src/stdlib/indexed_seq.rs`](../../crates/dynlang/src/stdlib/indexed_seq.rs) — `IndexedSeq::register/emit_literal/emit_push/emit_get/emit_length_unboxed`, plus `SeqView` host-side accessor that runs through `dynalloc::follow_forwarding`.
- **Tests**: 6 tests in `stdlib::indexed_seq::tests`. Two register-time (layout, view-rejection); four end-to-end IR tests via the interpreter (literal + view, length unboxed, get-at-index, push extends by one).
- **Migration**: beagle dropped `ArrayLayout` struct, `ARRAY_TYPE_NAME`, `Lowerer::array` field, three array-element free fns (`array_elem_addr`/`array_load_elem_at`/`array_store_elem_at`), and three `Lowerer` methods (`lower_array_literal`/`lower_push`/`array_load_at`). Host-side: `ArrayInfo`, `ARRAY_INFO` thread_local, `array_len_of`/`array_elem_at`/`decode_array_ptr` deleted. The latent `elem_base = len_offset + 8` bug in beagle's old host code (off by one varlen-count word) is fixed in passing — `IndexedSeq` uses `varlen_element_offset(0)` correctly.
- **Not implemented**: `Doubling` growth policy (only `CopyOnPush` ships, matching beagle's existing semantics). `BoundsPolicy::Trap` opt-in. Element-type parameter (still hardcoded `Type::I64` / NanBox).

## Problem

Every dynamic-language port needs a "growable indexed sequence" — JS arrays,
Python lists, Lua tables-as-arrays, Lox arrays, Beagle arrays. dynobj gives
you `varlen_values` and `Raw64` field kinds, but assembling a sequence type
from them is an exercise the embedder is forced to repeat.

In beagle:

- **Type registration** ([`lower.rs:191-196`](../../crates/beagle/src/lower.rs#L191))
  declares a synthetic `__Array__` obj-type with a `len: Raw64` field plus
  `varlen_values()`. The fact that a sequence is "Raw64 len + varlen
  payload" is reverse-engineered from the dynobj field-kind primitives.
- **Layout extraction** ([`lower.rs:245-255`](../../crates/beagle/src/lower.rs#L245))
  pulls `len_offset` and `elem_base = varlen_element_offset(0)` out of
  the registered type and stashes them in an `ArrayLayout` struct so the
  lowerer can do address arithmetic without re-borrowing the `DynModule`.
- **Lowerers for `[…]`, `push`, `get`, `length`** all hand-rolled:
  - `lower_array_literal` ([`lower.rs:1007-1041`](../../crates/beagle/src/lower.rs#L1007))
    — alloc + len store + per-element store, with the doc-02 root dance.
  - `lower_push` ([`lower.rs:1048-1109`](../../crates/beagle/src/lower.rs#L1048))
    — alloc-bigger + manual i64 stack-slot loop counter + copy + append.
    The loop counter is a `create_stack_slot(8, false)` because we don't
    have block-arg-based induction variables here; an embedder reading
    this code thinks the block-arg style is the answer until they discover
    it conflicts with the safepoint contract.
  - `array_load_at` / `array_load_elem_raw` / `array_store_elem`
    ([`lower.rs:1115-1137`](../../crates/beagle/src/lower.rs#L1115)) — the
    `base + idx*8` math, repeated three times.
- **Host-side recognition** ([`main.rs:55-76`, `345-388`](../../crates/beagle/src/main.rs#L55))
  — `ext_length` and `ext_get` need to recognize array NanBoxes vs scalars,
  so `Lowered` exposes `array_type_id_u16` and `array_len_offset` for the
  host to read. The `elem_base = len_offset + 8` invariant is hand-derived
  in `array_elem_at` ([`main.rs:362`](../../crates/beagle/src/main.rs#L362))
  because the host doesn't get `varlen_element_offset(0)`.

That's ~150 LOC implementing the same data structure every port wants.

## Proposed API

A first-class `IndexedSeq` type in `dynlang` (or a new `dynstdlib` crate),
sitting on dynobj's primitives:

```rust
use dynlang::stdlib::IndexedSeq;

// Register once during module setup. Returns a handle the lowerer uses
// for codegen and the host uses for runtime recognition.
let arr = IndexedSeq::register(&mut dm, "Array");

// During lowering:
let v = arr.emit_literal(f, &elements);          // [a, b, c]
let v = arr.emit_push(f, src, val);              // push(arr, x)
let v = arr.emit_get(f, src, idx_box);           // arr[i]
let v = arr.emit_length(f, src);                 // length(arr)

// Host-side (e.g. for builtin print/inspect):
if let Some(view) = arr.view(nanbox_bits) {
    let len = view.len();
    let elem = view.get(idx);
}
```

`emit_*` methods bake in the doc-02 GC root dance and the doc-03
forwarding chase. `view` is the host-side reader: returns `Option<SeqView>`
that internally checks the type tag and follows forwarding.

### Capacity policy

Beagle's `push` reallocates every call (capacity = old_len + 1). That's
fine for a benchmark but pathological for any real program. `IndexedSeq`
should support an opt-in growth policy:

```rust
let arr = IndexedSeq::builder(&mut dm, "Array")
    .growth(GrowthPolicy::Doubling)  // or .growth(GrowthPolicy::CopyOnPush)
    .build();
```

`Doubling` adds a `cap: Raw64` field and amortizes — typical
`Vec<T>::push` cost. `CopyOnPush` matches beagle's current persistent-ish
semantics. Default to `Doubling`.

## Implementation plan

1. **`dynlang::stdlib::IndexedSeq` module.** Owns:
   - `register(&mut DynModule, &str) -> IndexedSeq` — registers obj-type
     with the right field shape, captures `ObjTypeId`, `len_offset`,
     `elem_base`. Builder variant for growth policy.
   - `emit_literal(&mut DynFunc, &[Value])` — uses `with_rooted` from
     doc 02 to pin elements across the alloc.
   - `emit_push(&mut DynFunc, src: Value, val: Value) -> Value` — for
     `Doubling`, branches on `len < cap` (in-place store) vs
     `len == cap` (alloc bigger + copy); for `CopyOnPush`, always allocs.
   - `emit_get(&mut DynFunc, src: Value, idx_box: Value) -> Value` —
     unbox idx, base + idx*8, load.
   - `emit_length(&mut DynFunc, src: Value) -> Value` — load len,
     box as NanBox int.
   - `view(bits: u64) -> Option<SeqView>` — host-side reader. Internally
     calls `NanBox::decode_live_ptr` (doc 03), checks type id, returns
     a wrapper exposing `len()` / `get(i)` / `iter()`.

2. **Loop counter helper.** `lower_push`'s manual stack-slot counter
   exists because the safepoint live set conflicts with naive
   block-arg induction variables. Either:
   - Document the safepoint contract for block args (and use them).
   - Provide a `f.scalar_loop_counter()` helper that does the right thing.

   The first cut of `IndexedSeq::emit_push` should pick one and document
   why. Block-args are SSA-clean; stack-slot counters survive any
   GC-policy change. Lean stack-slot for now.

3. **Beagle migration.** Replace `__Array__` registration, `ArrayLayout`,
   `lower_array_literal`, `lower_push`, `array_load_at`,
   `array_load_elem_raw`, `array_store_elem`, the `ARRAY_INFO`
   thread-local, and the `array_*` helpers in main.rs. Estimated delta:
   −150 LOC.

## Open questions / risks

- **Element type.** Beagle stores raw NanBox `i64`s; some embedders may
  want unboxed `f64` storage (e.g. Lua's number-only arrays). Make the
  element type a parameter: `IndexedSeq::register(...).element(Type::I64)`.
  Default to `I64` since that's NanBox-native.
- **Persistence vs mutation.** Doubling is a mutable contract; beagle's
  current `push` is "build a new array" (functional). The choice affects
  user-visible semantics. Make this explicit per language at registration
  time.
- **Bounds checking.** `emit_get` today is unchecked (beagle is
  benchmark-grade). The builder should offer
  `.bounds_check(BoundsPolicy::Trap | None)`. Default `None` to match
  current behavior, opt-in trap once a real embedder needs it.
- **`view`'s lifetime.** Host-side reader holds a raw pointer that's
  invalidated if a GC runs. Either return `SeqView<'a>` borrowed from a
  GC-pinning guard, or document that `view`'s output is single-use
  (no allocations between obtaining and reading). Match dynalloc's
  existing root-handle convention if one exists; otherwise document.
