# GC Library Integration (v0 Draft)

This document describes how the language runtime will use `../claude-experiments/gc-library` for garbage collection.

## 1. Integration Strategy

- The language runtime is Rust (stage0) and links directly to `gc-library` as a Rust dependency.
- Generated code calls a small C-ABI runtime shim (`gc_allocate`, `gc_pollcheck_slow`, etc.).
- We do **not** use `gc-library`'s C FFI layer (`gc_library.h`) in v0; that FFI is for external C clients and assumes the example object model.

This keeps the object model and ABI under our control while leveraging the GC algorithms.

## 2. GC Library Traits We Implement

We will implement these `gc-library` traits for our runtime object model:

- `GcTypes`
- `TaggedPointer`
- `ObjectKind`
- `GcObject`
- `ForwardingSupport` (needed for moving collectors)
- `HeaderOps` (if we use its helpers)
- `RootProvider` (frame chain)

## 3. Object Model and Header Layout

We will keep the language ABI header defined in `docs/v0/RUNTIME_ABI.md`:

```
struct ObjectHeader {
  void* meta;     // TypeInfo pointer
  u32   gc_flags; // mark/forwarding/size bits
  u32   aux;      // array length or extra info
}
```

- Header size is 16 bytes.
- `GcObject::header_size()` will return 16.
- Marking / forwarding bits live in `gc_flags`.

This means we will **not** use the example header format in `gc-library/src/example.rs`.

## 4. Field Layout (Pointer Section First)

`GcObject::get_fields()` must return a contiguous slice of pointer-sized fields so the GC can update them in place.

We will lay out object payload as:

```
[ ObjectHeader | pointer_fields[] | raw_bytes[] ]
```

- `pointer_fields` are `usize` slots containing tagged values.
- `raw_bytes` are non-pointer data (integers, floats, bytes).

This allows us to support mixed field types while still giving the GC a contiguous pointer slice.

The compiler will emit `TypeInfo` containing:

- number of pointer fields
- total object size
- offsets of non-pointer fields

## 5. Tagged Value Representation

We will use a tagged-pointer representation internally so that heap fields are always `usize` values:

- `TaggedPointer` encodes:
  - heap object pointer
  - immediate integer (I64)
  - null

Tag bits: 3 low bits (8-byte alignment required).

This aligns with the expectations of `gc-library` (`TaggedPointer::is_heap_pointer`).

Non-pointer data stored in heap objects:

- `I64` may be stored as tagged values in pointer slots
- `F64` or larger values stored in raw bytes section

## 6. Root Enumeration

We will expose a runtime frame chain, consistent with `docs/v0/RUNTIME_ABI.md`:

```
struct Frame {
  Frame* parent;
  FrameOrigin* origin;
  TaggedValue* roots[N];
}
```

`RootProvider` will:

- Walk the frame chain
- For each root slot, call GC with `(slot_addr, tagged_value)`
- Update slot values in-place on relocation

## 7. Allocation and GC Flow

Runtime entrypoints used by generated code:

- `gc_allocate(thread, meta, size_bytes)`
- `gc_allocate_array(thread, meta, length)`
- `gc_pollcheck_slow(thread, frame_origin)`
- `gc_write_barrier(thread, obj, value)`

Internally, these map to `gc-library` APIs:

- `Allocator::try_allocate(...)` for fast allocation
- `Allocator::gc(&roots)` when allocation fails or at safepoints
- `Allocator::write_barrier(...)` for generational GC

## 8. GC Algorithm Choice

Phase 0 default:

- `GenerationalGC` with `LibcMemoryProvider`

We can switch to `MarkAndSweep` if we need a non-moving baseline.

## 9. Safety and Invariants

We must uphold `gc-library` trait invariants:

- All heap pointers are 8-byte aligned.
- `TaggedPointer::is_heap_pointer` is correct for all heap values.
- `GcObject::get_fields()` returns all pointer slots and only pointer slots.
- `GcObject::full_size()` returns exact allocation size.
- Root enumeration is complete.

## 10. Open Decisions

- Exact bit layout in `gc_flags`.
- Whether we allow unboxed floats in the raw bytes section during v0.
- Whether we allow tagged integers in all pointer slots (recommended for v0).

