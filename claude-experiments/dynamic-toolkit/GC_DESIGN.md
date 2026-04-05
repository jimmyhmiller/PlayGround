# Design: GC Integration in dynlang

## Problem

Every language implementor using the toolkit does the same manual work:
1. Defines object types as Rust enums → leaks memory (no GC)
2. Stores objects in `Vec<MyObj>` → never freed
3. Ignores `Type::GcPtr`, `Safepoint`, `InterpRootManager` → leaves perf on the table
4. If they DO want GC, they need to manually: create TypeInfo, implement PtrPolicy, track roots, emit Safepoints, wire up SemiSpace — hundreds of lines of error-prone glue

The toolkit already has all the pieces (dynir has GcPtr + Safepoint, dynobj has TypeInfo, dynalloc has SemiSpace). They just aren't connected through dynlang.

## Design Principle

**GC configuration is required, not optional.** You must choose upfront.

```rust
// GcConfig is a required parameter — no default, no forgetting
let dm = DynModule::new(GcConfig::leak(), tags);              // explicit: bump alloc, never free
let dm = DynModule::new(GcConfig::semi_space(1 << 20), tags); // real GC with collection

// DynModule::new() without GcConfig does not exist.
```

`GcConfig::leak()` is the "no GC" escape hatch. It uses a bump allocator that never
collects — zero overhead, no safepoints emitted, no root tracking. But objects are still
allocated through the same `gc_alloc` / `gc_load_field` / `gc_store_field` API, so
switching from `leak()` to `semi_space()` is a one-line change.

This means:
- Object type declarations are always the API (not `Vec<MyObj>` in Rust)
- `gc_alloc` is always how you create objects (not `heap.push(...)`)
- Field access is always through typed helpers (not `obj.fields.insert(...)`)
- The only difference between leak and semi_space is whether memory gets reclaimed

## API: Object Type Declarations

When GC is enabled, you declare object types. dynlang generates TypeInfo automatically.

```rust
// Declare types on the module (before declaring functions)
let string_ty = dm.obj_type("String")
    .varlen_bytes()            // variable-length byte array (untraced)
    .build();

let closure_ty = dm.obj_type("Closure")
    .field("func_ptr", Raw64)  // untraced 64-bit word
    .field("arity", Raw8)      // untraced byte
    .varlen_values()           // variable-length GC-traced value array (upvalues)
    .build();

let upvalue_ty = dm.obj_type("Upvalue")
    .field("value", Value)     // GC-traced value slot
    .build();

let instance_ty = dm.obj_type("Instance")
    .field("class", Value)     // GC-traced pointer to class
    .varlen_values()           // fields stored as [key, val, key, val, ...] pairs
    .build();
```

Under the hood, `build()` creates a `&'static TypeInfo` (leaked into a static, or stored in a registry) with the correct header size, field count, varlen kind, alignment.

Each `ObjTypeId` is an opaque handle the language uses to allocate and access objects.

## API: Allocation in IR

```rust
// In a DynFunc:
let len = f.fb.iconst(Type::I64, 5);
let obj = f.gc_alloc(string_ty, len);    // returns Value of type GcPtr
// This emits:
//   safepoint([all live GcPtrs])         — GC can run here
//   v = call __gc_alloc__(type_id, len)  — allocates via SemiSpace
//   ; v : GcPtr

// Field access:
f.gc_store_field(obj, closure_ty, "func_ptr", func_ptr_val);
f.gc_store_field(obj, closure_ty, "arity", arity_val);
let fp = f.gc_load_field(obj, closure_ty, "func_ptr");

// Varlen access:
f.gc_store_elem(obj, closure_ty, index, upvalue_val);
let uv = f.gc_load_elem(obj, closure_ty, index);
```

Key behaviors:
- `gc_alloc` **automatically emits a Safepoint** before the allocation. The safepoint lists all live GcPtr values in the current function. dynlang tracks these as it builds (every `gc_alloc` return value is a GcPtr, every `gc_load_field` of a Value field returns GcPtr).
- Field access computes offsets from the TypeInfo at build time — no runtime lookup.
- Store to a Value field may need a **write barrier** (for generational GC). dynlang emits this automatically.

## API: Tagging GC Pointers

The tag scheme and GC need to agree on how to identify pointers in values.

```rust
// The module already knows:
// - NanBoxTags { ptr: 0 } means tag 0 = heap pointer
// - GcConfig uses SemiSpace
//
// dynlang derives PtrPolicy automatically:
// - try_decode_ptr: check if NanBox tag == ptr tag, extract payload, shift
// - encode_ptr: shift, NanBox encode with ptr tag
//
// This is used by the GC when scanning values to find pointers.
```

No manual `PtrPolicy` implementation needed. The tag scheme IS the pointer policy.

## Runtime: How GC Actually Works

### Interpreter path

The `ModuleInterpreter` already has `InterpRootManager` support. When GC is enabled:

1. dynlang provides a `GcRootManager` struct that implements `InterpRootManager`
2. It wraps a `SemiSpace` (or `Heap`) allocator
3. When the interpreter hits a `Safepoint` instruction, it calls `roots.collect()` if needed
4. The root manager tracks all live GcPtr values across interpreter frames
5. After collection, all GcPtr values in all frames are updated (pointers may have moved)

```rust
// User code (in vm.rs):
let gc = DynGcRuntime::new(GcConfig::semi_space(1 << 20), &built.module);
let interp = ModuleInterpreter::<NanBox, _>::new(&built.module, &gc.root_manager());
// bind externs...
interp.run(entry, &[]);
// GC happens automatically at safepoints
```

### JIT path

The JIT already supports safepoints via `compile_with_gc`:
1. At each `Safepoint`, the JIT emits a call to the safepoint handler
2. The handler checks if GC is needed
3. If yes: scan roots from stack maps, collect, update pointers
4. The `GcPtr` values in registers/stack are updated via the stack map

dynlang's `with_gc` configuration feeds into the JIT compilation:
```rust
// Automatic — compile_with_gc is used when GC is configured
let jit = JitModule::compile_with_gc::<NanBox>(&module, &externs, gc.safepoint_handler());
```

## Runtime: Extern Allocation Function

`gc_alloc` in the IR calls an extern `__gc_alloc__(type_id: I64, varlen_len: I64) -> GcPtr`.

The extern implementation:
```rust
extern "C" fn gc_alloc(type_id: u64, varlen_len: u64) -> *mut u8 {
    with_gc(|gc| {
        let info = gc.type_info(type_id as usize);
        let ptr = gc.heap.alloc_obj::<Compact>(info, varlen_len as usize);
        if ptr.is_null() {
            // Allocation failed — trigger GC was already done at safepoint
            panic!("out of memory after GC");
        }
        ptr
    })
}
```

## What This Means for Lox

With GC integrated into dynlang, the Lox implementation changes from:

```rust
// BEFORE: manual Vec<HeapObj>, no GC, memory leaks
let closure_val = vm.heap.alloc(HeapObj::Closure { ... });  // Rust-side
```

To:

```rust
// AFTER: GC-managed objects, automatic collection
let obj = f.gc_alloc(closure_ty, num_upvalues);             // in IR
f.gc_store_field(obj, closure_ty, "func_ptr", func_idx);    // in IR
// GC traces and collects automatically
```

The key shift: object allocation moves from **Rust runtime code** (extern functions) into the **IR itself**. The IR knows about object layout, can emit efficient field access, and the GC can trace everything.

## What Changes in Each Crate

### dynlang (the main work)

Changes:
- `DynModule::new()` signature changes to require `GcConfig`
- `GcConfig` enum: `Leak` (bump allocator, no collection) or `SemiSpace { size }` (copying GC)
- `ObjTypeBuilder` → `ObjTypeId` with auto-generated TypeInfo
- `DynFunc::gc_alloc`, `gc_load_field`, `gc_store_field`, `gc_load_elem`, `gc_store_elem`
- Automatic GcPtr tracking (every alloc/load returns GcPtr, tracked for safepoint emission)
- Auto-derived PtrPolicy from NanBoxTags
- `DynGcRuntime` — runtime support struct for interpreter/JIT
- When `GcConfig::Leak`: safepoints are NOT emitted, alloc never fails
- When `GcConfig::SemiSpace`: safepoints emitted before alloc, collection happens automatically

### dynir (minimal changes)

Already has `Type::GcPtr`, `Safepoint`, `InterpRootManager`. May need:
- Helper to compute live GcPtrs at a given point (for automatic safepoint emission)
- Possibly nothing — dynlang handles this at build time

### dynobj (no changes)

TypeInfo, field access, scanning — all already correct.

### dynalloc (no changes)  

SemiSpace, BumpAllocator, root scanning — all already correct.

### dynlower (minimal changes)

Already has `compile_with_gc` and safepoint emission. May need:
- Ensure `GcPtr` values in registers are properly included in stack maps
- The `Safepoint` instruction already lists live values — lowerer already handles this

## Open Questions

1. **Write barriers**: For generational GC, stores to old-gen objects that point to new-gen need a write barrier. Should dynlang emit these automatically on `gc_store_field`, or require explicit barrier calls?

2. **Hash tables**: Lox needs hash tables for instance fields and class methods. Should we provide a GC-managed hash table type, or use sorted arrays of (key, value) pairs in varlen?

3. **Strings**: Should strings be GC-managed (varlen bytes in a GC object) or use Rust's `String` with a pinning/rooting mechanism?

4. **Large objects**: SemiSpace copying GC copies every live object. Large objects (big strings, big arrays) are expensive to copy. Should we have a separate large-object space?

5. **Finalization**: When a GC object is collected, does the language need a callback? (e.g., closing file handles)
