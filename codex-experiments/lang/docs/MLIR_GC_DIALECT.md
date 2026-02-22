# GC Dialect in MLIR via IRDL + PDL + Transform, Implemented with Melior

This document describes how to recast Lang's garbage collector infrastructure as an
MLIR dialect, defined dynamically with IRDL, lowered with PDL rewrite patterns, and
orchestrated with the Transform dialect — all driven from Rust using Melior.

## Table of Contents

1. [Why This Architecture](#why-this-architecture)
2. [The GC Dialect (IRDL)](#the-gc-dialect-irdl)
3. [Frontend Emission](#frontend-emission)
4. [Lowering with PDL](#lowering-with-pdl)
5. [Orchestration with Transform Dialect](#orchestration-with-transform-dialect)
6. [Melior Implementation](#melior-implementation)
7. [End-to-End Pipeline](#end-to-end-pipeline)
8. [What We Don't Need C++ For](#what-we-dont-need-c-for)
9. [Open Questions and Limitations](#open-questions-and-limitations)

---

## Why This Architecture

Lang's current codegen does a pre-scan of each function's AST to count GC roots, then
emits LLVM IR directly — frame allocation, root slot management, write barriers,
pollchecks, and allocation calls are all interleaved with regular codegen.

The key insight is that **the frontend already knows everything**: which locals are GC
refs, how many root slots each function needs, where safepoints go. No MLIR-side
analysis pass is required. The frontend can emit fully-explicit GC operations, and then
everything downstream is mechanical 1:1 lowering.

This means:
- **IRDL** defines the dialect shape (types + ops + constraints) — no C++, no TableGen
- **PDL** defines the lowering rewrites (each GC op → LLVM dialect ops) — declarative
- **Transform dialect** sequences the PDL patterns into a pipeline — also declarative
- **Melior** (Rust) drives all of it: loads the IRDL definitions, builds IR using
  `OperationBuilder`, runs the transform pipeline, then hands off to LLVM

Zero C++ in the GC pipeline.

---

## The GC Dialect (IRDL)

The dialect is defined as plain `.mlir` text, loaded at runtime.

### Types

```mlir
irdl.dialect @gc {

  // A GC-managed pointer. Parameterized by a type_id (integer attribute)
  // so we can track what kind of object it points to.
  // In practice all GC pointers are opaque i8* at the LLVM level,
  // but the type parameter lets the frontend distinguish them.
  irdl.type @ref {
    %0 = irdl.any
    irdl.parameters(%0)
  }

  // Thread handle — opaque, passed as first arg to every function.
  irdl.type @thread {}
```

### Operations

```mlir
  // --- Allocation ---

  // gc.alloc(%total_fields, %ptr_fields, %type_id) -> !gc.ref<T>
  //
  // Allocates a GC-managed object.
  //   total_fields: number of 8-byte slots
  //   ptr_fields:   how many of those are GC pointers (must come first in layout)
  //   type_id:      runtime type identifier
  irdl.operation @alloc {
    %i64 = irdl.is i64
    %result = irdl.any
    %gc_ref = irdl.parametric @gc::@ref<%result>
    irdl.operands(%i64, %i64, %i64)
    irdl.results(%gc_ref)
  }

  // --- Write Barrier ---

  // gc.write_barrier(%thread, %obj, %val)
  //
  // Must be called before storing a GC ref into an object field.
  // Notifies the generational collector of old→young pointer creation.
  irdl.operation @write_barrier {
    %thread = irdl.parametric @gc::@thread<>
    %obj = irdl.any
    %val = irdl.any
    irdl.operands(%thread, %obj, %val)
    irdl.results()
  }

  // --- Frame Management ---

  // gc.frame_push(%thread, %num_roots) -> !gc.frame
  //
  // Allocates a stack frame with N root slots, links it into the thread's
  // frame chain, and zeroes the root array.
  irdl.operation @frame_push {
    %thread = irdl.parametric @gc::@thread<>
    %i64 = irdl.is i64
    irdl.operands(%thread, %i64)
    irdl.results()
    // The frame pointer and root base are implicitly available;
    // they're materialized during lowering as stack allocations.
    // Alternatively, we could return them as results.
  }

  // gc.frame_pop(%thread)
  //
  // Restores the parent frame pointer in the thread struct.
  irdl.operation @frame_pop {
    %thread = irdl.parametric @gc::@thread<>
    irdl.operands(%thread)
    irdl.results()
  }

  // --- Root Slot Access ---

  // gc.root_store(%thread, %value, {slot = N})
  //
  // Stores a GC reference into root slot N of the current frame.
  // Volatile store — optimizer must not cache/reorder across safepoints.
  irdl.operation @root_store {
    %thread = irdl.parametric @gc::@thread<>
    %val = irdl.any
    irdl.operands(%thread, %val)
    irdl.results()
    // slot index is an i64 attribute, not an operand
  }

  // gc.root_load(%thread, {slot = N}) -> !gc.ref<T>
  //
  // Loads a GC reference from root slot N. Volatile load.
  irdl.operation @root_load {
    %thread = irdl.parametric @gc::@thread<>
    %result = irdl.any
    %gc_ref = irdl.parametric @gc::@ref<%result>
    irdl.operands(%thread)
    irdl.results(%gc_ref)
  }

  // --- Safepoint ---

  // gc.safepoint(%thread)
  //
  // Polls the GC state flag. If collection is requested, calls into
  // the slow path (gc_pollcheck_slow).
  irdl.operation @safepoint {
    %thread = irdl.parametric @gc::@thread<>
    irdl.operands(%thread)
    irdl.results()
  }

  // --- Field Access ---

  // gc.field_ptr(%obj, {index = N}) -> ptr
  //
  // Returns a pointer to field N of a GC object.
  // Byte offset = 8 (header) + N * 8.
  irdl.operation @field_ptr {
    %obj = irdl.any
    %ptr = irdl.base "!llvm.ptr"
    irdl.operands(%obj)
    irdl.results(%ptr)
  }
}
```

### Full IRDL Definition File

Save this as `gc_dialect.mlir` and load it at runtime via Melior's
`load_irdl_dialects`.

---

## Frontend Emission

The `.lang` frontend already does the hard work:
1. Pre-scans each function AST to count GC-ref locals (`count_roots_in_fn`)
2. Assigns root slot indices incrementally as it encounters let-bindings
3. Knows which types are GC refs (`is_gc_ref_type`)

Instead of emitting LLVM IR directly, it emits GC dialect ops:

### Example: Before (current LLVM IR emission)

```
define void @my_fn(i8* %thread) {
entry:
  %frame = alloca {i8*, i8*, [3 x i8*]}          ; 3 root slots
  %frame.parent = getelementptr ... %frame, 0, 0
  %old_top = load i8*, i8** %thread.top
  store i8* %old_top, i8** %frame.parent
  store i8* @__frame_origin_my_fn, i8** %frame.origin
  call void @llvm.memset(... roots ..., 0, 24)
  store i8* %frame, i8** %thread.top

  ; ... body ...

  %parent = load i8*, i8** %frame.parent
  store i8* %parent, i8** %thread.top
  ret void
}
```

### Example: After (GC dialect ops)

```mlir
func.func @my_fn(%thread: !gc.thread) {
  gc.frame_push %thread, 3              // 3 root slots

  %x = gc.alloc 4, 2, 7 : !gc.ref<MyStruct>
  gc.root_store %thread, %x {slot = 0}
  gc.safepoint %thread

  %y = gc.alloc 2, 1, 3 : !gc.ref<Pair>
  gc.root_store %thread, %y {slot = 1}
  gc.write_barrier %thread, %x, %y      // storing y into x's field
  gc.safepoint %thread

  %loaded = gc.root_load %thread {slot = 0} : !gc.ref<MyStruct>

  gc.frame_pop %thread
  return
}
```

The IR is **fully explicit**. No analysis needed downstream.

---

## Lowering with PDL

Each GC op lowers to a sequence of LLVM dialect ops. These are all local,
pattern-based rewrites — PDL's sweet spot.

### gc.alloc → llvm.call @gc_alloc

```mlir
pdl.pattern @lower_gc_alloc : benefit(1) {
  %total   = pdl.operand : i64
  %ptr_f   = pdl.operand : i64
  %type_id = pdl.operand : i64
  %res_ty  = pdl.type

  %op = pdl.operation "gc.alloc"(%total, %ptr_f, %type_id : !pdl.value, !pdl.value, !pdl.value)
    -> (%res_ty : !pdl.type)

  pdl.rewrite %op {
    // Emit: %result = llvm.call @gc_alloc(%total, %ptr_f, %type_id) : (i64, i64, i64) -> !llvm.ptr
    %call = pdl.apply_native_rewrite "emit_gc_alloc_call"(%total, %ptr_f, %type_id : !pdl.value, !pdl.value, !pdl.value)
      : !pdl.value
    pdl.replace %op with(%call : !pdl.value)
  }
}
```

### gc.write_barrier → llvm.call @gc_write_barrier

```mlir
pdl.pattern @lower_gc_write_barrier : benefit(1) {
  %thread = pdl.operand
  %obj    = pdl.operand
  %val    = pdl.operand

  %op = pdl.operation "gc.write_barrier"(%thread, %obj, %val : !pdl.value, !pdl.value, !pdl.value)

  pdl.rewrite %op {
    %call = pdl.apply_native_rewrite "emit_write_barrier_call"(%thread, %obj, %val
      : !pdl.value, !pdl.value, !pdl.value) : !pdl.operation
    pdl.erase %op
  }
}
```

### gc.safepoint → pollcheck sequence

This one is more involved — it lowers to a conditional branch:

```mlir
pdl.pattern @lower_gc_safepoint : benefit(1) {
  %thread = pdl.operand

  %op = pdl.operation "gc.safepoint"(%thread : !pdl.value)

  pdl.rewrite %op {
    // Lower to:
    //   %state_ptr = llvm.getelementptr %thread[0, 1] : !llvm.ptr  (offset 8 = gc_state)
    //   %state = llvm.load %state_ptr : i32
    //   %needs_gc = llvm.icmp "ne" %state, 0
    //   llvm.cond_br %needs_gc, ^slow, ^cont
    // ^slow:
    //   llvm.call @gc_pollcheck_slow(%thread, %frame_origin)
    //   llvm.br ^cont
    // ^cont:
    //   ...
    pdl.apply_native_rewrite "emit_pollcheck_sequence"(%thread : !pdl.value) : !pdl.operation
    pdl.erase %op
  }
}
```

### gc.frame_push → alloca + memset + thread linkage

```mlir
pdl.pattern @lower_gc_frame_push : benefit(1) {
  %thread    = pdl.operand
  %num_roots = pdl.operand

  %op = pdl.operation "gc.frame_push"(%thread, %num_roots : !pdl.value, !pdl.value)

  pdl.rewrite %op {
    // Lower to:
    //   %frame = llvm.alloca {ptr, ptr, [N x ptr]}
    //   %parent_slot = llvm.getelementptr %frame[0, 0]
    //   %origin_slot = llvm.getelementptr %frame[0, 1]
    //   %old_top = llvm.load %thread.top_frame
    //   llvm.store %old_top, %parent_slot
    //   llvm.store @__frame_origin_<fn>, %origin_slot
    //   if N > 0:
    //     %roots = llvm.getelementptr %frame[0, 2]
    //     llvm.intr.memset %roots, 0, (N * 8)
    //   llvm.store %frame, %thread.top_frame
    pdl.apply_native_rewrite "emit_frame_push"(%thread, %num_roots
      : !pdl.value, !pdl.value) : !pdl.operation
    pdl.erase %op
  }
}
```

### gc.frame_pop → restore parent frame

```mlir
pdl.pattern @lower_gc_frame_pop : benefit(1) {
  %thread = pdl.operand

  %op = pdl.operation "gc.frame_pop"(%thread : !pdl.value)

  pdl.rewrite %op {
    // Lower to:
    //   %parent = llvm.load %frame.parent
    //   llvm.store %parent, %thread.top_frame
    pdl.apply_native_rewrite "emit_frame_pop"(%thread : !pdl.value) : !pdl.operation
    pdl.erase %op
  }
}
```

### gc.root_store / gc.root_load → volatile GEP + load/store

```mlir
pdl.pattern @lower_gc_root_store : benefit(1) {
  %thread = pdl.operand
  %val    = pdl.operand

  %op = pdl.operation "gc.root_store"(%thread, %val : !pdl.value, !pdl.value)

  pdl.rewrite %op {
    // Lower to:
    //   %slot = llvm.getelementptr %root_base[0, <slot_index>]
    //   llvm.store volatile %val, %slot
    pdl.apply_native_rewrite "emit_root_store"(%thread, %val : !pdl.value, !pdl.value)
      : !pdl.operation
    pdl.erase %op
  }
}

pdl.pattern @lower_gc_root_load : benefit(1) {
  %thread = pdl.operand
  %res_ty = pdl.type

  %op = pdl.operation "gc.root_load"(%thread : !pdl.value) -> (%res_ty : !pdl.type)

  pdl.rewrite %op {
    // Lower to:
    //   %slot = llvm.getelementptr %root_base[0, <slot_index>]
    //   %val = llvm.load volatile %slot
    %val = pdl.apply_native_rewrite "emit_root_load"(%thread : !pdl.value)
      : !pdl.value
    pdl.replace %op with(%val : !pdl.value)
  }
}
```

### Note on `apply_native_rewrite`

The PDL patterns above use `apply_native_rewrite` for multi-instruction sequences
that are awkward to express as pure PDL (e.g., creating new basic blocks for the
safepoint conditional). These native rewrites are registered in Melior as Rust
functions — not C++. More on this in the Melior section.

For simpler patterns (gc.alloc, gc.write_barrier), you could also construct the
replacement ops inline in PDL using `pdl.operation "llvm.call"(...)`.

---

## Orchestration with Transform Dialect

The Transform dialect sequences the PDL patterns:

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @lower_gc(%module: !transform.any_op {transform.readonly}) {

    // Step 1: Lower frame management first (frame_push creates
    // the alloca that root_store/root_load need to reference)
    transform.apply_patterns to %module {
      transform.apply_patterns.pdl @lower_gc_frame_push
    } : !transform.any_op

    // Step 2: Lower root access (depends on frame being materialized)
    transform.apply_patterns to %module {
      transform.apply_patterns.pdl @lower_gc_root_store
      transform.apply_patterns.pdl @lower_gc_root_load
    } : !transform.any_op

    // Step 3: Lower allocations and write barriers
    transform.apply_patterns to %module {
      transform.apply_patterns.pdl @lower_gc_alloc
      transform.apply_patterns.pdl @lower_gc_write_barrier
    } : !transform.any_op

    // Step 4: Lower safepoints (can reference frame_origin from step 1)
    transform.apply_patterns to %module {
      transform.apply_patterns.pdl @lower_gc_safepoint
    } : !transform.any_op

    // Step 5: Lower frame pop
    transform.apply_patterns to %module {
      transform.apply_patterns.pdl @lower_gc_frame_pop
    } : !transform.any_op

    // Step 6: Standard LLVM lowering for everything else
    transform.apply_registered_pass "convert-func-to-llvm" to %module : (!transform.any_op) -> !transform.any_op
    transform.apply_registered_pass "convert-arith-to-llvm" to %module : (!transform.any_op) -> !transform.any_op
    transform.apply_registered_pass "reconcile-unrealized-casts" to %module : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
```

---

## Melior Implementation

### Project Setup

```toml
# Cargo.toml
[dependencies]
melior = "0.26"
mlir-sys = "210.0"
```

Requires LLVM/MLIR 21 installed. Set `MLIR_SYS_210_PREFIX` if needed.

### Step 1: Context and Dialect Loading

```rust
use melior::{
    Context,
    dialect::DialectRegistry,
    ir::{Module, Location},
    utility::{register_all_dialects, load_irdl_dialects},
};

fn create_context() -> Context {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    // Load our GC dialect from IRDL definition
    let gc_dialect_mlir = include_str!("gc_dialect.mlir");
    let irdl_module = Module::parse(&context, gc_dialect_mlir)
        .expect("failed to parse GC dialect IRDL");
    load_irdl_dialects(&irdl_module);

    context
}
```

### Step 2: Building GC IR from the Frontend

```rust
use melior::ir::{
    Block, Region, Operation, Value, Type,
    attribute::{IntegerAttribute, StringAttribute, TypeAttribute},
    r#type::{FunctionType, IntegerType},
    operation::OperationBuilder,
};

/// Emit gc.frame_push for a function with `num_roots` GC-ref locals.
fn emit_gc_frame_push<'c>(
    context: &'c Context,
    block: &'c Block,
    thread: Value<'c, '_>,
    num_roots: i64,
    location: Location<'c>,
) {
    let i64_ty = IntegerType::new(context, 64).into();
    let num_roots_val = block.append_operation(
        melior::dialect::arith::constant(
            context,
            IntegerAttribute::new(i64_ty, num_roots).into(),
            location,
        )
    );

    block.append_operation(
        OperationBuilder::new("gc.frame_push", location)
            .add_operands(&[thread, num_roots_val.result(0).unwrap().into()])
            .build()
            .unwrap()
    );
}

/// Emit gc.alloc for a struct/enum with known layout.
fn emit_gc_alloc<'c>(
    context: &'c Context,
    block: &'c Block,
    total_fields: i64,
    ptr_fields: i64,
    type_id: i64,
    result_type: Type<'c>,
    location: Location<'c>,
) -> Value<'c, '_> {
    let i64_ty = IntegerType::new(context, 64).into();

    let total = block.append_operation(melior::dialect::arith::constant(
        context,
        IntegerAttribute::new(i64_ty, total_fields).into(),
        location,
    ));
    let ptrs = block.append_operation(melior::dialect::arith::constant(
        context,
        IntegerAttribute::new(i64_ty, ptr_fields).into(),
        location,
    ));
    let tid = block.append_operation(melior::dialect::arith::constant(
        context,
        IntegerAttribute::new(i64_ty, type_id).into(),
        location,
    ));

    let alloc = block.append_operation(
        OperationBuilder::new("gc.alloc", location)
            .add_operands(&[
                total.result(0).unwrap().into(),
                ptrs.result(0).unwrap().into(),
                tid.result(0).unwrap().into(),
            ])
            .add_results(&[result_type])
            .build()
            .unwrap()
    );

    alloc.result(0).unwrap().into()
}

/// Emit gc.root_store to pin a value in a root slot.
fn emit_gc_root_store<'c>(
    context: &'c Context,
    block: &'c Block,
    thread: Value<'c, '_>,
    value: Value<'c, '_>,
    slot: i64,
    location: Location<'c>,
) {
    let i64_ty = IntegerType::new(context, 64).into();
    block.append_operation(
        OperationBuilder::new("gc.root_store", location)
            .add_operands(&[thread, value])
            .add_attributes(&[(
                melior::ir::Identifier::new(context, "slot"),
                IntegerAttribute::new(i64_ty, slot).into(),
            )])
            .build()
            .unwrap()
    );
}

/// Emit gc.root_load to retrieve a value from a root slot.
fn emit_gc_root_load<'c>(
    context: &'c Context,
    block: &'c Block,
    thread: Value<'c, '_>,
    slot: i64,
    result_type: Type<'c>,
    location: Location<'c>,
) -> Value<'c, '_> {
    let i64_ty = IntegerType::new(context, 64).into();
    let load = block.append_operation(
        OperationBuilder::new("gc.root_load", location)
            .add_operands(&[thread])
            .add_results(&[result_type])
            .add_attributes(&[(
                melior::ir::Identifier::new(context, "slot"),
                IntegerAttribute::new(i64_ty, slot).into(),
            )])
            .build()
            .unwrap()
    );
    load.result(0).unwrap().into()
}

/// Emit gc.write_barrier before a GC-ref field store.
fn emit_gc_write_barrier<'c>(
    block: &'c Block,
    thread: Value<'c, '_>,
    obj: Value<'c, '_>,
    val: Value<'c, '_>,
    location: Location<'c>,
) {
    block.append_operation(
        OperationBuilder::new("gc.write_barrier", location)
            .add_operands(&[thread, obj, val])
            .build()
            .unwrap()
    );
}

/// Emit gc.safepoint (pollcheck).
fn emit_gc_safepoint<'c>(
    block: &'c Block,
    thread: Value<'c, '_>,
    location: Location<'c>,
) {
    block.append_operation(
        OperationBuilder::new("gc.safepoint", location)
            .add_operands(&[thread])
            .build()
            .unwrap()
    );
}

/// Emit gc.frame_pop before function return.
fn emit_gc_frame_pop<'c>(
    block: &'c Block,
    thread: Value<'c, '_>,
    location: Location<'c>,
) {
    block.append_operation(
        OperationBuilder::new("gc.frame_pop", location)
            .add_operands(&[thread])
            .build()
            .unwrap()
    );
}
```

### Step 3: Lowering GC Ops — ExternalPass Approach

PDL's `apply_native_rewrite` requires registering callbacks through the C API, which
Melior wraps with `ExternalPass`. For the multi-instruction lowerings (safepoint,
frame_push), an `ExternalPass` that walks the IR is the most practical approach:

```rust
use melior::pass::external::{create_external, ExternalPass, RunExternalPass};
use melior::ir::r#type::TypeId;
use std::sync::OnceLock;

static LOWER_GC_PASS_ID: OnceLock<()> = OnceLock::new();

#[derive(Clone)]
struct LowerGcOps;

impl<'c> RunExternalPass<'c> for LowerGcOps {
    fn initialize(&mut self, _context: ContextRef<'c>) {}

    fn run(&mut self, op: OperationRef<'c, '_>, pass: ExternalPass<'_>) {
        // Walk all operations in the module.
        // For each gc.* operation, replace it with the equivalent
        // sequence of llvm.* operations.
        //
        // The walk order matters:
        //   1. gc.frame_push  → alloca + GEP + memset + store
        //   2. gc.root_store  → GEP into frame roots + volatile store
        //   3. gc.root_load   → GEP into frame roots + volatile load
        //   4. gc.alloc       → llvm.call @gc_alloc
        //   5. gc.write_barrier → llvm.call @gc_write_barrier
        //   6. gc.safepoint   → load state + cond_br + call slow path
        //   7. gc.frame_pop   → load parent + store to thread
        //
        // Each lowering produces LLVM dialect ops that further passes
        // will handle.

        walk_and_lower(op);
    }
}

fn lower_gc_pass(context: &Context) -> melior::pass::Pass {
    create_external(
        LowerGcOps,
        TypeId::create(&LOWER_GC_PASS_ID),
        "lower-gc-ops",
        "lower-gc",
        "Lower gc.* operations to LLVM dialect",
        "builtin.module",
        &[],  // dependent dialects: llvm
    )
}
```

### Step 4: Full Pipeline

```rust
use melior::pass::{self, PassManager};

fn run_pipeline(context: &Context, module: &mut Module) {
    let pm = PassManager::new(context);
    pm.enable_verifier(true);

    // Phase 1: Lower GC ops to LLVM dialect
    pm.add_pass(lower_gc_pass(context));

    // Phase 2: Lower standard dialects to LLVM
    pm.add_pass(pass::conversion::create_func_to_llvm());

    let func_pm = pm.nested_under("func.func");
    func_pm.add_pass(pass::conversion::create_arith_to_llvm());
    func_pm.add_pass(pass::conversion::create_index_to_llvm());

    pm.add_pass(pass::conversion::create_scf_to_control_flow());
    pm.add_pass(pass::conversion::create_control_flow_to_llvm());
    pm.add_pass(pass::conversion::create_reconcile_unrealized_casts());

    // Phase 3: Cleanup
    pm.add_pass(pass::transform::create_canonicalizer());
    pm.add_pass(pass::transform::create_cse());

    pm.run(module).expect("pipeline failed");
}
```

### Step 5: AOT Compilation or JIT

```rust
use melior::ExecutionEngine;

// --- JIT path ---
fn jit_execute(module: &Module) {
    let engine = ExecutionEngine::new(module, 2, &[], false);

    // Register GC runtime symbols
    unsafe {
        engine.register_symbol("gc_alloc", gc_alloc as *mut ());
        engine.register_symbol("gc_write_barrier", gc_write_barrier as *mut ());
        engine.register_symbol("gc_pollcheck_slow", gc_pollcheck_slow as *mut ());
    }

    unsafe {
        engine.invoke_packed("main", &mut []).unwrap();
    }
}

// --- AOT path ---
fn aot_compile(module: &Module, output: &str) {
    let engine = ExecutionEngine::new(module, 2, &[], true);
    engine.dump_to_object_file(output);
    // Then link with: gcc output.o -o binary -lgc_bridge -lruntime ...
}
```

---

## End-to-End Pipeline

```
                    ┌─────────────┐
                    │  .lang src  │
                    └──────┬──────┘
                           │
                    lex → parse → qualify → resolve → typecheck
                           │
                           ▼
                 ┌─────────────────┐
                 │  GC Dialect IR  │  ← frontend emits explicit gc.* ops
                 │  (func + arith  │     count_roots already done
                 │   + gc.*)       │     slot indices already assigned
                 └────────┬────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
         IRDL verify   PDL lower   Transform
         (type check   (gc.* →     (sequence
          gc ops)      llvm.*)     the passes)
              │           │           │
              └───────────┼───────────┘
                          ▼
                 ┌─────────────────┐
                 │  LLVM Dialect   │  ← all gc.* ops are gone
                 │  (pure llvm.*)  │
                 └────────┬────────┘
                          │
                    LLVM backend
                          │
                    ┌─────┴─────┐
                    ▼           ▼
                  JIT         .o file
              (register      (link with
              symbols)       gc_bridge.o)
```

---

## What We Don't Need C++ For

| Concern | How it's handled | C++ needed? |
|---------|------------------|-------------|
| Dialect definition | IRDL `.mlir` file | No |
| Type verification | IRDL constraints | No |
| Lowering gc.alloc | PDL pattern or ExternalPass (Rust) | No |
| Lowering gc.write_barrier | PDL pattern or ExternalPass (Rust) | No |
| Lowering gc.safepoint | ExternalPass (Rust) | No |
| Lowering gc.frame_push/pop | ExternalPass (Rust) | No |
| Lowering gc.root_store/load | ExternalPass (Rust) | No |
| Pipeline orchestration | Transform dialect or PassManager | No |
| Root counting | Frontend (already exists) | No |
| Slot index assignment | Frontend (already exists) | No |
| Safepoint placement | Frontend (already exists) | No |
| LLVM codegen | Melior → LLVM backend | No |

---

## Open Questions and Limitations

### IRDL Limitations

- **No interfaces/traits**: Can't mark `gc.alloc` as having memory side effects.
  The LLVM optimizer might try to DCE or reorder it. Workaround: mark all GC ops
  with an `"memory"` attribute and use the ExternalPass to ensure they have proper
  LLVM-level side effects after lowering.

- **No custom assembly format**: GC ops will print in generic MLIR syntax
  (`"gc.alloc"(...) : ...`) rather than a nice custom syntax. Cosmetic only.

- **No parameter introspection**: Can't programmatically walk type parameters of
  `!gc.ref<T>` from IRDL. The lowering pass treats all `!gc.ref<T>` as `!llvm.ptr`.

### Melior Limitations

- **Alpha API**: Melior is 0.26.x and subject to breaking changes.
- **LLVM IR translation**: Not wrapped at the Melior level. Need to drop to
  `mlir_sys::mlirTranslateModuleToLLVMIR` for direct LLVM IR output.
- **PDL API rough edges**: Issue #260 notes the PDL API needs improvement.
  Building PDL patterns programmatically via `OperationBuilder` works but is
  verbose. Parsing them from `.mlir` text is cleaner.

### Design Choices

- **gc.frame_push as a single op vs explicit alloca**: We chose a single high-level
  op that the lowering pass expands. Alternative: emit the alloca/GEP/memset directly
  in the frontend and only use GC ops for alloc/barrier/safepoint.

- **Root slot index as attribute vs operand**: Using an attribute (`{slot = N}`)
  is cleaner since the index is always a compile-time constant. Using an operand
  would work too but adds unnecessary SSA values.

- **Thread as explicit operand everywhere**: Matches the current ABI where thread
  is the implicit first parameter. Could alternatively use a thread-local or module
  attribute, but the explicit operand makes the data flow visible in IR.

### PDL vs ExternalPass Trade-off

For simple 1:1 rewrites (gc.alloc → llvm.call), PDL patterns parsed from `.mlir`
text are the cleanest approach. For multi-instruction lowerings that create new
basic blocks (gc.safepoint's conditional branch), a Rust `ExternalPass` is more
practical. You can mix both — use PDL for the easy cases and ExternalPass for the
rest, running them in sequence via the PassManager.

### Alternative: Pure ExternalPass

If PDL proves too fiddly, a single `ExternalPass` in Rust that walks all ops and
lowers every `gc.*` operation is perfectly viable. You lose the declarative
elegance but gain full control. The IRDL dialect definition is still valuable
regardless — it gives you type verification for free during IR construction.
