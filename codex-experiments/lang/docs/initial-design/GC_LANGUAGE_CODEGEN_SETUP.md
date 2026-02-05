# GC Language Codegen Setup (`gc-experiment`)

This document describes how `../../claude-experiments/gc-experiment` wires code generation, runtime stack maps, and GC integration so the language can allocate and collect managed objects.

## 1. High-level architecture

The project is a small language pipeline:

1. Build a typed AST in Rust (`src/ast/*`, currently a binary-trees benchmark in `src/main.rs`).
2. Compile AST to LLVM IR with Inkwell (`src/codegen/*`).
3. JIT the module and execute with a runtime `Thread` pointer passed as arg0 to every function.
4. Route allocations and safepoints into `gc-library` (`GenerationalGC`) through FFI shims in `src/main.rs`.

Core idea: generated code cooperates with GC by maintaining an explicit linked list of stack frames, each with root slots for GC references.

## 2. Value and object model

### GC references in the language

`Type::Struct(_)` and `Type::Array(_)` are GC-managed refs (`Type::is_gc_ref()`), while ints/bools are primitives.

### Object layout

Codegen and runtime agree on a 16-byte object header:

- `meta: ptr` (type metadata pointer)
- `gc_flags: u32` (mark/forwarding/opaque/size bits)
- `aux: u32` (array length or extra info)

Payload starts at offset 16.

This exact shape is represented in both:

- LLVM side: `LLVMTypes::object_header_ty` (`src/codegen/types.rs`)
- Runtime side: `ObjectHeader` / `OurHeader` (`src/runtime/object.rs`, `src/main.rs`)

## 3. Generated GC frame model (stack roots)

Every compiled function gets a stack frame object with layout:

- `parent: *mut Frame`
- `origin: *const FrameOrigin`
- `roots: [ptr; N]`

Where `N` is computed at compile time per function by counting GC-typed params/locals.

### What codegen emits

In `FunctionCompiler` (`src/codegen/compiler.rs`):

1. `create_frame_origin()` emits a per-function global constant `{ num_roots, function_name }`.
2. `emit_prologue()`:
   - allocates frame on stack (`alloca`)
   - memset zeroes root slots
   - links frame into `thread.top_frame` (escape point)
   - stores `origin`
3. `emit_epilogue()` pops frame: `thread.top_frame = frame.parent`.

GC refs are stored in root slots (`VarStorage::RootSlot`), primitives in allocas (`VarStorage::Alloca`).

Loads/stores for root slots are marked volatile so LLVM does not move/eliminate them across safepoints.

## 4. Safepoints and collection handshake

### Fast path check in generated code

`emit_pollcheck()` emits:

- load `thread.state`
- if non-zero -> call `gc_pollcheck_slow(thread, frame_origin)`

Safepoints are inserted:

- after allocations (`NewStruct`, `NewArray`)
- after function calls (`Expr::Call`)
- on while-loop backedges

### Slow path runtime function

`gc_pollcheck_slow` (in `src/main.rs`):

- clears thread flag
- builds a root provider over the frame chain (`FrameChainRoots`)
- runs `gc.gc(&roots)` on `GenerationalGC`

`FrameChainRoots` walks `thread.frames()` and yields actual slot addresses so moving GC can update pointers in place.

## 5. Allocation path

Generated code calls runtime FFI declarations:

- `gc_allocate(thread, meta, payload_size)`
- `gc_allocate_array(thread, meta, length)`

These are declared in `RuntimeFunctions` (`src/codegen/types.rs`) and mapped to Rust symbols before JIT execution.

In runtime (`src/main.rs`):

- `gc_allocate` uses `gc.try_allocate(...)`
- on `AllocateAction::Allocated(ptr)`: initializes header + zeroes payload fields
- on `AllocateAction::Gc`: runs GC with current frame-chain roots, then retries

So allocation is tightly integrated with the collector and always root-safe.

## 6. Type metadata for scanning

For each struct, codegen creates LLVM globals containing `ObjectMeta`:

- object kind (struct/array)
- number of GC pointer fields
- offsets of pointer fields
- type name

This metadata is emitted in `LLVMTypes::create_struct_meta()` and shared by allocation/runtime scanning logic.

Array metadata is emitted once (`__meta_array`).

To prevent optimizer from deleting metadata globals used indirectly by runtime/GC, compiler puts frame origins + metas into `llvm.compiler.used` during optimization passes.

## 7. Write barrier strategy in this setup

The runtime has a `gc_write_barrier` entrypoint and exposes `YOUNG_GEN_START`/`YOUNG_GEN_END` globals for inline checks, but current codegen intentionally skips barriers for construction stores.

Reason documented in code: construction order is arranged so child expressions evaluate before parent allocation, then field stores happen immediately after allocation (young object construction case). Mutation barriers are left as future work.

## 8. JIT integration details

`main()` does:

1. Build AST program.
2. Compile LLVM module.
3. Optionally run LLVM pass pipeline (`TEST_PASS`).
4. Remove `llvm.compiler.used` before creating execution engine (JIT quirk).
5. Create JIT engine (Aggressive opt level).
6. Map runtime symbols/globals:
   - `gc_pollcheck_slow`, `gc_allocate`, `gc_allocate_array`, `gc_write_barrier`, `print_int`
   - `YOUNG_GEN_START`, `YOUNG_GEN_END`
7. Call JIT `main(thread*)`.

## 9. Why this enables a GC language

This setup gives the language the three requirements for safe GC:

1. Precise roots at safepoints: explicit frame chain + typed root slots.
2. Object shape knowledge: emitted per-type metadata and consistent object headers.
3. Runtime cooperation: allocation and pollcheck slow paths call into collector with live roots.

Together, codegen-generated stack maps (frames), runtime metadata, and GC callbacks make managed heap references viable in compiled/JITed code.
