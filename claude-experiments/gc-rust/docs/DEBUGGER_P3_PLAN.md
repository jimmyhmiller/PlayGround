# Debugger P3 — DWARF locals + reflection pretty-printers

> **STATUS: FIRST INCREMENT DONE + GREEN** — scalar locals/params inspectable.
> `gcr build --debug` produces an unoptimized executable carrying DWARF
> local-variable + formal-parameter DIEs, and **lldb's `frame variable` shows
> gc-rust locals by their source names with correct values**: breaking at
> `prog.gcr:4` in `add(7, 35)` prints `x = 7`, `y = 35`, `sum = 42`,
> `doubled = 84`. All scalar reprs decode (`i8..i64`, `u8..u64`, `f32/f64`,
> `bool`, `char`). Verified: dwarf 3/0 (new `full_debug_emits_named_local_and_param_dies`),
> debugger_lldb 1/0 (new end-to-end lldb `frame variable` value check), lib
> 158/0, aot/ffi/threads/reflect/modules/generational/heap_* all green, default
> (non-`--debug`) builds unchanged. **NEXT:** non-scalar locals (structs/enums)
> via reflection-paired lldb pretty-printers (`Point { x: 3, y: 4 }`), real
> subroutine types, the §5 moving-GC re-inspect check.

Builds on P2 (DWARF line tables, `DWARFEmissionKind::LineTablesOnly` + per-source
DIFiles). Goal: `gcr build --debug` carries **full** DWARF — local/parameter
variable DIEs pointing at frame slots — so LLDB (and any DAP IDE via `lldb-dap`)
inspects gc-rust values, the rung where "≥ JVM inspection" is realized.

## What shipped (first increment)

1. **Source names threaded to Core.** `CoreFn.local_names: Vec<Option<String>>`
   (parallel to `locals`), populated in lowering by `FnLowerer::bind_mut` (the
   first binding of a slot names it; shadows get fresh slots, so a name is never
   clobbered). Carried for ordinary fns, closures (captures named too), and the
   extern path (empty). Compiler temps stay `None`.

2. **`DebugLevel` { None, LineTables, Full }.** Replaces the old
   `codegen_with_debug(_, _, bool)`. `None` = JIT / `emit llvm`; `LineTables` =
   default `gcr build` (P2, kept under O2); `Full` = `gcr build --debug`. Wired
   through `codegen_with_debug` → `codegen_aot_object_level` →
   `build_executable_level`; `gcr build --debug` selects `Full`. The 2-arg
   `codegen_aot_object` / 3-arg `build_executable` wrappers are kept (tests +
   existing callers unchanged).

3. **Unoptimized full-debug codegen.** `Full` skips the IR `optimize_module`
   **and** builds the backend `TargetMachine` at `OptimizationLevel::None`
   (`host_target_machine_opt`). Both matter: an optimizing backend promotes the
   (still-present) allocas into registers and the `dbg.declare` frame-slot
   locations go stale — the bug that first showed up as garbage `frame variable`
   values. `LineTables` keeps the full O2 pipeline (debug locations survive it,
   so stepping is unaffected).

4. **Variable DIEs.** In `define_fn`, after params are stored, for each named
   scalar local: `create_parameter_variable` (params, 1-based `arg_no`) or
   `create_auto_variable` (other locals), typed via `di_type_for_repr`
   (`create_basic_type` with the right `DW_ATE_*` encoding), with a `dbg.declare`
   record pointing at the local's alloca slot. Non-scalar locals (Ref/Value) are
   skipped — they await the reflection pretty-printers.

## The LLVM 21 `insert_declare` landmine (and the fix)

inkwell's `DebugInfoBuilder::insert_declare_at_end`, under the `llvm21-1`
feature, calls the **record** C API (`LLVMDIBuilderInsertDeclareRecordAtEnd`,
returns an `LLVMDbgRecordRef`) but then casts that to `LLVMValueRef` and builds
an `InstructionValue`, whose `::new` asserts `is_instruction()`. A `DbgRecord` is
not a value, so this is UB — it panicked **nondeterministically** (~1 build in 5)
at `instruction_value.rs:205`. Setting the module's debug-info format flag does
NOT help: inkwell always calls the record API in LLVM 21.

Fix: bypass the wrapper. Call
`inkwell::llvm_sys::debuginfo::LLVMDIBuilderInsertDeclareRecordAtEnd` directly
(via `DebugInfoBuilder::as_mut_ptr`, the metadata types' `as_mut_ptr`,
`PointerValue::as_value_ref`, `BasicBlock::as_mut_ptr`) and **discard the
return** — we only need the side effect (the record attached to the entry block).
The record API requires the *new* format, which is LLVM 21's default, so no flag
change is needed. No edit to the shared inkwell fork.

## Scope boundary / open work

- **Scalars only.** Struct/enum (Ref/Value) locals are skipped. Next slice:
  lldb Python pretty-printers that read an object's header `type_id` → `TypeMeta`
  and render `Point { x: 3, y: 4 }`, reusing `gc::dump::render_object` — the
  JVM-grade piece (design §3).
- **Subroutine types** still `(None, &[])` — backtraces don't show param types
  yet. Cheap follow-up (build from `f.ret`/`f.params` via `di_type_for_repr`).
- **Variable lines** use the function's def line (no per-local span yet) — fine
  for scope membership; refine when statement-level spans land.
- **Moving-GC re-inspect** (design §5): break → `frame variable` a heap local →
  force GC → continue → re-inspect. To verify when Ref locals are inspectable
  (their DWARF location must point at the *frame root slot* the collector
  updates, not a cached pointer).
- **AOT only** (inherited from P2): JIT DWARF needs the LLDB JIT-registration
  interface.

## Verify

- `cargo test --test dwarf` — `full_debug_emits_named_local_and_param_dies`
  dwarfdumps a `Full` object and asserts named variable/parameter DIEs + base
  type are present, and that a `LineTables` object has none.
- `cargo test --test debugger_lldb` (macOS) — builds a `--debug` executable,
  drives `xcrun lldb` to break at a body line and `frame variable`, asserts the
  decoded values (`x = 7`, `sum = 42`, …). Skips if lldb is unavailable.
