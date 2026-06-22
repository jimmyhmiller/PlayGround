# Debugger P3 — DWARF locals + reflection pretty-printers

> **STATUS: SCALARS + HEAP STRUCTS DONE + GREEN (incl. the moving-GC check).**
> `gcr build --debug` produces an unoptimized executable carrying DWARF
> local/param variable DIEs, and **lldb's `frame variable` shows gc-rust locals
> by source name with correct values** — for scalars *and heap structs*:
> - `add(7,35)` at line 4 → `x = 7`, `y = 35`, `sum = 42`, `doubled = 84`
>   (all scalar reprs: `i8..i64`, `u8..u64`, `f32/f64`, `bool`, `char`).
> - `let p = Point{x:3,y:4}` → `(Point) p = { x = 3, y = 4 }` *directly* (gc-rust
>   reference semantics: the local IS the struct); nested structs expand with
>   `-P2` (`Line` → `a = { x=1, y=2 }`, …).
> - `let s = Shape::Rect(3,4)` → `(Shape) s = (tag = Rect)` — heap enums show
>   their active variant (the u32 tag as a DWARF enumeration). Payload not yet
>   rendered.
> - **§5 moving-GC check PASSES**: after ~millions of allocations relocate `p`
>   (its address changes `0x…e6000000` → `0x…c6000168`), `frame variable p` still
>   reads `{ x = 42, y = 7 }` — because the DWARF location *derefs the live GC
>   frame slot* the collector updates, never a cached object address.
>
> Verified: dwarf 3/0 (variable + struct DIEs present in Full, absent in
> LineTables), debugger_lldb 3/0 (scalar values, struct-by-field, moving-GC),
> lib 158/0, aot/ffi/threads/reflect/modules/generational green, examples
> (binary_trees/shapes/calculator/fib) build+run under `--debug`, default
> (non-`--debug`) builds unchanged. **NEXT:** enums (tagged unions, variant
> parts or reflection printer), value-aggregate fields, inline rendering of
> nested *Ref* fields (needs a synthetic-children/pretty-printer — DWARF members
> can't carry a deref), real subroutine types.

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

## Heap structs (native DWARF composite types)

Rather than lldb Python pretty-printers, heap structs use **native DWARF struct
types** built from the reflection metadata (`DiTypeBuilder` in `codegen.rs`,
driven off each `Layout.meta`): one memoized `DW_TAG_structure_type` per
`type_id` with members at their absolute (header-included) offsets. A `Ref` local
is then typed as that struct (not a pointer) with location
`DW_OP_plus_uconst <slot-off>, DW_OP_deref` — i.e. *the struct living at the
address the GC frame slot points to*. lldb formats it natively (no script), and
the `deref` re-reads the slot on every stop, which is exactly what makes the
moving-GC check pass. Field types: scalars → basic types, `Ref` fields → pointer
to the referent's struct DIE (lldb shows the address; `-P2` expands it).
Recursion (`List` → `List`) is broken by an in-progress guard → degrades that
pointer to a raw address. Enum / opaque / varlen (String/Array) layouts and
value-aggregate fields are left unmodeled → rendered as an address.

## Scope boundary / open work

- **Enum payloads.** The active variant NAME shows (`tag` enumeration member),
  but not its fields — `Rect(3,4)` reads as `(tag = Rect)`, not `Rect(3, 4)`.
  The C API has no `DW_TAG_variant_part` (checked llvm-sys 211 / inkwell), so the
  payload needs a reflection-driven synthetic provider: read the tag → active
  `VariantMeta` → its fields at their absolute offsets (the Python pretty-printer
  the design §3 envisioned, reusing `gc::dump::render_object`'s logic).
- **Inline nested-Ref rendering.** A struct member that is a `Ref` shows as a
  pointer/address (lldb won't auto-deref aggregate members; DWARF members can't
  carry the `deref` a local can). One-line inline rendering needs a
  synthetic-children / summary provider — the Python pretty-printer the design
  envisioned, reusing `gc::dump::render_object`.
- **Value aggregates** (`#[value]` fields/locals) skipped — no per-value offset
  metadata wired into the DWARF builder yet.
- **Subroutine types** DONE — full-debug `DISubprogram`s carry a real prototype
  (return + param types via `di_signature_type`; `Ref` → pointer-to-struct,
  `Value` → opaque address, `Unit` → omitted). `add`'s DIE now has
  `DW_AT_type "i64"`.
- **Variable lines** use the function's def line (no per-local span yet) — fine
  for scope membership; refine when statement-level spans land.
- **Moving-GC re-inspect** (design §5): DONE/VERIFIED — see the status block and
  the `lldb_heap_struct_survives_moving_gc` test.
- **AOT only** (inherited from P2): JIT DWARF needs the LLDB JIT-registration
  interface.

## Verify

- `cargo test --test dwarf` — `full_debug_emits_named_local_and_param_dies`
  dwarfdumps a `Full` object and asserts named variable/parameter DIEs + base
  type are present (and a `Point` struct DIE with ≥2 members), and that a
  `LineTables` object has none.
- `cargo test --test debugger_lldb` (macOS, skips if lldb absent) — builds
  `--debug` executables and drives `xcrun lldb`:
  - `lldb_frame_variable_shows_correct_scalar_locals` — scalar values.
  - `lldb_frame_variable_renders_heap_struct_by_field` — `(Point) p = { x=3, y=4 }`.
  - `lldb_frame_variable_shows_enum_variant` — `(Shape) s = (tag = Rect)`.
  - `lldb_heap_struct_survives_moving_gc` — `p` reads correctly after a
    relocating GC (the §5 property).
