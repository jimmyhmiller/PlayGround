# Debugger P3 — DWARF locals + reflection pretty-printers

> **STATUS: DONE + GREEN — scalars, heap structs, enums (incl. payloads),
> nested refs, the moving-GC check, and typed signatures.** Two layers:
> **(1) native DWARF** (`gcr build --debug`, no extra tooling) renders scalars
> and structs and shows the active enum variant; **(2) the reflection
> pretty-printer** (`tools/gcr_lldb.py`, `command script import …`) adds enum
> PAYLOADS and inline nested refs by decoding the baked `gcrust_type_meta` blob.
> - `add(7,35)` → `x = 7`, `y = 35`, `sum = 42`, `doubled = 84` (all scalar
>   reprs: `i8..i64`, `u8..u64`, `f32/f64`, `bool`, `char`).
> - `Point{x:3,y:4}` → `(Point) p = { x = 3, y = 4 }` natively; with the printer
>   `Point { x: 3, y: 4 }` and nested inline:
>   `Line { a: Point { x: 1, y: 2 }, b: Point { x: 5, y: 6 } }`.
> - `Shape::Rect(3,4)` → `(tag = Rect)` natively; **with the printer the full
>   payload**: `Shape::Rect(3, 4)`, and recursively `Tree::Node(5, Tree::Leaf,
>   Tree::Leaf)`.
> - **§5 moving-GC check PASSES** (both layers): after ~millions of allocations
>   relocate `p` (address `0x…e6000000` → `0x…c6000168`), it still reads
>   `{ x = 42, y = 7 }` — the location *derefs the live GC frame slot* the
>   collector updates, and the printer reads object memory fresh at the stop.
> - Flattened `#[value]` aggregate fields render inline too:
>   `Holder { s: Slot { b: Box { val: 9 }, tag: 7 } }`.
> - Full-debug `DISubprogram`s carry typed prototypes (return + param types).
>
> Verified: dwarf 3/0 (variable + struct + enum DIEs), debugger_lldb 5/0 (scalar,
> struct-by-field, enum-variant, reflection enum-payload+nested+value, moving-GC),
> lib 158/0, aot/ffi/threads/reflect/modules/generational green, examples
> build+run under `--debug`, default builds unchanged. **Remaining (minor):**
> auto-loading the printer without a manual `command script import`; generic
> type-name matching for the summary registration.

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

## The reflection pretty-printer (`tools/gcr_lldb.py`, design §3)

What native DWARF can't express (enum payloads — no `DW_TAG_variant_part` in the
C API; inline nested-Ref rendering — DWARF members can't carry a `deref`) is
delivered by an lldb Python script that mirrors `gc::dump::render_object`:

1. **Decode the blob.** The compiler bakes the reflection table into every binary
   as the `gcrust_type_meta` data symbol (LE, self-describing — format in
   `gc/reflect.rs`). The script reads it via the *symbol table* + `target.ReadMemory`
   (NOT `FindFirstGlobalVariable` — the blob has no debug-info DIE), pre-`run`,
   and decodes it to `{type_id → struct/enum/opaque}`.
2. **Register summaries.** For every struct/enum type name it does
   `type summary add -F gcr_lldb.gcr_summary "<Name>"`.
3. **Render at a stop.** The summary takes the value's load address (= object
   base, since a Ref local's location derefs the live slot), reads `type_id` at
   `base+8`, and renders: struct → `Name { f: v, … }` (tuple structs →
   `Name(v, …)`); enum → read u32 tag at `base+tag_offset` → active variant →
   `Name::Variant(payload…)`; `Ref` fields recurse (depth-limited, so cyclic
   types degrade to an address). Moving-GC-safe: it re-reads object memory each
   stop. There is also a `gcrv <expr>` command.

Import with `command script import tools/gcr_lldb.py`.

## Scope boundary / open work

- **Auto-load the printer** without a manual `command script import` (e.g. a
  `<binary>.py` next to the dSYM, or `target.load-script-from-symbol-file`).
- **Type-name collisions**: the summary attaches by DWARF type name; a non-gc
  type sharing a gc-rust type's name would be mis-rendered (unlikely in practice;
  the summary returns "" when the name isn't in the reflection table).
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
  - `lldb_reflection_printer_renders_enum_payload_and_nested` — imports
    `gcr_lldb.py`; asserts `Shape::Rect(3, 4)` and inline nested structs.
  - `lldb_heap_struct_survives_moving_gc` — `p` reads correctly after a
    relocating GC (the §5 property).
