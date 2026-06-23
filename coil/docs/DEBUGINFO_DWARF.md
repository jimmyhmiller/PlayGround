# DWARF debug info (FUTURE_WORK §3.2) — increment 1: function-granularity line tables

`coil build -g <file>` / `coil run -g <file>` now emit DWARF so `lldb`/`gdb` can
map the program to its source. Built on inkwell's `DebugInfoBuilder`.

## What works (verified)

- A **compile unit** (`DICompileUnit`, producer `coil`, `DWARFSourceLanguage::C` —
  the honest closest) over a `DIFile` for the source path, with the
  `Debug Info Version` and `Dwarf Version` module flags LLVM requires (inkwell
  doesn't add them).
- A **`DISubprogram` per function** at its real source line, attached to the LLVM
  function. Functions carry a span (`Func.span`, stamped from the `(defn …)` form
  in the parser, preserved through check + mono). A function read from an
  included/imported file carries `Span::DUMMY` → it gets no line info, never a
  wrong one.
- **Debug locations** on the body's instructions (function line), so a function
  has the `!dbg` the verifier requires once it has a subprogram. Cleared between
  functions so a no-info (dummy-span) function never inherits a stale scope.
- **`lldb` reads it**: `breakpoint set --name <fn>` resolves to `file:line` and
  the breakpoint **hits**; backtraces show `fn at file:line`. Verified in
  tests/debuginfo.rs (dwarfdump assertions + an lldb breakpoint test, both gated
  on the tools).
- **Opt-in**: without `-g`, output is byte-for-byte the old `-O3` build (no DWARF).
  A regression test asserts a release build has no `DW_TAG_subprogram`.

## The optimization tension (and the resolution)

Coil *always* optimizes (the README's `cc -O3` parity), and it *must* run at least
mem2reg + tail-call elimination — Coil's only loop is self-tail-recursion, so
without TCE a loop overflows the stack. But heavy `-O3` inlining makes most
functions vanish from the debugger (a breakpoint resolves but never hits). So a
`-g` build:

1. steps the pipeline down to **`default<O1>`** (keeps mem2reg + TCE; the
   `alwaysinline` `(llvm-ir …)` helpers still inline, so zero-overhead intrinsics
   are unaffected), and
2. marks user functions **`noinline`** (the cost-based inliner is what removes
   them; `alwaysinline` is a different pass, so helpers are untouched).

Together these make user-function breakpoints reliably resolve and hit. Frames
still show `[opt]`, and a tail call still replaces its caller's frame (correct
TCO behavior, just visible).

## macOS packaging

On Mach-O the DWARF stays in the `.o`; the linker records a debug map pointing at
it. A `-g` build therefore runs **`dsymutil`** to gather a `<exe>.dSYM` next to
the executable (best-effort — skipped if absent, e.g. ELF hosts where DWARF lands
in the binary), then removes the `.o`.

# Increment 2 — per-statement line tables (DONE)

Now that every `Expr` carries a source span (the per-Expr span foundation),
codegen sets a **per-statement debug location**: `emit_expr` calls `set_dbg_loc`
from each expression's span (scoped to the function's `DISubprogram`, threaded on
`Cg.cur_sp`), so each statement's instructions map to its own source line. In
lldb, `next` walks the body line by line (verified — tests/debuginfo.rs
`lldb_steps_line_by_line`), and a function breakpoint now lands on the first body
statement rather than the `defn` line.

## The pipeline, again — `-g` goes almost-empty

Per-statement stepping is destroyed by optimization (constant folding collapses
`(iadd 1 2)` to a constant; mem2reg promotes locals out of memory). So a `-g`
build now runs only **`function(tailcallelim),always-inline`**:

- `tailcallelim` is non-negotiable — Coil's only loop is self-tail-recursion, so
  without TRE a recursive program overflows the stack (verified: 100M-deep tail
  recursion exits cleanly under `-g`).
- `always-inline` keeps the `(llvm-ir …)` zero-overhead helpers inlined (a
  different pass from the cost-based inliner, which stays off).
- Everything else (mem2reg, instcombine, GVN, the inliner) is OFF, so statements
  survive as steppable instructions and `alloca`-backed locals stay in memory.

Note: pure-arithmetic statements over constants still produce no instructions
(immutable `let`s are SSA values, not stores), so there is nothing to step
*onto* for them — stepping is most useful across calls / effects / control flow.

# Increment 3 — scalar-parameter locals (DONE)

`frame variable` / `p a` now show a function's **scalar/pointer parameters**.
Each is given a `DILocalVariable` (`create_parameter_variable`) over a `-g`-only
debug `alloca` it is spilled into (the minimal `-g` pipeline keeps the `alloca` in
memory), with a `Coil-Type → DIType` map for the scalar/pointer cases
(`di_type`); aggregates (struct/sum/slice/array/fnptr) are skipped — no entry
rather than a wrong type. The `DISubprogram`/CU are marked **not optimized** under
`-g` so lldb trusts the DWARF instead of heuristically skipping the prologue.

This required **moving the whole build to the latest LLVM**: the `llvm.dbg.declare`
inserter (`LLVMDIBuilderInsertDeclareAtEnd`) was removed in LLVM 19+ (the
"debug-records" migration). Bumping `inkwell 0.5`/`llvm18-0` → `inkwell 0.9`/
`llvm21-1` (matching the Homebrew LLVM 21.1 actually linked) maps it to the new
`…InsertDeclareRecordAtEnd` symbol. See "LLVM 21 / inkwell 0.9" notes below.

**Caveat (AArch64):** a breakpoint set on a *function name* lands just after the
frame setup but *before* the parameter spill stores, so it shows stale stack
bytes for one instruction; a **line** breakpoint inside the body (the common
case), or one `next`/`step`, shows the correct values. Verified at a line
breakpoint (`tests/debuginfo.rs::lldb_frame_variable_shows_params`).

# Increment 4 — `let`-binding locals (DONE)

`frame variable` also shows scalar/pointer `let` bindings now (`create_auto_variable`
+ a debug spill in `emit_expr`'s Let arm). Three things had to line up:

- **Spill slot in the entry block** (`entry_alloca`) — a `let` inside a loop would
  otherwise re-`alloca` each iteration and grow the stack. The alloca is emitted
  with *no* line (save/unset/restore the debug location) so it doesn't pollute the
  line table and mis-place line breakpoints.
- **Backend at `-O0` for `-g`** (`OptimizationLevel::None` on the target machine,
  not just a minimal IR pipeline) — at `Aggressive` the *backend* still merges and
  reschedules the otherwise-dead debug-spill stores away from their statement, so a
  local read as stale at a line breakpoint. The IR pipeline being empty wasn't
  enough; the codegen opt level matters too.
- **A raw-FFI `dbg.declare`** (`dbg_declare_at_end`) — inkwell 0.9's
  `insert_declare_at_end` wraps the LLVM-19+ *DbgRecord* return value as an
  `InstructionValue` and trips `debug_assert!(is_instruction())`, an *intermittent*
  hard panic in dev/test builds (it reads whatever the record ref classifies as).
  We don't use the return, so we call
  `LLVMDIBuilderInsertDeclareRecordAtEnd` directly via `inkwell::llvm_sys` and
  ignore it. (This also de-flaked the parameter path from increment 3.)

Verified at a line breakpoint (`tests/debuginfo.rs::lldb_frame_variable_shows_let_locals`).
The lldb tests are serialized (a `Mutex`) and load the `.dSYM` explicitly
(`add-dsym`, since Spotlight doesn't reliably index a freshly-built temp dSYM).

## Deferred (increment 5+)

- **Aggregate `DIType`s** (struct members, sum/enum variants, slice/array element
  types), so `frame variable` can print structs/slices, not just scalars.
- **Typed `DISubroutineType`** (parameter/return DWARF types); currently one
  shared opaque `() -> ?` signature.
- A **debug-info path for included/imported functions** (multi-source spans), once
  the reader stamps real (source-id, offset) spans across files rather than DUMMY.

---

# Toolchain — LLVM 21 / inkwell 0.9

The project targets the latest LLVM (21.1, the Homebrew default `llvm`) via
`inkwell = { version = "0.9", features = ["llvm21-1"] }`. Migrating from
`inkwell 0.5`/`llvm18-0` (which "mostly worked" against LLVM 21 libs but broke on
removed/renamed symbols) needed three API adjustments:

- `Context::custom_width_int_type` now takes a `NonZeroU32` and returns a
  `Result` — wrapped in `codegen::int_width(ctx, bits)` (Coil widths are
  parse-validated nonzero, so it can't fail).
- `CallSiteValue::try_as_basic_value()` returns a `ValueKind` enum, not `Either`:
  `.left()` → `.basic()` (both `Option<BasicValueEnum>`).
- `MemoryBuffer::create_from_memory_range[_copy]` now *asserts* the input slice
  ends in a NUL byte (and decrements length past it), so the `(llvm-ir …)` helper
  text is handed an explicitly NUL-terminated copy.
- **Typed `DISubroutineType`** (parameter/return DWARF types); currently one
  shared opaque `() -> ?` signature.
- A **debug-info path for included/imported functions** (multi-source spans), once
  the reader stamps real (source-id, offset) spans across files rather than DUMMY.
