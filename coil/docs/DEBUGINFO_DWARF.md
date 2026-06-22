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

## Deferred (honest — increment 3+)

- **Local-variable / parameter DI** (`DILocalVariable` + `llvm.dbg.declare` /
  `dbg.value`), so `frame variable` / `p x` work. The codegen side was prototyped
  (a `Type` → `DIType` map for scalars/pointers + a `-g`-only spill of each scalar
  param to a debug `alloca`), but it is **BLOCKED by a toolchain mismatch in this
  environment**: inkwell 0.5's `insert_declare_at_end`/`insert_dbg_value_before`
  call the C symbols `LLVMDIBuilderInsertDeclareAtEnd` / `…InsertDbgValueBefore`,
  which were **removed in LLVM 19+** (the "debug records" migration renamed them
  to `…InsertDeclareRecordAtEnd`, etc.). The build links Homebrew LLVM **21** libs
  (despite the `llvm18-0` feature — most symbols still match, so increments 1–2
  link fine; these two do not), so the link fails with an undefined symbol.
  Unblock by one of: building against real LLVM ≤18 libs (set
  `LLVM_SYS_181_PREFIX=/opt/homebrew/opt/llvm@18` and rebuild `llvm-sys`); an
  inkwell version exposing the `InsertDeclareRecord*` API; or emitting the
  `@llvm.dbg.declare` intrinsic call by hand. (Coil also binds immutable `let`s /
  scalar params as SSA values, not `alloca`s, so a `-g`-only spill or `dbg.value`
  is needed regardless — but the symbol mismatch is the immediate blocker.)
- **Typed `DISubroutineType`** (parameter/return DWARF types); currently one
  shared opaque `() -> ?` signature.
- A **debug-info path for included/imported functions** (multi-source spans), once
  the reader stamps real (source-id, offset) spans across files rather than DUMMY.
