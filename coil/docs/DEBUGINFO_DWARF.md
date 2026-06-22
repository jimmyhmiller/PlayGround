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

## Deferred (honest — increment 2+)

- **Per-statement / per-expression line tables** — the headline of "step through
  code line by line." This needs a source **span on every `Expr`**, which the AST
  does not carry today (spans live only on `Sexp`; 220 `Expr::` match sites, and
  several checker invariants — `is_literal`, place detection, tail analysis —
  inspect raw `Expr` shapes, so a naïve `{kind, span}` wrapper would disturb them).
  This is its own deliberate slice, and it also unblocks **checker-level
  diagnostics carrying spans** (today they're bare messages) — the same
  foundation serves both.
- **Local-variable / parameter DI** (`DILocalVariable` + `llvm.dbg.declare`), so
  `frame variable` / `p x` work. Needs the variable→alloca mapping threaded with
  types into DWARF type entries.
- **Typed `DISubroutineType`** (parameter/return DWARF types); currently one
  shared opaque `() -> ?` signature.
- A **debug-info path for included/imported functions** (multi-source spans), once
  the reader stamps real (source-id, offset) spans across files rather than DUMMY.
