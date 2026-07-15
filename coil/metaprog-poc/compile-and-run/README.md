# Compile-and-run metaprograms — the two load-bearing mechanisms, proven

Run from the repo root: `metaprog-poc/compile-and-run/run.sh [coil-binary]`

## Why

A metaprogram is run by the comptime **interpreter**, which is a second
implementation of Coil's semantics. Every gap in it (generics, `fnptr-of`,
`call-ptr`, FFI, raw memory) is a feature the real compiler already has. Closing
those gaps one at a time is a treadmill that never reaches parity, and it cannot
ever reach "run a GUI at comptime", because the interpreter has no memory model:
`CtVal` has no raw-pointer variant, only `CRef`, a pointer to an interpreter cell.

The reason a metaprogram is not a normal program is that it is executed by a
different engine. So: **don't interpret metaprograms — compile them and run them.**
Then everything works, because it is the same pipeline that builds any other Coil
program. Generics work because `mono.coil` already does them. FFI works because it
is real linked code. `malloc` works because it is `malloc`.

This directory proves the two mechanisms that design rests on, using the compiler
exactly as it is today. No JIT bindings, no LLVM ORC, no new backend.

## 1. Callback — `plugin.coil` + `host.coil`

A coil-built dylib calls **back** into the coil-built host process. `host_add` is
undefined at link time in the dylib and bound at `dlopen` against the host's
exported symbols.

    coil build plugin.coil --shared -o plugin.dylib --link-flag -Wl,-undefined,dynamic_lookup
    coil build host.coil -o host --link-flag -Wl,-export_dynamic
    ./host        # plugin_entry(20) = 42

This is how a compiled metaprogram reaches compiler state that cannot be passed as
data: `report`, `warn`, `type-of`, `code-decl`.

## 2. A metaprogram as a normal program — `meta.coil` + `mhost.coil`

`meta.coil` has **no `Code` type, no `ECodeOp`, no special forms, no interpreter**.
It imports the compiler's real `selfhost/src/reader.coil` and `parser.coil` and
walks a real `Sexp` that the host built with the real reader.

    ./mhost       # metaprogram walked (a b c d e) -> 5 children

This is the whole design in one file. `Sexp` is already an ordinary Coil struct and
`sx-len`/`sx-at`/`sx-tag` are already ordinary Coil functions, so a metaprogram needs
no special language surface at all.

## What this implies for the real change

- `Code` becomes `(ptr Sexp)`; `code-nth`/`code-count`/… become ordinary library
  functions. Existing metaprogram source keeps working: same names, same shapes.
- `ECodeOp` and `TCode` are deleted rather than lowered. Today they hard-fail in
  every backend (`codegen.coil:730`, `codegen_a64.coil:1508`) and `mono.coil:713`
  drops `Code`-typed functions before mono. None of that is needed if the metaprogram
  is a normal program: the *sub-program* keeps its Code functions and gets compiled,
  the main program drops them exactly as it does now.
- Only compiler **state** stays extern, via mechanism 1.
- Quasiquote stays sugar, expanding to `Sexp` constructor calls — itself a macro,
  compiled the same way.
- The 1892-line interpreter in `comptime.coil` deletes.

## THE ENGINE IS BUILT — `COIL_META=compiled`

The design above is now real (with one deliberate deviation): set
`COIL_META=compiled` and expand-stage3 compiles the checked metaprogram sub-program
into a dylib and runs every macro/checker/transform entry as native code. The
interpreter stays the default engine and the oracle.

    COIL_META=compiled coil run app.coil        # any program; same output
    metaprog-poc/compile-and-run/parity.sh ./coil   # the proof

The pieces (all in `selfhost/src/`):

- **`metalower.coil`** — after `check-program`, rewrites the sub-program:
  `Code` -> `(ptr i8)` (opaque host-Sexp handle), `ECodeOp` -> shim arg-builder
  chains typed from the checker's type map, `EQuote`/`QLit` -> a host quote
  registry (original nodes, exact spans), `EQuasi` -> nested builder calls,
  `(error …)` -> a never-typed call-then-`(loop 0)`.
- **`metashim.coil`** — injected into the sub-program; ptr/i64/f64-only crossings.
- **`metahost.coil`** — the host side wraps arguments back into real CtVals and
  dispatches to the interpreter's own `code-op`: one semantics, by construction.
  An op `Err` (and `(error …)`) records the Diag and `pthread_exit`s the
  metaprogram thread; the engine joins and reports it like the interpreter would.
- **`metaengine.coil`** — dlopen + a **vtable handshake** (`coil_mp_init`) instead
  of `-export_dynamic`/compiler `export-c` (the arm64 backend has no export-c, and
  the compiler must keep bootstrapping through it). Each entry call runs on its own
  32 MiB pthread.

The deviation from the sketch above: `ECodeOp`/`TCode` are **lowered**, not
deleted, and quasiquote lowers in `metalower`, not as a macro — so the interpreter
and the compiled engine coexist and can be diffed. Deleting the interpreter comes
after the tower (below).

**Verified**: `parity.sh` — 112/112 files (examples/, lib/, all of metaprog-poc:
checkers, transforms + fixpoint, `code-decl`/`type-of`, `report`/`warn`/`error`,
quasiquote hygiene, gensym) produce **byte-identical** emit-ir output and
byte-identical diagnostics under both engines. The compiler compiling **itself**
under the compiled engine emits byte-identical IR at the same wall clock. The
rebootstrap fixpoint and both gates pass untouched.

**The payoff** — `arbitrary.coil` / `arbitrary_test.coil` (run.sh mechanism 4): a
macro that builds a generic `ArrayList`, looks up a compile-time string-keyed
`HashMap` (through `fnptr-of` KeyOps — exactly where the interpreter dies), calls
**libc `strlen` at expansion time**, and builds binder names with `StrBuf`. The
interpreter cannot run any of it; the compiled engine runs all of it, because it is
just a program.

## The remaining hard part: the tower

Metaprogram BODIES still cannot *call macros* (`fmt`, `when`, `try!` inside a
macro's own body) — the sub-program is resolved unexpanded, same as the
interpreter's restriction today. Fixing it means recursively expanding the
sub-program's forms before resolve (macro definitions form a DAG bottoming out in
core forms, so it terminates), with the gensym counter snapshotted around the inner
expansion so the main program's expansion stays byte-identical, and per-level
expansion tables kept coherent for diagnostics. That, plus a content-addressed
dylib cache keyed on the closure source (today the dylib is rebuilt per compile),
is what stands between here and deleting the 1892-line interpreter.
