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

## The hard part (not proven here)

The compiler bootstraps on macros: `try!` appears 56 times in `resolve.coil`, 45 in
`check.coil`, and it is a library macro over `block`/`return-from`, themselves macros
over `loop`/`break`. To expand `try!` you must run it; to run it you must compile it;
compiling it needs macros expanded.

It terminates because macro definitions form a DAG whose leaves bottom out in core
forms, so the tower compiles in dependency order, each stratum built with the strata
below it. The cost is a compile+link+dlopen per build, answered by a content-addressed
cache keyed on the closure source: same macros, same dylib, reuse it. Paid once per
macro-library version, not per build.
