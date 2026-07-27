# New metaprograms — a Display facility + two real checkers/transforms

This set was built to answer "make some cool and useful metaprograms for Coil." It also
enables a small but real language improvement: **`Code` can now carry trait impls.**

## The foundation: `show.coil` — a `Display` trait for diagnostics

`warn`/`report` take only a `(slice u8)`. Building messages that mix raw text, a `Code`
node (a symbol, a type, a form), an int, and a bool meant hand-rendering each. `show.coil`
makes rendering a **trait**, `Display`, and gives variadic macros that dispatch per argument:

```coil
(warn!   node "match on " ty " (" (code-variant-count ty) " variants, first=" vname ")")
(report! node "function " name " is marked pure but reaches extern '" ext "'")
```

- `(deftrait Display [Self] (dfmt [(x Self) (sb (mut StrBuf))] …))` with impls for
  `(slice u8)`, `Code` (via `code-str`), `i64` (plain runtime arithmetic — NOT the
  comptime-only `int->str`), and `bool`. Extend it to your own types with one `impl`.
- `msg!` / `report!` / `warn!` are variadic macros; `report!`/`warn!` expand to
  `(do (report/warn …) 0)` so they drop into `(if cond (warn! …) 0)` with no type clash.

### The compiler change under it

`impl Display Code` required lifting a restriction: `Code` was bucketed with
`void/ref/never/externref` as "can't carry an impl" in the single dispatch-key source of
truth, `selfhost/src/parser.coil::impl-base-name`. Adding `(TCode [] "code")` there fixes
it everywhere (check/mono delegate to that one function). It is safe because a
`Display`-for-`Code` method takes a `Code` param, so it is comptime-only and already
dropped before mono by `drop-code-funcs`. Verified with a full `selfhost/rebootstrap.sh`
(fixpoint + all gates, byte-exact against the reference corpus).

## `effects.coil` — an effect / purity checker (whole-program call graph)

A function whose name ends in `-pure` must not transitively reach an `extern` (FFI is the
effect boundary). Runs after typecheck, so `(code-decl CALL)` resolves every call to its
exact module + kind (`fn` vs `extern`) across modules and the bundled stdlib. Builds an
index of every function body, then does a transitive DFS from each `-pure` root.

```
coil run app.coil --use effects.coil
# (defn g-pure …) that calls (defn e …) that calls (extern putchar …)
#   → error: function g-pure is marked pure but reaches extern 'putchar'
```

**Scope:** follows DIRECT (named) calls; does not follow indirect calls through function
pointers (the standard limit of static call-graph analysis) — so effects reached only via a
vtable, e.g. buffered stdout behind the `Writer` API that `println`/`fmt` lower to, are not
seen. Direct externs and direct call chains are.

## `profile.coil` — an auto-profiler transform

A whole-program transform that wraps every user function body with entry/exit timing and a
tiny runtime that prints per-function totals when `main` returns. Zero source edits:

```
coil run app.coil --use profile.coil
# === profile: function / calls / total ns / avg ns ===
#   fib   calls=21891  total=12737000ns  avg=581ns
#   main  calls=1      total=951000ns    avg=951000ns
```

The rewrite is idempotent (transforms run to a fixpoint — an already-wrapped body is
recognized by its `__prof_t0__` binding). The function name is baked into the injected
`prof-record` call as a string literal via `code-str` (spliced as a value). Times are
INCLUSIVE (a function's total includes its callees). Skips `Code`-returning
(comptime) functions and its own runtime module.

## Two ideas that turned out to be already built in

- **Match exhaustiveness** — Coil's `match` builtin is already checked by the typechecker
  (missing variant, unknown variant, duplicate arm all error before a metaprogram runs).
- **`fmt` format-string checking** — `fmt` is a macro that expands to typed calls, so both
  argument COUNT and argument TYPES are already checked by the compiler.

The remaining format-string gap is the **C `printf` family** (variadic externs): `printf
c"%s\n" 12345` segfaults at runtime, unchecked. A `-Wformat` checker over those is the
natural next metaprogram (not yet built).
