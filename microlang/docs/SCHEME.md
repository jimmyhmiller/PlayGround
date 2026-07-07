# The Scheme frontend

`scheme/` is an R7RS-flavored Scheme built **entirely on the core's public API**.
The core (`microlang`) has no idea Scheme exists — the same execution tiers that
run the core's own Lisp run this. It is the live proof of the library/language
split ([LIBRARY_LANGUAGE_SPLIT.md](LIBRARY_LANGUAGE_SPLIT.md)): a whole language
as a *library on top of the toolkit*.

## Pipeline

```
Scheme source ─▶ reader ─▶ syntax-rules expand ─▶ desugar ─▶ core Val form ─▶ analyze ─▶ Ir ─▶ CodeSpace
              (scheme/src/lib.rs)   (syntax_rules.rs)   (lib.rs)              (core)
```

Everything above `analyze` is the frontend crate; everything at and below it is
the shared core. The frontend adds:

- **Reader** — Scheme lexical syntax the core reader lacks: `#t`/`#f`, `'quote`,
  `` `quasiquote ``/`,unquote`/`,@`, string literals, `#\char` literals.
- **`syntax-rules`** — pattern→template macros, **hygienic** (see below).
- **`desugar`** — surface forms → core forms (`define`→`def`, `car`→`first`,
  variadic `+` folds to binary prims, `let*`/`letrec`/named-`let`/`cond`/`case`,
  `and`/`or`, `reset`/`shift`, …).
- **A prelude** — standard-library procedures written *in Scheme itself* on the
  core's list primitives (`map`, `for-each`, `append`, `reverse`, `length`,
  `apply`, `call-with-values`, `dynamic-wind`, and first-class `+ - * < =`),
  auto-injected before every program.

## Conformance

`scheme/tests/conformance.rs` is a **61/61**-passing R7RS suite that doubles as
the roadmap. Every expected value is oracle-checked against Chicken Scheme (`csi`)
when it is installed, so a wrong *expectation* fails as loudly as a wrong
*implementation*. A live case is correct iff our output equals Chicken's.

Covered:

- Arithmetic, comparison, booleans; a numeric tower up through **arbitrary
  precision** — a dependency-free `BigInt` (`src/bigint.rs`) kicks in when the
  `i128` fast path overflows, so `10^20 * 10^20 = 10^40` is exact.
- Lists, `vector`s, `char`s, `string`s; `eq?`/`eqv?` (identity) vs `equal?`
  (structural).
- `let` / `let*` / `letrec` / named `let` / `cond` / `case` / `when` / `unless`,
  tail calls, mutual recursion.
- `map` / `append` / `apply` / `call-with-values` / `values`; quasiquote.
- **Hygienic `syntax-rules`**: identifiers a template *binds* (via `let`,
  `let*`, `letrec`, `lambda`, named `let`) are alpha-renamed to fresh names
  before instantiation, so a macro's introduced `t` cannot capture the caller's
  `t`. Pattern variables and free references (`let`, `if`, `+`) are left alone.
  The classic `my-or` capture test returns `5`, not `#f`.
- The full continuation story — `call/cc`, `shift`/`reset`, multi-shot, and
  GC-survivable ([CONTINUATIONS.md](CONTINUATIONS.md)).

## Running on the tiers

The functional core (arithmetic, recursion, `let`, `lambda`, prims, the prelude,
quasiquote, bignums) runs identically on `TreeWalk`, `ClosureComp`, `BytecodeVm`,
and `CekMachine` — same source, same `Ir`, same answer. The conformance suite
pins itself to `CekMachine` because the continuation-using cases (`call/cc`,
`shift`/`reset`), plus `apply` and the `(gc)` safepoint, are CEK-only; on other
tiers those specific primitives raise a clear error.

```
cargo run --bin scheme -p scheme        # demo: runs Scheme on TreeWalk + BytecodeVm
cargo test -p scheme --test conformance  # the 61/61 suite (oracle-checked if csi present)
```

## Honest edges

- `dynamic-wind` is correct for non-escaping use but does not yet re-run its
  guards across continuation jumps (needs a wind stack in the CEK machine).
- Hygiene covers introduced *bindings* for the common binding forms; it is not a
  full referential-transparency implementation.
- The numeric tower stops at exact integers — no rationals or exact/inexact
  contagion yet. `BigInt` is the foundation those would build on.
- The suite is a representative R7RS roadmap, not the entire standard (no ports,
  full string/char libraries, `case-lambda`, etc.).
