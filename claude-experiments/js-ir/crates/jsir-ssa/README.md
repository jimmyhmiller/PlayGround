# jsir-ssa: MLIR-style CFG + SSA + React-compiler analyses

A control-flow-graph + SSA layer built **on top of** the reversible JSIR IR,
plus the start of the React Compiler's analysis pipeline. JSIR (like upstream
`google/jsir`) is deliberately *not* in variable-SSA form — it keeps name-based
`identifier`/`identifier_ref` reads/writes so it can round-trip to Babel AST.
This crate adds the SSA substrate the React-Compiler-style analyses need, while
leaving the reversible IR untouched.

## Pipeline

```
source ──(jsir-swc)──▶ JSIR IR ──(lower)──▶ pre-SSA CFG ──(ssa::construct)──▶ SSA CFG
                                                                  │
                          mutability::analyze ◀───────────────────┤
                          scopes::infer / analyze ◀────────────────┘
```

## Modeled after MLIR

- **Block arguments are phi nodes.** A branch passes operands; the successor's
  block parameters are the merge points. No explicit `phi` op (exactly MLIR).
- **`scf`-to-`cf` lowering** (`lower.rs`): jshir's structured control flow
  (`if`/`while`/`?:`/`&&`/`||`) becomes basic blocks + branches. Expression value
  ops (already SSA in jsir) map almost 1:1.
- **`mem2reg`** (`ssa.rs`): name-based variables (`ReadVar`/`WriteVar`) are
  promoted to SSA values via **Braun et al.** (the same algorithm React
  Compiler's `EnterSSA` uses), with on-the-fly trivial-phi removal for minimal
  SSA.
- **Verifier** (`verify.rs`): MLIR-style well-formedness — single definition,
  def dominates use, block-argument arity — via iterative dominators.

## Source of truth (robust testing)

Two independent checks, because SSA bugs must not slip through:

1. **Executable oracle** (`tests/oracle.rs`): for ~20 programs × many inputs we
   interpret the **pre-SSA** CFG and the **SSA** CFG and require both to equal
   **Node** running the original JavaScript. 500+ (program, input) pairs,
   covering loops, nested control flow, reassignment, `?:`/`&&`/`||`, objects,
   arrays, and member reads/writes. Any lowering or SSA bug becomes a value
   mismatch.
2. **Dominance verifier**: every SSA CFG produced must be well-formed.

## React-compiler analyses (`mutability.rs`, `scopes.rs`)

- **Mutable-range inference** — for each value, the program-point range over
  which it (or an alias) may still be mutated. Sound/over-approximate: `obj.x=v`
  and calls mutate their object/arguments; capturing a value (`{a:v}`, `[v]`,
  `o.x=v`, call args) aliases it (union-find) so a later mutation widens the set.
  Pure object construction collapses to its definition → freely memoizable.
- **Reactive-scope inference** — merges overlapping mutable ranges into scope
  boundaries (the one place the grouping is forced). Pure values form
  independent singleton scopes.
- **Dependency inference** — each scope's reactive inputs (the memo cache key)
  and outputs (cached values used later). This is the memo interface React
  emits:

  ```js
  // function f(a,b){ let style={color:a}; let el={size:b,props:style}; return el; }
  const style = useMemo(() => ({color:a}),            [a]);        // scope, deps=[a]
  const el    = useMemo(() => ({size:b, props:style}),[b, style]); // scope, deps=[b, style]
  ```

  Our analysis derives exactly those scopes and dependency sets.

- **Memoization codegen** (`codegen.rs`) — for straight-line components, emits
  the React Compiler's `useMemoCache` output verbatim:

  ```js
  function C(a, b) {
    const $ = _c(5);
    let _v3, _v6;
    if ($[0] !== a)               { $[0] = a; _v3 = {color: a};            $[1] = _v3; } else { _v3 = $[1]; }
    if ($[2] !== b || $[3] !== _v3){ $[2]=b; $[3]=_v3; _v6={size:b,props:_v3}; $[4]=_v6; } else { _v6 = $[4]; }
    return _v6;
  }
  ```

  Verified under Node (`tests/codegen.rs`): the emitted code computes the same
  result as the original for every input **and** actually memoizes (object
  references are reused when dependencies are unchanged, recomputed when they
  change).

## JSX + matching the real React Compiler

Real components are handled: JSX is desugared to `React.createElement` calls in
`jsir-swc` (`jsx.rs`), so the existing CFG pipeline analyzes them unchanged.
`codegen::compile(src)` detects components (Capitalized) / hooks (`use*`), emits
the real `import { c as _c } from "react/compiler-runtime"` and the memoized
function, and passes other functions through.

Two refinements make the **scope structure match the React Compiler exactly**:

- **Pure-constructor calls are not mutations.** `React.createElement`/`jsx` read
  their args, so they don't mark `style`/`data` mutable (otherwise nothing would
  memoize).
- **`MergeScopesThatInvalidateTogether`.** A producer scope folds into its single
  consumer only when its deps ⊆ the consumer's (so they genuinely invalidate
  together — no over-invalidation, the producer keeps its observable identity).
  Plus the JSX props object (a transient with no independent identity) always
  folds into its element scope. On the canonical `Foo` component this yields the
  **same `_c(8)`, same 3 scopes, same dependency keys** as `react-compiler-e2e`.

`tests/jsx.rs` verifies (under Node, against the createElement-desugared
baseline) that the memoized component computes the same element tree for every
prop set **and** reuses element references when deps are unchanged.

## Status / next

Done and oracle-verified end-to-end (source → CFG → SSA → mutable ranges →
reactive scopes → merge → dependency inference → **memoized JS, run under
Node**), including **real JSX components** whose scope layout matches the React
Compiler. Verified against the actual `react-compiler-e2e` CLI.

Not yet: type inference (precise hook/primitive distinction), control-flow
codegen (codegen bails on loops/branches; the analyses already handle them), and
preserving JSX literally in the output (we emit `createElement`, React keeps the
JSX — semantically identical).
