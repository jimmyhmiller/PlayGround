# DCE capability study: ours vs. terser / esbuild on real-world JS

Goal: understand what our IR + dataflow can and cannot analyze *today*, using real
code. The existing JS tools are a reference for "what good DCE removes" — not an
oracle. Our pass is intentionally **sound** (never removes anything that could
change observable behavior); the comparison shows where soundness, scope, and
missing analyses leave value on the table.

## What our DCE does (`jsir_transforms::eliminate_dead_code`)

Driven by the constant-propagation dataflow analysis over the JSIR IR:

1. **Constant-condition `if`** — folds to the taken branch when the test
   propagates to a constant. More than upstream jsir's DCE, which only matches a
   *syntactic* `true`/`false`; we fold computed constants: `2 > 1`,
   `'production' !== 'production'`, a const-valued variable, etc.
2. **Constant-false `while`** — removed (a constant-true loop is infinite; kept).
3. **Unreachable code** via **completion analysis** — statements after a
   statement that always completes abruptly (`return`/`throw`/`break`/`continue`,
   a block ending that way, or an `if` whose both branches do) are dropped.
   Hoisted `var`/`function` in the dead tail are preserved.
4. **Unused variables** — `var`/`let`/`const` never read and never reassigned,
   with a side-effect-free initializer, are removed. Cascades (`var b = a` →
   removing `b` frees `a`).
5. **Unused functions** — `function f(){…}` whose name is referenced nowhere is
   removed. Cascades through the call graph.

These compose: folding a `if (false)` guard removes the references inside it,
which makes the functions it called unused, which removes them.

## Result 1 — idiomatic source has ~no statically-dead code

Ran on unmodified library source (lodash 544 KB, react.development 87 KB, redux,
debug, ms, classnames):

- **All 6 round-trip through our pipeline and re-parse cleanly** (source → IR →
  DCE → IR → source). Robustness on real code holds at 544 KB.
- **Our DCE eliminated 0 statements in every file.** Hand-written source has no
  `if (2>1)`, no unreachable tails, and its top-level bindings are exported/used.

This is the honest headline: the dead code people care about in shipped JS is not
*in* the source — it is **created** by the build (a `define` pass turning
`process.env.NODE_ENV` into a literal; bundling that exposes unused exports).

## Result 2 — the bundler scenario, where our analysis fires

Real `lodash-es`, import 12 functions, call 3, with the other calls behind a
`if (false)` guard (esbuild bundle, tree-shaking off → all the code is present):

| tool                | dead fns removed | semantics | notes |
|---------------------|------------------|-----------|-------|
| **ours (DCE only)** | debounce, isEmpty + dead deps | **identical** (`[[1,2],[3,4]]`) | no minification |
| esbuild `--minify`  | same set | identical | + minifies/renames |
| terser `-c -m`      | same set | identical | + minifies/renames |

We converge to the **same surviving function set** as esbuild and terser (exactly
`chunk`'s dependency closure), and input vs. output execute identically. The byte
gap (ours ~7.7 KB vs. ~4.8 KB) is **minification, not dead code** — renaming and
whitespace, which is out of scope for a DCE pass.

`process.env.NODE_ENV` → literal substitution then folds the dev/prod branch the
same way (`if ('production' !== 'production')` → drop the dev branch), matching
`esbuild --define`.

## What we cannot do yet (the limits)

- **No constant inlining / copy propagation.** terser turns `var x=1; return x`
  into `return 1` and drops `x`; we keep `var x` (it is read). Pure DCE, no
  value substitution.
- **No cross-module tree-shaking.** Bundlers tree-shake via the **ESM module
  graph** (resolve imports, drop unused exports). We operate on one already-
  resolved file. We re-derive the same removals *within* a bundle, but we don't
  resolve modules.
- **No property/object analysis.** `require('lodash')` + 3 methods is not
  reducible by anyone here (lodash UMD attaches methods to an object) — it needs
  object/property liveness we don't model. This is *why* `lodash-es` exists.
- **No minification** (renaming, whitespace, syntax collapse) — not DCE.
- **Conservative cases:** self-recursive functions never called externally are
  kept; function elimination is name-based (keeps a dead `f` if any other `f` is
  referenced under shadowing); arithmetic purity assumes no side-effecting
  `valueOf`/`toString` (the standard DCE assumption).
- **More aggressive than terser in one spot:** we fold top-level const bindings
  (`var DEBUG=false; if (DEBUG)…` → gone); terser keeps these without
  `--toplevel`, being cautious about external mutation of script globals. Sound
  for a single self-contained program.

## Result 3 — cross-module tree-shaking (the missing piece, now built)

`jsir_transforms::tree_shake(sources, entry)` resolves the ESM import graph,
drops unreachable modules, un-exports/removes dead named exports, runs the
per-module DCE (pinning live exports), and drops unused imports.

Tested on **real `lodash-es` (644 ESM modules)**, entry imports 3 functions
(`chunk`, `debounce`, `isEmpty`):

| metric | ours | esbuild |
|--------|------|---------|
| lodash-es modules retained | **50** (+1 entry = 51) | **50** |
| modules dropped | 663 of 714 | — |
| kept bytes | 41 KB of 3.96 MB | — |
| output runs identically | `[[[1,2],[3,4]],null,true]` | same |

We reach the **same module set as esbuild's tree-shaker**, all 644 files parse
and round-trip through our IR, and the shaken output executes identically under
node. Dead named-export/import counts are 0 here because lodash-es files are
single-default-export and the dependency closure is all-live (same finding as
Result 2). The dead-export/import elimination is exercised by the unit tests
(`tests/treeshake.rs`).

Limit vs. a real bundler: we keep module boundaries (no scope-hoisting into one
bundle), keep `export default` and `export *` conservatively, and resolve only
relative specifiers (bare npm specifiers are treated as external).

This depended on first fixing a converter bug: `export function`/`class` dropped
their declaration (see `PARITY_GAPS.md` "Recently fixed").

## Reproduce

```
cargo test -p jsir-transforms          # 22 DCE + 4 tree-shake tests
cargo test -p jsir-analyses            # constprop soundness oracle (real node exec)
cargo run --release --example dce -p jsir-transforms -- <file.js>
cargo run --release --example treeshake_dir -p jsir-transforms -- <root> <entry-rel> [outdir]
```
