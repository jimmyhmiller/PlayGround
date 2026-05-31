# JSIR transforms & dataflow: roadmap and oracle workflow

This doc plans the next phase after byte-exact `ast2hir` parity (currently ~96.8%
match on test262) and round-trip stability: porting upstream's **transform passes**
and **dataflow analysis**, using upstream `jsir_gen` as the oracle exactly as we
did for conversion.

## Why this is tractable

Every transform pass is a pure **IR → IR rewrite** that emits the *same generic
MLIR text format we already reproduce byte-for-byte*. So the oracle loop is the
one we already trust:

```
our_transformed_ir  ==  upstream_transformed_ir   (byte-for-byte, modulo offsets)
```

No new printer, no new comparison logic. We extend the corpus generator to capture
each pass's output, implement the pass against our `jsir-ir` model, and diff.

The dataflow **analysis** dump (`--output_type=analysis`) is a separate, harder
tier — it needs a second printer (pretty form + per-op state annotations) and the
analysis engine itself. Tackle it after the transforms.

## What upstream exposes

`jsir_gen` CLI:

- `--passes=<comma list>` — run a pipeline of passes. `--output_type=ir` prints
  the resulting IR (our format).
- `--jsir_analysis=constant_propagation --output_type=analysis` — run the dataflow
  analysis and print the annotated dump.

### Transform passes (CLI string → semantics)

| CLI `--passes` name | difficulty | what it does |
|---|---|---|
| `peelparens` | easy | drop `ParenthesizedExpression` wrappers (`jsir.parenthesized_expression`) |
| `remove_directives` | easy | strip directive prologues (`"use strict"`) |
| `erase_comments` | easy | clear the `comments` array + per-node trivia comments |
| `split_sequence_expressions` | easy–med | `a, b, c` statement → separate statements; in `for` headers too |
| `split_declaration_statements` | easy–med | `var a, b` → `var a; var b` |
| `normalizeobjprops` | med | canonicalize object property forms (shorthand/computed) |
| `movenamedfuncs` | med | hoist/relocate named function declarations |
| `constprop` | med–hard | constant propagation **as a rewrite** (folds constants, still emits IR) |
| `dead_code_elimination` | med–hard | DCE driven by reachability/constprop |
| `extract_prelude` | special | needs a prelude input; lower priority |
| `dynconstprop` | special | dynamic constprop, needs `--dynamic_prelude_path`; lower priority |

(`kNormalizeMemberExpressions` exists in the enum but has **no CLI string**, so it
is not oracle-able via the binary — skip.)

Each pass also ships a handful of golden fixtures under
`vendor/jsir-upstream/maldoca/js/ir/transforms/<pass>/tests/<case>/`
(`input.js` + `output.generated.txt` in FileCheck form) — use these as unit tests
in addition to the test262-scale differential oracle.

### Dataflow analysis (the deeper tier)

- Only **`constant_propagation`** is wired to the CLI (`StringToJsirAnalysisKind`).
- Framework: `analyses/dataflow_analysis.h`, `conditional_forward_dataflow_analysis.h`,
  `per_var_state.h` — a general sparse conditional forward dataflow; constprop is the
  shipped instantiation.
- Output is a **different printer**: the pretty/custom op form plus per-op lattice
  annotations, e.g.
  ```
  %4 = jsir.numeric_literal {#jsir<numeric_literal_extra "1", 1.000000e+00 : f64>, 1.000000e+00 : f64}
    // %4 = 1.000000e+00 : f64
    // State [default = <unknown>] { }
  ```
- 26 golden fixtures under `analyses/constant_propagation/tests/`.

## Recommended sequencing

1. **Round-trip must be green first** (see `cargo run --example parity -- --roundtrip`).
   Transforms that lift back to AST depend on a faithful round-trip.
2. **Easy structural rewrites** (validate the whole transform-oracle loop cheaply):
   `peelparens` → `remove_directives` → `split_sequence_expressions` →
   `split_declaration_statements` → `erase_comments`.
3. **Structural-but-fiddly**: `normalizeobjprops`, `movenamedfuncs`.
4. **Dataflow-powered rewrites** (need a real, if small, analysis but still emit
   plain IR): `constprop`, then `dead_code_elimination`.
5. **Annotated analysis dump**: the `--output_type=analysis` constprop form —
   second printer + state annotations. Biggest lift, do last.

## How to work on a transform with the oracle

### 1. Capture the oracle output for the pass
Extend `gen_oracle.py` (on the box) to also run the pass and store its IR. Add a
column per pass, e.g.:

```python
# in convert(), after the base ast2hir run:
out_peel = run([GEN, f"--input_file={f}",
                "--passes=source2ast,ast2hir,peelparens",
                "--output_type=ir", f"--output_file={tmp}"])
res["peelparens"] = read(tmp) if ok else None
```

Regenerate and pull `test262_oracle.jsonl.gz` (the harness already reads it locally).
Capture all the easy passes in one regen so you don't re-run the box repeatedly.

### 2. Implement the pass as an IR→IR rewrite
New crate `crates/jsir-transforms`. Each pass is a function
`fn peel_parentheses(op: &mut Op)` (or returning a new `Op`) that walks the
`jsir-ir` tree (ops / regions / blocks) and rewrites. These are MLIR-free tree
rewrites — no analysis needed for the structural ones. Re-number SSA values after
any op deletion/insertion so the printer output matches upstream's numbering.

### 3. Diff against the oracle
Add a `--pass <name>` mode to `examples/parity.rs` mirroring the existing loop:
run `source_to_ir`, apply the pass, `op.print()`, compare to the corpus's
`<name>` column with the same `norm`/`deep` categorization (match / differ_offset /
differ_struct / ours_fail) and the same signature histograms.

```
cargo run --release --example parity -- --pass peelparens
```

### 4. Iterate
Use the `differ_struct` / failure-reason histograms to drive fixes, exactly as we
did for `ast2hir` (this is how conversion went 67% → 97%). Keep the per-pass golden
fixtures as fast unit tests in `jsir-transforms`.

### 5. Round-trip the transformed IR (optional but recommended)
For passes that should remain reversible, also check
`transform(ir) → ir_to_source → source_to_ir` stability, reusing the
`--roundtrip` machinery.

## Dataflow analysis: extra work beyond a transform

To oracle `--output_type=analysis` (constant propagation) we need, in order:

1. **The analysis engine**: a forward dataflow fixpoint over the JSHIR CFG (regions
   as nested CFGs), a constant lattice (`Uninitialized` ⊑ const ⊑ `Unknown`), and
   transfer functions per op (literals → const, binary ops fold, joins at merges).
   Mirror `analyses/constant_propagation/analysis.{h,cc}`.
2. **The annotated/pretty printer**: a *second* IR printer producing the custom op
   form (`%4 = jsir.numeric_literal {…}`) plus the `// %N = <value>` and
   `// State […] { … }` comment annotations. This is distinct from our generic
   printer and is the bulk of the work.
3. **Oracle**: capture `--jsir_analysis=constant_propagation --output_type=analysis`
   per file; diff our annotated dump.

Until (2) exists, the cheapest *real-dataflow* signal is the `constprop` /
`dead_code_elimination` **transforms** (step 4 above): they exercise the same
analysis but emit plain IR we can already diff.

## Quick reference: oracle commands (on the box)

```bash
# transform pipeline, IR output
jsir_gen --input_file=X.js --passes=source2ast,ast2hir,peelparens \
         --output_type=ir --output_file=out.mlir

# dataflow analysis, annotated dump
jsir_gen --input_file=X.js --jsir_analysis=constant_propagation \
         --output_type=analysis --output_file=out.txt
```
