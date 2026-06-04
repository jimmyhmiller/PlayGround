# React parity status

Current source of truth for React Compiler work in this crate.

## Measured state

Last audited: 2026-06-03.

Command:

```sh
cd crates/jsir-ssa
./oracle/run-corpus.sh --json
```

`run-corpus.sh` saves the upstream React compiler result under
`oracle/.react-cache/` via `REACT_CACHE_DIR`, so the first run populates the
official oracle outputs and later runs only rebuild/re-run our side. Delete that
directory to refresh upstream output after changing `react-cc.js`, fixture
contents, or React compiler dependencies.

Representative result. Current runs still fluctuate slightly, observed in the
**236-237 / 715** range after deterministic ordering fixes in scope sorting and
escape DFS traversal, with upstream output cached:

```json
{"total":1122,"react_memoized_universe":715,"agree":236,"mismatch":194,"react_only":285,"parser_skip":10,"ours_only":111,"neither":286,"panic":0,"agreement_pct":33.01}
```

`cargo test -q -p jsir-ssa` passes. The corpus is the gate that matters for
React parity.

## Diagnosis

Do not restart the whole repository. Keep:

- the JSIR copy and SWC converter,
- CFG/SSA lowering and verification,
- the in-repo official React oracle wrapper,
- the in-place JSIR rewrite path.

Do restart the React parity methodology. The current React-specific scope
analysis is a partial/custom blend and should be treated as WIP reference code,
not as a complete compiler pipeline.

The dominant misses are:

- wrong escape/output decisions,
- missing dependency propagation, especially access paths and control-flow reads,
- block-scope alignment gaps for control flow,
- lowering gaps for unsupported JavaScript forms.

## Rules for future work

1. Port upstream React Compiler passes faithfully instead of reverse-engineering
   individual fixtures.
2. Land one pass at a time behind the corpus gate.
3. Before each change, snapshot:

   ```sh
   ./oracle/run-corpus.sh --list agree | sort > /tmp/agree_before.txt
   ```

4. After each change, snapshot:

   ```sh
   ./oracle/run-corpus.sh --list agree | sort > /tmp/agree_after.txt
   comm -23 /tmp/agree_before.txt /tmp/agree_after.txt
   ```

   The `comm -23` output must be empty unless the regression is deliberately
   explained and accepted.

5. Keep unhandled shapes as loud bails or pass-throughs. Do not silently emit
   approximate memoization.

## Next implementation order

1. Faithful `PruneNonEscapingScopes`: replace the current partial
   `prune_non_escaping` / output filter in `scopes::analyze`.
2. Faithful `PropagateScopeDependencies`: replace operand-derived deps with
   access-path dependencies, including control-flow reads inside a scope.
3. Block-scope alignment: make surviving scopes line up with whole structured
   control-flow regions before relaxing memoize-plan multi-block bails.
4. Only then revisit emitter placement gaps such as anonymous multi-output and
   non-contiguous source statements.

The old string emitter in `codegen.rs` is retained for tests/reference only.
`codegen::compile` should continue using `memoize_plan::memoize_inplace`.
