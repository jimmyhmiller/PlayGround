# Ground truth source: rolldown + esbuild agreement (NOT unbundled Node)

This fixture is an eager mixed ESM/CJS cycle: `esm.js` statically imports
`legacy.cjs`, and `legacy.cjs` calls `require("./esm.js")` at module top
level while `esm.js` is still evaluating.

Unbundled Node v22 refuses to run this graph at all:

    Error [ERR_REQUIRE_CYCLE_MODULE]: Cannot require() ES Module
    .../esm.js in a cycle. ... A cycle involving require(esm) is not
    allowed to maintain invariants mandated by the ECMAScript
    specification.

That is a module-loader admission policy, not an execution semantics we can
observe, so Node cannot serve as the execution ground truth here. Every
bundler (including diffpack's own oracle for the equivalent fixture)
linearizes this cycle and executes it successfully.

Per the suite's documented fallback, `expected.txt` for this fixture is
generated from the agreement of rolldown 1.2.0 and esbuild (the runner's
`--update-expected` mode builds with both, runs both, and requires identical
stdout and exit class before recording). The presence of this file is the
marker that switches `--update-expected` to that mode.

Interpretation note: a PASS here means "matches the consensus bundler
linearization of the cycle", not "matches Node". Node itself would reject
the program.
