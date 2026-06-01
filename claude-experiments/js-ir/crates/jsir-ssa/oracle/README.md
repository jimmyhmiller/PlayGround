# In-repo React Compiler oracle

The parity corpus (`crates/jsir-ssa/examples/corpus.rs`) compares our memoization
output against the React Compiler. It needs two things that used to live only in
`/tmp/react-rust` (a Rust port of the React Compiler whose origin was never
recorded, and which got wiped whenever `/tmp` was cleared). They now live here,
version-controlled:

- `fixtures/` — React's `babel-plugin-react-compiler` test fixtures (the `.js`
  inputs the corpus compiles on both sides), vendored from `facebook/react`.
- `react-cc.js` — the oracle CLI. Wraps the **official**
  `babel-plugin-react-compiler` (the ground truth the old Rust port approximated),
  mirroring React's own `snap` test transform: it reads the fixture's first line
  for config pragmas (`parseConfigPragmaForTests`, `compilationMode: 'all'`),
  parses with the right dialect (typescript/flow + jsx), and runs the plugin.
  Reads source on stdin, prints compiled JS on stdout, exits nonzero on a
  compiler error (which the corpus buckets as a React bail).

## Setup

```sh
cd crates/jsir-ssa/oracle && npm install
```

## Run the corpus

```sh
crates/jsir-ssa/oracle/run-corpus.sh                 # summary
crates/jsir-ssa/oracle/run-corpus.sh --json          # machine-readable
crates/jsir-ssa/oracle/run-corpus.sh --list mismatch # inspect a bucket
```

The script sets `REACT_CC` and `REACT_FIXTURES` to the in-repo paths, so no
`/tmp` clone is required. `corpus.rs` itself is unchanged (its default paths
still point at `/tmp`; the script overrides them via env).

## Fidelity notes

The comparison is **structural**: `_c(N)` cache size + count of `if ($[…])` memo
blocks. The oracle differs from React's full `snap` runner in two minor,
documented ways that don't affect that structure for the common case:

- it skips the `fbt`/`idx` babel plugins (only relevant to a handful of fbt
  fixtures), and
- it omits the shared-runtime `moduleTypeProvider` (extra type hints a few
  fixtures rely on), so a small number of type-driven fixtures may bail where the
  full runner would memoize.
