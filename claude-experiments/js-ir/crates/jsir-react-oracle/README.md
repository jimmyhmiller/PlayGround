# jsir-react-oracle

The byte-exact gate for porting the React Compiler onto JSLIR (see
`~/.claude/plans/cuddly-squishing-shore.md`). It answers one question for every fixture:
**does our pipeline's final JS match the upstream React Compiler's, byte-for-byte?**

## What's the oracle

`cache/` holds the in-tree TS compiler's output for every snap fixture, extracted from each
`.expect.md`'s `## Code` (compiled) / `## Error` (bail) block at the pinned commit in
`UPSTREAM.lock` (NOT npm `1.0.0`, which skews from the Rust port we copy pass logic from).
`fixtures/` holds the matching inputs. `## Code` is already snap-normalized, so it is the
ground truth verbatim — 1426 code fixtures, 299 error fixtures.

Byte-exact comparison only works through a shared formatter, because our only JS emission path is
swc (no native AST→JS printer) and swc formats differently from babel. Both sides pass through
`tools/normalize.js` — the exact snap formatter, `prettier.format(code, {semi:true, parser:
'babel-ts'|'flow'})`, pinned to prettier 3.3.3. Proof it matches: `normalize(## Code) == ## Code`
for all 1426 code fixtures (idempotent).

## Use

```sh
npm install                  # once (prettier + babel)
./run-oracle.sh              # summary: match / mismatch / our_error / error_fixture
./run-oracle.sh --json
./run-oracle.sh --list mismatch
./run-oracle.sh --filter useMemo --limit 50
./run-oracle.sh --regen      # re-extract cache/ from the pinned upstream
```

## Scope: ES + JSX only

TypeScript and Flow are explicitly **out of scope** (memoization/reactivity is dialect-agnostic;
type syntax is incidental). Flow fixtures (`@flow` / `.flow`) and any fixture that doesn't parse as
ES+JSX (TypeScript type syntax) are skipped and excluded from the denominator.

## Buckets

In scope (ES+JSX):
- **match** — `normalize(ours) == cache` ✅
- **mismatch** — code fixture, output differs (the frontier: needs JSLIR compilation)
- **our_error** — parsed as ES+JSX but lowering/emit failed (a real gap)
- **normalize_error** — our emitted JS didn't parse
- **error_fixture** — oracle expects a compiler bail; not modeled until passes emit errors

Out of scope (skipped):
- **skip_flow** — `@flow` / `.flow`
- **skip_unparsed** — TypeScript type syntax (not ES+JSX-parseable)

## Status

The pipeline seam (`compile_fixture` in `src/main.rs`) is currently an **identity round-trip**
(`jsir_swc::source_to_ir` → `ir_to_source`) as a placeholder. As the JSLIR port lands
(BuildJSLIR → passes → LiftJSLIR), replace that one function; `match` climbs and `frontier`
shrinks. Baseline today: 23 match, 1284 frontier, TS/Flow skipped (56 flow + 82 TS), and **0
in-scope pipeline errors** — every ES+JSX fixture either matches or is a clean mismatch awaiting
JSLIR. The final milestone is re-earning upstream's output on every in-scope fixture.

Two `jsir-swc` fixes landed while standing this up: JSX string attributes with a literal newline are
now JS-escaped on emit (`jsx.rs`, was an unterminated string), and `source_to_ir` now rejects parses
that only succeeded via swc error-recovery `Invalid` nodes (`from_swc.rs`), so TypeScript that swc
silently recovers (e.g. `f<T>()`) is correctly classified as out-of-scope instead of failing in
lowering.
