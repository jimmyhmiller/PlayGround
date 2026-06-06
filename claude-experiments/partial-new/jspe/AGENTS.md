# jspe — agent task conventions

Porting RustPE to JS. Read `../docs/jspe-port-plan.md` for the big picture.

## The pattern: every task is one dispatch-table entry (a pure-ish function + a test)

We do NOT write big functions with internal `switch`es. Instead each variant is an
independent entry in a shared table:

```js
// dispatcher (committed in the skeleton; agents DON'T touch it)
function emit(e) { return EMIT[e.tag](e, emit); }   // EMIT is the table

// one task = implement one entry:
EMIT.Bin = (e, emit) => "(" + emit(e.a) + " " + e.op + " " + emit(e.b) + ")";
```

Tables auto-assemble, so the modules stitch together with zero merge coordination.
A bug is localized to one cell.

## What a task is

Each task in `tasks.json` has:
- `id` — stable id, also the test filename.
- `file` — the file to edit (the table lives there, pre-stubbed).
- `symbol` — the exact entry to implement, e.g. `EMIT.Bin` or `STEP.GetIndex`.
- `signature` — the exact JS signature + the shapes it sees (from `contracts.js`).
- `spec` — what it must do.
- `rustRef` — where the behavior lives in the Rust source (the oracle of truth).
- `deps` — task ids that must land first (usually `[]`; tables are independent).
- `test` — `test/<id>.test.js`, which the agent ALSO writes (cases from `rustRef`).
- `wave` — scheduling band.

## How an agent does a task (your script wires this up)

1. Read `contracts.js` (frozen data shapes — never edit), the stub in `file`, the
   `spec`, and the Rust at `rustRef`.
2. Replace the stub `throw UNIMPLEMENTED(...)` with the implementation.
3. Write `test/<id>.test.js` with `>= 3` cases derived from `rustRef`.
4. `node test/run.js <id>` must pass. For integration-level tasks, the differential
   harness (`node test/harness.js`) must stay green.

## Rules

- Edit ONLY your `file` and your `test/<id>.test.js`. Never edit `contracts.js`,
  the dispatchers, other entries, or the runner.
- Pure functions where possible; if you need a State, use the fixtures in
  `test/fixtures.js`.
- The Rust at `rustRef` is the spec. When in doubt, match its behavior exactly.
- Leave a one-line comment citing `rustRef`.
