# jsir-jslir

JSLIR — a **CFG/SSA dialect on the generic `jsir-ir` op substrate**, the IR the
ported React Compiler passes will run on (see `~/.claude/plans/cuddly-squishing-shore.md`).

## Idea

JSHIR (what `jsir-swc::source_to_ir` produces) already lowers every JS expression
to flat `jsir.*` ops with use-def chains, but keeps control flow **structured**
(`jshir.if_statement`/`while_statement`/… with nested regions). JSLIR keeps the
`jsir.*` instruction ops verbatim and replaces the structured control-flow ops
with a real **CFG**: basic blocks ending in `jslir.*` terminators that name
successor blocks (via the generic `Op.successors`/`BlockId` added to `jsir-ir` in
Phase 1). This is exactly MLIR's `scf → cf` conversion, and it lets us reuse all
of `source_to_ir` / `hir2ast` / `ast2source` unchanged.

```text
source ─source_to_ir→ JSHIR ─build_jslir→ JSLIR ─[React passes]→ JSLIR ─lift_jslir→ JSHIR ─ir_to_source→ JS
```

- **`build_jslir`** flattens structured control flow into a CFG.
- **`lift_jslir`** rebuilds the structured form (so existing codegen emits JS).
- Terminators: `jslir.return`, `jslir.br`, `jslir.cond_br`. Structured-reconstruction
  metadata (the merge/join block, loop marker, brace flags) rides on `cond_br` as
  ordinary attrs — the SPIR-V `OpSelectionMerge`/`OpLoopMerge` approach.

## Graceful degradation

A function `build_jslir` can't lower yet is **passed through unchanged** (still
valid JSHIR), so the round-trip never regresses. `Stats` reports how many
functions actually became a CFG. The oracle (`jsir-react-oracle`) reports this as
"JSLIR coverage".

## Status

**Function shapes lowered:** `function` declarations/expressions, arrows (both
**expression** and block bodies), and object/class methods. **Control flow:**
straight-line, `return`, `if`/`else` (braced + unbraced, nested), `while`, and the
canonical `for (let i = ..; test; update)`. **~94% of all corpus functions** lower
to a CFG with **0 round-trip errors** and no match regression (oracle "JSLIR
coverage"). Every lowered CFG is checked by `verify::verify_cfg`.

**Within-block flattening (Phase 3):**
- Single-binding `let/const/var x = e` declarations → block-level value ops + a
  `jslir.store_local` (React HIR's `StoreLocal`). The lift expands it back to a
  `variable_declaration` referencing the block-level value (`hir2ast`'s global
  value→def index resolves it, no manual re-nesting).
- **Expression control flow** `&&` / `||` / `?:` → real CFG diamonds where the
  result is a **merge-block argument (a phi)** — the first use of block arguments
  / SSA-style value merging (`expr_flatten.rs`). Logicals/ternaries in loop-edge
  blocks (header/back-edge/preheader/latch) stay coarse to avoid tangling loop
  annotations. Validated by `tests/logical.rs` + the round-trip at scale.

Passed through / coarse (next): `for-in`/`for-of`, `do-while`, `switch`,
`try`/`throw`, `break`/`continue`, optional chaining, multi-binding/destructuring
declarations, logicals/ternaries inside loop conditions.

## Next (Phase 3)

- Optional chaining; multi-binding declarations; remaining control flow.
- **SSA for named variables** (`enter_ssa` analog): version `store_local`/reads,
  insert variable phis (block args) at merges.
- Then port the first real React passes onto the flat CFG, gated by hand-written
  per-pass equivalence tests against the upstream Rust port.

## Layout

- `dialect.rs` — op-name constants + terminator constructors + structured metadata.
- `build.rs` — JSHIR → JSLIR (the CFG builder + per-construct lowering).
- `lift.rs` — JSLIR → JSHIR (guided structured reconstruction).
- `verify.rs` — CFG well-formedness.
- `tests/` — round-trip and verifier tests.
