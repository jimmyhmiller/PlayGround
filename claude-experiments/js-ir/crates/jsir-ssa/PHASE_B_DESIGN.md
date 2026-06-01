I have enough to write the design doc. The `def(op, ...)` at lower.rs:407 takes the originating `&IrOp` already — that's the threading point for provenance. Here is the synthesized design.

---

# Phase B Design: IR-Rewrite Memoization (jsir-transforms-style, byte-equivalent to current emit, then control-flow general)

## 0. Goal and non-goals

**Goal.** Replace the string emitter (`crates/jsir-ssa/src/codegen.rs::emit_memoized`, lines 20–97) with a *JSIR→JSIR transform*: read the existing reactive-scope analysis (`ScopeInfo` from `scopes.rs:163`), synthesize new `jsir_ir::Op` nodes (cache decl, dep-compare `if`, hoisted `let`s, slot reads/writes), splice them into the function body op-tree, and print through the existing reversible path `jsir_swc::ir_to_source` (`jsir-swc/src/lib.rs:54`). Same clone-recurse-splice pattern the DCE pass already uses (`jsir-transforms/src/dce.rs:134`).

**Hard constraints (from feedback + findings).**
- No relooper. We do not reconstruct structured control flow from the CFG. The reversible JSIR op-tree *already is* structured; we rewrite it in place and keep the CFG/SSA layer purely as the *analysis oracle* (where scopes/deps/outputs come from).
- No string-codegen extension. `emit_memoized` is frozen as the parity reference but is not the production path going forward; the transform must reproduce its output through real IR.
- No JSIR-purity violation. Every synthesized op must be lowerable by `hir2ast`'s `fn stmt`/`fn expr` (`hir2ast.rs:120`/`:316`); we reuse only existing op names (`jsir.variable_declaration`, `jshir.if_statement`, `jsir.member_expression`/`_ref`, `jsir.binary_expression`, `jsir.assignment_expression`, `jsir.call_expression`, `jsir.identifier`/`_ref`, `jsir.numeric_literal`, `jshir.block_statement`, `jshir.labeled_statement`, `jshir.break_statement`, `jsir.import_declaration`).
- Stubs are hard errors. Any unhandled op shape `return Err(...)` (matching the existing `emit_memoized` hard errors at codegen.rs:22/27/202), never a silent skip.

**The parity gate** is `examples/corpus.rs` `structure(code)` = `(_c(N) cache size, memo-block count)` (corpus.rs:56–71), bucketed agree/mismatch/react_only/ours_only (corpus.rs:296–322). Every step must not regress `agree` and must not introduce `panic`.

---

## 1. Transform interface and data structures

### 1.1 New crate module: `jsir-transforms/src/memoize.rs`

Mirror DCE's entry shape (`dce.rs`). Public entry:

```rust
// crates/jsir-transforms/src/memoize.rs
pub struct MemoizeResult {
    pub file: jsir_ir::Op,      // rewritten program op-tree (ready for ir_to_source)
    pub added_import: bool,     // whether react/compiler-runtime import was prepended
}

/// Rewrite every component/hook function in `file` to its memoized form.
/// `file` is the program op-tree from jsir_swc::source_to_ir.
pub fn memoize_file(file: &jsir_ir::Op) -> Result<MemoizeResult, String>;
```

It walks the program body (the structure from `lower.rs:18–29`: `file.regions[0].blocks[0].ops[0]` is the `jsir.program`, whose `regions[0].blocks[0].ops` is the statement list), finds each `jsir.function_declaration` (descending `export`/`export default` wrappers exactly like `find_function` at lower.rs:33), and for each one whose name `is_component_or_hook` (reuse codegen.rs:242), rewrites its body.

> **Why not route through `jsir-ssa`?** Because the analysis lives there. The transform crate calls *into* `jsir-ssa` for analysis but does the synthesis itself. To avoid a cycle, the analysis-producing functions need to be exposed; see §1.3.

### 1.2 The missing piece: scope → JSIR-statement mapping (`SrcRef`)

Finding `scope-mapping` is the crux: there is **no** back-map today. `ScopeInfo` is expressed entirely in `cfg::Value` and `mutability::Point` (RPO instr index), with zero reference to the JSIR op that produced the instruction. We must thread provenance from JSIR → CFG and preserve it through SSA. Concretely:

**(a) Add a stable per-op id to JSIR.** `jsir_ir::Op` has no NodeId today (findings, `jsir-ir/src/lib.rs:27`). Add:

```rust
// crates/jsir-ir/src/lib.rs (Op struct, line 27)
pub struct Op {
    ...
    pub trivia: Option<Trivia>,
    pub node_id: Option<u32>,   // NEW: stable origin id, None for synthetic ops
}
```

Assign `node_id` monotonically during `ast2hir` build (the `Builder` already mints `ValueId`s; piggyback a parallel counter). This is *not* printed (the textual printer ignores everything but name/operands/attrs/regions per finding `printer`; confirm `print.rs` doesn't touch it) and is ignored by `hir2ast` structural decisions, so it cannot break the 44 golden fixtures or `corpus_hir2ast_round_trip`. **De-risk:** gate this single change behind the round-trip test before anything else (Step 0 below).

**(b) Carry `SrcRef` onto `cfg::Instr`.** Today `Instr = { result, op }` (cfg.rs:57). Add:

```rust
// crates/jsir-ssa/src/cfg.rs:57
pub struct Instr {
    pub result: Option<Value>,
    pub op: Op,
    pub src: Option<SrcRef>,    // NEW
}
pub struct SrcRef {
    pub stmt_node_id: u32,      // the enclosing *statement* op's node_id
    // (span optional; not needed for the structure metric)
}
```

`Cfg::push`/`push_effect` (cfg.rs:252/259) gain a `src: Option<SrcRef>` param (or a `push_with_src`). **The threading point is `Lower::def`** (lower.rs:407): it already receives `op: &IrOp`. But `def` receives the *expression* op, not the enclosing statement. So `Lower` carries `cur_stmt_node_id: Option<u32>`, set at the top of `lower_op` when the op is a statement root (0-result ops: `jsir.expression_statement`, `jsir.variable_declaration`, `jshir.if_statement`, etc. — the same set `hir2ast.stmts()` selects at hir2ast.rs:83), and read inside `def`/`push`/`push_effect`. Synthetic SSA values (undef at ssa.rs:250, phis at ssa.rs:182) get `src: None`.

**(c) Preserve through SSA.** `ssa::materialize` (ssa.rs:266) keeps `Instr` objects and only rewrites operands/deletes ReadVar/WriteVar — retain the `src` field across the rewrite (it rides the surviving `Instr`, not a value-keyed side-table, exactly as finding `scope-mapping` recommends, because SSA mints new values and deletes instrs).

**(d) Derive the scope→statement range.** Given a `ScopeInfo`, the set of owning instructions is *already computed* by `emit_memoized` (codegen.rs:38–52: an instr is owned if its result ∈ `scope.values` or it is a `StoreMember` whose `obj` ∈ `scope.values`). The transform reuses that exact ownership rule, then maps each owned instr → `instr.src.stmt_node_id`, and unions to a **contiguous set of statement node_ids**. That set identifies the JSIR statements to lift out of the body and re-emit inside the synthesized `if`.

### 1.3 Exposing the analysis from `jsir-ssa`

Add a single façade in `jsir-ssa` that returns everything the transform needs *without* the string emit:

```rust
// crates/jsir-ssa/src/lib.rs  (new pub fn)
pub struct MemoPlan {
    pub fn_name: String,
    pub infos: Vec<scopes::ScopeInfo>,
    pub cfg: cfg::Cfg,                       // carries Instr.src now
    pub single_block: bool,                  // blocks.len()==1 && Term::Ret
}
pub fn plan(file_fn: &jsir_ir::Op) -> Result<MemoPlan, String>;
```

`plan` runs `lower_function` → `ssa::construct` → `mutability::analyze` → `scopes::analyze` (the same chain as `compile` at codegen.rs:257–267) and returns the data. The transform consumes `MemoPlan` and never touches strings.

### 1.4 IR builder helpers (the synthesis toolbox)

A small `Builder` in `memoize.rs` modeled on `jsir-convert`'s `Builder{next:u32}.fresh()` (finding `jsir-statements`, lib.rs:1498), seeded *past the current max ValueId in the file* so synthesized `results` never collide with `hir2ast.index()` (finding gotcha). Helpers, each citing the ast2hir shape it must mirror:

| Helper | Produces | Mirrors |
|---|---|---|
| `import_c()` | `jsir.import_declaration` (`source`=StringLiteralKey `react/compiler-runtime`, `specifiers`=`[ImportSpecifier{imported:c, local:_c}]`) | lib.rs:305, hir2ast.rs:973, to_swc.rs:65 |
| `cache_decl(n)` | `const $ = _c(n);` = `jsir.variable_declaration` kind=`const` → declarator(id `$`, init `jsir.call_expression`(`_c`, numeric n)) → `jsir.exprs_region_end` | lib.rs:436, call lib.rs:557, numeric lib.rs:473 |
| `let_hoist(names)` | `let a, b;` = `jsir.variable_declaration` kind=`let`, declarators with id only (no init) | lib.rs:436 |
| `cache_read(i)` | `$[i]` = `jsir.member_expression` computed, obj=`jsir.identifier $`, prop=`jsir.numeric_literal i` | lib.rs:848, hir2ast.rs:543 |
| `cache_write_ref(i)` | `$[i]` as l-value = `jsir.member_expression_ref` | lib.rs:848 |
| `dep_cmp(read_i, dep_expr)` | `$[i] !== dep` = `jsir.binary_expression` operator_=`!==` | lib.rs:494 |
| `sentinel_cmp(read_i)` | `$[i] === Symbol.for("react.memo_cache_sentinel")` | call+member+bin |
| `or_chain([..])` | `a \|\| b \|\| c` = nested `jsir.binary_expression` `\|\|` | lib.rs:494 |
| `if_stmt(test, then_ops, else_ops)` | `jshir.if_statement` operand[0]=test, region[0]=consequent (a `jshir.block_statement`), region[1]=alternate (`Region::default()` if empty — finding gotcha) | lib.rs:204 |
| `assign_stmt(lhs_ref, rhs)` | `$[i] = x;` = `jsir.assignment_expression` operator_=`=` wrapped in `jsir.expression_statement` | lib.rs:545 |

All synthesized ops carry `trivia: None, node_id: None` (legal; matches how `jsir.exprs_region_end` is built with plain `Op::new` — finding `printer`). Identifier/import *attribute* structs (`IdentifierAttr`, `StringLiteralKeyAttr`, `ImportSpecifierAttr`) need their mandatory span fields set to `0`/dummy — safe because `to_swc` ignores them (finding `printer`, attr.rs:183).

**Critical: a re-expression problem.** The string emitter prints expressions from the *CFG* (`expr()` at codegen.rs:167) using `_v{id}` names. The IR transform must instead *move the original JSIR expression ops* (which already exist in the body block) into the `if`-consequent, not re-synthesize them. So for the straight-line case the synthesis is mostly **relocation + wrapping**, not expression rebuilding — we only build the scaffolding (cache decl, `if`, slot reads/writes, hoists). This is what keeps us byte-faithful to source identifiers (no `_v` renaming) and is *more* correct than the string path.

---

## 2. Straight-line algorithm (reproduce current emit through IR)

Operates on one function's body statement-list: `block_statement.regions[0].blocks[0].ops` (finding `jsir-statements`: function region[1] → block_statement → region[0]). Precondition mirrors codegen.rs: `MemoPlan.single_block == true`, else `Err` (no fabrication).

Let `emitted = infos.filter(|i| !i.outputs.is_empty())` (codegen.rs:34). Build, per emitted scope, the owned-instr set and `stmt_node_id` set via §1.2(d). Then:

1. **Prepend import** to the program body (once per file): `import_c()` as first statement (finding `printer`).
2. **Prepend cache decl** as the function body's first statement: `cache_decl(N)`, `N = Σ (deps.len()+outputs.len())` over emitted scopes (codegen.rs:64). Slot numbering threads a running counter across scopes in scope-emit order (codegen.rs:74, 108–111) — preserve exactly.
3. **Hoist outputs:** collect all `outputs` values across emitted scopes, map each to its JSIR-declared name (the original `jsir.identifier_ref`/declarator name carried in the moved statements), dedupe, **sort lexicographically by rendered name string** (codegen.rs:67–68 — this is load-bearing and "buggy-looking": `_v10 < _v2`; but since we now use *real source names*, the sort is over real names, which is what React does anyway — verify the metric doesn't depend on it; if `agree` drops, re-impose the exact sort). Emit one `let_hoist([...])` after the cache decl.
4. **Walk the original statement list.** For each statement op, determine its owning scope by its `node_id` (via the per-scope `stmt_node_id` sets). 
   - Statement **not owned** by any scope → keep in place verbatim (clone, like DCE).
   - Statement owned by scope `s` and it is the **last** statement of `s`'s range → emit `s`'s memo `if` here, splicing *all* of `s`'s owned statements (in original order) into the consequent block, with slot reads/writes appended.
   - Statement owned by scope `s` but **not last** → drop from this position (it is relocated into the `if`). This mirrors codegen.rs:80.
5. **Build the memo `if`** for scope `s` (mirror `emit_scope`, codegen.rs:99):
   - **test:** deps non-empty → `or_chain( deps.map(|(i,d)| dep_cmp(cache_read(dep_base+i), <dep expr op>)) )`; deps empty → `sentinel_cmp(cache_read(out_base))`. The dep "expr op" is the *existing* value-producing op in the body — but a dep is a value defined *before* the scope, so its identifier already exists in scope; reference it by re-using a `jsir.identifier` to that name (deps are always reactive non-const values, i.e. params/derived vars with names; if a dep has no source name this is an `Err`, not a silent skip).
   - **consequent block ops:** `[ assign_stmt($[dep_base+i] = dep) for each dep ] ++ [ the relocated owned statements ] ++ [ assign_stmt($[out_base+j] = output) for each output ]`.
   - **alternate block ops:** `[ assign_stmt(output = $[out_base+j]) for each output ]`.
6. **Return statement** stays verbatim (it references a hoisted `let` name, which survives — finding `string-codegen-baseline`).

**Sentinel change (in scope, deliberate).** The string path uses `_e = Symbol('empty')` (codegen.rs:235). The IR path must emit `Symbol.for("react.memo_cache_sentinel")` for the empty-deps guard (finding `react-controlflow-emit` RULE 6) to be byte-closer to React. Because the **parity metric is `structure()`** (cache size + block count), the sentinel text does not affect the gate — so this can land in Step 1 without risking `agree`. Do NOT change the early-return sentinel until §3.2.

**Gate for §2:** `corpus` `agree` count must be **≥** the current emit_memoized baseline (the IR path replaces the memoized branch of `compile`; the pass-through branch at codegen.rs:263 stays as-is — do not route pass-through through the transform, finding gotcha). `panic` must stay 0.

---

## 3. Generalization to control flow

The CFG/SSA analysis already handles branches/loops (lower.rs builds CondBr/Br, `reactive_values` propagates through block-args at scopes.rs:120–137). What's missing is (a) `MemoPlan.single_block` is false, so today we `Err`; (b) the scope→statement-range derivation must cope with statements spanning branches. Because we rewrite the **JSIR op-tree** (which is *already* structured), we never need a relooper — we wrap *whole JSIR control constructs*.

### 3.1 if/else (no early return) — RULE 2

A scope's `stmt_node_id` set, when the scope's mutable range spans both branches plus the join (e.g. `obj-mutated-after-if-else`), will include the node_id of the **enclosing `jshir.if_statement`** itself plus the post-join statement (because the owning instrs come from inside both regions and after). The transform:
- Computes the *minimal contiguous statement range* in the body that covers all owned statements. If that range includes a `jshir.if_statement` whose *interior* statements are owned, the whole `if_statement` op is relocated into the memo consequent unchanged (it is just another statement to move — §2 step 4 already moves statement ops; an `if_statement` is one).
- **Contrast** `obj-literal-cached-in-if-else`: here each branch builds an independent object with disjoint deps and no post-join mutation. The analysis (`scopes.rs` merge logic, lines 300–359) keeps them as separate scopes, each owning only statements *inside one branch*. So the memo `if` for each scope nests *inside* that branch, and there is no outer guard. This falls out naturally: the per-scope statement range is confined to one region, so we recurse the §2 walk **into each region's block.ops** (clone-recurse, exactly DCE's `transform_block_ops` at dce.rs:146).

So §3.1 is: make the body walk **recursive over regions** (like DCE), and let each scope's memo `if` be emitted at the block level where its owned statements live. No new rule beyond "recurse into regions; emit each scope's guard at the deepest block containing all its owned statements."

**Gate:** `agree` non-regressing; expect new agrees on `obj-mutated-after-if-else`, `obj-literal-cached-in-if-else`.

### 3.2 Early returns — RULE 3 (sentinel + labeled break)

When a scope's owned statement range contains a `jsir.return_statement` whose value must be cached, we cannot `return` inside the guard. Transform (finding `react-controlflow-emit` RULE 3, fixture `early-return-within-reactive-scope.expect.md:43`):
- Allocate output temp name `t0` (compiler temp; reuse React's `t{n}` naming for synthesized temps — this is new naming, but only for *synthetic* outputs; user vars keep source names).
- `let t0;` hoist (§2 step 3 already hoists; add synthetic temps).
- First op in consequent: `assign_stmt(t0 = Symbol.for("react.early_return_sentinel"))`.
- Wrap the scope body in `jshir.labeled_statement` label=`bb0` (finding catalog) whose body region is a `jshir.block_statement`.
- Rewrite each `jsir.return_statement` inside that block: `return e` → `[assign_stmt(t0 = e), jshir.break_statement(label bb0)]`; bare `return` → `t0 = undefined; break bb0;`. (This is a structural rewrite of the moved statements before splicing — operate on the cloned op subtree.)
- After the guard: `if_stmt( bin(t0 !== Symbol.for("react.early_return_sentinel")), [return_statement(t0)], [] )`.
- All early returns in one scope share `t0`/`bb0` (RULE 3). When both early-return and fall-through outputs are live, cache both in separate slots and restore both in the else (RULE 3, `partial-early-return`).

Requires: confirming `hir2ast` lifts `jshir.labeled_statement` and `jshir.break_statement` (finding catalog lists them as existing op names — verify `fn stmt` handles them; if not, that's a real `hir2ast` addition, not a stub). The **early-return sentinel** (`react.early_return_sentinel`) is a *distinct* symbol from the memo-cache sentinel (finding RULE 6 gotcha) — keep them separate.

**Gate:** `early-return-within-reactive-scope`, `conditional-early-return`, `partial-early-return*`, `repro-useMemo-if-else-both-early-return`. `early-return.expect.md` (trivial `if(cond) return;` with nothing to memoize) must stay **NoMemo** — do not over-apply (finding RULE 8 gotcha).

### 3.3 Scope alignment — RULE 5

The hardest rule. A scope that *starts inside* one branch but is *used after* the join must have its guard hoisted to wrap the entire control construct (fixtures `align-scope-starts-within-cond`, `align-scopes-reactive-scope-overlaps-if/-label`). The algorithm (RULE 5, `align-scopes-nested-block-structure.expect.md:118`): recursively visit all scopes accessed between a block and its fallthrough, and **extend the range of any scope overlapping an active block/fallthrough pair** (transitive).

This is fundamentally a **mutable-range / scope analysis** decision, not a codegen decision — so it belongs in `scopes.rs` (extend `analyze` to compute alignment, widening `Scope.start/end` and pulling the control-construct's statements into the scope's owned set), **not** in the transform. The transform then just sees a scope whose `stmt_node_id` set already includes the whole `if`/`label` block, and §3.1/§3.2 machinery wraps it. De-risk by implementing alignment behind the existing `Point`-range model (`mutability.rs:22`) and validating against the three `align-*` fixtures one at a time.

Where alignment forces an else-branch `return null` to become the sentinel early-return (`align-scope-starts-within-cond`), it composes with §3.2.

### 3.4 Loops — RULE 7

Non-returning loop: wrapped like any statement range — the `jshir.for_of_statement`/`for_statement` op is relocated into the memo consequent (§2 walk handles it as one statement). Loop with cached return: the `bb0` label is placed *on the loop* (`bb0: for (...) {...}`), and `return` inside → `t1 = e; break bb0;` (RULE 7, `repro-memoize-for-of-collection-when-loop-body-returns`). Same labeled-break machinery as §3.2, just the label attaches to the loop op rather than a synthetic block. The loop's memoized iterable, if any, is computed in a prior guard and passed as a dep (already handled by the dep machinery). **Gate:** `for-of-nonmutating-loop-local-collection`, `repro-memoize-for-of-collection-when-loop-body-returns`.

---

## 4. Closures (memoizing a nested function as a value)

A nested `jsir.function_expression`/arrow that is a memoized output is just another allocation value flowing through. The key enabler is that **the function-expression op survives the transform intact** (we relocate it, never rebuild it), so its parameter and body identifiers print correctly through `hir2ast`→`to_swc` (finding `printer`: names come from the original ops, not from CFG `_v` naming). Concretely:
- `mutability.rs`/`scopes.rs` must treat a `jsir.function_expression` result as an allocation candidate (today `allocations` at scopes.rs:83 covers `MakeObject`/`MakeArray`/`Call` — **add the function-expression op-kind** so a closure becomes a memoizable scope value). This is a real analysis fix, gated by a closure fixture.
- The closure's **captured deps** are its free reactive variables; `scope_deps` (scopes.rs:307) already computes deps from operands, but a function-expression's captured names are not CFG operands today (the body is lowered separately, if at all). This is the one place needing care: the lowering of a nested function into the CFG must expose its free-variable reads as operands of the closure value (or the analysis must scan the function body region for free reactive names). Until that exists, **closures with captures must `Err`**, not silently miss deps (purity rule). Implement capture-as-operand, then the existing dep/output machinery memoizes the closure value with no further codegen changes.

The transform side needs **zero** new logic for closures beyond §2 (relocate the function op into the guard, cache its value) — the win is entirely that the IR printer preserves names. This is the structural payoff of doing an IR rewrite instead of string codegen.

---

## 5. Incremental, independently-gated plan

Each step is a real fix, gated by `cargo run -p jsir-ssa --example corpus -- --json` (agree non-decreasing, panic == 0). No relooper, no string-codegen extension, no purity violation.

| Step | Change | Files (anchors) | Gate |
|---|---|---|---|
| **0** | Add `node_id: Option<u32>` to `jsir_ir::Op`; assign in `ast2hir`; ensure printer/`hir2ast` ignore it. | jsir-ir/src/lib.rs:27; jsir-convert builder | `corpus_hir2ast_round_trip` + 44 golden fixtures unchanged; corpus unchanged |
| **1** | Thread `SrcRef` onto `cfg::Instr`; `Lower` tracks `cur_stmt_node_id`; preserve through `ssa::materialize`. Expose `jsir_ssa::plan()`. | cfg.rs:57/252; lower.rs:407/194; ssa.rs:266; jsir-ssa/src/lib.rs | corpus unchanged (analysis-only; no emit change yet) |
| **2** | New `jsir-transforms/src/memoize.rs`: builder helpers + straight-line transform; route `compile`'s memoized branch through it; switch empty-deps guard to `Symbol.for("react.memo_cache_sentinel")`. | memoize.rs (new); codegen.rs:257 (reroute) | `agree` ≥ baseline; panic 0; structure-identical on all currently-agreeing fixtures |
| **3** | Recursive region walk → if/else (RULE 2). | memoize.rs | new agrees: `obj-mutated-after-if-else`, `obj-literal-cached-in-if-else`; no regression |
| **4** | Early-return sentinel + labeled break (RULE 3, 6); verify/extend `hir2ast` for `labeled_statement`/`break_statement`. | memoize.rs; hir2ast.rs:120 | new agrees: `early-return-within-reactive-scope`, `conditional-early-return`, `partial-early-return*`; `early-return` stays NoMemo |
| **5** | Scope alignment in analysis (RULE 5). | scopes.rs:174/300 (analyze); mutability.rs | new agrees: `align-scope-starts-within-cond`, `align-scopes-*-if/-label` |
| **6** | Loops (RULE 7). | memoize.rs | new agrees: `for-of-nonmutating-*`, `repro-memoize-for-of-*` |
| **7** | Closures: function-expression as allocation + capture-as-operand (RULE: closures). | scopes.rs:83; lower.rs (nested fn lowering) | new agrees on closure fixtures; captures-without-support `Err` not silent |

The two-level node verification (`tests/codegen.rs:39`, `tests/jsx.rs:43`: run emitted JS under node for semantic + reference-stability) must also pass at each step that changes emit — keep those tests pointed at the new path.

---

## 6. Top risks and de-risking

1. **`node_id` breaks byte-exact round-trip / golden fixtures.** *Risk:* printer or `hir2ast` accidentally observes the new field, or serialization changes. *De-risk:* Step 0 is isolated; field is `Option`, defaults `None` for all existing build paths until `ast2hir` opt-in; gate solely on `corpus_hir2ast_round_trip` + the 44 fixtures *before* any transform work. If round-trip moves, the field is the only suspect.

2. **Scope→statement mapping is non-contiguous or empty after SSA.** *Risk:* SSA deletes/renames instrs (ssa.rs:266), so an owned instr's `src` is lost, or owned instrs span a hole. *De-risk:* preserve `src` on the surviving `Instr` (not a value-keyed map — finding `scope-mapping` gotcha); add an assertion in `memoize.rs` that each scope's `stmt_node_id` set is contiguous in the body block (else `Err` with the fixture name — surfaces as a known coverage gap, not a panic). Validate the mapping on the *currently-agreeing* straight-line fixtures in Step 2 (where the answer is already known from emit_memoized).

3. **The lexicographic `let`-sort / slot-numbering is load-bearing but "buggy".** *Risk:* re-implementing with real source names changes `agree` via cache-size or block-count drift. *De-risk:* Step 2 reproduces slot numbering (codegen.rs:74,108–111) and the sort (codegen.rs:67) *exactly first*; only after `agree` ≥ baseline do we consider normalizing. The metric (`structure()`) ignores name order, so this is likely a non-issue — confirm empirically, don't assume.

4. **Scope alignment (RULE 5) under-/over-hoists.** *Risk:* hardest analysis; wrong hoist → wrong block count → mismatch (worse than react_only). *De-risk:* implement in `scopes.rs` (analysis, where ranges live), one `align-*` fixture at a time; if a fixture can't be aligned soundly yet, `Err` (stays react_only) rather than emitting a wrong structure (which would show as `mismatch`). Mismatch is the signal we watch.

5. **Closure capture deps silently missing.** *Risk:* a closure memoized without its captured-variable deps → unsound (cached stale). *De-risk:* implement capture-as-operand before enabling closure memoization; until then, function-expression scopes with any free reactive variable `Err`. Never emit a closure guard with incomplete deps.

6. **`hir2ast` lacks a synthesized op shape.** *Risk:* `labeled_statement`/`break_statement`/`member_expression_ref` not lifted → "unsupported op" at print. *De-risk:* before Step 4, grep `hir2ast.rs fn stmt` (hir2ast.rs:120) for each op name we synthesize; any gap is a real `hir2ast` case to add (not a stub) and is itself round-trip-gated. Build a unit test that round-trips a hand-built memo `if`/labeled-break op subtree through `ir_to_source` *in isolation* before wiring it into the pass.

7. **Concurrent worktree edits (this task could not build).** *Risk:* anchors drift. *De-risk:* all anchors above are by symbol name + current line; the implementation workflow should re-grep symbol names (`emit_memoized`, `emit_scope`, `Lower::def`, `ScopeInfo`, `allocations`) rather than trust absolute lines.

---

**Bottom line:** the design threads one new stable id (`node_id`) JSIR→CFG→SSA, reuses the *existing* reactive-scope analysis verbatim as the oracle, and replaces only the *synthesis* half — relocating original JSIR statement ops into synthesized `jshir.if_statement`/`jshir.labeled_statement` scaffolding via the proven DCE clone-recurse-splice pattern, printing through the real reversible path. Control flow is handled by wrapping whole JSIR constructs (no relooper); alignment lives in analysis; closures fall out for free once names survive the printer.

Key files: `crates/jsir-ir/src/lib.rs:27`, `crates/jsir-ssa/src/cfg.rs:57`, `crates/jsir-ssa/src/lower.rs:407`, `crates/jsir-ssa/src/ssa.rs:266`, `crates/jsir-ssa/src/scopes.rs:163`/`:83`/`:174`, `crates/jsir-ssa/src/codegen.rs:20`/`:99`/`:242`/`:257` (parity reference, frozen), `crates/jsir-transforms/src/dce.rs:134` (rewrite template), `crates/jsir-convert/src/lib.rs:436`/`:204`/`:305`/`:557` (synthesis shapes), `crates/jsir-convert/src/hir2ast.rs:120`/`:83` (inverse contract), `crates/jsir-swc/src/lib.rs:54` (print path), `crates/jsir-ssa/examples/corpus.rs:56` (gate).