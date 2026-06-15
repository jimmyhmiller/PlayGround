# React Compiler HIR vs. JSIR (JSHIR / JSLIR): a true comparison

> Question being answered: can we write pass code that runs on **both** the React
> Compiler's HIR and our JSLIR without converting between them — and if not, how
> easy is the conversion?
>
> Short answer: **the CFG skeleton is genuinely shared and pass logic that lives
> at the basic-block / terminator level can be written once against a common
> interface.** But the two disagree on three load-bearing things — the
> *instruction model* (React is fully scalar/SSA; JSLIR is coarse AST-faithful
> ops, SSA as a side table), *value identity* (React `Place`→`Identifier` vs JSIR
> `ValueId`+resolved symbol), and the *analysis substrate* (React carries
> effects / mutable-ranges / reactive-scopes; JSIR carries none). So "works on
> both unmodified" is true only for the CFG layer; the instruction and analysis
> layers need an adapter. The conversion itself is mechanical in the
> React→JSLIR direction and is essentially what `jsir-jslir` is already building.

This doc is grounded on **both sides by reading (and compiling) the actual
source**: the JSIR side as it exists in this repo (`crates/jsir-ir`,
`crates/jsir-jslir`), and the React Compiler HIR — both the TypeScript original
(`facebook/react` @ `e730b5e`,
`compiler/packages/babel-plugin-react-compiler/src/HIR/`: `HIR.ts`,
`HIRBuilder.ts`, `BuildHIR.ts`, `EnterSSA.ts`) **and its Rust port**, which is now
**vendored into this repo** at `vendor/react-compiler-rust` (`facebook/react`
branch `pr-36173` @ `0dc7f2e`). The Rust `react_compiler_hir` crate compiles here
and is wired up as a *real second `Cfg` backend* (see §8) — so the
"works on both" claim below is not hypothetical; it's exercised by
`crates/jsir-jslir/tests/cross_backend.rs`. Every React-side type citation is from
that source, not from memory.

---

## 0. Provenance — why they look different

| | React Compiler HIR | JSIR (JSHIR / JSLIR) |
|---|---|---|
| Origin | `babel-plugin-react-compiler`, purpose-built for React memoization | Google maldoca `maldoca/js` MLIR dialect (`vendor/jsir-upstream`); Rust port in `jsir-ir` + a React-oriented CFG dialect `jsir-jslir` |
| Reason to exist | Drive reactive-scope inference + memoization | Byte-exact JS round-trip + general analyses (DCE, constprop); `jsir-jslir` then adds a CFG to host the React passes |
| Representation | In-memory TS objects (`HIRFunction`, `Map<BlockId, BasicBlock>`) | MLIR-style op tree (`Op` with regions/blocks/operands/attrs), AoS in `jsir-ir`, plus an SoA `Module` |
| Fidelity goal | Regenerate *equivalent* code (own codegen → Prettier) | Regenerate **byte-identical** code (`trivia` + brace metadata on every op) |

The fidelity goal already explains a lot of the divergence: JSLIR keeps source
trivia and brace/paren structure on every op because its contract is byte-exact
lift back to JS. React HIR throws that away — it never reproduces the input, it
emits fresh code.

---

## 1. The container — *this part lines up well*

**React:**
```
HIRFunction { id, params, returns, body: HIR, ... }
HIR         { entry: BlockId, blocks: Map<BlockId, BasicBlock> }
BasicBlock  { kind, id: BlockId, instructions: Instruction[], terminal: Terminal,
              preds: Set<BlockId>, phis: Set<Phi> }
```

**JSLIR** (on the generic `jsir-ir` substrate):
```
Op (jsir.function_declaration, …) { regions: [Region], ... }
Region { blocks: Vec<Block> }
Block  { id: BlockId, args: Vec<ValueId>, ops: Vec<Op> }   // terminator op carries `successors`
```

Both are a **CFG of basic blocks with a stable block identity** (`BlockId` is a
number on both sides, stable across edits so a pass can hold it while inserting/
removing blocks — `jsir-ir/src/lib.rs` says this explicitly, and React relies on
it the same way). A function body is a region/`HIR` of those blocks. The
`jsir-jslir` builder (`build.rs`) even notes it "mirrors the upstream HIR
builder's reserve/terminate model."

**This is the shared spine.** Anything that only needs *blocks + edges* —
dominators, post-order/RPO, reachability, predecessor maps, structured-CFG
reconstruction — can be written once against an interface both satisfy.

Two encoding differences to bridge:
- React stores `preds` and `phis` **on the block**; JSLIR derives predecessors
  from terminator `successors` (see `ssa.rs`) and keeps phis in a side table.
- React's blocks live in a `Map`; JSLIR's in a `Vec<Block>` inside a `Region`.
  A `BlockId → &Block` index (`ssa.rs` builds exactly this) makes them equivalent.

A nice convergence: `HIR.blocks` is documented (`HIR.ts:304`) as stored in
**reverse postorder** "to facilitate forward data flow analysis" — which is
exactly the RPO that JSLIR's `ssa.rs` computes on the fly (`analyze_cfg`). Both
IRs agree on the traversal order their analyses want; one bakes it into storage,
the other derives it.

---

## 2. The instruction model — *this is the crux of the mismatch*

This is the single biggest reason "the same code on both" does not work out of
the box.

**React HIR is fully scalar + SSA.** Every instruction is
```
Instruction { id: InstructionId, lvalue: Place, value: InstructionValue, loc }
```
(verified at `HIR.ts:651`) with the hard invariant: **one lvalue, and every
operand is a `Place`** — never a nested expression. `a = f(b + c)` becomes a
*list*:
```
$0 = LoadLocal b
$1 = LoadLocal c
$2 = BinaryExpression $0 "+" $1
$3 = LoadLocal f
$4 = CallExpression $3 ($2)
$5 = StoreLocal a = $4
```
`InstructionValue` is a **44-variant** closed tagged union (counted at the
checkout: `LoadLocal`, `StoreLocal`, `LoadGlobal`, `StoreGlobal`, `LoadContext`,
`BinaryExpression`, `CallExpression`, `MethodCall`, `NewExpression`,
`PropertyLoad`/`PropertyStore`, `ComputedLoad`/`ComputedStore`, `Destructure`,
`ObjectExpression`, `ArrayExpression`, `JsxExpression`, `JSXText`,
`FunctionExpression`, `ObjectMethod`, `Primitive`, `TemplateLiteral`, `Await`,
`StartMemoize`/`FinishMemoize`, …). This maximal flatness is *the* property the
reactive-scope analysis depends on: every intermediate value is named and
individually rangeable.

**JSLIR instructions are AST-faithful `jsir.*` ops, reused verbatim and coarse.**
`jsir-jslir/src/lib.rs` is explicit: block instructions "reuse the AST-faithful
`jsir.*` ops verbatim (coarse blocks for now; within-block flattening to a
fully-scalar instruction stream is a later refinement)." So a
`jsir.binary_expression` op still *nests* its operands as ops feeding results
into operands; the block is an op tree, not a one-lvalue-per-line stream. The
only flattening done so far is `expr_flatten.rs`, and only for **control-flow**
expressions (`&&`, `||`, and ternary/optional "next") — because those have to
become real edges; pure data sub-expressions stay nested.

| | React HIR instruction | JSLIR block instruction |
|---|---|---|
| Granularity | one scalar op per value (`InstructionValue`) | whole AST-shaped `jsir.*` op, operands nested |
| Operands | always `Place` | `ValueId` (op results) — but the producing op is the nested op, not a prior line |
| Temporaries | every subexpression is a named temp | only where `expr_flatten` split for control flow |
| Tag space | one closed `InstructionValue` enum (~50) | open set of `jsir.*` op names (the full JS AST) |

**Consequence for "shared code":** any pass that walks the instruction stream
expecting `lvalue/value` scalar form (which is *most* React passes:
`InferMutableRanges`, `InferReactiveScopeVariables`, `PropagateScopeDependencies`,
const-propagation over temporaries, …) cannot run on JSLIR until JSLIR has the
"fully-scalar instruction stream" refinement its own header promises. Conversely
a pass written for JSLIR's nested ops won't see React's temporaries.

The `InstructionValue` ↔ `jsir.*` mapping itself is a clean enumerable table
(`BinaryExpression`↔`jsir.binary_expression`, `CallExpression`↔
`jsir.call_expression`, `LoadGlobal`↔`jsir.identifier`+global resolution,
`StoreLocal`↔`jslir.store_local`, …). The gap is **not** the vocabulary — it's
the *shape* (scalar list vs nested tree).

---

## 3. SSA & value identity — *materialized vs side-table*

**React:** SSA is **in the IR**. `EnterSSA` rewrites the function so each
`Place` (`HIR.ts:1165`) points at an `Identifier` with a unique `IdentifierId`;
phis are real members of `BasicBlock.phis` (`Phi { kind, place, operands:
Map<BlockId, Place> }`, `HIR.ts:789`). Crucially, `Identifier` carries **two**
identities (`HIR.ts:1253`): `id: IdentifierId` is the *SSA instance* (distinct per
reassignment after `EnterSSA`), while `declarationId: DeclarationId` is the
*original source variable* (stable across reassignments). Type, scope and mutable
range all hang off `Identifier`.

**JSLIR:** SSA is a **pure side-table analysis** (`ssa.rs::enter_ssa →
SsaInfo`). The IR is *not* in SSA form — it stays in named-variable form, and
`SsaInfo` computes `reaching: ValueId → ValueId` (def-use) plus `phis: Vec<Phi>`
*without mutating the IR* (the round-trip must be preserved). Variable identity
is recovered from `trivia.referenced_symbol` (`name` + `def_scope_uid`), not from
a first-class id. And the analysis is deliberately partial today: it bails (`None`,
never wrong) on shadowing, nested/multiple loops, and compound/destructuring
writes.

Two further mismatches here:
- **Phi form.** React uses classic **phi nodes** (`pred BlockId → Place`). The
  JSIR substrate is **MLIR block-argument** form (`Block.args` + `Successor.args`
  on the branch). These are duals — `cond_br_logical` in `dialect.rs` already
  uses block args as phis — but a shared pass must pick one and adapt the other.
- **Where identity lives.** React: `IdentifierId` (intrinsic, SSA-instance) plus
  `declarationId` (intrinsic, source-variable). JSLIR: resolved symbol
  name/`def_scope_uid` (extrinsic, via trivia) maps to React's *`declarationId`*
  level; the SSA-instance level is what `SsaInfo`'s reaching-def values stand in
  for. So a shared "what variable is this" query needs an adapter that exposes
  both levels on the JSLIR side.

So to make JSLIR look like React HIR you must (a) materialize SSA from `SsaInfo`
into block args/results, and (b) synthesize stable identifier ids from resolved
symbols. Both are mechanical once `SsaInfo` covers the construct.

---

## 4. Control flow & terminals — *same structure, different encoding*

This is the place the two are **closest in spirit**, which is encouraging.

**React** keeps high-level, *structured* terminals as distinct variants: `if`,
`switch`, `for`, `while`, `do-while`, `for-of`, `for-in`, `return`, `throw`,
`goto` (with `break`/`continue` flavor), `label`, `ternary`, `logical`,
`optional`, `sequence`, `try`, `maybe-throw`, `scope`, `branch`, `unreachable`.
The loop terminals point at their `init`/`test`/`update`/`loop` blocks. That
retained structure is *why it's called HIR*.

**JSLIR** has only three terminator ops — `jslir.br`, `jslir.cond_br`,
`jslir.return` — but carries the same structure as **attributes** on them
(`dialect.rs`):
- `cond_br_if` → `merge` (join block) + `then_braced`/`else_braced`
- `cond_br_loop` → `loop`, `kind="while"`, `merge`, `body_braced`
- `cond_br_for` → `kind="for"`, `preheader`, `latch`, `update_val`
- `cond_br_ternary` / `cond_br_logical` → `condexpr` / `logexpr` operator + `merge`
- `br_loop_exit` → `loopjump="break"|"continue"`
- `ret(..., implicit)` / `ret_expr` (arrow expression body)

The doc comments call this out as the **SPIR-V `OpSelectionMerge`/`OpLoopMerge`
model** — explicit structure carried *on* a generic CFG. React's model is
"structure as terminal variants"; JSLIR's is "generic branch + structured
metadata." They encode the same information (enough to rebuild source-level
`if`/`for`/`while`), just factored differently.

**Implication:** a structured-control-flow pass is very portable — but it must
read structure through an accessor (`is_if_header`, `merge_of`, `is_for_header`,
… on JSLIR; the terminal variant on React). Define that accessor once and the
loop/branch passes are shared.

One real semantic gap: React models exceptions richly (`try`, `maybe-throw`,
`MaybeThrowTerminal` on potentially-throwing instructions). JSLIR currently
*passes through* `try`/`throw` functions uncompiled (`CONTROL_FLOW_STMTS` in
`lib.rs` lists them as not-yet-lowered). So exception control flow is React-only
for now.

---

## 5. The analysis substrate — *React-only, and that's the whole point*

React HIR is saturated with memoization metadata that JSIR simply does not have:

- `Place.effect: Effect` — the 8-value enum at `HIR.ts:1512` (`Read`, `Mutate`,
  `Store`, `Capture`, `Freeze`, `ConditionallyMutate`,
  `ConditionallyMutateIterator`, `Unknown`). The dataflow currency of the compiler.
- A newer, richer **aliasing-effect** layer on top: `Instruction.effects:
  Array<AliasingEffect> | null` (`HIR.ts:656`) and a function-level
  `HIRFunction.aliasingEffects` (`HIR.ts:296`). So the effect substrate is even
  heavier than "one effect per Place" — it's per-instruction alias graphs.
- `Identifier.mutableRange` — an `[InstructionId, InstructionId)` range; the
  reason instructions carry a globally-monotonic `InstructionId`.
- `Identifier.scope: ReactiveScope` — the memoization unit; `BasicBlock.kind`
  (`HIR.ts:324`) includes the expression-block kinds `value`/`loop`/`sequence`,
  and the terminal union has `ReactiveScopeTerminal`/`PrunedScopeTerminal`.
- `reactive: boolean` on `Place`, mutable-range-based aliasing, etc.

JSIR has **none** of this — it's a general-purpose JS IR. There is no effect, no
mutable range, no reactive scope, no global instruction numbering (JSLIR's
`node_id` is an origin tag for mapping back to source, not an analysis clock).

This is not a defect; it's the gap `jsir-jslir` exists to fill. `passes.rs`
states the plan plainly: the passes "mirror the upstream Rust port pass-for-pass,
but consume our `SsaInfo` / JSLIR instead of the React HIR." So the substrate is
meant to be **added as side tables** keyed on `ValueId`/`BlockId`, exactly like
`SsaInfo` is — not baked into the op the way React bakes it into `Identifier`.

---

## 6. Side-by-side mapping table

| Concept | React Compiler HIR | JSLIR (this repo) | Convertibility |
|---|---|---|---|
| Function | `HIRFunction` | `jsir.function_*` op + body `Region` | trivial |
| CFG | `HIR { entry, blocks: Map }` | `Region { blocks: Vec<Block> }` | trivial (+ build `BlockId→Block` index) |
| Block id | `BlockId` (number) | `BlockId(u32)` | trivial |
| Predecessors | `block.preds` (stored) | derived from `successors` | mechanical |
| Instruction | scalar `{lvalue, value}` | nested `jsir.*` op | **hard** — needs full scalarization (not yet built) |
| Value kind | `InstructionValue` (~50 enum) | `jsir.*` op name (open) | enumerable table |
| Operand | `Place` | `ValueId` (op result) | needs SSA + identity adapter |
| Variable id | `Identifier.id` (SSA-instance) + `declarationId` (source var) | `trivia.referenced_symbol` (≈ declarationId) + `SsaInfo` reaching def (≈ id) | synthesize ids |
| SSA / phi | materialized, `Phi` nodes | `SsaInfo` side table, MLIR block-args | materialize from `SsaInfo` |
| `if`/`while`/`for` | structured terminal variants | `cond_br` + structured attrs | accessor shim, mechanical |
| `return`/`throw`/`break` | terminal variants | `jslir.return`, `br_loop_exit`, (throw TBD) | mechanical (throw pending) |
| `&&`/`||`/`?:` | `logical`/`ternary` value blocks | `expr_flatten` → `cond_br_*` + block-arg phi | same move, partially done |
| effects / ranges / scopes | on `Place`/`Identifier`/`BasicBlock` | **absent** → add as side tables | additive |
| Source fidelity | none (regenerates) | byte-exact (`trivia` + braces) | extra info JSLIR keeps |

---

## 7. So: one codebase, or convert?

### "Code that works on both without converting"
**Partially achievable, and worth doing — but only at the CFG layer.** Factor the
genuinely-shared spine behind a trait:

```
trait Cfg {
    fn entry(&self) -> BlockId;
    fn blocks(&self) -> impl Iterator<Item = BlockId>;
    fn successors(&self, b: BlockId) -> &[BlockId];
    fn preds(&self, b: BlockId) -> &[BlockId];
    // structured-control-flow view (the SPIR-V-style accessors):
    fn terminal(&self, b: BlockId) -> TerminalView;   // If{merge,..} | Loop{kind,latch,..} | Return | Br | ...
}
```
Dominators, RPO, reachability, DCE-by-reachability, structured reconstruction,
loop-finding, and the `if`/loop-shaped passes can all be written **once** against
`Cfg`/`TerminalView` and run on either side. This is the realistic "works on
both" win.

What **cannot** be shared without conversion is everything below the terminator:
the instruction stream (scalar vs nested), value/identity (`Place`/`Identifier`
vs `ValueId`/symbol), and the effect/range/scope analyses. A pass like
`InferMutableRanges` is defined entirely in terms of React's scalar SSA +
effects; there is no JSLIR shape for it to read yet.

### "Easily convert"
**Yes — and the React→JSLIR direction is exactly what `jsir-jslir` is already
doing.** The mapping table above is the conversion spec. The mechanical pieces
(container, block ids, terminals, value-kind table) are straightforward. The two
real costs:

1. **Scalarization.** To present JSLIR as React-style scalar SSA you must do the
   "within-block flattening to a fully-scalar instruction stream" that
   `lib.rs` already flags as a later refinement, then materialize `SsaInfo` into
   results/block-args and synthesize identifier ids. This is the load-bearing
   work; until it exists, JSLIR→React (and instruction-level shared passes) is
   blocked.
2. **Substrate.** Effects / mutable-ranges / reactive-scopes must be added as
   `ValueId`-keyed side tables (the `SsaInfo` pattern). Additive, no IR change.

The reverse (JSLIR→React-faithful enough to *emit* React's output) is harder only
because React discards source fidelity that JSLIR carries — but that's a *lift*,
not a *loss*, so it's not on the critical path.

---

## 8. Recommendation

1. **Build the shared `Cfg`/`TerminalView` trait.** ✅ *Done, with both backends
   real.* `crates/jsir-jslir/src/cfg.rs` defines the backend-neutral `Cfg` trait
   (generic over an associated `Value` type, since React names values by
   `IdentifierId` and JSLIR by `ValueId`) + `TerminalView` (Return / Goto /
   If / While / For / Ternary / Logical / Branch / **Other** / Open), plus generic
   `analysis::{predecessors, reverse_postorder, reachable, immediate_dominators}`
   (Cooper–Harvey–Kennedy) written *only* against the trait.
   - `JslirCfg` backs it over a lowered `Region` (`tests/cfg.rs`).
   - `cfg/react.rs` (`--features react-hir`) backs it over the **real vendored
     `react_compiler_hir::HIR`**, reading its actual `Terminal`s and
     `each_terminal_successor` semantics.
   - `tests/cross_backend.rs` runs the *identical* dominator/RPO/reachability code
     on a real React-HIR diamond and a JSLIR one — the "write once, run on both"
     claim, compiled and green.

   The honest boundary: React's structured-terminal vocabulary is *richer* than
   JSLIR's current dialect (loops encoded via a separate `test` block, plus
   `switch`/`try`/`for-of`/`scope`/…), so those map to `TerminalView::Other`
   rather than being forced into a JSLIR-shaped variant. The generic algorithms
   are unaffected — they consult only `successors()`, which is exact on both. The
   shared *structured* core that lines up precisely today is
   Return / Goto / If / Branch.
2. **Land within-block scalarization in JSLIR** (the refinement its own header
   promises). This is the gate to instruction-level shared code and to a faithful
   JSLIR→React-HIR adapter. Until then, accept that instruction-walking passes are
   per-IR.
3. **Materialize SSA + synthesize identifier ids** from `SsaInfo` behind the same
   trait, so a `Value`/`Identifier` abstraction reads uniformly on both.
4. **Model effects/ranges/scopes as side tables**, never as fields on `Op` — keep
   the byte-exact round-trip untouched (the non-negotiable JSIR contract).
5. **Keep one enumerable `InstructionValue ↔ jsir.*` table** as the single source
   of truth for the value-kind mapping; generate both directions from it.

Net: a single *conceptual* HIR with two backends is realistic at the CFG +
structured-control-flow level, which is a meaningful fraction of the React passes.
"Truly one representation, no conversion" for the instruction/effects layers is
not realistic given JSIR's byte-exact-round-trip contract and React's
scalar-SSA-plus-effects contract — those are genuinely different data models
serving different masters. The pragmatic target is **one shared CFG interface +
a thin, mechanical React↔JSLIR adapter**, which is the trajectory `jsir-jslir` is
already on.

---

## 9. Could we run the *existing* React passes on JSIR with the right interface?

This is the load-bearing question, and now that the Rust port is vendored we can
answer it from the actual pass code, not by analogy.

**No — not through an interface/trait.** The passes are not generic. Every one is
a concrete function over concrete data:

```rust
// react_compiler_ssa/src/enter_ssa.rs, .../eliminate_redundant_phi.rs
pub fn enter_ssa(func: &mut HirFunction, env: &mut Environment) { … }
// react_compiler_inference, react_compiler_optimization, react_compiler_reactive_scopes …
pub fn infer_reactive_scope_variables(func: &mut HirFunction, env: &mut Environment) -> … { … }
pub fn dead_code_elimination(func: &mut HirFunction, env: &mut Environment) { … }
```

~30 pass entry points, **161 references to `HirFunction`** across the pass crates,
and they `match` the concrete `Terminal` / `InstructionValue` / `Place` enums and
mutate the `Environment` arenas directly. A `Cfg`-style trait abstracts only code
*you* write (the algorithms in `cfg.rs`); it cannot retro-fit an abstraction onto
30 functions that own a concrete type. So "make JSIR implement an interface and
the passes just work" is **not** on the table.

**Yes — through a converter.** The only way to reuse the *real* passes is to
produce an actual `HirFunction` (+ an `Environment`), run them unmodified, and
convert back: `JSIR → HirFunction → [real react passes] → HirFunction → JSIR`.
Two findings make this genuinely viable rather than a rewrite:

1. **The storage layout already matches.** The Rust port is SoA, not the TS
   pointer-soup: `HirFunction.instructions: Vec<Instruction>` is a flat arena,
   `BasicBlock.instructions: Vec<InstructionId>` indexes it, and
   `Instruction { id, lvalue: Place, value: InstructionValue }`. That is *exactly*
   the shape JSLIR would have after the "within-block scalarization" it already
   plans — so the destination is friendly.
2. **The CFG/terminal half is done.** §1/§4 + `cfg.rs` already map blocks, edges,
   and the structured terminals both ways.

### How far off, layer by layer (converter surface)

| Layer | Distance | Why |
|---|---|---|
| CFG blocks + edges | **~0 (done)** | shared `Cfg`, proven on both backends |
| Structured terminals | **small** | `TerminalView` mapping exists; React's extra kinds (loops-via-test-block, switch, try, for-of, scope) are mechanical additions |
| Instruction stream | **the bulk of the work** | scalarize JSLIR's nested `jsir.*` ops → flat `Vec<Instruction>` (one lvalue `Place` per `InstructionId`); map the **44** `InstructionValue` kinds ↔ `jsir.*` op names |
| Value identity / SSA | **moderate** | materialize `SsaInfo` into `Place`/`Identifier` + populate `Environment.identifiers` |
| `Environment` | **moderate, tiered** | `Environment::new()` is cheap (counters + arenas); `globals`/`shapes`/type registries are only needed by the *type-dependent* passes |

### The strategic consequence

There are two ways to get React's memoization onto JSIR, and they are mutually
exclusive trajectories:

- **(A) Re-port** each pass against JSLIR + `SsaInfo` — what `jsir-jslir`'s
  `passes.rs` does today. No converter, but you re-implement and then *maintain
  parity for all ~30 passes forever* as upstream evolves.
- **(B) Convert** once (`JSIR ⇄ HirFunction`/`Environment`) and call the **real**
  passes unmodified — they come for free and track upstream automatically. The
  converter cost is the table above; the biggest single chunk is instruction
  scalarization. ✅ *A first cut of this direction now exists*:
  `crates/jsir-jslir/src/to_react_hir.rs` (`--features react-hir`) converts a
  JSLIR body into a real `react_compiler_hir::HirFunction` — flat scalar
  `Instruction`s (literals, identifier loads, binary/unary, calls, store/assign),
  real `Terminal::If`/`Goto`/`Return`, allocated `IdentifierId`s via the
  `Environment` arena, and populated predecessors. `tests/to_react_hir.rs` proves
  it on straight-line and `if`-diamond functions; unsupported constructs (loops,
  member/computed access, objects, …) return a descriptive `Err` rather than a
  wrong HIR. **A key discovery while building it:** JSLIR is *already* a flat
  per-block SSA-value op stream (each `jsir.*` op yields a result `ValueId`;
  operands are earlier results), so the feared "scalarization" is largely
  pre-done by jsir's lowering — the converter is mostly a per-op mapping. The
  remaining work to run the actual passes is breadth (more op kinds) plus
  vendoring a pass crate (e.g. `react_compiler_ssa`) to call `enter_ssa` on the
  result.

Before the Rust port existed, (B) was impossible (the passes were TypeScript).
Now that `react_compiler_hir` compiles in *this* workspace, (B) is a real option:
you can already call `react_compiler_ssa::enter_ssa(&mut func, &mut env)` from our
crate — the only missing piece is a `func` produced from JSIR. The tiered
`Environment` means you don't need the whole thing at once: the **structural
passes** (`enter_ssa`, `eliminate_redundant_phi`, `dead_code_elimination`,
`prune_unused_*`) need little more than counters + arenas, so a thin converter +
minimal `Environment` could run them end-to-end first, and the type/shape
registries only become necessary when you reach `infer_types` and the
reactive-scope inference.

**Bottom line on "how far off":** not far in *kind* — they share the CFG exactly
and now share the storage *philosophy* (flat instruction arena). They differ in
three *materializations* — scalar instructions, SSA/`Place` identity, and the
`Environment` substrate — none of which is a fundamental impedance mismatch; all
are "lower/convert," not "incompatible." The honest measure is: **a JSIR→HIR
converter is a bounded, well-understood piece of work whose long pole is
instruction scalarization — and it buys you the entire real pass pipeline instead
of a perpetual re-port.**
