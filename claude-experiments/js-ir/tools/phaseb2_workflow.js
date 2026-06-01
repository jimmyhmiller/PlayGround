export const meta = {
  name: 'phase-b2-analysis',
  description: 'Re-sequenced Phase B per PHASE_B_REDESIGN.md: analysis-first (escape-prune, align, deps) then control-flow codegen, per-step gated + anti-hack',
  whenToUse: 'Execute the re-sequenced Phase B plan in the isolated js-ir-parity worktree',
  phases: [{ title: 'Measure' }, { title: 'Implement' }, { title: 'Verify' }, { title: 'Gate' }],
}

const WT = '/Users/jimmyhmiller/Documents/Code/PlayGround-worktrees/js-ir-parity/claude-experiments/js-ir'
const ENV = 'REACT_CC=/tmp/react-rust/compiler/target/release/react-compiler-e2e'
const DESIGN = `${WT}/crates/jsir-ssa/PHASE_B_REDESIGN.md`
const REACT = '/tmp/react-rust/compiler/crates'
const GATE = `cd ${WT} && ${ENV} cargo run --release -p jsir-ssa --example corpus -- --json 2>/dev/null | tail -1`

const RULES = `
HARD RULES (monorepo others use; correctness over cleverness):
- Work ONLY inside ${WT}. Absolute paths. NEVER edit the gate/oracle (examples/corpus.rs, structure(), react CLI, fixtures).
- This is the RE-SEQUENCED plan in ${DESIGN}. READ the relevant step section. It ports the SEMANTICS of React's passes;
  the React Rust source is at ${REACT} — read the cited pass to get the algorithm right, port the semantics (not the code).
- Re-grep symbol names (mutability::analyze, scopes::infer/analyze, term_operands, build_layout/check_soundness,
  emit_scope) rather than trusting line numbers.
- ARCHITECTURE: analysis fixes live in mutability.rs/scopes.rs/cfg.rs/lower.rs; emission stays an IR->IR transform
  (memoize.rs/memoize_plan.rs) printed via hir2ast->ast2source. NO relooper, NO string-codegen revival, keep reversible JSIR pure.
- NO stubs / silent fallbacks: any unhandled shape MUST return Err(...) with a clear message, never a fake value or silent skip.
  Incompleteness behind a clean Err is fine; silent-wrong output is NOT. Verify memoized cases under Node (value + ref stability).`

const MEASURE = {
  type: 'object',
  required: ['build_ok', 'tests_pass', 'agree', 'panic', 'mismatch', 'ours_only', 'universe'],
  properties: {
    build_ok: { type: 'boolean' }, tests_pass: { type: 'boolean' },
    tests_summary: { type: 'string' }, build_error: { type: 'string' },
    agree: { type: 'number' }, panic: { type: 'number' }, mismatch: { type: 'number' },
    ours_only: { type: 'number' }, universe: { type: 'number' },
  },
}
const VERDICT = {
  type: 'object', required: ['hacked', 'confidence', 'evidence'],
  properties: { hacked: { type: 'boolean' }, confidence: { type: 'string', enum: ['low', 'medium', 'high'] }, evidence: { type: 'string' } },
}

// gate: agree in {up, up_or_same, same}; ours_only in {down, down_or_same, same, any}
const STEPS = [
  { id: '0a', title: 'number terminals in Point space',
    spec: `Per design Step 0a: extend mutability::analyze so RPO numbering assigns a Point to each block TERMINATOR too ` +
      `(terminal point = after the block's last instr); add Ranges.term_point: HashMap<BlockId,Point>. Pure plumbing — ` +
      `nothing observable should change.`,
    tests: 'cargo test --release -p jsir-ssa', gate: { agree: 'same', ours_only: 'same' } },
  { id: '0b', title: 'reconstruct structured join/fallthrough info',
    spec: `Per design Step 0b: in lower.rs, when lowering if/for/while/switch/ternary/logical, record head block -> join ` +
      `(merge) block in a new Cfg.joins: HashMap<BlockId,BlockId> (+ a BlockKind: Block/Loop). Hard-error in lower.rs if a ` +
      `construct is lowered without a recordable join (don't silently omit). Metadata only — nothing observable changes.`,
    tests: 'cargo test --release -p jsir-ssa', gate: { agree: 'same', ours_only: 'same' } },
  { id: '1', title: 'PruneNonEscapingScopes (THE fix for 86 ours_only)',
    spec: `Per design Step 1 (THE big one). Port PruneNonEscapingScopes: add scopes::prune_non_escaping run inside analyze ` +
      `BEFORE the merge fold, replacing the WRONG outputs predicate. Our current bug: outputs = "used by any later instr OR ` +
      `any terminator" (term_operands includes CondBr cond + branch args) — React's escape set = only values transitively ` +
      `aliased into a RETURN value or a HOOK arg. Seed escape roots from Term::Ret operands (+ hook args when callee matches ` +
      `use[A-Z]). Build the MemoizationLevel lattice + compute_memoized_identifiers DFS + force_memoize_scope_dependencies ` +
      `(read prune_non_escaping_scopes.rs:69,391,1043,1098,1155). Keep a scope iff it declares a value in the memoized set ` +
      `(or the keep-exceptions). Set surviving scopes' outputs = escaping∩memoized. Assume aliasing where unknown (never under-memoize).`,
    tests: 'cargo test --release -p jsir-ssa', gate: { agree: 'up_or_same', ours_only: 'down' } },
  { id: '2', title: 'PruneAlwaysInvalidatingScopes',
    spec: `Per design Step 2: add scopes::prune_always_invalidating after the merge fold. always_invalidating lvalues = ` +
      `MakeArray/MakeObject/JSX/Call-new results; unmemoized = produced outside any scope. If any scope dependency is in ` +
      `unmemoized, demote the scope (drop its cache). Read prune_always_invalidating_scopes.rs:50.`,
    tests: 'cargo test --release -p jsir-ssa', gate: { agree: 'up_or_same', ours_only: 'down_or_same' } },
  { id: '4', title: 'PropagateScopeDependencies (property paths)',
    spec: `Per design Step 4: replace ScopeInfo.deps: Vec<Value> with Vec<Dep{root, path:Vec<(PropKey,optional)>}>. Record a ` +
      `Member chain off an outer value as ONE dep (temporaries sidemap) instead of the intermediate SSA temp. Add ` +
      `check_valid_dependency (def Point strictly before scope.range.start) + ref.current->ref collapse + minimal-dep ` +
      `derivation. Sort deps by name+path STRING key (not SSA id). Codegen: fold a path into member_expression ops in ` +
      `emit_scope; optional-member form if any entry optional. Read propagate_scope_dependencies_hir.rs:34,250,343,1844,1896 + ` +
      `codegen_reactive_function.rs:3132,3673. Stage hoistable-non-null fixpoint as a later sub-step; until then render only ` +
      `fully-unconditional paths, else fall back to the intermediate temp (sound, over-fragmented).`,
    tests: 'cargo test --release -p jsir-ssa', gate: { agree: 'up', ours_only: 'down_or_same' } },
  { id: '5', title: 'AlignReactiveScopesToBlockScopes (enables if/else agree)',
    spec: `Per design Step 5: add scopes::align_to_block_scopes called inside infer after the interval merge, before allocation ` +
      `assignment. Forward walk with active_scopes + a stack of active_block_fallthrough_ranges keyed on Cfg.joins: entering a ` +
      `join block pulls active scopes' start back to the construct head; a terminal with a join extends active scopes out to ` +
      `the join's first point. Drop React's value-block machinery initially (our ternary/logical are CondBr diamonds). Then ` +
      `RELAX check_soundness's multi-block rejection (scopes are now block-aligned). Keep the "mutation crosses guard" check as ` +
      `a hard-error post-condition assertion. Read align_reactive_scopes_to_block_scopes_hir.rs:86,103,180,207.\n` +
      `GOAL IS AGREE, NOT JUST MEMOIZE: a newly-memoized if/else fixture only counts if its (cache_size, block_count) ` +
      `MATCHES React (else it's a mismatch, which fails this gate). If matching requires correcting dependency ` +
      `representation that affects cache size (deduping a.b vs a.b.c to a minimal dep set, property-path deps — design ` +
      `step 4, currently deferred), do that here too. Verify each newly-agreeing fixture's structure against the REAL ` +
      `react CLI and its semantics under Node. Cases you can't make sound+matching yet MUST hard-error (stay react_only), ` +
      `never emit a wrong structure (mismatch) or wrong semantics.`,
    tests: 'cargo test --release -p jsir-ssa', gate: { agree: 'up', ours_only: 'down_or_same' } },
  { id: '6', title: 'MergeOverlappingReactiveScopes (nesting-aware) + nested emit',
    spec: `Per design Step 6: replace the flat interval merge with the LIFO-stack crosser-union on aligned half-open ranges + ` +
      `the "mutate outer scope while inner active -> union" rule; preserve insertion-order union roots (don't sort by id). ` +
      `Extend memoize_plan emission so a scope can NEST inside another. Read merge_overlapping_reactive_scopes_hir.rs:146,185,250.`,
    tests: 'cargo test --release -p jsir-ssa', gate: { agree: 'up_or_same', ours_only: 'down_or_same' } },
  { id: '3', title: 'generalize MergeReactiveScopesThatInvalidateTogether',
    spec: `Per design Step 3 (now that Step 4/6 give a block-ordered scope list): extend the merge fold to React's three ` +
      `can_merge_scopes rules (equal dep-sets; producer-decls==consumer-deps; always-invalidating-typed output consumed next) ` +
      `with the reassignment gate + adjacency (only consecutive scopes). Read merge_..._that_invalidate_together.rs:440,554.`,
    tests: 'cargo test --release -p jsir-ssa', gate: { agree: 'up_or_same', ours_only: 'down_or_same' } },
  { id: '7', title: 'PropagateEarlyReturns (the inverted case, now sound)',
    spec: `Per design Step 7: remove the early-return hard-errors. In recover_regions, a "return inside a SURVIVING scope's ` +
      `range" lowers to labeled-break: redirect Ret to the scope's post-block carrying the value. Add ` +
      `ScopeInfo.early_return_value attached to the OUTERMOST enclosing scope (bubbling up, idempotent). The early-return value ` +
      `is an ordinary cached output (own slot), initialized at scope head to Symbol.for("react.early_return_sentinel") (distinct ` +
      `from memo_cache_sentinel). emit_scope: AFTER the memo if/else, emit "if (name !== Symbol.for('react.early_return_sentinel')) return name;" ` +
      `— NOT folded into the dep-keyed test. Top-level / pruned-scope returns stay plain Ret. Read propagate_early_returns.rs:26,77,112,185 + codegen_reactive_function.rs:861-905.`,
    tests: 'cargo test --release -p jsir-ssa', gate: { agree: 'up', ours_only: 'down_or_same' } },
  { id: '8', title: 'loops (FlattenReactiveLoops) + hooks (FlattenScopesWithHooksOrUse)',
    spec: `Per design Step 8: detect back-edges in recover_regions and FLATTEN scopes wrapping loops (don't memoize across a ` +
      `loop body) instead of hard-erroring. Add is_hook detection (use[A-Z] / known hooks) and demote scopes containing a hook ` +
      `call. Read flatten_scopes_with_hooks_or_use_hir.rs.`,
    tests: 'cargo test --release -p jsir-ssa', gate: { agree: 'up', ours_only: 'down_or_same' } },
]

async function measure(step, label) {
  return agent(
    `Measure the gate for ${label}.\n${RULES}\n\nRun from ${WT}:\n` +
    `  1. cargo build --release -p jsir-ssa -p jsir-convert -p jsir-transforms 2>&1 | tail -20\n` +
    `  2. ${step.tests}\n  3. ${GATE}  (one JSON line)\n` +
    `Report build_ok; tests_pass (+tests_summary); and corpus agree/panic/mismatch/ours_only/universe. ` +
    `Build failed -> build_ok=false, first error in build_error, counts 0. Do not edit files.`,
    { schema: MEASURE, phase: 'Measure', label }
  )
}

async function implement(step, base) {
  return agent(
    `Implement Phase B (re-sequenced) STEP ${step.id} (${step.title}) for REAL in ${WT}.\n${RULES}\n\n` +
    `READ ${DESIGN} (the Step ${step.id} section) and the cited React pass, then re-grep the symbols.\n\n` +
    `STEP ${step.id} SPEC: ${step.spec}\n\n` +
    `Current gate: agree=${base.agree}, ours_only=${base.ours_only}, panic=${base.panic}, mismatch=${base.mismatch}.\n` +
    `THIS STEP'S GATE: agree must be "${step.gate.agree}", ours_only must be "${step.gate.ours_only}", panic stays 0, no regression.\n\n` +
    `Steps: implement generally + soundly (unhandled shapes hard-error); build (cargo build --release -p jsir-ssa ` +
    `-p jsir-convert -p jsir-transforms); run ${step.tests}; sanity-check the gate (${GATE}) and a few fixtures with ` +
    `corpus --show. Do NOT commit. End by stating build/tests pass + a one-line summary + the agree/ours_only you observed.`,
    { phase: 'Implement', label: `step${step.id}:${step.title.slice(0, 22)}` }
  )
}

async function verify(step, lens) {
  const lenses = {
    correctness: `Run the REAL React oracle (${ENV}) on the cases this step claims to handle and confirm OUR structure matches ` +
      `React's (cache size + memo-block count) AND the emitted JS is semantically correct + reference-stable under Node. Flag ` +
      `hacked=true for: silently-wrong output, fake values, output that DISAGREES with the React oracle while tests pin it as ` +
      `correct, or narrow fixture-fitting. Cleanly-hard-errored incompleteness is fine.`,
    semantics: `Confirm this is a faithful port of React's pass SEMANTICS (read the cited React source). For Step 1 esp.: is the ` +
      `escape set truly "transitively aliased into a return or hook arg" (not "used by any later instr/terminator")? Is ` +
      `force_memoize_scope_dependencies applied? Flag hacked=true if the port is a shape-fitted approximation that diverges from ` +
      `React's actual algorithm in a way that will mis-prune.`,
    stub: `Hunt for stubs / silent fallbacks / TODO / unimplemented! / fake values where an unhandled shape should hard-error.`,
    gaming: `Hunt for metric-gaming: fixture-name/text special-casing, hardcoded sizes, fake _c(N), any edit to gate/oracle/fixtures, ` +
      `weakened tests, or DEAD CODE that exists only to make a fixture appear to pass. Confirm git diff touches only real ` +
      `analysis/codegen src (+ legit tests). NOTE: a faithful port of a React pass that simply does NOT trigger on the current ` +
      `corpus (changes no numbers) is NOT gaming and NOT a hack — it fakes nothing; do NOT flag it. Only flag code that FAKES or ` +
      `SPECIAL-CASES the metric, or pins a wrong result as correct.`,
  }
  return agent(
    `Adversarially review the uncommitted diff in ${WT} for Phase B STEP ${step.id} (${step.title}). Default hacked=true if ` +
    `unconvinced.\nRun: cd ${WT} && git diff\nLENS (${lens}): ${lenses[lens]}\nCite file:line evidence.`,
    { schema: VERDICT, phase: 'Verify', label: `verify:${step.id}:${lens}` }
  )
}

async function commitStep(step, base, now) {
  return agent(
    `In ${WT}, commit Phase B step ${step.id}. Run:\n  cd ${WT} && git add -A crates/ && git commit -q -m ` +
    `"phase-b2 step ${step.id}: ${step.title} (agree ${base.agree}->${now.agree}, ours_only ${base.ours_only}->${now.ours_only})"\n` +
    `Report the new HEAD hash. Nothing else.`,
    { phase: 'Gate', label: `commit:${step.id}` }
  )
}
async function revertStep(step, reason) {
  return agent(
    `In ${WT}, DISCARD all uncommitted changes (Phase B step ${step.id} rejected: ${reason}). Run: cd ${WT} && ` +
    `git checkout -- crates/ && git clean -fdq crates/jsir-transforms/src crates/jsir-ssa/src 2>/dev/null; git status --short\n` +
    `(clean only those two src dirs for new untracked source; NEVER touch the corpus .gz or examples/.) Confirm clean. Report status.`,
    { phase: 'Gate', label: `revert:${step.id}` }
  )
}

function checkGate(step, base, now) {
  if (!now.build_ok || !now.tests_pass) return `build/tests failed (${now.build_error || now.tests_summary || '?'})`
  if (now.agree < base.agree) return `agree regressed ${base.agree}->${now.agree}`
  if (now.panic > base.panic) return `panic rose ${base.panic}->${now.panic}`
  const g = step.gate
  if (g.agree === 'up' && !(now.agree > base.agree)) return `expected agree UP, stayed ${now.agree}`
  if (g.agree === 'same' && now.agree !== base.agree) return `expected agree UNCHANGED, ${base.agree}->${now.agree}`
  if (g.ours_only === 'down' && !(now.ours_only < base.ours_only)) return `expected ours_only DOWN, stayed ${now.ours_only}`
  if (g.ours_only === 'same' && now.ours_only !== base.ours_only) return `expected ours_only UNCHANGED, ${base.ours_only}->${now.ours_only}`
  if (g.ours_only === 'down_or_same' && now.ours_only > base.ours_only) return `expected ours_only not to rise, ${base.ours_only}->${now.ours_only}`
  return null // ok
}

// --------------------------------- driver ----------------------------------
// Committed: 0a, 0b, 1 (ours_only 123->51). DEFERRED (both structurally-invisible on our
// (cache_size, block_count) gate, so they can't move agree and aren't worth a gated round now):
//   - step 2 (PruneAlwaysInvalidating): faithful port but inert on this corpus.
//   - step 4 (PropagateScopeDependencies/property-paths): changes dep EXPRESSIONS, not structure.
// Resume at the real agree-driver — step 5 (block-scope alignment) — which changes block count.
const START = '5'
const startIdx = STEPS.findIndex(s => s.id === START)
phase('Measure')
let base = await measure(STEPS[0], 'baseline')
if (!base.build_ok) { log(`Baseline build broken: ${base.build_error}. Abort.`); return { aborted: true, base } }
log(`Phase B2 baseline (from step ${START}): agree=${base.agree}/${base.universe}, ours_only=${base.ours_only}, panic=${base.panic}, mismatch=${base.mismatch}`)
const baselineAgree = base.agree, baselineOurs = base.ours_only

const done = []
for (const step of STEPS.slice(startIdx)) {
  try {
    phase('Implement')
    await implement(step, base)
    phase('Measure')
    const now = await measure(step, `step${step.id}`)

    const gateFail = checkGate(step, base, now)
    let hacked = false, verdicts = []
    if (!gateFail) {
      phase('Verify')
      verdicts = (await parallel(
        ['correctness', 'semantics', 'stub', 'gaming'].map(lens => () => verify(step, lens))
      )).filter(Boolean)
      hacked = verdicts.some(v => v.hacked && v.confidence !== 'low')
    }

    phase('Gate')
    if (!gateFail && !hacked) {
      const head = await commitStep(step, base, now)
      log(`KEPT step ${step.id} (${step.title}): agree ${base.agree}->${now.agree}, ours_only ${base.ours_only}->${now.ours_only}, mismatch ${base.mismatch}->${now.mismatch}. ${head?.slice?.(0, 80) ?? ''}`)
      base = now
      done.push({ id: step.id, title: step.title, kept: true, agree: now.agree, ours_only: now.ours_only })
    } else {
      const why = gateFail || `anti-hack: ${verdicts.filter(v => v.hacked).map(v => v.evidence).join(' | ').slice(0, 300)}`
      await revertStep(step, why)
      log(`BLOCKED at step ${step.id} (${step.title}): ${why}. Stopping — later steps build on this one.`)
      done.push({ id: step.id, title: step.title, kept: false, why })
      break
    }
  } catch (e) {
    await revertStep(step, `errored: ${String(e).slice(0, 120)}`)
    log(`Step ${step.id} ERRORED: ${String(e).slice(0, 160)}. Stopping.`)
    done.push({ id: step.id, title: step.title, kept: false, why: `errored: ${String(e).slice(0, 160)}` })
    break
  }
}

return {
  baseline: { agree: baselineAgree, ours_only: baselineOurs },
  final: { agree: base.agree, ours_only: base.ours_only, mismatch: base.mismatch, universe: base.universe, panic: base.panic },
  steps: done,
  completed_all: done.length === STEPS.slice(startIdx).length && done.every(s => s.kept),
  blocked_at: done.find(s => !s.kept) || null,
}
