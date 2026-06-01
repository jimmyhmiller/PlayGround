export const meta = {
  name: 'phase-b-impl',
  description: 'Implement Phase B IR->IR memoization codegen step-by-step per PHASE_B_DESIGN.md, each step gated + anti-hack verified',
  whenToUse: 'Execute the Phase B IR-rewrite plan in the isolated js-ir-parity worktree',
  phases: [{ title: 'Measure' }, { title: 'Implement' }, { title: 'Verify' }, { title: 'Gate' }],
}

const WT = '/Users/jimmyhmiller/Documents/Code/PlayGround-worktrees/js-ir-parity/claude-experiments/js-ir'
const ENV = 'REACT_CC=/tmp/react-rust/compiler/target/release/react-compiler-e2e'
const DESIGN = `${WT}/crates/jsir-ssa/PHASE_B_DESIGN.md`
const GATE = `cd ${WT} && ${ENV} cargo run --release -p jsir-ssa --example corpus -- --json 2>/dev/null | tail -1`

const RULES = `
HARD RULES (monorepo others use; correctness over cleverness):
- Work ONLY inside ${WT}. Absolute paths. NEVER edit anything outside it. NEVER edit the gate/oracle
  (crates/jsir-ssa/examples/corpus.rs, structure(), the react CLI, the fixtures).
- This is the Phase B plan in ${DESIGN}. READ IT. Anchors there may have drifted — re-grep symbol names
  (emit_memoized, emit_scope, Lower::def, ScopeInfo, ast2hir builder, hir2ast fn stmt) rather than trust line numbers.
- ARCHITECTURE (non-negotiable, the whole point of Phase B): memoization is an IR->IR transform on the
  reversible JSIR, printed via hir2ast -> ast2source. NO relooper (do not reconstruct control flow from the
  CFG; the JSIR op-tree already IS structured — rewrite it in place). Do NOT extend the single-block string
  codegen (codegen.rs emit_memoized is the FROZEN parity reference, not the production path). Keep the
  reversible JSIR pure — synthesized ops must use only existing op names that hir2ast already lifts.
- NO stubs / silent fallbacks. Any unhandled op shape MUST return Err(...) with a clear message (matching the
  existing hard errors), never a fake value or silent skip. Partial-but-honest is the goal; do not over-reach.`

const MEASURE = {
  type: 'object',
  required: ['build_ok', 'tests_pass', 'agree', 'panic', 'mismatch', 'universe'],
  properties: {
    build_ok: { type: 'boolean' },
    tests_pass: { type: 'boolean', description: 'true if all requested cargo test suites passed' },
    tests_summary: { type: 'string', description: 'which suites ran and pass/fail counts; any failure detail' },
    roundtrip_ok: { type: 'boolean', description: 'jsir-convert round-trip + golden fixtures unchanged (only relevant Steps 0/1; else true)' },
    agree: { type: 'number' }, panic: { type: 'number' }, mismatch: { type: 'number' },
    universe: { type: 'number' }, ours_only: { type: 'number' },
    build_error: { type: 'string' },
  },
}

const VERDICT = {
  type: 'object',
  required: ['hacked', 'confidence', 'evidence'],
  properties: {
    hacked: { type: 'boolean' },
    confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
    evidence: { type: 'string' },
  },
}

// The 8 steps from PHASE_B_DESIGN.md section 5. `tests` is the extra suites the
// measure step must run for that step's gate (beyond corpus).
const STEPS = [
  { id: 0, title: 'node_id on jsir_ir::Op',
    spec: `Add \`node_id: Option<u32>\` to jsir_ir::Op; assign monotonically in ast2hir (piggyback the Builder counter). ` +
      `Ensure the textual printer and hir2ast IGNORE it (no behavioral dependence). This is pure infrastructure: ` +
      `the corpus numbers and byte-exact round-trip MUST be unchanged.`,
    tests: 'cargo test --release -p jsir-convert (round-trip + 44 golden fixtures); cargo test --release -p jsir-ssa',
    expect: 'no change to corpus or round-trip; roundtrip_ok must be true' },
  { id: 1, title: 'thread SrcRef JSIR->CFG->SSA + expose jsir_ssa::plan()',
    spec: `Add SrcRef { stmt_node_id } onto cfg::Instr; Lower tracks cur_stmt_node_id (set when entering a statement-root op) ` +
      `and stamps it via def/push/push_effect; preserve src across ssa::materialize (it rides the surviving Instr, NOT a ` +
      `value-keyed side table). Expose jsir_ssa::plan(file_fn) -> MemoPlan { fn_name, infos, cfg (with src), single_block }. ` +
      `Analysis-only: corpus MUST be unchanged.`,
    tests: 'cargo test --release -p jsir-ssa',
    expect: 'corpus unchanged (no emit change yet)' },
  { id: 2, title: 'straight-line IR->IR transform (reproduce emit through IR)',
    spec: `New crate module jsir-transforms/src/memoize.rs with the builder helpers + the straight-line algorithm from ` +
      `design section 2 (prepend import + cache decl, hoist outputs, walk statements, relocate each scope's owned ` +
      `statements into a synthesized jshir.if_statement with dep-compare/sentinel test, slot reads/writes). Route compile's ` +
      `MEMOIZED branch through it (leave the pass-through branch as-is). Reproduce emit_memoized's cache-size/slot/sort ` +
      `behavior EXACTLY first. Add the contiguity assertion (Err with fixture name if a scope's stmt_node_id set isn't contiguous).`,
    tests: 'cargo test --release -p jsir-ssa (incl. node-semantic codegen/jsx tests must point at / still pass on the new path)',
    expect: 'agree >= baseline; panic 0; structure-identical on all currently-agreeing fixtures' },
  { id: 3, title: 'if/else (recursive region walk, RULE 2)',
    spec: `Extend the transform to walk nested regions (jshir.if_statement consequent/alternate) so scopes within an ` +
      `if/else memoize, AND wire multi-block functions through the production \`compile\` path (today codegen.rs hard-errors ` +
      `all multi-block fns; relax that to route SOUND if/else through memoize_file so the gate's agree RISES). ` +
      `No relooper — wrap the existing JSIR if construct.\n\n` +
      `SOUNDNESS INVARIANT (enforce THIS, do NOT enumerate shapes — two prior attempts shipped silent-wrong bugs by ` +
      `guarding only specific shapes):\n` +
      `  A scope is memoizable ONLY IF the single emitted memo guard provably ENCLOSES EVERY instruction that mutates ` +
      `any of the scope's values (StoreMember on a scope value, or a call that may mutate a scope value passed as a ref ` +
      `arg). Equivalently: a scope is memoizable iff ALL its owned statements (alloc + every mutation) lie in ONE ` +
      `contiguous straight-line range that the guard wraps. If the scope's owned statements span a branch boundary — ` +
      `alloc in one region and mutation in another, or owned statements in both consequent and alternate, or any ` +
      `mutation reachable OUTSIDE the guard — you MUST hard-error (Err) and leave it react_only. Pure (mutation-free) ` +
      `value scopes are always fine. Be CONSERVATIVE: when unsure a scope is fully enclosed, Err. These deferred cases ` +
      `are handled by step 5 (alignment).\n\n` +
      `REQUIRED HARD-ERROR TESTS (both are confirmed silent-wrong counterexamples — they MUST return Err, not compile, ` +
      `and your test suite must assert that):\n` +
      `  1. \`function Component(props){ const a=[]; if(props.c){ a.push(props.x); } return a; }\`  (alloc before if)\n` +
      `  2. \`function Component(props){ let a; if(props.c){ a=[]; a.push(props.p0); } else { a=[]; } return Foo(a); }\`  (alloc inside branch)\n` +
      `For every case you DO memoize, verify the emitted JS under Node: same return value across many prop sets AND ` +
      `reference-stable when deps unchanged. The correctness verifier WILL build adversarial mutation-in-branch cases.`,
    tests: 'cargo test --release -p jsir-ssa',
    expect: 'agree RISES (fully-enclosed if/else scopes like deps-in-branch / jsx-in-branch / obj-literal-cached-in-if-else); ' +
      'EVERY mutation-crossing-branch case stays react_only via Err; no regression; no silent-wrong output' },
  { id: 4, title: 'early-return sentinel + labeled break (RULE 3/6)',
    spec: `Implement early-return memoization via the sentinel + labeled-break pattern from design 3.2. First verify/extend ` +
      `hir2ast to lift jshir.labeled_statement / jshir.break_statement (round-trip a hand-built subtree in isolation before wiring in).\n\n` +
      `DEP-RECONSTRUCTABILITY SOUNDNESS (a prior attempt shipped a silent-wrong bug here): a scope's memo guard re-emits ` +
      `each dependency expression at the guard site. This is only sound if the dep expression can be reconstructed from ` +
      `names IN SCOPE at the guard. A COMPUTED-member dep \`a[k]\` whose key temp \`k\` is defined INSIDE the relocated ` +
      `block is NOT reconstructable (the temp is out of scope at the guard) — emitting it references an undefined name / ` +
      `wrong value. Any dependency that is not reconstructable from in-scope names (computed-member key defined inside the ` +
      `range, or any value not live at the guard) MUST cause the scope to hard-error (Err), NOT emit a broken dep. Only ` +
      `memoize early-return scopes whose deps are all reconstructable (static-member, param, simple in-scope var). Verify ` +
      `each memoized case under Node (value + reference stability); the verifier WILL build computed-member adversarial cases.`,
    tests: 'cargo test --release -p jsir-convert; cargo test --release -p jsir-ssa',
    expect: 'agree RISES (early-return-within-reactive-scope, conditional-early-return, ...); computed-member deps Err; ' +
      'plain early-return (no reactive scope) stays NoMemo; no regression; no silent-wrong output' },
  { id: 5, title: 'scope alignment (RULE 5)',
    spec: `Implement React's scope-alignment in the ANALYSIS (scopes.rs): a scope starting inside one branch but used after ` +
      `the join hoists to the join point. If a fixture can't be aligned soundly yet, Err (stays react_only) rather than ` +
      `emitting a wrong structure (which would show as mismatch).`,
    tests: 'cargo test --release -p jsir-ssa',
    expect: 'new agrees (align-scope-starts-within-cond, align-scopes-*); watch mismatch does not grow' },
  { id: 6, title: 'loops (RULE 7)',
    spec: `Extend the transform to memoize scopes around for-of/for/while by wrapping the existing JSIR loop construct.`,
    tests: 'cargo test --release -p jsir-ssa',
    expect: 'new agrees (for-of-nonmutating-*, repro-memoize-for-of-*); no regression' },
  { id: 7, title: 'closures as memoizable values',
    spec: `Memoize a nested function-expression as an allocation with its captured variables as operands/deps (design ` +
      `section 4). A function-expression scope with any free reactive variable whose capture is not yet supported MUST Err ` +
      `(never emit a closure guard with incomplete deps -> would cache stale).`,
    tests: 'cargo test --release -p jsir-ssa',
    expect: 'new agrees on closure fixtures; unsupported captures Err, not silent' },
]

async function measure(step, label) {
  return agent(
    `Measure the Phase B gate for ${label}.\n${RULES}\n\n` +
    `Run, from ${WT}:\n` +
    `  1. Build: cargo build --release -p jsir-ssa -p jsir-convert -p jsir-transforms 2>&1 | tail -20\n` +
    `  2. Tests for this step: ${step.tests}\n` +
    `  3. Corpus gate: ${GATE}  (one JSON line)\n` +
    `Report: build_ok; tests_pass (+ tests_summary); roundtrip_ok (true unless a round-trip/golden suite FAILED — ` +
    `relevant for Steps 0/1); and the corpus fields agree/panic/mismatch/universe/ours_only. If the build failed, ` +
    `set build_ok=false with the first error in build_error and leave counts at 0. Do not edit files.`,
    { schema: MEASURE, phase: 'Measure', label }
  )
}

async function implement(step, base) {
  return agent(
    `Implement Phase B STEP ${step.id} (${step.title}) for REAL in ${WT}.\n${RULES}\n\n` +
    `READ ${DESIGN} first (esp. the sections relevant to this step), and re-grep the symbols you'll touch.\n\n` +
    `STEP ${step.id} SPEC: ${step.spec}\n` +
    `EXPECTED GATE OUTCOME: ${step.expect}\n` +
    `Current gate: agree=${base.agree}, panic=${base.panic}, mismatch=${base.mismatch}.\n\n` +
    `Steps:\n` +
    `1. Implement it generally and soundly. Unhandled shapes hard-error (Err), never fake/silent.\n` +
    `2. Build: cd ${WT} && cargo build --release -p jsir-ssa -p jsir-convert -p jsir-transforms 2>&1 | tail -25\n` +
    `3. Run this step's tests (${step.tests}) and confirm they pass.\n` +
    `4. Sanity-check against the gate (${GATE}) and a couple of the expected fixtures with the corpus --show.\n` +
    `Do NOT commit. Do NOT edit the gate/oracle. End by stating whether the build + tests passed and a one-line summary.`,
    { phase: 'Implement', label: `step${step.id}:${step.title.slice(0, 24)}` }
  )
}

async function verify(step, lens) {
  const lenses = {
    architecture: `THE central Phase B check: confirm this is a genuine IR->IR transform on the reversible JSIR (synthesized ` +
      `jsir_ir::Op spliced into the op-tree, printed via ir_to_source). Flag hacked=true for ANY relooper (CFG->JS), ` +
      `any extension of the string codegen (emit_memoized) to do real work, any new op name hir2ast can't lift, or React ` +
      `semantics baked into the base reversible IR. The string codegen must remain a frozen reference, not the live path.`,
    stub: `Hunt for stubs / silent fallbacks: TODO, unimplemented!, todo!, a fake/empty value or silent skip where an ` +
      `unhandled op shape should hard-error. Per the rules, unfinished shapes MUST be Err(...). Incompleteness behind a ` +
      `clean Err is ACCEPTABLE — only flag silent-wrong / fake values.`,
    gaming: `Hunt for metric-gaming: fixture-name/text special-casing, hardcoded cache sizes, fake _c(N), any edit to the ` +
      `gate/oracle/fixtures, or weakening a test. Confirm git diff touches only real src logic + (for steps that say so) tests.`,
    correctness: `Check the emitted memoization is SOUND for inputs it does NOT hard-error on: correct deps (no stale ` +
      `caching), correct slot numbering, statements relocated in correct order, outputs hoisted correctly. Verify via the ` +
      `node-semantic tests (emitted JS runs + is reference-stable). Flag hacked=true ONLY for silently-wrong output / fake ` +
      `values / narrow fitting — NOT for cleanly-hard-errored incompleteness.`,
  }
  return agent(
    `Adversarially review the uncommitted diff in ${WT} for Phase B STEP ${step.id} (${step.title}). Default hacked=true ` +
    `if unconvinced.\n\nRun: cd ${WT} && git diff\n\nLENS (${lens}): ${lenses[lens]}\n\nCite file:line evidence; read ` +
    `surrounding code to judge intent.`,
    { schema: VERDICT, phase: 'Verify', label: `verify:${step.id}:${lens}` }
  )
}

async function commitStep(step, base, now) {
  return agent(
    `In ${WT}, commit Phase B step ${step.id}. Run:\n` +
    `  cd ${WT} && git add -A crates/ && git commit -q -m "phase-b step ${step.id}: ${step.title} (agree ${base.agree}->${now.agree}, panic ${now.panic})"\n` +
    `Report the new HEAD hash. Nothing else.`,
    { phase: 'Gate', label: `commit:${step.id}` }
  )
}

async function revertStep(step, reason) {
  return agent(
    `In ${WT}, DISCARD all uncommitted changes (Phase B step ${step.id} rejected: ${reason}).\n` +
    `Run: cd ${WT} && git checkout -- crates/ && git clean -fdq crates/jsir-transforms/src crates/jsir-ssa/src 2>/dev/null; git status --short\n` +
    `(git clean only the two src dirs to drop any NEW untracked source file the step added; NEVER touch the corpus .gz or ` +
    `examples/.) Confirm the tree is clean. Report status.`,
    { phase: 'Gate', label: `revert:${step.id}` }
  )
}

// --------------------------------- driver ----------------------------------
// Resume point: steps 0-3 are committed (straight-line transform + sound if/else
// region walk). Bump this as steps land. (Hardcoded — args passing proved unreliable.)
const START = 4
phase('Measure')
let base = await measure(STEPS[0], 'baseline')
if (!base.build_ok) { log(`Baseline build broken: ${base.build_error}. Abort.`); return { aborted: true, base } }
log(`Phase B baseline (resuming at step ${START}): agree=${base.agree}/${base.universe}, panic=${base.panic}, mismatch=${base.mismatch}`)
const baselineAgree = base.agree

const done = []
for (const step of STEPS.filter(s => s.id >= START)) {
  let outcome
  try {
    phase('Implement')
    await implement(step, base)

    phase('Measure')
    const now = await measure(step, `step${step.id}`)

    // Gate: never regress (agree non-decreasing, panic 0), build + tests pass,
    // round-trip intact (Steps 0/1), and anti-hack clean. Incremental steps need
    // only be no-regression (agree gains often arrive a step later); a step that
    // builds clean, passes tests, and doesn't regress is kept.
    const built = now.build_ok && now.tests_pass
    const regressed = !built || now.agree < base.agree || now.panic > base.panic || now.roundtrip_ok === false

    let hacked = false, verdicts = []
    if (built && !regressed) {
      phase('Verify')
      verdicts = (await parallel(
        ['architecture', 'stub', 'gaming', 'correctness'].map(lens => () => verify(step, lens))
      )).filter(Boolean)
      hacked = verdicts.some(v => v.hacked && v.confidence !== 'low')
    }

    phase('Gate')
    if (built && !regressed && !hacked) {
      const head = await commitStep(step, base, now)
      log(`KEPT step ${step.id} (${step.title}): agree ${base.agree}->${now.agree}, panic ${now.panic}, mismatch ${base.mismatch}->${now.mismatch}. ${head?.slice?.(0, 80) ?? ''}`)
      base = now
      outcome = { id: step.id, title: step.title, kept: true, agree: now.agree, panic: now.panic }
      done.push(outcome)
    } else {
      const why = !built ? `build/tests failed (${now.build_error || now.tests_summary || 'see implement log'})`
        : hacked ? `anti-hack: ${verdicts.filter(v => v.hacked).map(v => v.evidence).join(' | ').slice(0, 300)}`
        : now.roundtrip_ok === false ? `round-trip / golden fixtures regressed`
        : `regression (agree ${base.agree}->${now.agree}, panic ${now.panic})`
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

const lastKept = done.filter(s => s.kept).pop()
return {
  baseline_agree: baselineAgree,
  final_agree: base.agree,
  steps: done,
  completed_all: done.length === STEPS.length && done.every(s => s.kept),
  blocked_at: done.find(s => !s.kept) || null,
}
