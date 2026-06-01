export const meta = {
  name: 'react-parity',
  description: 'Drive jsir-ssa toward React-Compiler parity, one cluster per round, anti-hack + regression gated',
  whenToUse: 'Make forward progress on React Compiler corpus parity in the isolated js-ir-parity worktree',
  phases: [
    { title: 'Measure' },
    { title: 'Triage' },
    { title: 'Implement' },
    { title: 'Verify' },
    { title: 'Gate' },
    { title: 'Benchmark' },
  ],
}

// ---- fixed environment (ISOLATED worktree — never touch the shared tree) ----
const WT = '/Users/jimmyhmiller/Documents/Code/PlayGround-worktrees/js-ir-parity/claude-experiments/js-ir'
const ENV = 'REACT_CC=/tmp/react-rust/compiler/target/release/react-compiler-e2e'
const GATE = `cd ${WT} && ${ENV} cargo run --release -p jsir-ssa --example corpus -- --json 2>/dev/null | tail -1`
const RULES = `
HARD RULES (this is a monorepo others use; correctness over cleverness):
- Work ONLY inside ${WT}. Use absolute paths. NEVER edit anything outside it.
- NEVER edit the gate or oracle: crates/jsir-ssa/examples/corpus.rs, the react CLI,
  the React fixtures, or the structure() extractor. The React side is ground truth.
- NO stubs / placeholders / silent fallbacks. Per the project rule, an unfinished
  path MUST be a hard error with a clear message, never a fake value or a no-op.
- NO metric-gaming: never special-case fixture names/text, never emit fake _c(N),
  never hardcode cache sizes. A fix must be a real, general implementation.
- ARCHITECTURE (PARITY.md hard requirement): memoization codegen must be an IR->IR
  transform on the reversible JSIR. Do NOT build a CFG->JS relooper and do NOT
  extend the single-block string codegen to fake control flow.
- Keep the reversible JSIR pure: React semantics live in the analysis/transform
  layer only, never baked into the base IR.`

const COUNTS = {
  type: 'object',
  required: ['ok', 'universe', 'agree', 'mismatch', 'react_only', 'ours_only', 'panic'],
  properties: {
    ok: { type: 'boolean', description: 'true if it built and produced JSON; false on compile error' },
    build_error: { type: 'string', description: 'first compiler error if ok=false, else empty' },
    universe: { type: 'number' }, agree: { type: 'number' }, mismatch: { type: 'number' },
    react_only: { type: 'number' }, parser_skip: { type: 'number' }, ours_only: { type: 'number' },
    neither: { type: 'number' }, panic: { type: 'number' }, agreement_pct: { type: 'number' },
  },
}

async function measure(label) {
  return agent(
    `Run the React-parity gate and report its numbers verbatim.\n${RULES}\n\n` +
    `Run exactly:\n  ${GATE}\n\n` +
    `It prints one JSON line. If instead you see a Rust compile error (the build failed), ` +
    `set ok=false and put the first error line in build_error, with counts left at 0. ` +
    `Otherwise set ok=true and copy each field from the JSON. Do not edit any files.`,
    { schema: COUNTS, phase: 'Measure', label: label || 'measure' }
  )
}

const PLAN = {
  type: 'object',
  required: ['target', 'rationale', 'representative_fixtures', 'approach'],
  properties: {
    target: { type: 'string', description: 'the cluster to fix this round (one coherent feature)' },
    rationale: { type: 'string', description: 'why this is the highest-value, cleanest next win' },
    representative_fixtures: { type: 'array', items: { type: 'string' }, description: '3-6 fixture basenames in this cluster' },
    approach: { type: 'string', description: 'concrete plan: which files/functions, what real implementation' },
    est_fixtures: { type: 'number', description: 'rough count of fixtures this could move' },
  },
}

const IMPL = {
  type: 'object',
  required: ['summary', 'files_changed', 'build_ok'],
  properties: {
    summary: { type: 'string' },
    files_changed: { type: 'array', items: { type: 'string' } },
    build_ok: { type: 'boolean' },
    self_check: { type: 'string', description: 'how you verified it is a real fix, not a hack' },
  },
}

const VERDICT = {
  type: 'object',
  required: ['hacked', 'confidence', 'evidence'],
  properties: {
    hacked: { type: 'boolean', description: 'true if the diff hacks/games/stubs rather than really implementing' },
    confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
    evidence: { type: 'string', description: 'specific file:line evidence for the verdict' },
  },
}

async function triage(base, history) {
  const reverted = history.filter(h => !h.keep)
  const avoid = reverted.length
    ? `\nALREADY TRIED AND REVERTED THIS SESSION — do NOT re-attempt the same approach; pick a DIFFERENT, decoupled cluster:\n` +
      reverted.map(h => `  - "${h.target}"  -> rejected: ${h.why}`).join('\n') + '\n'
    : ''
  return agent(
    `You are triaging the NEXT cluster to fix to raise React-Compiler parity.\n${RULES}\n${avoid}\n` +
    `Current gate state: ${JSON.stringify(base)}.\n` +
    `Goal: maximize "agree" (fixtures where our memo structure matches React's), over the ` +
    `"universe" of React-memoized fixtures, WITHOUT raising panic or ours_only.\n\n` +
    `Investigate in ${WT}:\n` +
    `  ${ENV} cargo run --release -p jsir-ssa --example corpus -- --list react_only\n` +
    `  ${ENV} cargo run --release -p jsir-ssa --example corpus -- --list mismatch\n` +
    `  ${ENV} cargo run --release -p jsir-ssa --example corpus -- --list panic\n` +
    `  (the default summary prints the "why we miss" reason histogram — read it)\n` +
    `  ${ENV} cargo run --release -p jsir-ssa --example corpus -- --show <fixture-substring>  (diffs source/react/ours)\n` +
    `Read PARITY.md (it orders the workstreams) and the src to scope the fix.\n\n` +
    `Pick ONE coherent cluster that is the largest, cleanest real win right now. Prefer ` +
    `eliminating panics and lowering-breadth gaps (arrow/new/spread/patterns/control-flow) ` +
    `before deep type-inference work. A "cluster" is one real feature, not "fix fixture X". ` +
    `Return a concrete plan. Do not edit files yet.\n\n` +
    `KNOWN LEADS (from prior runs — use to save investigation, but re-confirm):\n` +
    `  CLEAN, LANDABLE (decouple these — do ONE per round, never bundle):\n` +
    `  - DESTRUCTURING: desugar jsir.array_pattern_ref / jsir.object_pattern_ref (in BOTH declarator\n` +
    `    and function-parameter position) to member-reads + WriteVars in lower.rs. Verified correct in\n` +
    `    a prior run. Rest-elements (...r) and pattern defaults that you can't do soundly must HARD-ERROR,\n` +
    `    not approximate. This is a clean win on its OWN — do NOT add closures to it.\n` +
    `  - Other lowering breadth, each its own small round: jsir.new_expression (~10), jsir.spread_element\n` +
    `    (~5), template literals (~5), jsir.optional_member_expression (~4), regexp/await (~1 each).\n` +
    `  - FIDELITY: the ${base.mismatch} mismatch fixtures (both memoize, (cache,blocks) differ) are\n` +
    `    scope-merge / dependency-precision bugs that usually need NO new lowering. Use --show to see\n` +
    `    where our scope count diverges from React's; many are single-fixture fidelity fixes.\n` +
    `  KNOWN TRAP — DO NOT DO THIS (reverted 3x already):\n` +
    `  - Lowering arrow/function expressions as "opaque values carrying verbatim SOURCE TEXT" is a HACK.\n` +
    `    The single-block string codegen renames SSA values to _vN, but a verbatim closure source keeps\n` +
    `    the ORIGINAL captured names, so the emitted closure references nonexistent vars and silently\n` +
    `    produces wrong output. Real closure support needs proper captured-name substitution or the\n` +
    `    Phase B IR->IR codegen (so real names survive) — it is a SEPARATE HARD cluster, NOT a quick win.\n` +
    `    If a fixture needs closures to memoize correctly, SKIP it this round; pick something landable.\n` +
    `IMPORTANT: expect ours_only/mismatch to RISE as coverage grows (over-memoization / not-yet-perfect\n` +
    `structure on newly-lowered fixtures). That is EXPECTED — separate later clusters. The gate only\n` +
    `cares that agree goes UP and panic does NOT. Do NOT avoid a real fix because ours_only/mismatch rises,\n` +
    `and do NOT bundle a risky feature with a clean one (a hack in either half reverts the whole round).`,
    { schema: PLAN, phase: 'Triage', label: 'triage' }
  )
}

// implement() is schema-LESS on purpose: it does long Rust work and would often
// end its turn without calling StructuredOutput (which throws and kills the run).
// We don't need its structured output — measure() re-runs the gate and tells us
// authoritatively whether it built (now.ok) and what moved.
async function implement(plan, base) {
  return agent(
    `Implement this React-parity cluster for REAL in ${WT}.\n${RULES}\n\n` +
    `TARGET: ${plan.target}\nAPPROACH: ${plan.approach}\n` +
    `REPRESENTATIVE FIXTURES: ${plan.representative_fixtures.join(', ')}\n` +
    `GATE: the round is kept iff agree goes UP (from ${base.agree}) and panic does NOT go up (from ` +
    `${base.panic}), and the anti-hack review is clean. ours_only (${base.ours_only}) and mismatch ` +
    `(${base.mismatch}) are informational — do NOT hold back a real, general fix to keep them low; ` +
    `over-memoization is a separate later cluster. Maximize real agree gain.\n\n` +
    `Steps:\n` +
    `1. Edit the real implementation (likely crates/jsir-ssa/src/{lower,ssa,mutability,scopes,codegen}.rs ` +
    `   and possibly the jsir-swc converter). Implement the feature GENERALLY for all inputs of its kind.\n` +
    `2. Build: cd ${WT} && cargo build --release -p jsir-ssa --example corpus 2>&1 | tail -20\n` +
    `3. Sanity-check a representative fixture with --show. The output must be a faithful ` +
    `   memoization, matching React's intent — not just enough to fool the (cache,blocks) counters.\n` +
    `4. Run the existing tests so you don't regress them: ` +
    `cd ${WT} && ${ENV} cargo test --release -p jsir-ssa 2>&1 | tail -15\n` +
    `PARTIAL IS FINE, SILENT-WRONG IS NOT: you do NOT need to handle every shape of the feature. Any case ` +
    `you don't fully and soundly support (e.g. nested pattern combos like {x:[a,b]}, rest elements, defaults) ` +
    `MUST hard-error with a clear message — never emit wrong output or a fake value for it. A feature that ` +
    `correctly handles the common shapes and cleanly hard-errors the rest WILL pass the gate and is the goal; ` +
    `do not over-reach into shapes you can't get right.\n` +
    `Do NOT commit. Do NOT edit the gate/oracle.\n\n` +
    `End by stating clearly whether the build succeeded and a one-line summary of what you changed.`,
    { phase: 'Implement', label: `impl:${plan.target.slice(0, 30)}` }
  )
}

async function verify(plan, lens) {
  const lenses = {
    stub: `Hunt for STUBS / placeholders / silent fallbacks: TODO, unimplemented!, todo!, returning a ` +
      `fake/empty value where real work belongs, a no-op that pretends to handle a case, returning -1 ` +
      `or None to dodge an error. The project rule: unfinished paths must be HARD ERRORS, not fake values.`,
    gaming: `Hunt for METRIC-GAMING: special-casing on fixture names/paths/source text, hardcoded cache ` +
      `sizes, emitting fake _c(N), any edit to the gate/oracle (examples/corpus.rs, structure(), the react ` +
      `CLI path, fixtures), or weakening a test/assertion. Confirm the diff touches only real src logic.`,
    architecture: `Check the PARITY.md hard requirement: codegen must be an IR->IR transform on the ` +
      `reversible JSIR. Flag any CFG->JS relooper, any extension of the single-block string codegen to ` +
      `fake control flow, and any React semantics baked into the base reversible IR.`,
    correctness: `Check the implementation produces CORRECT output for every input it does NOT hard-error on. ` +
      `Flag hacked=true ONLY for: silently WRONG output, fake/placeholder/sentinel values returned instead of ` +
      `real results, or narrow fitting to the representative fixtures (fixture-name/text special-casing). ` +
      `CRUCIAL: incompleteness is ACCEPTABLE when it hard-errors. If an unhandled shape (e.g. a nested pattern ` +
      `combo like {x:[a,b]}, a rest element, or a default) returns Err(...) with a clear message, that is the ` +
      `project's sanctioned way to be partial — do NOT flag it as hacked. Only flag if an unhandled case ` +
      `produces wrong output or a fake value INSTEAD of erroring. A partial-but-honest feature must PASS.`,
  }
  return agent(
    `Adversarially review the uncommitted diff in ${WT} for ONE failure mode. Default to hacked=true if ` +
    `you are not convinced it is a real fix.\n\nRun: cd ${WT} && git diff\n\n` +
    `TARGET claimed: ${plan.target}\nLENS (${lens}): ${lenses[lens]}\n\n` +
    `Cite specific file:line evidence. Read the surrounding code to judge intent, not just the diff hunk.`,
    { schema: VERDICT, phase: 'Verify', label: `verify:${lens}` }
  )
}

async function commitRound(plan, base, now) {
  return agent(
    `In ${WT}, commit the current changes. Run:\n` +
    `  cd ${WT} && git add -A crates/ && git commit -q -m "react-parity: ${plan.target} (agree ${base.agree}->${now.agree}, panic ${base.panic}->${now.panic})"\n` +
    `Then update PARITY.md's "Current baseline" section with the new numbers ` +
    `(universe=${now.universe} agree=${now.agree} mismatch=${now.mismatch} react_only=${now.react_only} ` +
    `ours_only=${now.ours_only} panic=${now.panic}) and amend the commit: ` +
    `git add crates/jsir-ssa/PARITY.md && git commit -q --amend --no-edit. ` +
    `Report the new HEAD hash. Do nothing else.`,
    { phase: 'Gate', label: 'commit' }
  )
}

async function revertRound(reason) {
  return agent(
    `In ${WT}, DISCARD all uncommitted changes (the last round was rejected: ${reason}).\n` +
    `Run: cd ${WT} && git checkout -- crates/ && git status --short\n` +
    `Confirm the working tree is clean (only gitignored corpus may remain). ` +
    `Do NOT run git clean. Do NOT touch the corpus .gz. Report the clean status.`,
    { phase: 'Gate', label: 'revert' }
  )
}

// --------------------------------- driver ----------------------------------
phase('Measure')
let base = await measure('baseline')
if (!base.ok) {
  log(`Baseline build is broken: ${base.build_error}. Aborting — fix the worktree build first.`)
  return { aborted: true, base }
}
log(`Baseline: agree=${base.agree}/${base.universe} (${base.agreement_pct}%), mismatch=${base.mismatch}, react_only=${base.react_only}, ours_only=${base.ours_only}, panic=${base.panic}`)

const MAX_ROUNDS = 6
const STALL_LIMIT = 2
let stall = 0
const history = []

for (let round = 1; round <= MAX_ROUNDS; round++) {
  if (base.agree >= base.universe) { log(`Reached parity: agree ${base.agree} == universe ${base.universe}.`); break }
  if (stall >= STALL_LIMIT) { log(`Stalled: ${STALL_LIMIT} consecutive rounds with no kept progress. Stopping for human review.`); break }

  // A single agent throwing (e.g. ending without StructuredOutput) must NOT kill
  // the whole run — catch it, revert any partial edits, count a stall, continue.
  try {
    phase('Triage')
    const plan = await triage(base, history)
    log(`Round ${round}: ${plan.target} (~${plan.est_fixtures ?? '?'} fixtures) — ${plan.rationale}`)

    phase('Implement')
    await implement(plan, base)

    phase('Measure')
    const now = await measure(`round${round}`)

    // The ONLY hard invariants: agree never decreases, panic never increases.
    // ours_only/mismatch are INFORMATIONAL — they rise naturally as coverage grows
    // (over-memoized newly-compiled fixtures, or newly-lowered fixtures not yet at
    // React's exact structure). Over-memoization is its own later cluster. Don't gate on it.
    const built = now.ok
    const regressed = !built || now.agree < base.agree || now.panic > base.panic
    const improved = built && (now.agree > base.agree || (now.agree === base.agree && now.panic < base.panic))

    // Anti-hack: only spend verifier tokens when the change actually moved the gate forward.
    let hacked = false, verdicts = []
    if (built && improved && !regressed) {
      phase('Verify')
      verdicts = (await parallel(
        ['stub', 'gaming', 'architecture', 'correctness'].map(lens => () => verify(plan, lens))
      )).filter(Boolean)
      hacked = verdicts.some(v => v.hacked && v.confidence !== 'low')
    }

    phase('Gate')
    const keep = built && improved && !regressed && !hacked
    if (keep) {
      const head = await commitRound(plan, base, now)
      log(`KEPT round ${round}: agree ${base.agree}->${now.agree}, panic ${base.panic}->${now.panic}, mismatch ${base.mismatch}->${now.mismatch}. ${head?.slice?.(0, 80) ?? ''}`)
      base = now
      stall = 0
      history.push({ round, target: plan.target, keep: true, agree: base.agree, panic: base.panic, mismatch: base.mismatch })
    } else {
      const why = !built ? `build failed (${now.build_error || 'see implement log'})`
        : hacked ? `anti-hack rejected: ${verdicts.filter(v => v.hacked).map(v => v.evidence).join(' | ').slice(0, 300)}`
        : regressed ? `regression (agree ${base.agree}->${now.agree}, panic ${base.panic}->${now.panic})`
        : `no improvement (agree stayed ${base.agree}, panic stayed ${base.panic})`
      await revertRound(why)
      log(`REVERTED round ${round}: ${why}`)
      stall++
      history.push({ round, target: plan.target, keep: false, why, agree: base.agree, panic: base.panic, mismatch: base.mismatch })
    }
  } catch (e) {
    log(`Round ${round} ERRORED (${String(e).slice(0, 160)}); reverting partial edits and continuing.`)
    await revertRound(`round errored: ${String(e).slice(0, 120)}`)
    stall++
    history.push({ round, target: '(errored)', keep: false, why: String(e).slice(0, 160), agree: base.agree, panic: base.panic, mismatch: base.mismatch })
  }
}

// ----------------------- benchmark + optimize (only at parity) -----------------------
let benchmark = null
if (base.agree >= base.universe) {
  phase('Benchmark')
  benchmark = await agent(
    `Parity reached. Benchmark OUR compiler vs the real react-compiler-e2e CLI over the comparable ` +
    `fixtures (the ones both memoize) in ${WT}.\n${RULES}\n\n` +
    `Build a fair wall-clock comparison: for each comparable fixture, time our codegen::compile vs the ` +
    `react CLI (account for process startup honestly — e.g. compile a batch in one of our processes vs ` +
    `the CLI's per-invocation cost, and also measure pure in-process compile time). Report median/total ` +
    `for each side and whether we are faster or slower, with the methodology. Do not edit src; this is ` +
    `measurement only. If we are SLOWER, also profile where our time goes and list the top hotspots.`,
    { phase: 'Benchmark', label: 'benchmark' }
  )
  log(`Benchmark complete. ${benchmark?.slice?.(0, 200) ?? ''}`)
}

return {
  baseline_agree: history.length ? history[0]?.agree : base.agree,
  final: { agree: base.agree, universe: base.universe, mismatch: base.mismatch, panic: base.panic, ours_only: base.ours_only, agreement_pct: base.agreement_pct },
  rounds: history,
  reached_parity: base.agree >= base.universe,
  benchmark,
}
