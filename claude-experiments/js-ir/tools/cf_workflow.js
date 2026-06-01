export const meta = {
  name: 'cf-parity',
  description: 'Fixture-driven control-flow parity: pick a react_only cluster, implement the whole chain to make it AGREE with React, gated + anti-hack',
  whenToUse: 'Move agree on control-flow fixtures, one cluster per round, in the isolated js-ir-parity worktree',
  phases: [{ title: 'Measure' }, { title: 'Triage' }, { title: 'Implement' }, { title: 'Verify' }, { title: 'Gate' }],
}

const WT = '/Users/jimmyhmiller/Documents/Code/PlayGround-worktrees/js-ir-parity/claude-experiments/js-ir'
const ENV = 'REACT_CC=/tmp/react-rust/compiler/target/release/react-compiler-e2e'
const DESIGN = `${WT}/crates/jsir-ssa/PHASE_B_REDESIGN.md`
const REACT = '/tmp/react-rust/compiler/crates'
const GATE = `cd ${WT} && ${ENV} cargo run --release -p jsir-ssa --example corpus -- --json 2>/dev/null | tail -1`
const BUILD = `cd ${WT} && cargo build --release -p jsir-ssa -p jsir-convert -p jsir-transforms 2>&1 | tail -20`

const RULES = `
HARD RULES (monorepo others use; correctness over cleverness):
- Work ONLY inside ${WT}. Absolute paths. NEVER edit the gate/oracle (examples/corpus.rs, structure(), react CLI, fixtures).
- The blueprint is ${DESIGN} (React's reactive-scope pipeline ported to us). The React Rust source is at ${REACT} —
  read the relevant pass and port its SEMANTICS. Re-grep symbol names; don't trust line numbers.
- COMMITTED FOUNDATION you build on: provenance (node_id, SrcRef, Cfg.joins, terminal numbering), the straight-line
  IR->IR memoization transform (jsir-transforms/src/memoize.rs + jsir-ssa/src/memoize_plan.rs), the if/else region walk,
  and PruneNonEscapingScopes (escape analysis matching React). Analysis lives in jsir-ssa/src/{mutability,scopes,cfg,lower}.rs.
- ARCHITECTURE: emission stays an IR->IR transform printed via hir2ast->ast2source. NO relooper, NO string-codegen revival,
  keep reversible JSIR pure. NO stubs: any shape you can't make BOTH sound AND structurally-matching MUST hard-error (Err),
  leaving the fixture react_only — never emit wrong structure (mismatch) or wrong semantics (silent-wrong).
- The gate metric is STRUCTURE: (_c(N) cache size, memo-block count) vs the REAL react CLI. A fixture only counts as "agree"
  when both match. Verify each newly-agreeing fixture against the react CLI AND under Node (value + reference stability).`

const COUNTS = {
  type: 'object', required: ['ok', 'agree', 'panic', 'mismatch', 'ours_only', 'universe'],
  properties: {
    ok: { type: 'boolean' }, build_error: { type: 'string' },
    agree: { type: 'number' }, panic: { type: 'number' }, mismatch: { type: 'number' },
    ours_only: { type: 'number' }, react_only: { type: 'number' }, universe: { type: 'number' },
  },
}
const PLAN = {
  type: 'object', required: ['cluster', 'rationale', 'fixtures', 'passes_needed', 'approach'],
  properties: {
    cluster: { type: 'string', description: 'the control-flow cluster to make agree this round' },
    rationale: { type: 'string' },
    fixtures: { type: 'array', items: { type: 'string' }, description: '3-8 react_only fixture basenames in this cluster' },
    passes_needed: { type: 'array', items: { type: 'string' }, description: 'which redesign passes this cluster needs (align/deps/merge/early-return/loops)' },
    approach: { type: 'string' },
  },
}
const VERDICT = {
  type: 'object', required: ['hacked', 'confidence', 'evidence'],
  properties: { hacked: { type: 'boolean' }, confidence: { type: 'string', enum: ['low', 'medium', 'high'] }, evidence: { type: 'string' } },
}

async function measure(label) {
  return agent(
    `Measure the parity gate.\n${RULES}\n\nRun: ${GATE}\nIt prints one JSON line. If you instead see a Rust compile error, ` +
    `set ok=false + build_error=first error, counts 0. Else ok=true and copy agree/panic/mismatch/ours_only/react_only/universe. Do not edit files.`,
    { schema: COUNTS, phase: 'Measure', label }
  )
}

async function triage(base, history) {
  const reverted = history.filter(h => !h.keep)
  const avoid = reverted.length
    ? `\nALREADY TRIED + REVERTED this session (pick a DIFFERENT cluster or a materially different approach):\n` +
      reverted.map(h => `  - "${h.cluster}": ${h.why}`).join('\n') + '\n'
    : ''
  return agent(
    `Triage the next CONTROL-FLOW cluster to make AGREE with React.\n${RULES}\n${avoid}\n` +
    `Current gate: ${JSON.stringify(base)}. Goal: raise "agree" by making a coherent cluster of react_only fixtures ` +
    `structurally match React, WITHOUT raising panic or ours_only.\n\n` +
    `Investigate in ${WT}:\n` +
    `  ${ENV} cargo run --release -p jsir-ssa --example corpus -- --list react_only   (the coverage gap)\n` +
    `  ${ENV} cargo run --release -p jsir-ssa --example corpus -- --show <fixture>    (source / react / ours)\n` +
    `Read ${DESIGN} to see which passes a cluster needs. Group fixtures by the SHAPE that blocks them (e.g. "if/else where a ` +
    `scope spans the branch", "early-return inside a reactive scope", "for-of nonmutating"). Pick ONE cluster that is the ` +
    `largest, cleanest real win and whose full chain you can implement this round. A cluster is a real shape, not one fixture. ` +
    `Return the plan; do not edit yet. NOTE: deps/alignment alone do NOT move the structural gate — only a COMPLETE memoization ` +
    `matching React's (cache,blocks) does, so scope the cluster to what you can take all the way to AGREE.`,
    { schema: PLAN, phase: 'Triage', label: 'triage' }
  )
}

async function implement(plan, base) {
  return agent(
    `Make the control-flow cluster AGREE with React, end-to-end, in ${WT}.\n${RULES}\n\n` +
    `CLUSTER: ${plan.cluster}\nFIXTURES: ${plan.fixtures.join(', ')}\nPASSES NEEDED: ${plan.passes_needed.join(', ')}\n` +
    `APPROACH: ${plan.approach}\n\n` +
    `Baseline: agree=${base.agree}, ours_only=${base.ours_only}, panic=${base.panic}. GATE: agree must RISE, panic stays 0, ` +
    `ours_only must not rise, anti-hack clean.\n\n` +
    `Implement the WHOLE chain these fixtures need (alignment / dep paths / nesting / early-return sentinel / loop flatten — ` +
    `per ${DESIGN} and the React source at ${REACT}). For EACH target fixture, drive it to structurally MATCH the real react ` +
    `CLI (same cache size + memo-block count) and be semantically correct under Node. Steps:\n` +
    `1. Implement (analysis in jsir-ssa/src/{scopes,mutability,cfg,lower}.rs; emission in jsir-transforms/src/memoize.rs + ` +
    `jsir-ssa/src/memoize_plan.rs). Hard-error any sub-shape you can't make sound+matching.\n` +
    `2. Build: ${BUILD}\n3. Run: cd ${WT} && ${ENV} cargo test --release -p jsir-ssa 2>&1 | tail -15\n` +
    `4. For each target fixture: corpus --show it, confirm ours==react structure, and run the emitted JS under Node.\n` +
    `5. Check the gate: ${GATE}\nDo NOT commit. End with whether build+tests pass and the agree/ours_only you observed.`,
    { phase: 'Implement', label: `impl:${plan.cluster.slice(0, 26)}` }
  )
}

async function verify(plan, lens) {
  const lenses = {
    correctness: `Run the REAL react CLI (${ENV}) on this cluster's fixtures and confirm OUR (cache_size, block_count) MATCHES ` +
      `React's, and the emitted JS is value-correct + reference-stable under Node. Flag hacked=true for: structure that ` +
      `DISAGREES with the oracle while counted as agree, silently-wrong output, fake values, or fixture-fitting. ` +
      `Cleanly-hard-errored fixtures (left react_only) are fine.`,
    semantics: `Confirm the implementation is a faithful port of the React passes it claims (read ${DESIGN} + the cited React ` +
      `source). Flag hacked=true if it's a shape-fitted approximation that will mis-handle other inputs of the same cluster.`,
    stub: `Hunt for stubs / silent fallbacks / TODO / unimplemented! / fake values where an unhandled shape must hard-error. ` +
      `Incompleteness behind a clean Err is fine; silent-wrong is not.`,
    gaming: `Hunt for metric-gaming: fixture-name/text special-casing, hardcoded cache sizes, fake _c(N), edits to ` +
      `gate/oracle/fixtures, weakened tests, or dead code that only makes a fixture appear to pass. A faithful pass that ` +
      `doesn't fire is NOT gaming. Confirm git diff touches only real analysis/codegen src (+ legit tests).`,
  }
  return agent(
    `Adversarially review the uncommitted diff in ${WT} for cluster "${plan.cluster}". Default hacked=true if unconvinced.\n` +
    `Run: cd ${WT} && git diff\nLENS (${lens}): ${lenses[lens]}\nCite file:line evidence; read surrounding code.`,
    { schema: VERDICT, phase: 'Verify', label: `verify:${lens}` }
  )
}

async function commitRound(plan, base, now) {
  return agent(
    `In ${WT}, commit. Run:\n  cd ${WT} && git add -A crates/ && git commit -q -m "cf-parity: ${plan.cluster} (agree ${base.agree}->${now.agree}, ours_only ${base.ours_only}->${now.ours_only})"\n` +
    `Then update crates/jsir-ssa/PARITY.md's baseline numbers to agree=${now.agree}/${now.universe}, ours_only=${now.ours_only}, ` +
    `panic=${now.panic} and amend: git add crates/jsir-ssa/PARITY.md && git commit -q --amend --no-edit. Report HEAD hash.`,
    { phase: 'Gate', label: 'commit' }
  )
}
async function revertRound(reason) {
  return agent(
    `In ${WT}, DISCARD all uncommitted changes (round rejected: ${reason}). Run: cd ${WT} && git checkout -- crates/ && ` +
    `git clean -fdq crates/jsir-ssa/src crates/jsir-ssa/tests crates/jsir-transforms/src 2>/dev/null; git status --short\n` +
    `(NEVER touch the corpus .gz or examples/.) Confirm clean. Report status.`,
    { phase: 'Gate', label: 'revert' }
  )
}

// --------------------------------- driver ----------------------------------
phase('Measure')
let base = await measure('baseline')
if (!base.ok) { log(`Baseline build broken: ${base.build_error}. Abort.`); return { aborted: true, base } }
log(`CF baseline: agree=${base.agree}/${base.universe}, ours_only=${base.ours_only}, panic=${base.panic}, mismatch=${base.mismatch}`)
const baselineAgree = base.agree

const MAX_ROUNDS = 3, STALL_LIMIT = 2
let stall = 0
const history = []
for (let round = 1; round <= MAX_ROUNDS; round++) {
  if (base.agree >= base.universe) { log(`Parity reached: agree ${base.agree} == universe.`); break }
  if (stall >= STALL_LIMIT) { log(`Stalled: ${STALL_LIMIT} rounds with no agree gain. Stopping for human review.`); break }
  try {
    phase('Triage')
    const plan = await triage(base, history)
    log(`Round ${round}: cluster "${plan.cluster}" (${plan.fixtures.length} fixtures, needs ${plan.passes_needed.join('+')}) — ${plan.rationale}`)

    phase('Implement')
    await implement(plan, base)
    phase('Measure')
    const now = await measure(`round${round}`)

    const built = now.ok
    const regressed = !built || now.agree < base.agree || now.panic > base.panic || now.ours_only > base.ours_only
    const improved = built && now.agree > base.agree

    let hacked = false, verdicts = []
    if (improved && !regressed) {
      phase('Verify')
      verdicts = (await parallel(['correctness', 'semantics', 'stub', 'gaming'].map(lens => () => verify(plan, lens)))).filter(Boolean)
      hacked = verdicts.some(v => v.hacked && v.confidence !== 'low')
    }

    phase('Gate')
    if (improved && !regressed && !hacked) {
      const head = await commitRound(plan, base, now)
      log(`KEPT round ${round}: "${plan.cluster}" agree ${base.agree}->${now.agree}, ours_only ${base.ours_only}->${now.ours_only}, mismatch ${base.mismatch}->${now.mismatch}. ${head?.slice?.(0, 80) ?? ''}`)
      base = now; stall = 0
      history.push({ round, cluster: plan.cluster, keep: true, agree: now.agree })
    } else {
      const why = !built ? `build failed (${now.build_error || 'see impl log'})`
        : hacked ? `anti-hack: ${verdicts.filter(v => v.hacked).map(v => v.evidence).join(' | ').slice(0, 300)}`
        : !improved ? `no agree gain (stayed ${base.agree})`
        : `regression (agree ${base.agree}->${now.agree}, panic ${now.panic}, ours_only ${base.ours_only}->${now.ours_only})`
      await revertRound(why)
      log(`REVERTED round ${round}: "${plan.cluster}": ${why}`)
      stall++
      history.push({ round, cluster: plan.cluster, keep: false, why })
    }
  } catch (e) {
    await revertRound(`errored: ${String(e).slice(0, 120)}`)
    log(`Round ${round} ERRORED: ${String(e).slice(0, 160)}; reverted, continuing.`)
    stall++
    history.push({ round, cluster: '(errored)', keep: false, why: String(e).slice(0, 160) })
  }
}

return {
  baseline_agree: baselineAgree,
  final: { agree: base.agree, ours_only: base.ours_only, mismatch: base.mismatch, universe: base.universe, panic: base.panic },
  rounds: history,
  agree_gain: base.agree - baselineAgree,
}
