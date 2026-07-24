export const meta = {
  name: 'tailwind-conformance-grind',
  description: 'Implement every remaining Tailwind v4 utility + variant in src/tailwind.rs, family by family, byte-exact against the real Tailwind oracle, until the conformance harness passes',
  phases: [
    { title: 'Implement', detail: 'sequential byte-exact implementation per family, gated by the harness + cargo test' },
    { title: 'Verify', detail: 'full conformance harness + lib tests + clippy' },
  ],
}

const A = typeof args === 'string' ? JSON.parse(args) : args
const REPO = A.repo
const HARNESS = A.harness
const DP = A.diffpack

function grepFor(prefixes) {
  // match the family's classes in candidates.txt: prefix at start, or after a
  // variant `:`, optionally negated with `-`.
  return prefixes.map((p) => `(^|:)-?${p}`).join('|')
}

function implPrompt(g) {
  return `You are extending diffpack's NATIVE Tailwind v4 compiler (Rust) so it BYTE-EXACTLY matches real Tailwind CSS v4.3.3. The compiler is one file: ${REPO}/src/tailwind.rs. This is a faithful reimplementation — correctness is byte-for-byte against the real Tailwind oracle, not "close enough".

YOUR GROUP: ${g.name}
SCOPE: every candidate class whose utility base matches one of these prefixes/names: ${g.prefixes.join(', ')}
${g.notes ? 'NOTES: ' + g.notes + '\n' : ''}
THE ORACLE + GATE — use these constantly:
- The full candidate surface (Tailwind's own test classes) is ${HARNESS}/candidates.txt.
- List YOUR group's classes:
    grep -E '${grepFor(g.prefixes)}' ${HARNESS}/candidates.txt
- Compare diffpack vs REAL Tailwind, byte-exact, for any classes:
    cd ${HARNESS} && DIFFPACK=${DP} node verify.mjs <class1> <class2> ...
  It prints "✅ <class>" when diffpack is byte-equal to real Tailwind, or "❌ <class>" followed by
    TW: <what real Tailwind emits>   and   DP: <what diffpack emits (or <ERROR> ...)>.
  Real Tailwind is the oracle. Match its output EXACTLY: selector text, declarations and their values, @media/@supports wrappers, whitespace, calc() forms, color-mix forms, everything.
- You MUST rebuild before every verify after editing:
    cd ${REPO} && cargo build --release
- No regressions allowed:
    cd ${REPO} && cargo test --release --lib tailwind    (must stay green)

METHOD:
1. cd ${REPO}. Grep candidates.txt for your group and run verify.mjs on ALL of them to see current ✅/❌ and the exact expected TW output for each ❌.
2. Read src/tailwind.rs to learn the patterns: render_utility (the dispatch), the per-family blocks, color_value, arbitrary_value, split_color_modifier, parse_variants, the static keyword match (\`let decls ... = match base {\`), Theme, TwProp. Match the existing style.
3. Implement your group's utilities so every one of your classes verifies ✅. Handle bare values, theme tokens, arbitrary [..] values, fractions, negatives (Tailwind often uses \`calc(<v> * -1)\`), and slash modifiers, exactly as the oracle emits them. Some utilities register an @property via TwProp — mirror how existing ones do it.
4. Rebuild, re-run verify.mjs on the WHOLE group, iterate until every class is ✅ (a class both TW and DP emit nothing for is also ✅ — Tailwind correctly rejects it).
5. Make sure \`cargo test --release --lib tailwind\` is green. If you regressed a test that asserted diffpack's OLD (non-Tailwind) output, update that assertion to the correct Tailwind-matching output (verify against the oracle) — but never weaken a test that guards correctness.

HARD RULES:
- NEVER silently drop, fake, or approximate output to make a class "pass" — the byte-exact oracle will show ❌. If you cannot make a class byte-exact, leave it hard-erroring and report it as remaining; do not weaken the hard-error path for unimplemented utilities.
- Keep your diff scoped to your group; cargo test guards the rest.
- Prefer small helper functions and follow the file's idioms.

When done, report exactly how many of your group's classes verify ✅ vs ❌, and for each remaining ❌ give the class and the precise TW-vs-DP difference and why it is hard.`
}

phase('Implement')
const results = []
for (const g of A.groups) {
  const r = await agent(implPrompt(g), {
    label: `impl:${g.name}`,
    phase: 'Implement',
    agentType: 'general-purpose',
    model: 'opus',
    effort: 'high',
    schema: {
      type: 'object',
      additionalProperties: false,
      required: ['group', 'passing', 'remaining', 'summary'],
      properties: {
        group: { type: 'string' },
        passing: { type: 'integer' },
        remaining: { type: 'integer' },
        remaining_classes: { type: 'array', items: { type: 'string' } },
        tests_green: { type: 'boolean' },
        summary: { type: 'string' },
      },
    },
  })
  results.push(r)
  log(`${g.name}: ${r ? `${r.passing} ✅ / ${r.remaining} ❌${r.tests_green === false ? ' (TESTS RED!)' : ''}` : 'agent returned null'}`)
}

phase('Verify')
const final = await agent(
  `Run the FULL Tailwind conformance measurement and report the numbers precisely.
1. cd ${REPO} && cargo build --release  (must succeed)
2. cd ${REPO} && cargo test --release --lib 2>&1 | tail -1   (report the pass/fail counts)
3. cd ${REPO} && cargo clippy --release --all-targets -- -D warnings 2>&1 | grep -cE '^error|^warning: [a-z]'   (0 = clean)
4. cd ${HARNESS} && DIFFPACK=${DP} node conformance.mjs 2>/dev/null | tail -20   (the PASS/FAIL category breakdown)
Return the conformance category counts and whether tests + clippy are green. If conformance is not 100% of generating utilities, also list the top remaining failing families (from ${HARNESS}/failures.json).`,
  {
    label: 'final-verify',
    phase: 'Verify',
    agentType: 'general-purpose',
    model: 'opus',
    effort: 'high',
    schema: {
      type: 'object',
      additionalProperties: false,
      required: ['overall_pass_pct', 'generated', 'unsupported', 'hard_error', 'mismatch', 'overgenerated', 'tests_green', 'clippy_clean', 'summary'],
      properties: {
        overall_pass_pct: { type: 'number' },
        total: { type: 'integer' },
        generated: { type: 'integer' },
        rejected: { type: 'integer' },
        unsupported: { type: 'integer' },
        hard_error: { type: 'integer' },
        mismatch: { type: 'integer' },
        overgenerated: { type: 'integer' },
        tests_green: { type: 'boolean' },
        clippy_clean: { type: 'boolean' },
        top_remaining_families: { type: 'array', items: { type: 'string' } },
        summary: { type: 'string' },
      },
    },
  },
)
return { families: results, final }
