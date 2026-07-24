// Tailwind conformance harness: for every candidate class in Tailwind's own test
// suite, compile it with REAL Tailwind (v4.3.3, full framework via @import) and with
// diffpack's native compiler, isolate the per-class utility rules, and compare.
//
// The comparison is per-class and format-agnostic: both outputs are parsed with
// postcss, the rules attributable to the class (selector mentions it, in any
// media/supports context) are collected, declarations are normalized (sorted,
// whitespace-stripped), and the two rule sets are compared. @property / theme :root /
// preflight infrastructure is excluded — this measures "does diffpack generate the
// correct UTILITY rule", which is the core of a Tailwind reimplementation.
import { readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import postcss from 'postcss';
import tw from '@tailwindcss/postcss';

const DP = process.env.DIFFPACK || new URL('../../target/release/diffpack', import.meta.url).pathname;
const candidatesPath = new URL('candidates.txt', import.meta.url).pathname;
const classes = readFileSync(candidatesPath, 'utf8').split('\n').map(s => s.trim()).filter(Boolean);

// --- diffpack: one batch -> NDJSON {class, ok, css|error} -----------------------
const dpRaw = execFileSync(DP, ['tailwind'], { input: classes.join('\n'), maxBuffer: 1 << 30 }).toString();
const dp = new Map();
for (const line of dpRaw.split('\n')) {
  if (!line.trim()) continue;
  const o = JSON.parse(line);
  dp.set(o.class, o);
}

// --- real Tailwind: compile each class fresh (full framework) -------------------
async function twCompile(cls) {
  const input = `@import "tailwindcss" source(none);@source inline(${JSON.stringify(cls)});`;
  const r = await postcss([tw()]).process(input, { from: 'x.css' });
  return r.css;
}

// Unescape a selector (drop CSS backslash escapes) so we can match a class name
// regardless of how each side escapes `:` `/` `[` `.` etc.
const unescape = (s) => s.replace(/\\(.)/g, '$1');
const normDecls = (rule) => {
  const decls = [];
  rule.walkDecls(d => decls.push(`${d.prop.trim()}:${d.value.replace(/\s+/g, ' ').trim()}`));
  return decls.sort().join(';');
};

// Collect the rules attributable to `cls`: any style rule whose (unescaped) selector
// contains `.<cls>`, tagged with its at-rule wrapper context (media/supports params).
function utilityRules(css, cls) {
  const root = postcss.parse(css);
  const needle = '.' + cls;
  const out = [];
  root.walkRules(rule => {
    // skip preflight/theme: only rules that mention this exact class
    const sel = unescape(rule.selector);
    // match the class as a whole token in the selector
    if (!sel.split(/[\s,>+~()]/).some(part => part === needle || part.startsWith(needle + ':') || part.startsWith(needle + '::') || part.startsWith(needle + '['))) {
      return;
    }
    const ctx = [];
    let p = rule.parent;
    while (p && p.type === 'atrule') {
      if (p.name !== 'layer') ctx.push(`@${p.name} ${p.params.replace(/\s+/g, ' ').trim()}`);
      p = p.parent;
    }
    out.push(`${ctx.sort().join('|')}||${normDecls(rule)}`);
  });
  return out.sort();
}

const cats = { pass_generated: 0, pass_rejected: 0, fail_unsupported: 0, fail_mismatch: 0, fail_overgenerated: 0, fail_dp_error: 0 };
const failures = [];
let i = 0;
for (const cls of classes) {
  i++;
  if (i % 500 === 0) process.stderr.write(`  ${i}/${classes.length}\n`);
  let twCss;
  try { twCss = await twCompile(cls); } catch (e) { continue; } // skip classes real TW itself can't process
  const twRules = utilityRules(twCss, cls);
  const d = dp.get(cls);
  const dpRules = d && d.ok ? utilityRules(d.css, cls) : [];
  const twHas = twRules.length > 0;
  const dpErr = d && !d.ok;

  if (!twHas) {
    // Tailwind generates nothing (invalid/rejected class).
    if (!dpErr && dpRules.length > 0) { cats.fail_overgenerated++; failures.push({ cls, kind: 'overgenerated', dpRules }); }
    else cats.pass_rejected++;
    continue;
  }
  // Tailwind generates a rule.
  if (dpErr) { cats.fail_dp_error++; failures.push({ cls, kind: 'dp_error', error: d.error }); continue; }
  if (dpRules.length === 0) { cats.fail_unsupported++; failures.push({ cls, kind: 'unsupported' }); continue; }
  const a = JSON.stringify(twRules), b = JSON.stringify(dpRules);
  if (a === b) cats.pass_generated++;
  else { cats.fail_mismatch++; failures.push({ cls, kind: 'mismatch', tw: twRules, dp: dpRules }); }
}

const total = Object.values(cats).reduce((a, b) => a + b, 0);
const pass = cats.pass_generated + cats.pass_rejected;
console.log('\n=== Tailwind v4.3.3 conformance (per-class, from its own test suite) ===');
console.log(`total classes tested: ${total}`);
console.log(`PASS: ${pass} (${(100 * pass / total).toFixed(1)}%)`);
console.log(`  - correctly generated: ${cats.pass_generated}`);
console.log(`  - correctly rejected:  ${cats.pass_rejected}`);
console.log(`FAIL: ${total - pass} (${(100 * (total - pass) / total).toFixed(1)}%)`);
console.log(`  - unsupported (TW generates, diffpack empty): ${cats.fail_unsupported}`);
console.log(`  - hard error (diffpack errors):               ${cats.fail_dp_error}`);
console.log(`  - mismatch (both generate, differ):           ${cats.fail_mismatch}`);
console.log(`  - overgenerated (TW empty, diffpack non-empty): ${cats.fail_overgenerated}`);
writeFileSync('failures.json', JSON.stringify(failures, null, 2));
console.log('\n(full failures -> failures.json)');
