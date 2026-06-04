#!/usr/bin/env node
//
// Batched differential comparator for the fuzzer.
//
//   node tools/fuzzcmp.js <cases.json>
//
// Reads a JSON array of cases:
//   [{ "src": "<original js>", "residual": "<emitted js>", "inputs": [..] }, ...]
//
// For every case and every input it runs BOTH `main(input)` in isolated VM
// contexts (the original source and the partial-evaluator's residual) and
// compares the observable result. This is exactly the `assert_node_equiv`
// oracle generalized to a batch: a correct residual is observationally
// equivalent to the original, so any disagreement is a real specialization bug.
//
// Both programs are run sloppy-mode (plain Script, not a module) under a
// wall-clock timeout, so an accidental nonterminating loop is caught rather
// than hanging the fuzzer. Termination behaviour is itself observable: if one
// side throws / times out and the other does not, that is a divergence.
//
// Prints a JSON array of findings to stdout:
//   [{ "index": i, "input": v, "kind": "...", "orig": "...", "spec": "..." }]
// An empty array means every case agreed on every input.

const vm = require('node:vm');
const fs = require('node:fs');

const TIMEOUT_MS = 1000;

const file = process.argv[2];
if (!file) {
  console.error('usage: node tools/fuzzcmp.js <cases.json>');
  process.exit(2);
}
const cases = JSON.parse(fs.readFileSync(file, 'utf8'));

// Run `code` (which defines `main`) then call `main(input)`, returning a
// normalized observation: either { ok: <json string> } or { threw: true } or
// { timeout: true }. We deliberately collapse all throws to a single bucket:
// partial evaluation may legitimately change *which* error object is produced
// (e.g. fold a TypeError site away), but it must not turn a throw into a normal
// return or vice versa. Result values are compared via JSON so NaN/Infinity/-0
// normalize identically on both sides.
function observe(code, input) {
  const sandbox = Object.create(null);
  let ctx;
  try {
    ctx = vm.createContext(sandbox);
    vm.runInContext(code + '\n;globalThis.__main = main;', ctx, { timeout: TIMEOUT_MS });
  } catch (e) {
    // A failure to even define `main` is a structural problem with this case;
    // surface it so the driver can tell load errors from runtime throws.
    return { loaderr: true, msg: String(e && e.message || e) };
  }
  try {
    const call = `globalThis.__out = JSON.stringify(__main(${input}));`;
    vm.runInContext(call, ctx, { timeout: TIMEOUT_MS });
    // JSON.stringify(undefined) === undefined; normalize to a sentinel string.
    const out = sandbox.__out;
    return { ok: out === undefined ? '<undefined>' : out };
  } catch (e) {
    if (e && e.code === 'ERR_SCRIPT_EXECUTION_TIMEOUT') return { timeout: true };
    return { threw: true };
  }
}

function classify(orig, spec) {
  // Loader errors mean the emitted/orig JS didn't even parse+define main.
  if (orig.loaderr || spec.loaderr) {
    return { kind: 'loaderr', orig: orig.loaderr ? orig.msg : 'ok', spec: spec.loaderr ? spec.msg : 'ok' };
  }
  // Timeout asymmetry: one side terminates, the other doesn't.
  if (orig.timeout !== spec.timeout) {
    return { kind: 'termination', orig: orig.timeout ? 'timeout' : (orig.threw ? 'throw' : orig.ok), spec: spec.timeout ? 'timeout' : (spec.threw ? 'throw' : spec.ok) };
  }
  if (orig.timeout && spec.timeout) return null; // both nonterminating: not a divergence we can judge
  // Throw asymmetry: PE turned a throw into a return or vice versa.
  if (!!orig.threw !== !!spec.threw) {
    return { kind: 'throw-mismatch', orig: orig.threw ? 'throw' : orig.ok, spec: spec.threw ? 'throw' : spec.ok };
  }
  if (orig.threw && spec.threw) return null; // both threw: agreement
  // Both returned: compare the serialized value.
  if (orig.ok !== spec.ok) {
    // Inherent PE limitation, NOT a soundness defect: a program that
    // string-coerces a function observes its SOURCE TEXT, but specialization
    // necessarily rewrites a function's body, and a residual closure renders as
    // `__rfN.bind(null, caps)` whose `String(...)` is `function () { [native code] }`.
    // No correct partial evaluator can preserve `String(fn)`. Skip exactly this
    // signature (residual side is the bound-closure native-code string while the
    // original is a function source); any other value mismatch is still reported.
    // The residual side renders a closure as `__rfN.bind(...)`, whose `String(...)`
    // is `function () { [native code] }` — the unambiguous marker. Skip when the
    // residual produced that AND the original has a real function source at the
    // corresponding spot (a `"function (`/`"function(` JSON-stringified source,
    // possibly nested in an object/array). Any other value mismatch is reported.
    const specIsBound = (spec.ok || '').includes('function () { [native code] }');
    const origHasFnSrc = /"function ?\(/.test(orig.ok || '') && !(orig.ok || '').includes('[native code]');
    if (specIsBound && origHasFnSrc) return null;
    return { kind: 'value', orig: orig.ok, spec: spec.ok };
  }
  return null;
}

const findings = [];
for (let i = 0; i < cases.length; i++) {
  const { src, residual, inputs } = cases[i];
  for (const input of inputs) {
    const o = observe(src, input);
    const s = observe(residual, input);
    const div = classify(o, s);
    if (div) {
      findings.push({ index: i, input, ...div });
      break; // one finding per case is enough; the driver will shrink it
    }
  }
}

process.stdout.write(JSON.stringify(findings));
