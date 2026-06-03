#!/usr/bin/env node
//
// Differential effect tracer for the JS partial evaluator.
//
//   node tools/difftrace.js <source.js>
//
// A correct residual is *observationally equivalent* to the original: it runs
// exactly the same sequence of calls to functions the program does not itself
// define (the "external" boundary: `Date`, `String.fromCharCode`,
// `new Uint8Array(...)`, the real `TextDecoder`, `console`, ...), with the same
// arguments, and terminates the same way. Partial evaluation only folds away
// *internal*, side-effect-free computation, so it can never change that external
// trace. Therefore the first point where the two traces differ is the first
// place the residual's observable behavior diverges from the original — which is
// where the specialization bug bit.
//
// This runs the original (its top-level code) and the residual (its `main(0)`)
// each in an isolated context whose globals are logging proxies, then prints the
// first divergence with a window of context around it.

const vm = require('node:vm');
const fs = require('node:fs');
const { execSync } = require('node:child_process');
const path = require('node:path');

const SRC = process.argv[2];
if (!SRC) {
  console.error('usage: node tools/difftrace.js <source.js>');
  process.exit(2);
}
const repo = path.resolve(__dirname, '..');

// --- get the residual JS from the frontend ---------------------------------
function residualOf(srcPath) {
  const out = execSync(
    `cargo run -q --release -p js-frontend --bin js-frontend -- --js ${JSON.stringify(srcPath)}`,
    { cwd: repo, maxBuffer: 256 * 1024 * 1024 },
  ).toString();
  // strip the "--- residual ... ---" banner line(s)
  return out.split('\n').filter((l) => !l.startsWith('--- ')).join('\n');
}

// --- run code in a context whose globals are logged ------------------------
//
// `has` returns true for every name so the VM routes all free identifiers
// through the proxy; `get` serves program-defined globals from the store, real
// globals as call-logging wrappers, and everything else as `undefined`; `set`
// stores assignments. We only log calls to *external* globals, so program-
// defined functions (handlers, `main`, ...) are silent — exactly the boundary
// that folding must preserve.
// Globals available to the program. `tracked` ones are *effectful or
// nondeterministic*, so the partial evaluator never folds them away — they must
// appear identically in both traces, so we wrap and log them. The rest are pure
// (and may be folded when their args are static, which legitimately removes them
// from the residual's runtime trace), so we expose them unwrapped and unlogged.
const TRACKED = new Set(['Date', 'console']);

function run(code, extraTracked) {
  const log = [];
  const store = Object.create(null);
  const tracked = new Set([...TRACKED, ...(extraTracked || [])]);
  const reals = {
    Date, String, Number, Boolean, Array, Object, Math, JSON,
    Uint8Array, Uint16Array, Int32Array, ArrayBuffer,
    TextDecoder, TextEncoder, parseInt, parseFloat, isNaN, isFinite,
    RegExp, Symbol, Error, TypeError, RangeError, Promise, console,
  };

  const ser = (v) => {
    try {
      if (v instanceof Uint8Array) v = `Uint8Array[${v.length}]:` + Array.from(v.slice(0, 24));
      let s = JSON.stringify(v, (k, x) =>
        typeof x === 'function' ? '[fn]'
          : x instanceof Uint8Array ? `Uint8Array[${x.length}]`
          : x);
      if (s === undefined) s = String(v);
      return s.length > 160 ? s.slice(0, 160) + `…(${s.length})` : s;
    } catch {
      return String(v);
    }
  };
  const args = (a) => Array.from(a).map(ser).join(', ');

  function wrap(name, fn) {
    return new Proxy(fn, {
      apply(t, thisArg, a) {
        log.push(`${name}(${args(a)})`);
        return Reflect.apply(t, thisArg, a);
      },
      construct(t, a, nt) {
        log.push(`new ${name}(${args(a)})`);
        return Reflect.construct(t, a, nt);
      },
      get(t, p) {
        const v = t[p];
        return typeof v === 'function' ? wrap(`${name}.${String(p)}`, v) : v;
      },
    });
  }

  const sandbox = new Proxy(store, {
    has() { return true; },
    get(s, p) {
      if (p === 'globalThis') return sandbox;
      if (p in s) return s[p];
      if (p in reals) {
        const r = reals[p];
        return typeof r === 'function' && tracked.has(String(p)) ? wrap(String(p), r) : r;
      }
      return undefined;
    },
    set(s, p, v) { s[p] = v; return true; },
  });

  const ctx = vm.createContext(sandbox);
  try {
    vm.runInContext(code, ctx, { timeout: 20000 });
    log.push('[completed]');
  } catch (e) {
    log.push(`[threw] ${e && e.constructor ? e.constructor.name : 'Error'}: ${e && e.message}`);
  }
  return log;
}

// --- diff the two traces ----------------------------------------------------
function firstDivergence(a, b) {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return i;
  return a.length === b.length ? -1 : n;
}

const src = fs.readFileSync(SRC, 'utf8');
const RF = process.argv.includes('--rf');
const residual = residualOf(SRC);

// --- internal residual-execution trace (for pre-effect bugs) ---------------
//
// When the divergence is before the first external effect, the external diff
// only says "diverges at the start". This mode injects an entry log into every
// generated residual function (`__rfN`, the specialized VM handlers) and runs
// `main(0)`, printing the sequence of handlers and a compact summary of each
// argument (a `{value}` cell shows its value) up to the throw — so you can see,
// e.g., a handler firing with an empty stack because the pushes that should
// precede it never ran.
if (RF) {
  const summary = (v) => {
    if (v && typeof v === 'object' && 'value' in v) {
      const x = v.value;
      if (Array.isArray(x)) return `cell[${x.length}]`;
      if (x && typeof x === 'object') return 'cell{obj}';
      return `cell(${JSON.stringify(x)})`;
    }
    if (typeof v === 'function') return 'fn';
    try { return JSON.stringify(v); } catch { return String(v); }
  };
  const trace = [];
  const inst = residual.replace(
    /function (__rf\d+)\(([^)]*)\)\s*\{/g,
    (m, name, params) =>
      `${m} __T(${JSON.stringify(name)}, [${params}]);`,
  );
  const code =
    `const __TR=[];function __T(n,a){__TR.push(n+'('+a.map(${summary.toString()}).join(', ')+')');}\n` +
    inst +
    `\n;globalThis.__dump=()=>__TR;try{main(0);}catch(e){__TR.push('[threw] '+e.message);}`;
  const ctx = vm.createContext({ console, JSON, Array, Object, Date, String, Uint8Array, TextDecoder, Math });
  vm.runInContext(code, ctx, { timeout: 20000 });
  const tr = ctx.__dump();
  console.log(`residual ran ${tr.length} handler entries; last 25 before end:`);
  for (const line of tr.slice(-25)) console.log('  ' + line);
  process.exit(0);
}

// extra effectful globals to track, e.g.  node difftrace.js f.js Math.random fetch
const extra = process.argv.slice(3);
// The residual is always a `function main(input)`; call it. The original either
// runs at top level (real programs, e.g. simple.js) or, if it explicitly defines
// `function main`, must be called the same way to be comparable.
const origCode = /function\s+main\s*\(/.test(src) ? src + '\n;main(0);' : src;
console.error('tracing original...');
const orig = run(origCode, extra);
console.error('tracing residual...');
const resid = run(residual + '\n;main(0);', extra);

console.log(`original effects: ${orig.length}, residual effects: ${resid.length}`);
const d = firstDivergence(orig, resid);
if (d === -1) {
  console.log('IDENTICAL external-effect traces ✓');
  process.exit(0);
}
console.log(`\nFIRST DIVERGENCE at effect #${d}:`);
const lo = Math.max(0, d - 4);
for (let i = lo; i <= d; i++) {
  const o = orig[i] === undefined ? '·' : orig[i];
  const r = resid[i] === undefined ? '·' : resid[i];
  const mark = i === d ? '>>' : '  ';
  console.log(`${mark} #${i}`);
  console.log(`     orig: ${o}`);
  console.log(`     res : ${r}`);
}
process.exit(1);
