#!/usr/bin/env node
// Warm throughput benchmark for the official React compiler: compile every
// fixture in ONE process (excludes Node startup), report ms/compile. Pair with
// `cargo run -p jsir-ssa --example bench` for a head-to-head. Set ITERS to vary.
const fs = require('fs');
const path = require('path');
const babel = require('@babel/core');
const RC = require('babel-plugin-react-compiler');
const plugin = RC.default;
const parse = RC.parseConfigPragmaForTests;

const dir = process.env.REACT_FIXTURES || path.join(__dirname, 'fixtures');
const files = fs.readdirSync(dir).filter((f) => f.endsWith('.js'));
const srcs = files.map((f) => fs.readFileSync(path.join(dir, f), 'utf8'));

function compile(src) {
  const nl = src.indexOf('\n');
  const first = nl < 0 ? src : src.slice(0, nl);
  const lang = first.includes('@flow') ? 'flow' : 'typescript';
  const cfg = parse(first, { compilationMode: 'all' });
  try {
    babel.transformSync(src, {
      filename: '/t.tsx',
      parserOpts: { plugins: lang === 'flow' ? ['flow', 'jsx'] : ['typescript', 'jsx'] },
      plugins: [[plugin, { ...cfg, environment: { ...cfg.environment } }]],
      sourceType: 'module',
      compact: false,
      configFile: false,
      babelrc: false,
    });
    return true;
  } catch (e) {
    return false;
  }
}

const iters = parseInt(process.env.ITERS || '10', 10);
const okset = []; let err = 0;
for (const s of srcs) (compile(s) ? okset.push(s) : err++); // warmup + partition
for (const [label, set] of [['ALL', srcs], ['OK-only', okset]]) {
  const t = process.hrtime.bigint();
  for (let i = 0; i < iters; i++) for (const s of set) compile(s);
  const el = Number(process.hrtime.bigint() - t) / 1e6;
  const total = set.length * iters;
  console.log(
    `REACT[${label}]: n=${set.length} ${el.toFixed(1)}ms => ${(el / total).toFixed(4)} ms/compile, ${(total / (el / 1000)).toFixed(0)} compiles/sec`,
  );
}
console.log(`REACT: fixtures=${srcs.length} ok=${okset.length} err=${err}`);
