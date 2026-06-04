#!/usr/bin/env node
// The shared normalizer. Byte-for-byte comparison only works if BOTH the oracle's
// `## Code` and OUR compiler output pass through the identical formatter snap used:
//
//   prettier.format(code, { semi: true, parser: 'babel-ts' | 'flow' })
//
// (see packages/snap/src/compiler.ts `format()` at the pinned commit). pinned
// prettier 3.3.3 — see package.json. Because `## Code` is already snap-formatted,
// normalize() is idempotent on it; that idempotency is our normalizer-correctness test.
//
// Usage: node normalize.js [--dialect typescript|flow] < code.js > normalized.js
//        (reads code on stdin, writes formatted code on stdout, exit 1 on parse error)

const prettier = require('prettier');

let dialect = 'typescript';
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === '--dialect' && i + 1 < argv.length) dialect = argv[++i];
}

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', d => (input += d));
process.stdin.on('end', async () => {
  try {
    const out = await prettier.format(input, {
      semi: true,
      parser: dialect === 'flow' ? 'flow' : 'babel-ts',
    });
    process.stdout.write(out);
  } catch (e) {
    process.stderr.write('normalize: ' + String((e && e.message) || e) + '\n');
    process.exit(1);
  }
});
