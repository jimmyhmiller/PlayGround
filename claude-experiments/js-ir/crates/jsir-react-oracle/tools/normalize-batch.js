#!/usr/bin/env node
// Batch normalizer: one node process for the whole corpus (per-fixture node startup
// would dominate runtime). Same formatter as normalize.js / snap:
//   prettier.format(code, { semi: true, parser: 'babel-ts' | 'flow' }).
//
// stdin : JSON array of { name, code, dialect }
// stdout: JSON array of { name, ok, code } | { name, ok:false, error }

const prettier = require('prettier');

let buf = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', d => (buf += d));
process.stdin.on('end', async () => {
  let items;
  try { items = JSON.parse(buf); } catch (e) {
    process.stderr.write('normalize-batch: bad JSON on stdin\n'); process.exit(2);
  }
  const out = [];
  for (const it of items) {
    try {
      const code = await prettier.format(it.code, {
        semi: true,
        parser: it.dialect === 'flow' ? 'flow' : 'babel-ts',
      });
      out.push({ name: it.name, ok: true, code });
    } catch (e) {
      out.push({ name: it.name, ok: false, error: String((e && e.message) || e).split('\n')[0] });
    }
  }
  process.stdout.write(JSON.stringify(out));
});
