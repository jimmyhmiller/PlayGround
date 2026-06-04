#!/usr/bin/env node
// Extract the committed final-JS oracle from the pinned upstream's snap fixtures.
//
// Each `<name>.expect.md` holds:
//   ## Input    -> the fixture source we compile          -> fixtures/<name>.js
//   ## Code     -> the in-tree TS compiler's output (ok)  -> cache/<name>.code.js
//   ## Error    -> the compiler's error (bail)            -> cache/<name>.error.txt
//
// The `## Code` block is already snap-normalized (babel-generate -> prettier), so it
// IS the byte-exact ground truth our port must reproduce. We store it verbatim;
// the runner normalizes OUR output the same way (tools/normalize.js) before diffing.
//
// Nested fixture dirs (e.g. gating/) are preserved in <name> to avoid collisions.
// Run: node tools/extract.js   (reads UPSTREAM.lock for the checkout path)

const fs = require('fs');
const path = require('path');

const HERE = __dirname;
const CRATE = path.dirname(HERE);

// Resolve the pinned checkout + fixtures dir from UPSTREAM.lock (kept in one place).
const lock = fs.readFileSync(path.join(CRATE, 'UPSTREAM.lock'), 'utf8');
function lockVal(key) {
  const m = lock.match(new RegExp('^' + key + '\\s*=\\s*(\\S+)', 'm'));
  if (!m) throw new Error(`UPSTREAM.lock missing ${key}`);
  return m[1];
}
const checkout = lockVal('local_checkout').replace(/^~/, process.env.HOME);
const fixturesDir = path.join(checkout, lockVal('fixtures_dir'));

const OUT_FIX = path.join(CRATE, 'fixtures');
const OUT_CACHE = path.join(CRATE, 'cache');

// Pull the first fenced ``` ... ``` block after a `## <heading>` line. Fences are
// matched only at column 0 (start of line) so that ``` inside indented JSDoc
// comments in the code itself (e.g. ` * ```js`) does NOT terminate the block early.
function section(md, heading) {
  const lines = md.split('\n');
  let i = 0;
  for (; i < lines.length; i++) if (lines[i].trimEnd() === `## ${heading}`) break;
  if (i === lines.length) return null;
  // find opening fence (line starting with ```)
  for (; i < lines.length; i++) if (lines[i].startsWith('```')) break;
  if (i === lines.length) return null;
  const start = i + 1;
  // find closing fence (line starting with ```)
  for (i = start; i < lines.length; i++) if (lines[i].startsWith('```')) break;
  if (i === lines.length) return null;
  return lines.slice(start, i).join('\n').replace(/\s*$/, '') + '\n';
}

function walk(dir) {
  const out = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...walk(p));
    else if (e.name.endsWith('.expect.md')) out.push(p);
  }
  return out;
}

function mkdirp(p) { fs.mkdirSync(path.dirname(p), { recursive: true }); }

const files = walk(fixturesDir).sort();
let okCount = 0, errCount = 0, skipCount = 0;
const manifest = [];

for (const file of files) {
  const rel = path.relative(fixturesDir, file).replace(/\.expect\.md$/, '');
  const md = fs.readFileSync(file, 'utf8');

  const input = section(md, 'Input');
  if (input == null) { skipCount++; continue; }

  const code = section(md, 'Code');
  const error = md.indexOf('## Error') !== -1;

  const inPath = path.join(OUT_FIX, rel + '.js');
  mkdirp(inPath);
  fs.writeFileSync(inPath, input);

  let kind;
  if (code != null) {
    const cPath = path.join(OUT_CACHE, rel + '.code.js');
    mkdirp(cPath);
    fs.writeFileSync(cPath, code);
    kind = 'code';
    okCount++;
  } else if (error) {
    // Store the raw `## Error` block text; the runner only checks bail vs no-bail
    // plus (optionally) the error reason, not the full code frame.
    const errText = section(md, 'Error') || '';
    const ePath = path.join(OUT_CACHE, rel + '.error.txt');
    mkdirp(ePath);
    fs.writeFileSync(ePath, errText);
    kind = 'error';
    errCount++;
  } else {
    kind = 'unknown';
    skipCount++;
  }

  // dialect: filename variant wins, else @flow pragma in the first line.
  const firstLine = input.slice(0, input.indexOf('\n') === -1 ? input.length : input.indexOf('\n'));
  const dialect = /\.flow$/.test(rel) || firstLine.includes('@flow') ? 'flow'
    : /\.ts$/.test(rel) ? 'typescript' : 'typescript';

  manifest.push({ name: rel, kind, dialect });
}

fs.mkdirSync(OUT_CACHE, { recursive: true });
fs.writeFileSync(
  path.join(CRATE, 'fixtures.manifest.json'),
  JSON.stringify(manifest, null, 2) + '\n'
);

console.log(`extracted ${files.length} fixtures: ${okCount} code, ${errCount} error, ${skipCount} skipped`);
console.log(`-> ${path.relative(process.cwd(), OUT_FIX)} , ${path.relative(process.cwd(), OUT_CACHE)}`);
console.log(`-> fixtures.manifest.json (${manifest.length} entries)`);
