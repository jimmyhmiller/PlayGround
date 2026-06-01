#!/usr/bin/env node
//! In-repo React Compiler oracle CLI for the jsir-ssa parity corpus.
//!
//! Replaces the old `/tmp/react-rust/.../react-compiler-e2e` Rust port (whose
//! origin was never recorded and which lived only in /tmp). This wraps the
//! OFFICIAL `babel-plugin-react-compiler` — the ground truth the Rust port was
//! approximating — and mirrors React's own `snap` test transform
//! (`transformFixtureInput` in compiler/packages/snap/src/compiler.ts): it reads
//! the first line for config pragmas via `parseConfigPragmaForTests`, parses with
//! the right dialect (typescript/flow + jsx), and runs the compiler plugin.
//!
//! Protocol (matches what crates/jsir-ssa/examples/corpus.rs expects):
//!   - reads fixture source on stdin
//!   - accepts/ignores `--frontend <x>` and `--filename <name>` flags
//!   - prints the compiled JS on stdout
//!   - exits nonzero if the compiler errors/throws (corpus buckets that as a bail)
//!
//! We emit only the "forget" (compiled) code, not the sprout evaluator output,
//! since the corpus compares memo *structure* only (`_c(N)` cache size + count of
//! `if ($[...])` memo blocks).

const babel = require('@babel/core');
const ReactCompiler = require('babel-plugin-react-compiler');

const plugin = ReactCompiler.default;
const parseConfigPragmaForTests = ReactCompiler.parseConfigPragmaForTests;

// --- args (only --filename is meaningful to us; --frontend is accepted/ignored)
let filename = 't.jsx';
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === '--filename' && i + 1 < argv.length) filename = argv[++i];
  else if (argv[i] === '--frontend' && i + 1 < argv.length) i++; // ignore value
}

function parseLanguage(firstLine) {
  return firstLine.indexOf('@flow') !== -1 ? 'flow' : 'typescript';
}

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', d => (input += d));
process.stdin.on('end', () => {
  try {
    const nl = input.indexOf('\n');
    const firstLine = nl === -1 ? input : input.slice(0, nl);
    const language = parseLanguage(firstLine);

    // Mirror snap's makePluginOptions: parse pragmas with compilationMode 'all'.
    const config = parseConfigPragmaForTests(firstLine, { compilationMode: 'all' });
    const options = {
      ...config,
      environment: { ...config.environment },
    };

    // Parser dialect mirrors snap's parseInput (minus Hermes for flow; babel's
    // flow plugin is structurally equivalent for these fixtures).
    const parserPlugins = language === 'flow' ? ['flow', 'jsx'] : ['typescript', 'jsx'];

    const result = babel.transformSync(input, {
      filename: '/' + filename,
      parserOpts: { plugins: parserPlugins },
      plugins: [[plugin, options]],
      sourceType: 'module',
      // Pretty (non-compact) output: the corpus `structure()` extractor counts
      // memo blocks by matching `if (` WITH a space, which babel's default
      // (non-compact) printer emits; `compact: true` would drop the space and
      // make every memo block invisible to the gate.
      compact: false,
      highlightCode: false,
      configFile: false,
      babelrc: false,
    });

    if (result == null || result.code == null) {
      process.stderr.write('react-cc: no code emitted\n');
      process.exit(2);
    }
    process.stdout.write(result.code);
  } catch (e) {
    process.stderr.write(String((e && e.message) || e) + '\n');
    process.exit(1);
  }
});
