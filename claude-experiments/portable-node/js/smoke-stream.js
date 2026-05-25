// Smoke test for node:stream — verifies the lifted internal/streams/* files
// actually do useful work, not just load.
(function () {
const log = [];
const ok = (name, val) => log.push(`OK   ${name}: ${val}`);
const fail = (name, e) => log.push(`FAIL ${name}: ${e.message || e}`);
const check = (name, fn) => { try { ok(name, JSON.stringify(fn())); } catch (e) { fail(name, e); } };

const { Readable, Writable, Transform, PassThrough, Duplex, pipeline } = require('stream');

check('Readable is a class',  () => typeof Readable === 'function');
check('Writable is a class',  () => typeof Writable === 'function');
check('Transform is a class', () => typeof Transform === 'function');
check('PassThrough is a class',() => typeof PassThrough === 'function');

// Readable.from(iterable) — does it produce a stream that emits 'data'?
check('Readable.from + sync collect', () => {
  let collected = '';
  const r = Readable.from(['hello ', 'world']);
  let finished = false;
  r.on('data', (chunk) => { collected += chunk; });
  r.on('end', () => { finished = true; });
  // Without a real event loop, data events fire synchronously on push for sync Readables.
  // If our streams need ticks, this won't emit.
  return { collected, finished };
});

// Custom Readable using _read
check('Custom Readable', () => {
  let collected = '';
  const r = new Readable({
    read() {
      this.push('alpha');
      this.push('beta');
      this.push(null);
    },
  });
  r.on('data', (c) => { collected += c; });
  r.read(); // kick off
  return collected;
});

// Custom Writable using _write
check('Custom Writable', () => {
  const chunks = [];
  const w = new Writable({
    write(chunk, enc, cb) { chunks.push(chunk.toString()); cb(); },
  });
  w.write('one');
  w.write('two');
  w.end('three');
  return chunks.join(',');
});

// PassThrough — write to it, read from it
check('PassThrough', () => {
  const pt = new PassThrough();
  let out = '';
  pt.on('data', (c) => { out += c; });
  pt.write('foo');
  pt.write('bar');
  pt.end();
  return out;
});

// Transform — uppercase each chunk
check('Transform uppercase', () => {
  let out = '';
  const t = new Transform({
    transform(chunk, enc, cb) { cb(null, chunk.toString().toUpperCase()); },
  });
  t.on('data', (c) => { out += c; });
  t.write('hello ');
  t.write('world');
  t.end();
  return out;
});

// pipe()
check('Readable.pipe(Writable)', () => {
  const collected = [];
  const r = new Readable({
    read() {
      this.push('A');
      this.push('B');
      this.push(null);
    },
  });
  const w = new Writable({
    write(chunk, enc, cb) { collected.push(chunk.toString()); cb(); },
  });
  r.pipe(w);
  r.read();
  return collected.join(',');
});

return log.join('\n');
})()
