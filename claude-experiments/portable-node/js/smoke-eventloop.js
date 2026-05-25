// Tests that exercise the event loop. Each schedules work asynchronously
// and stashes results into __asyncResults; the Rust driver runs the loop
// to idle, then a final post-loop eval reads __asyncResults out.

(function () {
globalThis.__asyncResults = {};
const R = globalThis.__asyncResults;

// 1. process.nextTick fires (via microtask).
R.tick = 'not fired';
process.nextTick(() => { R.tick = 'fired'; });

// 2. Promise.then fires.
R.promise = 'not resolved';
Promise.resolve().then(() => { R.promise = 'resolved'; });

// 3. setTimeout fires after delay.
R.timeout = 'not fired';
setTimeout(() => { R.timeout = 'fired'; }, 10);

// 4. setInterval fires repeatedly, then we clear it.
R.intervalCount = 0;
const ivId = setInterval(() => {
  R.intervalCount++;
  if (R.intervalCount >= 3) clearInterval(ivId);
}, 5);

// 5. setImmediate fires (modeled as 0ms timer).
R.immediate = 'not fired';
setImmediate(() => { R.immediate = 'fired'; });

// 6. Ordering: nextTick before setTimeout(0).
R.order = [];
setTimeout(() => { R.order.push('timeout-0'); }, 0);
process.nextTick(() => { R.order.push('nextTick'); });

// 7. async/await with setTimeout.
R.asyncAwait = 'not done';
(async () => {
  await new Promise((res) => setTimeout(res, 10));
  R.asyncAwait = 'done';
})();

// 8. Custom Readable emitting via push() + 'data' event (the streams test).
R.readable = '';
R.streamTrace = '';
const { Readable, Writable, Transform, PassThrough } = require('stream');
const r = new Readable({
  read() {
    R.streamTrace += '_read; ';
    this.push('alpha');
    this.push('beta');
    this.push(null);
  },
});
r.on('data', (c) => { R.readable += c; R.streamTrace += 'data:' + c + '; '; });
r.on('end',  ()  => { R.streamTrace += 'end; '; });
r.on('error',(e) => { R.streamTrace += 'err:' + e.message + '; '; });
R.streamCtor = r.constructor && r.constructor.name;
R.streamFlow = r._readableState && r._readableState.flowing;
R.streamReadable = r._readableState && r._readableState.readableListening;
// Try explicit read() to bypass flow-mode detection
setTimeout(() => {
  R.streamFlow2 = r._readableState && r._readableState.flowing;
  R.afterRead = r.read();
  R.streamFlow3 = r._readableState && r._readableState.flowing;
}, 20);

// 9. Transform — uppercase the input.
R.transform = '';
const t = new Transform({
  transform(chunk, enc, cb) { cb(null, chunk.toString().toUpperCase()); },
});
t.on('data', (c) => { R.transform += c; });
t.write('hello ');
t.write('world');
t.end();

// 10. PassThrough.
R.passthrough = '';
const pt = new PassThrough();
pt.on('data', (c) => { R.passthrough += c; });
pt.write('foo');
pt.end('bar');

// 11. pipe() — Readable into Writable.
R.pipe = '';
const src = new Readable({
  read() { this.push('one'); this.push('two'); this.push(null); },
});
const dst = new Writable({
  write(chunk, enc, cb) { R.pipe += chunk.toString(); cb(); },
});
src.pipe(dst);

return 'scheduled — see post-loop check for results';
})()
