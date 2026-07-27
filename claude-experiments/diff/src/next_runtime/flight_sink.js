// The streaming SSR destination for the next app-router adapter. This module is
// spliced verbatim into the generated SSR entry (see `ssr_entry_module` in
// src/next_adapter.rs) and is also imported directly by
// scripts/rsc/ssr-stream-integrity.mjs, so the regression test exercises the code
// that actually ships.
//
// Why this exists: react-dom packs HTML into a fixed 2048-byte view and calls
// `destination.write()` whenever that view fills. Those boundaries land ANYWHERE —
// including in the middle of `src="/vercel.svg"` — so anything interleaved after an
// arbitrary `write()` corrupts the document. The only HTML-token-safe boundary
// react-dom offers is the end of a flush cycle: `flushCompletedQueues()` calls
// `completeWriting()` (which drains the partial view) and then `flushBuffered()`
// (i.e. `destination.flush()`) in its `finally`. So the inline flight `<script>`s and
// the `useServerInsertedHTML` markup may ONLY be written from `flush()`, or from a
// macrotask — a flush cycle is fully synchronous, so a `setImmediate` callback is the
// same boundary (this is Next's own `atLeastOneTask()` trick).
import { Writable } from "node:stream";

// `res` is the node ServerResponse; `scriptQueue` is the shared array of inline
// `<script>` strings the flight pump appends to; `renderInserted()` returns the
// `useServerInsertedHTML` markup registered since the last boundary; `onFirstWrite()`
// marks the shell as started; `beforeEnd()` resolves once the flight pump has queued
// its terminal chunk.
export function createFlightSink({ res, scriptQueue, renderInserted, onFirstWrite, beforeEnd }) {
  let sawFlush = false;
  let started = false;
  let scheduled = false;

  // React's bytes go straight to `res` in call order. The write callback MUST be
  // invoked synchronously: deferring it makes the Writable buffer the remainder of
  // the flush cycle internally, and `flush()` would then inject ahead of bytes that
  // have not reached `res` yet — the same corruption by a different route.
  const forward = (chunk) => {
    if (!started) {
      started = true;
      onFirstWrite();
    }
    res.write(chunk);
  };

  // The ONLY place flight scripts / inserted HTML enter the byte stream.
  const drainQueues = () => {
    for (const html of renderInserted()) res.write(html);
    while (scriptQueue.length) res.write(scriptQueue.shift());
  };

  // A drain from a token-safe boundary. Before the shell starts, `res` has no head
  // written yet, so hold everything back — the first `flush()` picks it up.
  const drainAtBoundary = () => {
    scheduled = false;
    if (started) drainQueues();
  };

  const sink = new Writable({
    write(chunk, _enc, cb) {
      forward(chunk);
      cb();
    },
    // A corked burst: forward every chunk in order, never injecting between them.
    writev(chunks, cb) {
      for (const entry of chunks) forward(entry.chunk);
      cb();
    },
    // react-dom calls `destination.end()` after its last `flushBuffered()`; whatever
    // the pump queued after that (typically the terminal `push([0])`) goes out here.
    final(cb) {
      Promise.resolve()
        .then(beforeEnd)
        .then(() => {
          sink.assertFlushHookFired();
          drainQueues();
          res.end();
        })
        .then(() => cb(), cb);
    },
  });

  // react-dom's `flushBuffered()` — the end of a flush cycle, after `completeWriting()`
  // emptied the partial view. This is the injection boundary.
  sink.flush = () => {
    sawFlush = true;
    drainAtBoundary();
  };

  // Flight chunks can also arrive while React has nothing to flush; a macrotask can
  // never run inside React's synchronous flush cycle, so it is the same boundary.
  sink.scheduleDrain = () => {
    if (scheduled) return;
    scheduled = true;
    setImmediate(drainAtBoundary);
  };

  sink.assertFlushHookFired = () => {
    if (!sawFlush) {
      throw new Error(
        "diffpack next ssr (createFlightSink, src/next_runtime/flight_sink.js): react-dom never " +
          "called destination.flush(), so the HTML-token-safe injection boundary does not exist " +
          "in this react-dom build — refusing to serve possibly-corrupt HTML",
      );
    }
  };

  return sink;
}
