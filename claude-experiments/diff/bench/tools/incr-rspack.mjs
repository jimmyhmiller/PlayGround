// Incremental rebuilds with rspack's in-process watch mode
// (compiler.watch, aggregateTimeout 0). Timing is measured from the moment
// the edited file is written to the watch callback firing after the output is
// emitted, so it INCLUDES filesystem-watch detection latency — that is how
// rspack delivers rebuilds.
// usage: node incr-rspack.mjs '<json options>'
import { join } from "node:path";
import { rspack } from "@rspack/core";
import { makeRspackConfig } from "./rspack-config.mjs";
import { writeContentEdit, expectedFreshValue } from "../gen.mjs";
import { verifyBundleValue } from "../util.mjs";

const options = JSON.parse(process.argv[2]);
const { corpusDir, entry, outdir, warmupEdits, timedEdits } = options;
const outfile = join(outdir, "bundle.mjs");

const compiler = rspack(makeRspackConfig(entry, outdir));
const builds = buildQueue();
const watching = compiler.watch({ aggregateTimeout: 0 }, (error, stats) => {
  if (error) return builds.fail(error);
  if (stats.hasErrors()) return builds.fail(new Error(stats.toString({ colors: false })));
  builds.push(Date.now());
});

await builds.next();
verifyBundleValue(outfile, expectedFreshValue(options.moduleCount), "rspack initial");

const rebuildMs = [];
const totalEdits = warmupEdits + timedEdits;
for (let editIndex = 0; editIndex < totalEdits; editIndex += 1) {
  const edited = editIndex % 2 === 0;
  // rspack's watcher uses mtime; ensure the queue is idle before editing.
  await settle(builds);
  const expected = writeContentEdit(corpusDir, options, edited);
  const started = process.hrtime.bigint();
  await builds.next();
  const elapsed = Number(process.hrtime.bigint() - started) / 1e6;
  verifyBundleValue(outfile, expected, `rspack edit ${editIndex}`);
  if (editIndex >= warmupEdits) rebuildMs.push(elapsed);
}

await settle(builds);
const restored = writeContentEdit(corpusDir, options, false);
await builds.next();
verifyBundleValue(outfile, restored, "rspack restore");

await new Promise((resolve, reject) =>
  watching.close((error) => (error ? reject(error) : resolve())),
);
await new Promise((resolve, reject) =>
  compiler.close((error) => (error ? reject(error) : resolve())),
);

console.log(`RESULT ${JSON.stringify({ rebuildMs })}`);

// Drain any duplicate rebuild notifications before the next edit: wait for a
// full quiet window with no new build completions.
async function settle(queue) {
  for (;;) {
    await new Promise((resolve) => setTimeout(resolve, 150));
    if (!queue.drain()) return;
  }
}

function buildQueue() {
  const values = [];
  const waiters = [];
  let failure;
  return {
    push(value) {
      const waiter = waiters.shift();
      if (waiter) waiter.resolve(value);
      else values.push(value);
    },
    fail(error) {
      failure = error;
      for (const waiter of waiters.splice(0)) waiter.reject(error);
    },
    // Consume queued completions without blocking; true if any were queued.
    drain() {
      if (failure) throw failure;
      const had = values.length > 0;
      values.length = 0;
      return had;
    },
    next() {
      if (failure) return Promise.reject(failure);
      if (values.length > 0) return Promise.resolve(values.shift());
      return new Promise((resolve, reject) => waiters.push({ resolve, reject }));
    },
  };
}
