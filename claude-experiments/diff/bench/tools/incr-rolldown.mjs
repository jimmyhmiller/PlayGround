// Incremental rebuilds with rolldown's watch API and incrementalBuild: true —
// the same mechanism oracle/benchmark.mjs uses. The reported time is
// rolldown's own BUNDLE_END event.duration (build + write), which excludes
// filesystem-watch detection latency.
// usage: node incr-rolldown.mjs '<json options>'
import { watch } from "rolldown";
import { writeContentEdit, expectedFreshValue } from "../gen.mjs";
import { verifyBundleValue } from "../util.mjs";

const options = JSON.parse(process.argv[2]);
const { corpusDir, entry, outfile, warmupEdits, timedEdits } = options;

const builds = buildQueue();
const watcher = watch({
  input: entry,
  treeshake: true,
  incrementalBuild: true,
  logLevel: "silent",
  output: { file: outfile, format: "esm", codeSplitting: false, minify: false, sourcemap: false },
});
watcher.on("event", async (event) => {
  if (event.code === "BUNDLE_END") {
    await event.result.close();
    builds.push(event.duration);
  } else if (event.code === "ERROR") {
    builds.fail(event.error);
  }
});

await builds.next();
verifyBundleValue(outfile, expectedFreshValue(options.moduleCount), "rolldown initial");

const rebuildMs = [];
const totalEdits = warmupEdits + timedEdits;
for (let editIndex = 0; editIndex < totalEdits; editIndex += 1) {
  const edited = editIndex % 2 === 0;
  const expected = writeContentEdit(corpusDir, options, edited);
  const duration = await builds.next();
  verifyBundleValue(outfile, expected, `rolldown edit ${editIndex}`);
  if (editIndex >= warmupEdits) rebuildMs.push(duration);
}

const restored = writeContentEdit(corpusDir, options, false);
await builds.next();
verifyBundleValue(outfile, restored, "rolldown restore");
await watcher.close();

console.log(`RESULT ${JSON.stringify({ rebuildMs })}`);

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
    next() {
      if (failure) return Promise.reject(failure);
      if (values.length > 0) return Promise.resolve(values.shift());
      return new Promise((resolve, reject) => waiters.push({ resolve, reject }));
    },
  };
}
