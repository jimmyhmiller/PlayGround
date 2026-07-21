// Incremental rebuilds with diffpack's existing `diffpack watch` path (the
// same incremental engine oracle/benchmark.mjs exercises). Timing is measured
// from the moment the edited file is renamed into place to diffpack printing
// its "rebuilt ..." line, which it emits after writing the output bundle — so
// the measurement INCLUDES inotify detection latency and stdout pipe latency.
// usage: node incr-diffpack.mjs '<json options>'
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { writeContentEdit, expectedFreshValue } from "../gen.mjs";
import { verifyBundleValue } from "../util.mjs";

const options = JSON.parse(process.argv[2]);
const { corpusDir, entry, outfile, warmupEdits, timedEdits, diffpackBinary } = options;
const WAIT_TIMEOUT_MS = 120_000;

const child = spawn(diffpackBinary, ["watch", entry, outfile], {
  stdio: ["ignore", "pipe", "ignore"],
});

try {
  const lines = lineQueue(child);

  // Wait for the initial build ("watching ... wrote ...").
  await waitFor(lines, (line) => line.startsWith("watching "));
  verifyBundleValue(outfile, expectedFreshValue(options.moduleCount), "diffpack initial");

  const rebuildMs = [];
  const totalEdits = warmupEdits + timedEdits;
  for (let editIndex = 0; editIndex < totalEdits; editIndex += 1) {
    const edited = editIndex % 2 === 0;
    const expected = writeContentEdit(corpusDir, options, edited);
    const started = process.hrtime.bigint();
    await waitFor(lines, (line) => line.startsWith("rebuilt "));
    const elapsed = Number(process.hrtime.bigint() - started) / 1e6;
    verifyBundleValue(outfile, expected, `diffpack edit ${editIndex}`);
    if (editIndex >= warmupEdits) rebuildMs.push(elapsed);
    // If one rename surfaces as multiple filesystem events, drain the
    // duplicate rebuilds before the next timed edit.
    await drainQuiet(lines, 150);
  }

  const restored = writeContentEdit(corpusDir, options, false);
  await waitFor(lines, (line) => line.startsWith("rebuilt "));
  verifyBundleValue(outfile, restored, "diffpack restore");

  console.log(`RESULT ${JSON.stringify({ rebuildMs })}`);
} finally {
  child.kill("SIGKILL");
}
process.exit(0);

function lineQueue(process_) {
  const values = [];
  const waiters = [];
  let failure;
  const fail = (error) => {
    failure = error;
    for (const waiter of waiters.splice(0)) waiter.reject(error);
  };
  const reader = createInterface({ input: process_.stdout });
  reader.on("line", (line) => {
    const waiter = waiters.shift();
    if (waiter) waiter.resolve(line);
    else values.push(line);
  });
  process_.on("exit", (code) => fail(new Error(`diffpack watch exited early with code ${code}`)));
  process_.on("error", fail);
  return {
    next() {
      if (values.length > 0) return Promise.resolve(values.shift());
      if (failure) return Promise.reject(failure);
      return new Promise((resolve, reject) => {
        const waiter = { resolve, reject };
        waiters.push(waiter);
        setTimeout(() => {
          const position = waiters.indexOf(waiter);
          if (position !== -1) {
            waiters.splice(position, 1);
            reject(new Error(`timed out after ${WAIT_TIMEOUT_MS} ms waiting for diffpack watch output`));
          }
        }, WAIT_TIMEOUT_MS).unref();
      });
    },
    tryShift() {
      return values.shift();
    },
  };
}

async function waitFor(queue, predicate) {
  for (;;) {
    const line = await queue.next();
    if (predicate(line)) return line;
  }
}

async function drainQuiet(queue, quietMs) {
  for (;;) {
    await new Promise((resolve) => setTimeout(resolve, quietMs));
    let sawAny = false;
    while (queue.tryShift() !== undefined) sawAny = true;
    if (!sawAny) return;
  }
}
