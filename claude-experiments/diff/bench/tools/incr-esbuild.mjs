// Incremental rebuilds with esbuild's in-process context.rebuild() API.
// Timing covers the rebuild() call including writing the output file; it does
// NOT include filesystem-watch detection (esbuild rebuilds on demand).
// usage: node incr-esbuild.mjs '<json options>'
import esbuild from "esbuild";
import { writeContentEdit, expectedFreshValue } from "../gen.mjs";
import { verifyBundleValue } from "../util.mjs";

const options = JSON.parse(process.argv[2]);
const { corpusDir, entry, outfile, warmupEdits, timedEdits } = options;

const context = await esbuild.context({
  entryPoints: [entry],
  bundle: true,
  format: "esm",
  target: "esnext",
  outfile,
  write: true,
  minify: false,
  sourcemap: false,
  logLevel: "silent",
});

await context.rebuild();
verifyBundleValue(outfile, expectedFreshValue(options.moduleCount), "esbuild initial");

const rebuildMs = [];
const totalEdits = warmupEdits + timedEdits;
for (let editIndex = 0; editIndex < totalEdits; editIndex += 1) {
  const edited = editIndex % 2 === 0;
  const expected = writeContentEdit(corpusDir, options, edited);
  const started = process.hrtime.bigint();
  await context.rebuild();
  const elapsed = Number(process.hrtime.bigint() - started) / 1e6;
  verifyBundleValue(outfile, expected, `esbuild edit ${editIndex}`);
  if (editIndex >= warmupEdits) rebuildMs.push(elapsed);
}

// Restore the corpus to its original state and confirm.
const restored = writeContentEdit(corpusDir, options, false);
await context.rebuild();
verifyBundleValue(outfile, restored, "esbuild restore");
await context.dispose();

console.log(`RESULT ${JSON.stringify({ rebuildMs })}`);
