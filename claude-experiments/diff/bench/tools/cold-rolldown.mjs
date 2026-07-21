// Fresh-process cold build with rolldown 1.2.0 (same pin as oracle/).
// usage: node cold-rolldown.mjs <entry> <outfile>
import { rolldown } from "rolldown";

const [entry, outfile] = process.argv.slice(2);
const bundle = await rolldown({ input: entry, treeshake: true, logLevel: "silent" });
await bundle.write({
  file: outfile,
  format: "esm",
  codeSplitting: false,
  minify: false,
  sourcemap: false,
});
await bundle.close();
