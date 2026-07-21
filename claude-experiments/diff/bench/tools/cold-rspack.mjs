// Fresh-process cold build with @rspack/core.
// usage: node cold-rspack.mjs <entry> <outdir>
import { rspack } from "@rspack/core";
import { makeRspackConfig } from "./rspack-config.mjs";

const [entry, outdir] = process.argv.slice(2);
const compiler = rspack(makeRspackConfig(entry, outdir));
await new Promise((resolve, reject) => {
  compiler.run((error, stats) => {
    if (error) return reject(error);
    if (stats.hasErrors()) {
      return reject(new Error(stats.toString({ colors: false })));
    }
    compiler.close((closeError) => (closeError ? reject(closeError) : resolve()));
  });
});
