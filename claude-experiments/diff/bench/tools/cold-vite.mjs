// Fresh-process cold production build with vite (library mode: pure JS entry,
// single ESM chunk, minify off, sourcemap off, target esnext).
// usage: node cold-vite.mjs <root> <entry> <outdir>
import { join } from "node:path";
import { build } from "vite";

const [root, entry, outdir] = process.argv.slice(2);
await build({
  root,
  configFile: false,
  logLevel: "error",
  cacheDir: join(root, ".vite-cache"),
  build: {
    outDir: outdir,
    emptyOutDir: true,
    target: "esnext",
    minify: false,
    sourcemap: false,
    modulePreload: false,
    reportCompressedSize: false,
    lib: { entry, formats: ["es"], fileName: () => "bundle.mjs" },
  },
});
