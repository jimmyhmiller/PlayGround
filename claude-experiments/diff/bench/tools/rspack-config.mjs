// Shared rspack configuration: matched flags (minify OFF, sourcemaps OFF,
// tree shaking ON via production mode, ESM output via experiments.outputModule).
import { dirname } from "node:path";

export function makeRspackConfig(entry, outdir) {
  return {
    context: dirname(entry),
    entry: { bundle: entry },
    mode: "production",
    devtool: false,
    target: ["node20", "es2022"],
    experiments: { outputModule: true },
    output: {
      path: outdir,
      filename: "bundle.mjs",
      module: true,
      chunkFormat: "module",
      library: { type: "module" },
      clean: false,
    },
    optimization: {
      minimize: false,
    },
    resolve: { extensions: ["...", ".ts"] },
    module: {
      rules: [
        {
          test: /\.ts$/,
          loader: "builtin:swc-loader",
          options: { jsc: { parser: { syntax: "typescript" }, target: "esnext" } },
          type: "javascript/auto",
        },
      ],
    },
    stats: "errors-only",
    infrastructureLogging: { level: "error" },
  };
}
