// Compiles ONE `.mdx`/`.md` source with the APP's OWN MDX pipeline — its installed
// `@mdx-js/mdx` plus the exact `remarkPlugins` / `rehypePlugins` / `recmaPlugins` its
// `next.config` passes to `createMDX({ options })`.
//
// Why a Node process at all: diffpack's native MDX compiler (`src/mdx.rs`) is a
// markdown-rs -> JSX emitter and cannot run a unified plugin, which is an arbitrary
// JavaScript function operating on an mdast/hast. Reimplementing `remark-gfm` and every
// other plugin in Rust would be endless and would drift; running the app's own pipeline is
// the faithful answer and generalizes to every plugin that exists. This mirrors
// `src/less_stylus_runner.mjs` (the app's own `less`/`stylus`) and
// `src/postcss_runner.mjs`.
//
// Plugin *values* cannot be serialized out of `next-config-eval.mjs`, so this runner
// re-evaluates `next.config` itself and keeps the live functions in-process.
//
// Contract (values set by `crate::mdx`):
//   argv[2]                  absolute path to the app's next.config.*
//   env DIFFPACK_MDX_FILE    absolute path of the MDX source (plugins + errors see it)
//   env DIFFPACK_MDX_PROVIDER  absolute path of the app's `mdx-components.*`, or "" when
//                            the app has none. Non-empty becomes @mdx-js/mdx's
//                            `providerImportSource`, which is exactly what `@next/mdx`
//                            does via its `next-mdx-import-source-file` webpack alias.
//   cwd                      the project root (so `import("@mdx-js/mdx")` resolves the
//                            app's copy)
//   stdin                    the MDX source
//   stdout                   {"jsx": "<compiled JSX module>"}
//
// Output is JSX (`jsx: true`), NOT calls into a JSX runtime: the emitted module then flows
// through diffpack's normal oxc parse + Transformer + RSC pipeline exactly like the native
// emitter's output, so the app's own tsconfig/jsx settings still decide the runtime.
//
// Any failure exits non-zero with a message naming the file; `crate::mdx` turns that into a
// hard build error. There is no fall-back to the native compiler — silently compiling
// without the configured plugins is the defect this runner exists to remove.
import Module, { createRequire, register, registerHooks } from "node:module";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

// Same interception as `scripts/rsc/next-config-eval.mjs`: `require("@next/mdx")` is
// answered by a capturing shim so the config evaluates even when `@next/mdx` is not
// installed, and so the options object reaches us regardless of what the real package does
// with it. Both module systems are covered — a `next.config.mjs` that `import`s the real
// package never touches `Module._load`, and an un-shimmed load would silently hand us no
// plugins at all.
let pluginOptions = null;
const shim =
  (options = {}) =>
  (nextConfig = {}) => {
    pluginOptions = options || {};
    return nextConfig;
  };
globalThis.__DIFFPACK_CONFIG_SHIMS__ = { "@next/mdx": shim };
const origLoad = Module._load;
Module._load = function (request) {
  if (request === "@next/mdx") return shim;
  return origLoad.apply(this, arguments);
};
const SHIM_URL = "diffpack-config-shim:@next/mdx";
const SHIM_SOURCE = 'export default globalThis.__DIFFPACK_CONFIG_SHIMS__["@next/mdx"];';
if (typeof registerHooks === "function") {
  // Node >= 22.15: in-thread hooks, so the shim body is handed back directly.
  registerHooks({
    resolve(specifier, context, next) {
      if (specifier === "@next/mdx") return { url: SHIM_URL, shortCircuit: true };
      return next(specifier, context);
    },
    load(url, context, next) {
      if (url === SHIM_URL) return { format: "module", shortCircuit: true, source: SHIM_SOURCE };
      return next(url, context);
    },
  });
} else {
  try {
    // Node >= 20.6 (off-thread hooks). On older runtimes only `require()`/jiti are
    // intercepted; a config that reaches the real package through ESM then leaves
    // `pluginOptions` null, which `main()` turns into a hard error rather than a
    // plugin-free compile.
    register(
      "data:text/javascript," +
        encodeURIComponent(`
          export async function resolve(specifier, context, next) {
            if (specifier === "@next/mdx") {
              const body = ${JSON.stringify(SHIM_SOURCE)};
              return {
                url: "data:text/javascript," + encodeURIComponent(body),
                shortCircuit: true,
              };
            }
            return next(specifier, context);
          }
        `),
    );
  } catch {
    /* older node: CJS interception only */
  }
}

async function loadConfig(configPath) {
  const appRequire = createRequire(configPath);
  try {
    const { createJiti } = await import(pathToFileURL(appRequire.resolve("jiti")).href);
    const jiti = createJiti(pathToFileURL(configPath).href, { interopDefault: true });
    const m = await jiti.import(configPath);
    return (m && m.default) || m;
  } catch {
    const m = await import(pathToFileURL(configPath).href);
    return m.default || m;
  }
}

// `createMDX({ options: {...} })` is the documented shape; the same keys are also accepted
// at the top level (older @next/mdx, and configs written that way). Options-first.
function mdxOptions() {
  const plugin = pluginOptions || {};
  const nested = plugin.options && typeof plugin.options === "object" ? plugin.options : {};
  const merged = { ...plugin, ...nested };
  delete merged.options;
  delete merged.extension; // which files are MDX — diffpack's own routing decides that
  return merged;
}

// This script is executed from a temp path, so a bare `import("@mdx-js/mdx")` would resolve
// against THAT directory, not the app. Resolve from the app's next.config first (the same
// `createRequire` trick used for jiti above) and import the concrete file; the bare import
// is only a fallback for a layout `require.resolve` cannot express.
async function loadCompiler(file, configPath) {
  let mod;
  try {
    try {
      const resolved = createRequire(configPath).resolve("@mdx-js/mdx");
      mod = await import(pathToFileURL(resolved).href);
    } catch {
      mod = await import("@mdx-js/mdx");
    }
  } catch (error) {
    throw new Error(
      `MDX ${file}: next.config configures remark/rehype plugins, so diffpack must compile ` +
        `this file with the app's own MDX pipeline, but "@mdx-js/mdx" could not be loaded ` +
        `from the project (${(error && error.message) || error}). Install @mdx-js/mdx ` +
        `(it is an optional peer of @next/mdx) or remove the configured plugins.`,
    );
  }
  const compile = mod.compile || (mod.default && mod.default.compile);
  if (typeof compile !== "function") {
    const version = mod.sync || (mod.default && mod.default.sync) ? " (looks like @mdx-js/mdx v1)" : "";
    throw new Error(
      `MDX ${file}: the project's "@mdx-js/mdx" does not export compile()${version}. ` +
        `diffpack drives the v2/v3 compile() API; upgrade @mdx-js/mdx to >=2, or remove the ` +
        `configured remark/rehype plugins so the native compiler can be used.`,
    );
  }
  return compile;
}

async function main() {
  const configPath = process.argv[2];
  const file = process.env.DIFFPACK_MDX_FILE;
  const provider = process.env.DIFFPACK_MDX_PROVIDER;
  if (!configPath || !file) {
    throw new Error("mdx runner: next.config path / DIFFPACK_MDX_FILE not set");
  }
  await loadConfig(configPath);
  if (pluginOptions === null) {
    // diffpack only reaches this runner because the config eval SAW `createMDX` options it
    // must honour. Not seeing them here means the shim above was bypassed, and compiling
    // anyway would drop the app's plugins in silence — the whole defect this file fixes.
    throw new Error(
      `MDX ${file}: re-evaluating ${configPath} never reached @next/mdx's createMDX(), so the ` +
        `configured remark/rehype plugins could not be recovered. diffpack will not compile ` +
        `this file without them.`,
    );
  }
  const options = mdxOptions();
  const compile = await loadCompiler(file, configPath);
  const source = readFileSync(0, "utf8");
  const result = await compile(
    { value: source, path: file },
    {
      // The app's configured providerImportSource wins; otherwise the discovered
      // `mdx-components.*` stands in for @next/mdx's `next-mdx-import-source-file` alias.
      ...(provider ? { providerImportSource: provider } : {}),
      ...options,
      // Non-negotiable: diffpack owns the JS/JSX emit downstream.
      jsx: true,
      outputFormat: "program",
    },
  );
  process.stdout.write(JSON.stringify({ jsx: String(result) }));
}

main().catch((error) => {
  const message = (error && (error.message || error.stack)) || String(error);
  process.stderr.write(message + "\n");
  process.exit(1);
});
