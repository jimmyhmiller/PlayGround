// Evaluates a Vite config file and prints the resolved fields Diffpack needs as
// JSON on stdout. Diffpack (native Rust) spawns `node` with this piped to stdin,
// exactly as Vite itself evaluates its config in Node before handing resolved
// values to its bundler. This is a one-time config read; the actual build is
// entirely native. Inputs come from the environment:
//   DIFFPACK_VITE_CONFIG  absolute path to vite.config.{ts,js,mts,mjs}
//   DIFFPACK_VITE_MODE    build mode ("production")
import { registerHooks } from 'node:module';
import { existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { resolve as resolvePath, isAbsolute } from 'node:path';

// TypeScript and Vite allow extensionless relative imports; raw Node ESM does not.
// Fill in the extension a bundler resolver would, but only when Node's own
// resolution fails first, so package resolution is never altered.
const EXTENSIONS = ['.ts', '.tsx', '.mts', '.js', '.jsx', '.mjs', '/index.ts', '/index.tsx', '/index.js'];
registerHooks({
  resolve(specifier, context, nextResolve) {
    try {
      return nextResolve(specifier, context);
    } catch (error) {
      if (specifier.startsWith('.') || specifier.startsWith('/')) {
        const base = context.parentURL
          ? new URL(specifier, context.parentURL)
          : pathToFileURL(specifier);
        for (const extension of EXTENSIONS) {
          const candidate = new URL(base.href + extension);
          if (existsSync(fileURLToPath(candidate))) {
            return nextResolve(candidate.href, context);
          }
        }
      }
      throw error;
    }
  },
});

const configPath = process.env.DIFFPACK_VITE_CONFIG;
const mode = process.env.DIFFPACK_VITE_MODE || 'production';

const module = await import(pathToFileURL(configPath).href);
let config = module.default;
// `defineConfig` may export an object, a (possibly async) function of the build
// context, or a promise. Resolve it to a plain object the way Vite does.
if (typeof config === 'function') {
  config = await config({ mode, command: 'build', isSsrBuild: false, isPreview: false });
} else {
  config = await config;
}
config = config || {};

// Vite `define`: a string value is used verbatim as replacement source; any other
// value is JSON-stringified. Normalize to the raw replacement text.
const define = {};
for (const [key, value] of Object.entries(config.define || {})) {
  define[key] = typeof value === 'string' ? value : JSON.stringify(value);
}

// `resolve.alias`: both the object form ({ '@': '/abs/src' }) and the array
// form ([{ find, replacement }]). Only string finds are expressible to the
// native resolver; regex/function entries are counted, never silently dropped.
const alias = [];
let aliasSkipped = 0;
const aliasConfig = config.resolve?.alias;
if (Array.isArray(aliasConfig)) {
  for (const entry of aliasConfig) {
    if (entry && typeof entry.find === 'string' && typeof entry.replacement === 'string') {
      alias.push([entry.find, entry.replacement]);
    } else {
      aliasSkipped += 1;
    }
  }
} else if (aliasConfig && typeof aliasConfig === 'object') {
  for (const [find, replacement] of Object.entries(aliasConfig)) {
    if (typeof replacement === 'string') alias.push([find, replacement]);
    else aliasSkipped += 1;
  }
}

// `css.preprocessorOptions.scss.additionalData`: only the string form can be
// expressed to the native Sass compiler; a function is counted, never
// silently dropped.
let scssAdditionalData = null;
let scssAdditionalDataSkipped = 0;
const scssOptions = config.css?.preprocessorOptions?.scss;
if (scssOptions && scssOptions.additionalData !== undefined) {
  if (typeof scssOptions.additionalData === 'string') {
    scssAdditionalData = scssOptions.additionalData;
  } else {
    scssAdditionalDataSkipped = 1;
  }
}

// `build.rollupOptions.input`: the MULTI-PAGE entry set. Vite accepts a string, an
// array of strings, or an object `{ name: path }`; each value is a path resolved
// relative to `config.root` (falling back to the cwd, which Diffpack sets to the
// project root). Normalized to ordered `[name, absolutePath]` pairs. For an array or
// string the name is derived from the file's basename (sans extension), as Vite does.
const projectRoot = typeof config.root === 'string' ? resolvePath(config.root) : process.cwd();
const resolveInput = (value) => (isAbsolute(value) ? value : resolvePath(projectRoot, value));
const basenameNoExt = (p) => {
  const last = p.replace(/\\/g, '/').split('/').pop() || p;
  const dot = last.lastIndexOf('.');
  return dot > 0 ? last.slice(0, dot) : last;
};
const inputs = [];
const rawInput = config.build?.rollupOptions?.input;
if (typeof rawInput === 'string') {
  inputs.push([basenameNoExt(rawInput), resolveInput(rawInput)]);
} else if (Array.isArray(rawInput)) {
  for (const value of rawInput) {
    if (typeof value === 'string') inputs.push([basenameNoExt(value), resolveInput(value)]);
  }
} else if (rawInput && typeof rawInput === 'object') {
  for (const [name, value] of Object.entries(rawInput)) {
    if (typeof value === 'string') inputs.push([name, resolveInput(value)]);
  }
}

// `build.manifest`: Vite writes `.vite/manifest.json` when this is truthy (true or a
// custom file name). Diffpack emits its own manifest when it is truthy; a string name
// is honored as the manifest file name.
let manifest = false;
let manifestName = null;
if (config.build && config.build.manifest !== undefined && config.build.manifest !== false) {
  manifest = true;
  if (typeof config.build.manifest === 'string') manifestName = config.build.manifest;
}

// `resolve.conditions` / `resolve.mainFields` / `resolve.dedupe`: string arrays that
// tune module resolution. Passed through verbatim (non-string members dropped).
const stringArray = (value) =>
  Array.isArray(value) ? value.filter((entry) => typeof entry === 'string') : [];
const resolveConditions = stringArray(config.resolve?.conditions);
const mainFields = stringArray(config.resolve?.mainFields);
const dedupe = stringArray(config.resolve?.dedupe);

// `optimizeDeps.exclude`: dependencies Vite must not pre-bundle. Diffpack bundles every
// dependency natively from source (there is no separate pre-bundle step), so an exclude
// is satisfied by construction; it is surfaced so the caller can report it honestly.
const optimizeDepsExclude = stringArray(config.optimizeDeps?.exclude);

// `server.proxy`: the dev proxy table. Vite keys it by a path context; each value is a
// target string or an options object `{ target, changeOrigin, ws, secure, rewrite }`.
// Only the string-expressible fields cross to the native proxy; a `rewrite` FUNCTION
// cannot be serialized, so it is counted and surfaced, never silently dropped.
const proxy = [];
let proxyRewriteSkipped = 0;
const proxyConfig = config.server?.proxy;
if (proxyConfig && typeof proxyConfig === 'object') {
  for (const [context, value] of Object.entries(proxyConfig)) {
    if (typeof value === 'string') {
      proxy.push({ context, target: value, changeOrigin: false, ws: false });
    } else if (value && typeof value === 'object' && typeof value.target === 'string') {
      if (typeof value.rewrite === 'function') proxyRewriteSkipped += 1;
      proxy.push({
        context,
        target: value.target,
        changeOrigin: value.changeOrigin === true,
        ws: value.ws === true,
      });
    }
  }
}

process.stdout.write(
  JSON.stringify({
    base: typeof config.base === 'string' ? config.base : null,
    define,
    alias,
    aliasSkipped,
    scssAdditionalData,
    scssAdditionalDataSkipped,
    inputs,
    manifest,
    manifestName,
    resolveConditions,
    mainFields,
    dedupe,
    optimizeDepsExclude,
    proxy,
    proxyRewriteSkipped,
  }),
);
