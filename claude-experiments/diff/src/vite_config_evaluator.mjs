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
const env = { mode, command: 'build', isSsrBuild: false, isPreview: false };
// `defineConfig` may export an object, a (possibly async) function of the build
// context, or a promise. Resolve it to a plain object the way Vite does.
if (typeof config === 'function') {
  config = await config(env);
} else {
  config = await config;
}
config = config || {};

// --- plugin `config()` hooks ----------------------------------------------
// A Vite plugin contributes configuration: it is where `jsxImportSource` comes
// from for every scaffold that has no tsconfig (`@preact/preset-vite` returns
// `{ oxc: { jsx: { importSource: 'preact' } } }`, create-vite's JS preact
// template carries no tsconfig at all). Vite runs these hooks before it resolves
// its transform options, merging each returned partial over the config so far
// (`runConfigHook` -> `mergeConfig(conf, res)`, plugin result as the override).
//
// Two groups of keys are merged: the JSX keys (`oxc` / `esbuild`) and `resolve`.
//
// `resolve` is here because it is a module-RESOLUTION contract, not build machinery,
// and a preset plugin is exactly where it comes from. `@preact/preset-vite`'s
// `preact:config` plugin returns
// `{ resolve: { alias: { react: 'preact/compat', 'react/jsx-runtime':
// 'preact/jsx-runtime', 'react-dom': 'preact/compat', ... } } }`, and that alias is
// the ONLY thing that makes `import { forwardRef } from "react"` work in a Preact
// app. Dropping it either fails the build with "install react" on a project that
// legitimately has no react (Vite builds it), or — when react IS installed as a
// transitive dependency — silently resolves to REAL React and ships two frameworks
// in one bundle. Diffpack's native resolver already implements every field here
// (`alias`, `dedupe`, `conditions`, `mainFields`) and already honors them from the
// user's config; there is no reason to accept them from the user and refuse them
// from a plugin, and Vite draws no such distinction.
//
// Still NOT merged, and this remains the deliberate boundary: `define`,
// `build.rollupOptions.input`, `server.*`. A plugin's contribution to those
// describes machinery (virtual modules, dev middleware, SSR entry rewriting) a
// native build does not run; merging it would import half a plugin's build model.
// A plugin whose hook throws is reported by name, never swallowed.
const flattenPlugins = async (value, out = []) => {
  const resolved = await value;
  if (!resolved) return out;
  if (Array.isArray(resolved)) {
    for (const entry of resolved) await flattenPlugins(entry, out);
    return out;
  }
  out.push(resolved);
  return out;
};
// Vite's hook order: `enforce: 'pre'` plugins, then unenforced, then `'post'`.
const HOOK_ORDER = { pre: 0, post: 2 };
const hookOrder = (plugin) => {
  const hook = plugin.config;
  const declared = plugin.enforce ?? (hook && typeof hook === 'object' ? hook.order : undefined);
  return HOOK_ORDER[declared] ?? 1;
};
// `apply` restricts a plugin to one command; a build must not run a serve-only
// plugin's hooks.
const appliesToBuild = (plugin) => {
  if (typeof plugin.apply === 'function') return plugin.apply({ ...config, mode }, env) !== false;
  return plugin.apply === undefined || plugin.apply === 'build';
};
// Merge of the merged-through keys, plugin result overriding. DEEP, as Vite's
// `mergeConfig` is: `@vitejs/plugin-react` contributes the runtime from one
// plugin (`{ oxc: { jsx: { runtime, importSource } } }`) and Fast Refresh from a
// second (`{ oxc: { jsx: { refresh: false } } }`), so a shallow merge of `oxc`
// would drop the runtime the first one set.
const isPlainObject = (value) =>
  value !== null && typeof value === 'object' && !Array.isArray(value);
// `resolve.alias` accepts BOTH an object and an array of `{find,replacement}`, and
// a user config and a plugin may disagree on which. Vite's `mergeAlias` normalizes
// to the array form whenever they do, with the plugin's entries FIRST because
// aliases are matched top-down and the override must win; two objects merge by key
// for the same reason. A plain deep merge would turn a user's ARRAY into the
// plugin's object and lose every user alias.
const aliasEntries = (alias) =>
  Array.isArray(alias)
    ? alias
    : Object.entries(alias).map(([find, replacement]) => ({ find, replacement }));
const mergeAlias = (into, from) => {
  if (into === undefined || into === null) return from;
  if (from === undefined || from === null) return into;
  if (isPlainObject(into) && isPlainObject(from)) return { ...into, ...from };
  return [...aliasEntries(from), ...aliasEntries(into)];
};
const deepMerge = (into, from, key) => {
  if (key === 'alias') return mergeAlias(into, from);
  // `dedupe`, `conditions`, `mainFields`, `optimizeDeps.include`: Vite concatenates
  // rather than replacing, so one plugin's dedupe list never erases another's.
  if (Array.isArray(into) || Array.isArray(from)) {
    const arraify = (value) => (value === undefined ? [] : Array.isArray(value) ? value : [value]);
    return [...arraify(into), ...arraify(from)];
  }
  if (!isPlainObject(from)) return from;
  if (!isPlainObject(into)) return { ...from };
  const merged = { ...into };
  for (const [childKey, value] of Object.entries(from)) {
    merged[childKey] =
      value === undefined ? merged[childKey] : deepMerge(merged[childKey], value, childKey);
  }
  return merged;
};
const MERGED_KEYS = ['oxc', 'esbuild', 'resolve'];
const mergePluginConfig = (into, from) => {
  for (const key of MERGED_KEYS) {
    if (from[key] === undefined) continue;
    into[key] = deepMerge(into[key], from[key], key);
  }
};
const pluginErrors = [];
const pluginName = (plugin) =>
  typeof plugin.name === 'string' ? plugin.name : '(anonymous plugin)';
const runs = (plugin) => {
  if (!plugin || !plugin.config) return false;
  try {
    return appliesToBuild(plugin);
  } catch (error) {
    // A throwing `apply` cannot say whether the plugin runs; skipping it is the
    // safe half, and the reason is reported rather than swallowed.
    pluginErrors.push({
      plugin: pluginName(plugin),
      message: `its apply() threw: ${String((error && error.message) || error)}`,
    });
    return false;
  }
};
const plugins = (await flattenPlugins(config.plugins))
  .filter(runs)
  .map((plugin, index) => ({ plugin, index }))
  .sort((a, b) => hookOrder(a.plugin) - hookOrder(b.plugin) || a.index - b.index);
for (const { plugin } of plugins) {
  const hook = plugin.config;
  const handler = typeof hook === 'function' ? hook : hook.handler;
  if (typeof handler !== 'function') continue;
  try {
    // The hook may mutate `config` in place and return nothing, or return a
    // partial config; Vite supports both, so both are honored.
    const returned = await handler.call({ meta: { framework: 'diffpack' } }, config, env);
    if (returned && returned !== config) mergePluginConfig(config, returned);
  } catch (error) {
    pluginErrors.push({
      plugin: pluginName(plugin),
      message: String((error && error.message) || error),
    });
  }
}

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

// How JSX is LOWERED. Vite 8 runs this transform with oxc and derives `oxc.jsx`
// from the legacy `esbuild.*` fields when `oxc` is unset
// (`convertEsbuildConfigToOxcConfig`); an explicit `oxc` wins outright. Both shapes
// are read here and normalized to the same four values. Note that Vite consults
// `esbuild.jsxImportSource`/`jsxFactory`/`jsxFragment` ONLY once `esbuild.jsx` has
// selected a runtime, so neither does this — parity with the tool being matched
// beats guessing at intent.
const JSX_PRESETS = {
  react: { runtime: 'classic', pragma: 'React.createElement', pragmaFrag: 'React.Fragment' },
  'react-jsx': { runtime: 'automatic', importSource: 'react' },
};
let jsxOptions = null;
if (config.oxc && typeof config.oxc === 'object' && config.oxc.jsx !== undefined) {
  jsxOptions = config.oxc.jsx;
} else if (config.esbuild && typeof config.esbuild === 'object') {
  const { jsx: mode, jsxImportSource, jsxFactory, jsxFragment } = config.esbuild;
  if (mode === 'preserve') jsxOptions = 'preserve';
  else if (mode === 'automatic') jsxOptions = { runtime: 'automatic', importSource: jsxImportSource };
  else if (mode === 'transform') {
    jsxOptions = { runtime: 'classic', pragma: jsxFactory, pragmaFrag: jsxFragment };
  }
}
if (typeof jsxOptions === 'string' && jsxOptions !== 'preserve') {
  jsxOptions = JSX_PRESETS[jsxOptions] ?? null;
}
// `preserve` asks the transform to emit JSX unchanged. A bundle cannot ship JSX to a
// browser, so diffpack lowers it anyway; counted here so the caller can say so.
const jsxPreserve = jsxOptions === 'preserve';
const jsx = { runtime: null, importSource: null, factory: null, fragmentFactory: null };
if (jsxOptions && typeof jsxOptions === 'object') {
  if (jsxOptions.runtime === 'automatic' || jsxOptions.runtime === 'classic') {
    jsx.runtime = jsxOptions.runtime;
  }
  if (typeof jsxOptions.importSource === 'string') jsx.importSource = jsxOptions.importSource;
  if (typeof jsxOptions.pragma === 'string') jsx.factory = jsxOptions.pragma;
  if (typeof jsxOptions.pragmaFrag === 'string') jsx.fragmentFactory = jsxOptions.pragmaFrag;
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

// `build.outDir` / `build.assetsDir`: where the build is written, and the
// subdirectory of it that hashed assets go in. Vite's defaults are `dist` and
// `assets`; both are resolved by the caller against the project root. Passed
// through verbatim as strings (a non-string value is not expressible and is
// reported as absent, so the caller falls back to the Vite default rather than
// inventing one).
const outDir = typeof config.build?.outDir === 'string' ? config.build.outDir : null;
const assetsDir = typeof config.build?.assetsDir === 'string' ? config.build.assetsDir : null;

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
    jsx,
    jsxPreserve,
    inputs,
    outDir,
    assetsDir,
    manifest,
    manifestName,
    resolveConditions,
    mainFields,
    dedupe,
    optimizeDepsExclude,
    proxy,
    proxyRewriteSkipped,
    pluginErrors,
  }),
);
