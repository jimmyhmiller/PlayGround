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

process.stdout.write(
  JSON.stringify({
    base: typeof config.base === 'string' ? config.base : null,
    define,
    alias,
    aliasSkipped,
  }),
);
