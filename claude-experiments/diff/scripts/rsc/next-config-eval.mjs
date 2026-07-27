// Evaluate `next.config.{js,mjs,ts}` and print the routing rules diffpack's
// orchestrator applies — `redirects()`, `rewrites()`, `headers()` — plus the scalar
// routing surface (`basePath`, `assetPrefix`, `trailingSlash`, `i18n`) and the `images`
// block, as JSON:
//   { "redirects": [...], "rewrites": [...], "headers": [...], "images": {...},
//     "basePath": "", "assetPrefix": "", "trailingSlash": false, "i18n": null }
// Loaded via the app's own jiti when present (handles a `.ts` config / ESM+CJS mix);
// falls back to a plain dynamic import. Only these three async functions are called —
// the rest of the config (webpack, experimental, …) is never touched.
import { pathToFileURL } from "node:url";
import Module, { createRequire } from "node:module";
import { dirname, resolve } from "node:path";

// Build-only Next plugins that wrap next.config but whose runtime behavior diffpack
// implements NATIVELY (so the real package is unnecessary and often not installed).
// `@next/mdx` compiles `.mdx`/`.md` through a webpack loader; diffpack compiles MDX
// natively (src/mdx.rs), so its only config-visible effect we must preserve is the
// `pageExtensions` merge (adding `md`/`mdx`). We shim it here as a faithful identity-plus-
// pageExtensions wrapper so `require("@next/mdx")` never crashes the config eval — a crash
// would otherwise discard the ENTIRE config (redirects/rewrites/headers/i18n/basePath).
// A real installed `@next/mdx` is intentionally shadowed: diffpack owns the MDX pipeline.
const NATIVE_CONFIG_PLUGINS = {
  "@next/mdx": (_pluginOptions) => (nextConfig = {}) => {
    const exts =
      Array.isArray(nextConfig.pageExtensions) && nextConfig.pageExtensions.length
        ? nextConfig.pageExtensions
        : ["tsx", "ts", "jsx", "js"];
    const merged = exts.includes("mdx") ? exts : [...exts, "md", "mdx"];
    return { ...nextConfig, pageExtensions: merged };
  },
};
// Intercept `require`/`createRequire` (both funnel through Module._load) for the shimmed
// specifiers, so both a CJS `require("@next/mdx")` and jiti's delegated require are caught.
const origLoad = Module._load;
Module._load = function (request, parent, isMain) {
  if (Object.prototype.hasOwnProperty.call(NATIVE_CONFIG_PLUGINS, request)) {
    return NATIVE_CONFIG_PLUGINS[request];
  }
  return origLoad.apply(this, arguments);
};

const configPath = process.argv[2];
if (!configPath) {
  process.stdout.write(
    JSON.stringify({
      redirects: [],
      rewrites: [],
      headers: [],
      images: extractImages({}, "."),
      ...extractRouting({}),
    }),
  );
  process.exit(0);
}

async function load() {
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

// The `images` config the next/image shim needs (remote-host allow-list + loader).
// `loaderFile` is resolved to an ABSOLUTE path so the generated image-config module can
// `import` it into every graph.
function extractImages(config, configPath) {
  const img = (config && config.images) || {};
  let loaderFile = null;
  if (img.loaderFile) {
    loaderFile = resolve(dirname(configPath), img.loaderFile);
  }
  return {
    deviceSizes: img.deviceSizes || null,
    imageSizes: img.imageSizes || null,
    remotePatterns: img.remotePatterns || [],
    domains: img.domains || [],
    loader: img.loader || "default",
    loaderFile,
    path: img.path || "/_next/image",
    qualities: img.qualities || null,
    unoptimized: Boolean(img.unoptimized),
  };
}

// A basePath is a URL prefix: it MUST have a leading slash and MUST NOT carry a trailing
// slash, so `${basePath}${appRelativePath}` composes cleanly. An empty value = no prefix.
function normalizeBasePath(value) {
  if (typeof value !== "string" || value === "" || value === "/") return "";
  let p = value.replace(/\/+$/, "");
  if (!p.startsWith("/")) p = "/" + p;
  return p;
}
// assetPrefix may be a same-origin path OR a full CDN URL. A protocol(-relative) URL keeps
// its shape (only the trailing slash is trimmed); a path-only prefix gets a leading slash.
function normalizeAssetPrefix(value) {
  if (typeof value !== "string" || value === "" || value === "/") return "";
  let p = value.replace(/\/+$/, "");
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(p) || p.startsWith("//")) return p;
  if (!p.startsWith("/")) p = "/" + p;
  return p;
}

// The scalar routing surface. `i18n` is normalized to null unless it carries a non-empty
// `locales` array (app-router `next build` ignores next.config `i18n`, so diffpack treats
// a present, well-formed `i18n` as an explicit opt-in to its locale-routing EXTENSION —
// see next-server.mjs; an absent/empty `i18n` is a plain no-op).
function extractRouting(config) {
  const i18nRaw = config && config.i18n;
  let i18n = null;
  if (i18nRaw && Array.isArray(i18nRaw.locales) && i18nRaw.locales.length) {
    i18n = {
      locales: i18nRaw.locales,
      defaultLocale: i18nRaw.defaultLocale || i18nRaw.locales[0],
      localeDetection: i18nRaw.localeDetection !== false,
      domains: Array.isArray(i18nRaw.domains) ? i18nRaw.domains : [],
    };
  }
  return {
    basePath: normalizeBasePath(config && config.basePath),
    assetPrefix: normalizeAssetPrefix(config && config.assetPrefix),
    trailingSlash: Boolean(config && config.trailingSlash),
    i18n,
    pageExtensions: extractPageExtensions(config),
  };
}

// The `pageExtensions` the config declares (Next's default is tsx/ts/jsx/js; `@next/mdx`
// merges md/mdx — see the shim above). Returned as-is (lowercased, dot-stripped) so the
// adapter can honor / validate them; `null` when the config does not set the field (the
// adapter then uses its built-in superset default).
function extractPageExtensions(config) {
  const raw = config && config.pageExtensions;
  if (!Array.isArray(raw) || !raw.length) return null;
  const out = [];
  for (const e of raw) {
    if (typeof e !== "string") continue;
    const ext = e.trim().replace(/^\./, "").toLowerCase();
    if (ext) out.push(ext);
  }
  return out.length ? out : null;
}

const EMPTY = {
  redirects: [],
  rewrites: [],
  headers: [],
  images: extractImages({}, configPath),
  ...extractRouting({}),
};

try {
  let config = await load();
  if (typeof config === "function") config = await config("phase-production-server", {});
  config = config || {};
  const out = {
    redirects: [],
    rewrites: [],
    headers: [],
    images: extractImages(config, configPath),
    ...extractRouting(config),
  };
  if (typeof config.redirects === "function") out.redirects = (await config.redirects()) || [];
  if (typeof config.rewrites === "function") {
    const r = (await config.rewrites()) || [];
    // rewrites() may return an array OR { beforeFiles, afterFiles, fallback }.
    out.rewrites = Array.isArray(r)
      ? r
      : [...(r.beforeFiles || []), ...(r.afterFiles || []), ...(r.fallback || [])];
  }
  if (typeof config.headers === "function") out.headers = (await config.headers()) || [];
  process.stdout.write(JSON.stringify(out));
} catch (error) {
  // A config that throws (e.g. a missing env) must not break the build; report it and
  // emit empty rules so the app still serves.
  process.stderr.write(`next.config eval: ${error && error.message ? error.message : error}\n`);
  process.stdout.write(JSON.stringify(EMPTY));
}
