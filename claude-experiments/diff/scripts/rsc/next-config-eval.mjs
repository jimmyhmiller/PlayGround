// Evaluate `next.config.{js,mjs,ts}` and print the routing rules diffpack's
// orchestrator applies — `redirects()`, `rewrites()`, `headers()` — as JSON:
//   { "redirects": [...], "rewrites": [...], "headers": [...] }
// Loaded via the app's own jiti when present (handles a `.ts` config / ESM+CJS mix);
// falls back to a plain dynamic import. Only these three async functions are called —
// the rest of the config (webpack, experimental, …) is never touched.
import { pathToFileURL } from "node:url";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";

const configPath = process.argv[2];
if (!configPath) {
  process.stdout.write(JSON.stringify({ redirects: [], rewrites: [], headers: [], images: extractImages({}, ".") }));
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

const EMPTY = { redirects: [], rewrites: [], headers: [], images: extractImages({}, configPath) };

try {
  let config = await load();
  if (typeof config === "function") config = await config("phase-production-server", {});
  config = config || {};
  const out = { redirects: [], rewrites: [], headers: [], images: extractImages(config, configPath) };
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
