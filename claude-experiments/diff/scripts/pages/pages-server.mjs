// The pages-router production server — the Node runtime a diffpack-built Next
// pages-router app is served by. diffpack builds every graph natively (Rust); this
// orchestrator is plain Node that wires the emitted client `public/` assets and the
// SSR bundle (`server/server.mjs`, which runs the app's own bundled React) into a
// working HTTP server. It runs NO React itself.
//
//   GET  /<asset>   -> serve the client build's public/ files (client.js, css, ...)
//   GET  /<route>   -> handleRequest -> full HTML document (hydrates #__next)
//   GET  /<route>?__nextDataReq=1 -> page props as JSON (client navigation)
//   *    /api/<x>   -> the page-api handler's response

import { createServer } from "node:http";
import { existsSync, readFileSync, statSync } from "node:fs";
import { extname, join, normalize } from "node:path";
import { pathToFileURL } from "node:url";

// The emitted SSR chunks carry a `.map` each; Node only consumes them with
// source-map support on. See `next-server.mjs` for why this is unconditional.
process.setSourceMapsEnabled(true);

function fail(message) {
  console.error(`pages-server: ${message}`);
  process.exit(1);
}

const outputDir = process.argv[2];
const port = Number(process.argv[3] || "3000");
if (!outputDir) fail("usage: node pages-server.mjs <.diffpack-output dir> [port]");

const publicDir = join(outputDir, "public");
const ssrEntry = join(outputDir, "server", "server.mjs");
if (!existsSync(ssrEntry)) {
  fail(`SSR bundle not found at ${ssrEntry} — run the client + ssr builds first`);
}
if (!existsSync(publicDir)) {
  fail(`client public/ not found at ${publicDir} — run the client build first`);
}

const ssrModule = await import(pathToFileURL(ssrEntry).href);
const handleRequest =
  ssrModule.handleRequest ||
  (ssrModule.default && ssrModule.default.handleRequest);
if (typeof handleRequest !== "function") {
  fail("the SSR bundle does not export handleRequest");
}

// Seed the SSG/ISR cache from the build-time prerender manifest so getStaticProps
// pages are answered from cache with zero per-request data fetch (and regenerate on
// their revalidate window). Absent manifest = no prerendered pages (SSG opted out).
const seedPrerender =
  ssrModule.seedPrerender || (ssrModule.default && ssrModule.default.seedPrerender);
const prerenderManifest = join(outputDir, "prerender.json");
if (existsSync(prerenderManifest) && typeof seedPrerender === "function") {
  try {
    const data = JSON.parse(readFileSync(prerenderManifest, "utf8"));
    seedPrerender(data);
    const count = (data.entries || []).length;
    console.log(`pages-server: seeded ${count} prerendered SSG page(s)`);
  } catch (error) {
    fail(`cannot read prerender manifest ${prerenderManifest}: ${error}`);
  }
}

const MIME = {
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".txt": "text/plain; charset=utf-8",
};

function tryServeStatic(pathname, res) {
  if (pathname === "/" || pathname.includes("..")) return false;
  const filePath = normalize(join(publicDir, pathname));
  if (!filePath.startsWith(publicDir)) return false; // path traversal guard
  if (!existsSync(filePath) || !statSync(filePath).isFile()) return false;
  const body = readFileSync(filePath);
  res.writeHead(200, {
    "content-type": MIME[extname(filePath)] || "application/octet-stream",
  });
  res.end(body);
  return true;
}

function readBody(req) {
  return new Promise((resolve) => {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", () => resolve(""));
  });
}

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, "http://localhost");
    const pathname = decodeURIComponent(url.pathname);

    if (req.method === "GET" && tryServeStatic(pathname, res)) return;

    const query = Object.fromEntries(url.searchParams.entries());
    const body =
      req.method === "POST" || req.method === "PUT" || req.method === "PATCH"
        ? await readBody(req)
        : undefined;

    const result = await handleRequest(
      req.method,
      pathname,
      query,
      req.headers,
      body,
    );
    res.writeHead(result.status, result.headers || {});
    res.end(result.body || "");
  } catch (error) {
    res.writeHead(500, { "content-type": "text/plain; charset=utf-8" });
    res.end(`pages-server error: ${error && error.stack ? error.stack : error}`);
  }
});

server.listen(port, () => {
  console.log(`pages-server: listening on http://127.0.0.1:${port}`);
});
