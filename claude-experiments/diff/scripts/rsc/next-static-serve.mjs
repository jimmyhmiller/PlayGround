// The DUMB static file server for a diffpack SSG export — the honesty proof that
// serving a prerendered app is PURE FILES: it imports NEITHER RSC bundle (no
// rsc-render/server.mjs, no server/server.mjs) and spawns NO child process. It only
// reads `<static>/*.html` / `*.rsc` / assets off disk and streams them. If it ever
// rendered per request, the SSG claim would be false — so this must stay pure `fs`.
//
//   usage: node next-static-serve.mjs <static-dir> [port]
//
//   GET /            -> index.html
//   GET /x           -> x.html
//   GET /x/y         -> x/y.html
//   GET /x?__rsc=1   -> x.rsc   (content-type: text/x-component — the soft-nav source)
//   GET /<asset>     -> the file with its MIME (client.js, rsc.css, images, ...)
//   POST /_action/   -> 501 (no server on a static export)
//   unknown route recorded `dynamic` in the manifest -> 501 (clear message)
//   otherwise unknown -> 404

import { createServer } from "node:http";
import { readFileSync, existsSync, statSync } from "node:fs";
import { join, extname, normalize } from "node:path";

function die(message) {
  console.error(`next-static-serve: ${message}`);
  process.exit(1);
}

const staticDir = process.argv[2];
const port = Number(process.argv[3] || "0");
if (!staticDir) die("usage: node next-static-serve.mjs <static-dir> [port]");
if (!existsSync(staticDir)) die(`static dir ${staticDir} does not exist — run \`diffpack build-app <root> static\` first`);

// The prerender manifest tells us which unknown paths are dynamic (→ 501, honest) vs
// truly not-found (→ 404). Absent manifest → everything unknown is a 404.
let manifest = { static: [], dynamic: [] };
const manifestPath = join(staticDir, "prerender-manifest.json");
if (existsSync(manifestPath)) {
  try {
    manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  } catch {
    // a malformed manifest is non-fatal for serving; keep the empty default.
  }
}
const dynamicPaths = new Set((manifest.dynamic || []).map((d) => d.path));
const dynamicReason = new Map((manifest.dynamic || []).map((d) => [d.path, d.reason || "dynamic"]));

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".map": "application/json",
  ".rsc": "text/x-component",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".gif": "image/gif",
  ".avif": "image/avif",
  ".ico": "image/x-icon",
  ".txt": "text/plain; charset=utf-8",
};

// Resolve a request-relative path safely under staticDir (no `..` escape).
function safeJoin(rel) {
  const clean = normalize(rel).replace(/^(\.\.[/\\])+/, "");
  const full = join(staticDir, clean);
  if (!full.startsWith(staticDir)) return null;
  return full;
}

function serveFile(res, filePath, contentType) {
  const body = readFileSync(filePath);
  res.writeHead(200, { "content-type": contentType || MIME[extname(filePath)] || "application/octet-stream" });
  res.end(body);
}

function isFile(p) {
  return p != null && existsSync(p) && statSync(p).isFile();
}

const server = createServer((req, res) => {
  try {
    const url = new URL(req.url, "http://localhost");
    const pathname = url.pathname;

    if (req.method === "POST" && pathname === "/_action/") {
      res.writeHead(501, { "content-type": "text/plain" });
      res.end("no server on a static export: server actions require the orchestrator (next-server.mjs)");
      return;
    }
    if (req.method !== "GET") {
      res.writeHead(405, { "content-type": "text/plain" }).end("method not allowed");
      return;
    }

    // The route file stem for this pathname: "/" -> "index", "/x" -> "x", "/x/y" -> "x/y".
    const stem = pathname === "/" ? "index" : pathname.replace(/^\//, "").replace(/\/$/, "");

    // Soft-nav flight request: serve the prerendered .rsc (raw flight).
    if (url.searchParams.has("__rsc")) {
      const rscPath = safeJoin(`${stem}.rsc`);
      if (isFile(rscPath)) {
        serveFile(res, rscPath, "text/x-component");
        return;
      }
      // No prerendered flight (a dynamic route) — 404 so the client Router falls back
      // to a full navigation (which also 404s on a pure static host — honest scope).
      res.writeHead(404, { "content-type": "text/plain" }).end("no prerendered flight for this route (dynamic route on a static export)");
      return;
    }

    // A raw asset with an extension (client.js, rsc.css, images, ...) — serve verbatim.
    if (extname(pathname)) {
      const assetPath = safeJoin(pathname.replace(/^\//, ""));
      if (isFile(assetPath)) {
        serveFile(res, assetPath);
        return;
      }
    }

    // A prerendered page: <stem>.html.
    const htmlPath = safeJoin(`${stem}.html`);
    if (isFile(htmlPath)) {
      serveFile(res, htmlPath, "text/html; charset=utf-8");
      return;
    }

    // Unknown path: dynamic (recorded in the manifest) -> 501; else 404. Match against
    // the manifest's dynamic route patterns by prefix of the first segment too, so
    // e.g. /blog/anything maps to the /blog/[slug] dynamic entry.
    const firstSeg = "/" + (pathname.split("/").filter(Boolean)[0] || "");
    let isDynamic = dynamicPaths.has(pathname);
    let reason = dynamicReason.get(pathname);
    if (!isDynamic) {
      for (const dp of dynamicPaths) {
        // A dynamic pattern like /blog/[slug] or /go — match by its leading static seg.
        const dpFirst = "/" + (dp.split("/").filter(Boolean)[0] || "");
        if (dpFirst === firstSeg) {
          isDynamic = true;
          reason = dynamicReason.get(dp);
          break;
        }
      }
    }
    if (isDynamic) {
      res.writeHead(501, { "content-type": "text/plain" });
      res.end(
        `route ${pathname} is dynamic (${reason || "dynamic"}); a pure static export cannot serve it — ` +
          `use the orchestrator (next-server.mjs) for the hybrid static + dynamic surface`,
      );
      return;
    }
    res.writeHead(404, { "content-type": "text/plain" }).end(`404 — no prerendered page for ${pathname}`);
  } catch (error) {
    console.error(error && error.stack ? error.stack : String(error));
    res.writeHead(500, { "content-type": "text/plain" }).end("static-serve error");
  }
});

server.listen(port, () => {
  const actual = server.address().port;
  console.log(`next-static-serve listening on http://localhost:${actual} (static dir: ${staticDir})`);
});
