// The app-router RSC app server — the emitted Node runtime a diffpack-built Next
// app-router app is served by. It is a small fork of scripts/rsc/rsc-server.mjs
// (Slice E) with ONE difference: an app-router RootLayout owns the whole document
// (`<html>…`), so the SSR bundle renders the FULL document (via react-dom's
// bootstrap options, which inject the client module + the inlined flight so
// hydration matches) and this server returns it directly, instead of wrapping a
// fragment in a `#root` div.
//
// diffpack builds every graph natively (Rust); this orchestrator is plain Node that
// wires diffpack's three emitted bundles + manifests into a working app. It runs NO
// React itself — it only moves flight BYTES between graphs, each carrying its own
// inlined React. The react-server React is isolated from the SSR/browser React by
// process: the react-server render/action bundle runs in a spawned child.
//
//   GET /            -> spawn the react-server render child -> flight of the
//                       app-router document tree; feed the flight to the SSR bundle
//                       (client refs resolved via the divergent-id
//                       serverConsumerManifest) -> full HTML document (with the
//                       client bootstrap + inlined flight) -> return it.
//   POST /_action/   -> spawn the react-server action child -> result flight.
//   GET /<asset>     -> serve the client build's public/ assets.

import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { readFileSync, existsSync, statSync } from "node:fs";
import { join, extname } from "node:path";
import { pathToFileURL } from "node:url";
import os from "node:os";

// DEV (`diffpack dev`): the diffpack dev server re-emits the SSR bundle in place on a
// client-island edit, but Node caches an ESM module by URL forever. In dev we
// therefore re-import the SSR bundle with a fresh `?v=<mtime>` whenever its mtime
// changes, so a manual refresh after an island edit runs the fresh SSR code (the
// react-server render child is already spawned fresh per request). The production
// path (a single top-level import) is untouched.
const DEV = process.env.DIFFPACK_NEXT_DEV === "1";

// DEV lifetime: `diffpack dev` holds this process's stdin open as a pipe. When the
// dev server dies for ANY reason (including SIGKILL, where no Rust cleanup runs), the
// OS closes the pipe and stdin ends here — so exit instead of orphaning. Exiting also
// closes the persistent worker's stdin, cascading a clean shutdown to it.
if (DEV) {
  process.stdin.on("end", () => process.exit(0));
  process.stdin.on("close", () => process.exit(0));
  process.stdin.resume();
}

function fail(message) {
  console.error(`next-server: ${message}`);
  process.exit(1);
}

const outputDir = process.argv[2];
const port = Number(process.argv[3] || "0");
if (!outputDir) fail("usage: node next-server.mjs <.diffpack-output dir> [port]");

const publicDir = join(outputDir, "public");
const rscRenderEntry = join(outputDir, "rsc-render", "server.mjs");
const ssrEntry = join(outputDir, "server", "server.mjs");
const clientManifestPath = join(outputDir, "client-references-manifest.json");
const ssrManifestPath = join(outputDir, "server-references-manifest.json");

for (const [label, p] of [
  ["client public/", publicDir],
  ["react-server render bundle", rscRenderEntry],
  ["SSR bundle", ssrEntry],
  ["client-references manifest", clientManifestPath],
  ["ssr-references manifest", ssrManifestPath],
]) {
  if (!existsSync(p)) fail(`${label} not found at ${p} — build all three graphs first`);
}

// --- Manifest #2: the divergent-id ssrModuleMapping ------------------------------
const clientRefs = JSON.parse(readFileSync(clientManifestPath, "utf8"));
const ssrRefs = JSON.parse(readFileSync(ssrManifestPath, "utf8"));
const moduleMap = {};
for (const [moduleId, clientEntry] of Object.entries(clientRefs)) {
  const ssrEntryRef = ssrRefs[moduleId];
  if (!ssrEntryRef) {
    fail(`no SSR reference for ${moduleId}; the SSR graph did not bundle this "use client" module`);
  }
  moduleMap[String(clientEntry.id)] = {
    "*": { id: ssrEntryRef.id, chunks: ssrEntryRef.chunks, name: "*" },
  };
}
const serverConsumerManifest = {
  moduleMap,
  serverModuleMap: null,
  moduleLoading: { prefix: "", crossOrigin: null },
};

// --- The SSR bundle (in-process; its own inlined React) --------------------------
// Resolve `renderFlightToDocument` from the SSR bundle. In production it is imported
// once and cached; in dev it is re-imported (fresh `?v=<mtime>`) whenever the bundle
// changes on disk, so a client-island re-emit is picked up on the next request.
let __ssrCache = { key: null, fn: null };
function pickRender(mod) {
  const fn = mod.renderFlightToDocument || (mod.default && mod.default.renderFlightToDocument);
  if (typeof fn !== "function") fail("the SSR bundle does not export renderFlightToDocument");
  return fn;
}
async function getRenderFlightToDocument() {
  if (!DEV) {
    if (!__ssrCache.fn) __ssrCache.fn = pickRender(await import(pathToFileURL(ssrEntry).href));
    return __ssrCache.fn;
  }
  const key = statSync(ssrEntry).mtimeMs;
  if (__ssrCache.fn && __ssrCache.key === key) return __ssrCache.fn;
  const fn = pickRender(await import(pathToFileURL(ssrEntry).href + "?v=" + key));
  __ssrCache = { key, fn };
  return fn;
}
// Prime + validate the export up front (fail fast if the bundle is malformed), in
// both modes.
await getRenderFlightToDocument();

// --- Persistent react-server worker POOL (dev + production) -----------------------
// Spawning a fresh Node child per `?__rsc=1` render pays the whole cold-start cost on
// EVERY request — ruinous for production latency and memory. Instead we keep a pool of
// long-lived `serve` workers (`rsc-render/server.mjs serve`) that stay warm and answer
// render/action requests over newline-delimited JSON on stdin/stdout. DEV runs ONE
// worker (a single browser) which re-imports its bundle with `?v=<mtime>` on a diffpack
// re-emit so a server-component edit is picked up without a respawn; PRODUCTION runs a
// small pool (round-robined) so concurrent requests render in parallel, with the bundle
// mtime stable so each worker imports it exactly once. Same process isolation, no
// per-request spawn.
// Pool size trades memory for render concurrency. Each worker is a separate Node
// process (~one Node baseline + the react-server bundle), so memory scales with the
// count. Default to a small pool that keeps steady-state RSS below a single-process
// Next server while still rendering a couple of requests in parallel; override with
// DIFFPACK_RSC_WORKERS for high-concurrency deployments. DEV is always 1.
const POOL_SIZE = DEV
  ? 1
  : Math.max(1, Number(process.env.DIFFPACK_RSC_WORKERS) || 2);
let workerPool = null;
let poolCursor = 0;

function spawnWorker() {
  const child = spawn(process.execPath, [rscRenderEntry, "serve"], {
    stdio: ["pipe", "pipe", "inherit"], // worker stderr -> server log
  });
  const pending = new Map();
  let seq = 0;
  let buffer = "";
  const worker = { dead: false, call: null };
  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (chunk) => {
    buffer += chunk;
    let nl;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl);
      buffer = buffer.slice(nl + 1);
      if (!line.trim()) continue;
      let msg;
      try {
        msg = JSON.parse(line);
      } catch {
        continue; // ignore any non-protocol stdout line
      }
      const settle = pending.get(msg.id);
      if (settle) {
        pending.delete(msg.id);
        settle(msg);
      }
    }
  });
  const fail = (reason) => {
    worker.dead = true;
    for (const settle of pending.values()) settle({ error: reason });
    pending.clear();
  };
  child.on("exit", (code) => fail(`react-server worker exited (${code})`));
  child.on("error", (error) => fail(`react-server worker error: ${error}`));
  worker.call = (req) =>
    new Promise((resolve, reject) => {
      const id = ++seq;
      pending.set(id, (msg) => {
        if (msg.error) {
          reject(new Error(`react-server worker: ${msg.error}`));
          return;
        }
        resolve({
          flight: Buffer.from(msg.flight || "", "base64"),
          status: msg.status || 200,
          params: msg.params || {},
          redirect: msg.redirect,
          notFound: msg.notFound,
        });
      });
      child.stdin.write(JSON.stringify({ id, ...req }) + "\n");
    });
  return worker;
}

// The next warm worker, round-robined; a crashed worker is respawned on demand.
function nextWorker() {
  if (!workerPool) workerPool = Array.from({ length: POOL_SIZE }, spawnWorker);
  for (let i = 0; i < workerPool.length; i += 1) {
    if (workerPool[i].dead) workerPool[i] = spawnWorker();
  }
  const worker = workerPool[poolCursor % workerPool.length];
  poolCursor += 1;
  return worker;
}

// --- Spawn the react-server child for a flight (render or action) ----------------
// fd 3 is a status/params sidechannel the render op writes `{status,params}` to
// (guarded on the child side): a 404 renders its flight AND exits 0, carrying its
// HTTP status only over fd 3 — so we must NOT infer failure from a non-zero exit for
// a 404 (the child never exits non-zero for one). Resolves `{flight,status,params}`.
// Render + action always route through the persistent worker pool (dev AND prod), so
// no request pays a Node cold start; other ops fall through to a one-shot spawn.
function runReactServer(args, stdinBody) {
  const op = args[0];
  if (op === "render") {
    let reqCtx = {};
    if (stdinBody != null && String(stdinBody).trim()) {
      try {
        reqCtx = JSON.parse(String(stdinBody));
      } catch {
        reqCtx = {};
      }
    }
    return nextWorker().call({
      op: "render",
      pathname: args[1],
      manifestPath: args[2],
      reqCtx,
    });
  }
  if (op === "action") {
    return nextWorker().call({
      op: "action",
      actionId: args[1],
      manifestPath: args[2],
      body: stdinBody != null ? String(stdinBody) : "",
    });
  }
  // Unknown op falls through to a one-shot spawn (defensive; not reached today).
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [rscRenderEntry, ...args], {
      stdio: ["pipe", "pipe", "pipe", "pipe"],
    });
    const out = [];
    const err = [];
    const meta = [];
    child.stdout.on("data", (chunk) => out.push(Buffer.from(chunk)));
    child.stderr.on("data", (chunk) => err.push(Buffer.from(chunk)));
    if (child.stdio[3]) child.stdio[3].on("data", (chunk) => meta.push(Buffer.from(chunk)));
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`react-server child (${args.join(" ")}) exited ${code}:\n${Buffer.concat(err)}`));
        return;
      }
      let parsed = { status: 200, params: {} };
      const raw = Buffer.concat(meta).toString("utf8").trim();
      if (raw) {
        try {
          parsed = JSON.parse(raw);
        } catch {
          // leave the defaults; a malformed sidechannel is non-fatal.
        }
      }
      resolve({
        flight: Buffer.concat(out),
        status: parsed.status || 200,
        params: parsed.params || {},
        // Slice I: the render's control channel — a server-side redirect()/notFound()
        // captured from the flight render's onError. The orchestrator acts on these
        // (a real HTTP redirect / 404) instead of SSRing the errored flight tree.
        redirect: parsed.redirect,
        notFound: parsed.notFound,
      });
    });
    if (stdinBody != null) child.stdin.write(stdinBody);
    child.stdin.end();
  });
}

const MIME = {
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".map": "application/json",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".gif": "image/gif",
  ".avif": "image/avif",
  ".ico": "image/x-icon",
};

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, "http://localhost");
    // Server actions.
    if (req.method === "POST" && url.pathname === "/_action/") {
      const id = req.headers["x-diffpack-action-id"];
      if (!id) {
        res.writeHead(400).end("missing x-diffpack-action-id");
        return;
      }
      const body = [];
      for await (const chunk of req) body.push(Buffer.from(chunk));
      const { flight } = await runReactServer(["action", id, clientManifestPath], Buffer.concat(body));
      res.writeHead(200, { "content-type": "text/x-component" });
      res.end(flight);
      return;
    }
    // Static assets from the client build's public/ (checked before route render so
    // /client.js, /rsc.css, etc. are served, not treated as app-router paths).
    if (req.method === "GET") {
      const name = url.pathname.replace(/^\//, "");
      const filePath = join(publicDir, name);
      if (name && existsSync(filePath) && filePath.startsWith(publicDir)) {
        res.writeHead(200, { "content-type": MIME[extname(filePath)] || "application/octet-stream" });
        res.end(readFileSync(filePath));
        return;
      }
    }
    // Any other GET is an app-router route: the react-server render matches the
    // pathname to a route (nested-layout chain composed around its page); the SSR
    // bundle turns the flight into the full document.
    if (req.method === "GET") {
      // The per-request context the react-server render establishes (an
      // AsyncLocalStorage carrying the request url/headers/cookie, read by
      // next/headers cookies()/headers() inside async Server Components). Passed to the
      // render child on stdin. Array-valued headers (e.g. set-cookie) are joined so
      // `new Headers([[k,v]...])` accepts them.
      const reqCtx = JSON.stringify({
        url: "http://localhost" + req.url,
        headers: Object.entries(req.headers).map(([k, v]) => [
          k,
          Array.isArray(v) ? v.join(", ") : String(v),
        ]),
        cookie: req.headers.cookie || "",
      });
      const { flight, status, params, redirect, notFound } = await runReactServer(
        ["render", url.pathname, clientManifestPath],
        reqCtx,
      );
      // Server-side redirect(): issue a REAL HTTP redirect (do NOT SSR the errored
      // flight tree). Over the soft-nav channel (?__rsc=1) hand the client Router a
      // JSON redirect it can follow via history + a re-fetch.
      if (redirect) {
        if (url.searchParams.has("__rsc")) {
          res.writeHead(200, { "content-type": "application/json" });
          res.end(JSON.stringify({ __redirect: redirect }));
          return;
        }
        res.writeHead(status || 307, { location: redirect });
        res.end();
        return;
      }
      // Server-side notFound(): render the real not-found tree (a guaranteed
      // matchRoute miss yields app/not-found under the root layout) and serve it 404.
      if (notFound) {
        const nf = await runReactServer(
          ["render", "/__diffpack_notfound__", clientManifestPath],
          reqCtx,
        );
        const nfDoc = await (await getRenderFlightToDocument())(
          new Uint8Array(nf.flight),
          serverConsumerManifest,
          nf.flight.toString("base64"),
          {},
          { pathname: url.pathname, search: url.search },
        );
        res.writeHead(404, { "content-type": "text/html; charset=utf-8" });
        res.end(nfDoc);
        return;
      }
      // Soft-navigation: the client Router fetches `?__rsc=1` for the RAW flight of
      // the target route and diff-renders it in place (no full document load). The
      // static-asset check above ran first, so `?__rsc=1` never shadows an asset.
      // Raw flight is status-agnostic (a 404 body tree is still valid flight).
      if (url.searchParams.has("__rsc")) {
        res.writeHead(200, { "content-type": "text/x-component" });
        res.end(flight);
        return;
      }
      const doc = await (await getRenderFlightToDocument())(
        new Uint8Array(flight),
        serverConsumerManifest,
        flight.toString("base64"),
        params,
        { pathname: url.pathname, search: url.search },
      );
      res.writeHead(status || 200, { "content-type": "text/html; charset=utf-8" });
      res.end(doc);
      return;
    }
    res.writeHead(404).end("not found");
  } catch (error) {
    console.error(error && error.stack ? error.stack : String(error));
    res.writeHead(500, { "content-type": "text/plain" }).end(String(error && error.stack));
  }
});

server.listen(port, () => {
  const actual = server.address().port;
  console.log(`next-server listening on http://localhost:${actual}`);
});
