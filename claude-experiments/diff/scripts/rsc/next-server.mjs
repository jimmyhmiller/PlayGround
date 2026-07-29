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
import {
  readFileSync,
  readdirSync,
  existsSync,
  statSync,
  writeFileSync,
  renameSync,
  mkdirSync,
  createReadStream,
} from "node:fs";
import { pipeline } from "node:stream";
import { createGzip } from "node:zlib";
import { createHash } from "node:crypto";
import { join, extname } from "node:path";
import { pathToFileURL } from "node:url";
import os from "node:os";
import { MANIFEST_FILES, loadServerConsumerManifest } from "./ssr-module-map.mjs";

// The server graphs ship a `.map` beside every emitted chunk (Next emits server
// source maps unconditionally and so does diffpack — see
// `next_adapter::default_source_maps`). Node only READS those maps when source-map
// support is on, so without this the maps exist and nothing consumes them: an
// exception out of a Server Component or a route handler prints positions in
// `server.chunk-12.mjs` instead of the file the developer wrote. Node resolves a
// map lazily, only when a stack is actually formatted, so an app that never throws
// pays nothing.
process.setSourceMapsEnabled(true);

// DEV (`diffpack dev`): Node caches an ES module by URL forever, so an edit can only
// reach an already-loaded server graph through a URL Node has never seen. The dev
// server pushes each edit to `POST /__diffpack_dev/hot` as a tiny register-only
// micro-chunk holding ONLY the changed modules; this process applies it to the live
// SSR runtime and forwards it to the react-server worker. Nothing here polls a file's
// mtime — see `getSsrRenderers` for why entry-level re-importing cannot work.
const DEV = process.env.DIFFPACK_NEXT_DEV === "1";

// `DIFFPACK_BOOT_TRACE=1`: stamp each boot phase to stderr. Costs one env read
// otherwise; kept permanently so a slow start is diagnosable in the field.
const BOOT_TRACE = process.env.DIFFPACK_BOOT_TRACE === "1";
const bootMark = (label) => {
  if (BOOT_TRACE) console.error(`[boot] ${label} @ ${performance.now().toFixed(0)}ms`);
};

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

// A production server must survive a failing REQUEST. Node's default handlers turn an
// uncaught exception (or an unhandled rejection, since Node 15) into `process.exit(1)`,
// so ONE bad render — or one socket the peer reset mid-stream — takes the whole server
// down and every other in-flight request with it. Log loudly and keep serving; the
// failing request is already accounted for by `failRequest` below.
for (const [event, label] of [
  ["uncaughtException", "uncaught exception"],
  ["unhandledRejection", "unhandled rejection"],
]) {
  process.on(event, (error) => {
    console.error(
      `next-server: ${label} (request dropped, server stays up):`,
      error && error.stack ? error.stack : String(error),
    );
  });
}

// Reports a request-handling failure on `res` WITHOUT ever throwing.
//
// Once the streaming shell has gone out, the status line is spent: `res.writeHead(500)`
// on such a response throws `ERR_HTTP_HEADERS_SENT`, which (from an async handler) is an
// unhandled rejection — the crash this exists to prevent. After headers are sent the only
// truthful signal left is to END the response, so the client sees a truncated document
// rather than a hung socket.
function failRequest(res, error) {
  console.error(error && error.stack ? error.stack : String(error));
  try {
    if (res.headersSent || res.writableEnded) {
      if (!res.writableEnded) res.end();
      return;
    }
    res.writeHead(500, { "content-type": "text/plain; charset=utf-8" });
    res.end(String(error && error.stack ? error.stack : error));
  } catch (secondary) {
    // Reporting the failure failed (a destroyed socket, a response another writer
    // already finished). Drop the connection; never let this path throw.
    console.error(
      "next-server: could not report a request error:",
      secondary && secondary.message ? secondary.message : String(secondary),
    );
    try {
      res.destroy();
    } catch {}
  }
}

// The `'nonce-…'` value a strict Content-Security-Policy puts on `script-src` (falling
// back to `default-src`, exactly as Next's `getScriptNonceFromHeader` does). The header
// is read off the EFFECTIVE request headers — i.e. after middleware's
// `NextResponse.next({ request: { headers } })` overrides — because that is where the
// canonical strict-CSP recipe sets it. Every script the render emits must carry this
// value or the browser blocks it and the page never hydrates.
function scriptNonceFromHeaders(headerPairs) {
  let csp;
  for (const [name, value] of headerPairs) {
    const lower = String(name).toLowerCase();
    if (lower === "content-security-policy" || lower === "content-security-policy-report-only") {
      csp = String(value);
      if (lower === "content-security-policy") break;
    }
  }
  if (!csp) return undefined;
  // `script-src` first, `default-src` only as a fallback — a policy that declares both
  // (the canonical strict-CSP recipe does) carries the nonce on `script-src`.
  const directives = csp.split(";").map((part) => part.trim());
  const directive =
    directives.find((part) => part.startsWith("script-src")) ||
    directives.find((part) => part.startsWith("default-src"));
  if (!directive) return undefined;
  const token = directive
    .split(/\s+/)
    .slice(1)
    .find((part) => part.startsWith("'nonce-") && part.endsWith("'") && part.length > 8);
  return token ? token.slice("'nonce-".length, -1) : undefined;
}

const outputDir = process.argv[2];
const port = Number(process.argv[3] || "0");
if (!outputDir) fail("usage: node next-server.mjs <.diffpack-output dir> [port]");

const publicDir = join(outputDir, "public");
const rscRenderEntry = join(outputDir, "rsc-render", "server.mjs");
const ssrEntry = join(outputDir, "server", "server.mjs");

for (const [label, p] of [
  ["client public/", publicDir],
  ["react-server render bundle", rscRenderEntry],
  ["SSR bundle", ssrEntry],
  ...MANIFEST_FILES.map(([label, file]) => [label, join(outputDir, file)]),
]) {
  if (!existsSync(p)) fail(`${label} not found at ${p} — build all three graphs first`);
}

// --- Manifest #2: the divergent-id ssrModuleMapping ------------------------------
// Joined from the three graphs' manifests by ./ssr-module-map.mjs — the one place
// that decides which client references are resolvable (see its header).
let clientManifestPath;
let serverConsumerManifest;
try {
  ({ clientManifestPath, serverConsumerManifest } = loadServerConsumerManifest(outputDir));
} catch (error) {
  fail(error.message);
}
bootMark("manifests loaded");

// --- The SSR bundle (in-process; its own inlined React) --------------------------
// Resolve `renderFlightToDocument` from the SSR bundle. It is imported ONCE, in both
// modes. In dev, freshness after an edit does NOT come from re-importing this entry:
// see `applyHotUpdate` — the dev server pushes the changed modules as a micro-chunk
// and the live runtime swaps them in place.
//
// Re-importing the entry cannot work and must never be reintroduced. Two independent
// reasons, both verified:
//   1. An edit re-emits only the chunk that HOSTS the changed module, so the entry's
//      own mtime does not move — an mtime-keyed cache never even re-imports.
//   2. Even forcing a fresh `?v=` on the entry keeps serving stale code, because the
//      entry reaches its split chunks through `import("./server.chunk-N.mjs")`, whose
//      URL carries no query: Node returns the already-evaluated chunk from its ESM
//      cache, so the fresh entry binds against STALE factories (in the react-server
//      worker that surfaced as a hard `Module is not loaded: <id>` crash, because the
//      fresh runtime's id table and the cached chunk's registrations disagree).
// Only a URL Node has never seen re-evaluates, which is exactly what the per-edit
// micro-chunk is.
let __ssrCache = { fns: null };
function pickRender(mod) {
  const ns = mod.default && mod.default.renderFlightToDocument ? mod.default : mod;
  const doc = ns.renderFlightToDocument;
  if (typeof doc !== "function") fail("the SSR bundle does not export renderFlightToDocument");
  // Streaming export is required for the streaming GET path; buffered doc render (SSG,
  // notFound) still uses `doc`.
  const stream = ns.renderFlightToStream;
  if (typeof stream !== "function") fail("the SSR bundle does not export renderFlightToStream");
  // A hybrid app's `pages/api/**` endpoints run out of THIS bundle (Next's `api-node`
  // layer has no `react-server` condition, and neither does this graph). Not required to
  // exist — an app with no `pages/api` still has none — so it is only demanded at the
  // moment a request actually matches one of those patterns.
  return { doc, stream, pagesApi: ns.handlePagesApi };
}
async function getSsrRenderers() {
  if (!__ssrCache.fns) __ssrCache.fns = pickRender(await import(pathToFileURL(ssrEntry).href));
  return __ssrCache.fns;
}
async function getRenderFlightToDocument() {
  return (await getSsrRenderers()).doc;
}
async function getRenderFlightToStream() {
  return (await getSsrRenderers()).stream;
}
// The SSR bundle's `pages/api/**` dispatcher. Reached only after the route manifest said
// this path IS a pages API route, so a missing export is a build inconsistency, not a
// routing miss — fail loudly rather than answering a wrong 404.
async function getPagesApiHandler() {
  const fn = (await getSsrRenderers()).pagesApi;
  if (typeof fn !== "function") {
    // Thrown, not `fail()`ed: this is a per-request path, and a production server must
    // answer 500 for the request rather than take the whole process down.
    throw new Error(
      "next-server: the route manifest matched a pages/api route but the SSR bundle exports no handlePagesApi; " +
        "the pages-router API routes were not bundled into the SSR graph",
    );
  }
  return fn;
}
// Priming + validating the SSR bundle up front (fail fast if it is malformed)
// happens below, AFTER the worker pool is prespawned — the two most expensive
// boot steps are this process evaluating the SSR bundle and the react-server
// worker evaluating its own bundle, and prespawning first runs them in separate
// processes concurrently instead of back to back. See the prespawn site after
// `nextWorker` for the `await`.

// --- Persistent react-server worker POOL (dev + production) -----------------------
// Spawning a fresh Node child per `?__rsc=1` render pays the whole cold-start cost on
// EVERY request — ruinous for production latency and memory. Instead we keep a pool of
// long-lived `serve` workers (`rsc-render/server.mjs serve`) that stay warm and answer
// render/action requests over newline-delimited JSON on stdin/stdout. DEV runs ONE
// worker (a single browser), hot-patched in place by the `invalidate` op below so a
// server-component edit is picked up without a respawn; PRODUCTION runs a small pool
// (round-robined) so concurrent requests render in parallel, each importing the bundle
// exactly once. Same process isolation, no per-request spawn.
// Pool size trades memory for CPU-bound render parallelism. Each worker is a separate
// Node process (~one Node baseline + the react-server bundle), so memory scales with
// the count. Default to ONE: an RSC render is mostly async-I/O-bound, so a single
// worker already overlaps concurrent requests within its event loop (a slow awaited
// Server Component does not block others), and a lone worker's steady-state footprint
// sits well below a single-process Next server (measured ~26% less). Bump
// DIFFPACK_RSC_WORKERS only for CPU-bound render throughput on many-core hosts. DEV is
// always 1.
const POOL_SIZE = DEV
  ? 1
  : Math.max(1, Number(process.env.DIFFPACK_RSC_WORKERS) || 1);
let workerPool = null;
let poolCursor = 0;
// DEV: the react-server hot updates (see `POST /__diffpack_dev/hot`) the dev server has
// not yet re-emitted to disk in full, replayed into any worker spawned meanwhile.
let pendingHot = [];

function spawnWorker() {
  // `--enable-source-maps`, not `process.setSourceMapsEnabled()`: this worker's entry
  // IS an emitted chunk (`rsc-render/server.mjs`), so there is no diffpack-authored
  // line in it to make the call from, and source-map support does not cross a process
  // boundary. Without the flag the SERVER COMPONENT render — the layer whose
  // exceptions are hardest to place — reports positions in `server.chunk-N.mjs` while
  // its `.map` sits unread beside it. The worker's stderr is this server's log, so
  // this is the trace a developer actually reads.
  const child = spawn(process.execPath, ["--enable-source-maps", rscRenderEntry, "serve"], {
    stdio: ["pipe", "pipe", "inherit"], // worker stderr -> server log
  });
  const pending = new Map();
  // Streaming render ops (`render-stream`) get multiple replies per id: one streamMeta,
  // N streamChunk, one streamEnd. Their handlers live here, not in `pending`.
  const streamPending = new Map();
  let seq = 0;
  let buffer = "";
  const worker = { dead: false, call: null, callStream: null };
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
      // Streaming render reply (streamMeta/streamChunk/streamEnd), or an error for a
      // stream id: route to the stream handler and do NOT touch `pending`.
      const stream = streamPending.get(msg.id);
      if (stream) {
        if (msg.error) {
          streamPending.delete(msg.id);
          stream.onError(msg.error);
        } else if (msg.streamMeta !== undefined) {
          stream.onMeta(msg.streamMeta);
        } else if (msg.streamChunk !== undefined) {
          stream.onChunk(msg.streamChunk);
        } else if (msg.streamEnd !== undefined) {
          streamPending.delete(msg.id);
          stream.onEnd(msg.streamEnd);
        }
        continue;
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
    for (const stream of streamPending.values()) stream.onError(reason);
    streamPending.clear();
  };
  child.on("exit", (code) => fail(`react-server worker exited (${code})`));
  child.on("error", (error) => fail(`react-server worker error: ${error}`));
  // Resolves the RAW reply message; callers extract the fields for their op (a render
  // reply carries `flight`, a route reply `routeResult`, a routes reply `routes`).
  worker.call = (req) =>
    new Promise((resolve, reject) => {
      const id = ++seq;
      pending.set(id, (msg) => {
        if (msg.error) {
          reject(new Error(`react-server worker: ${msg.error}`));
          return;
        }
        resolve(msg);
      });
      child.stdin.write(JSON.stringify({ id, ...req }) + "\n");
    });
  // Streaming render: `handlers` = { onMeta, onChunk, onEnd, onError }. Returns nothing;
  // the caller drives its response from the callbacks.
  worker.callStream = (req, handlers) => {
    const id = ++seq;
    streamPending.set(id, handlers);
    child.stdin.write(JSON.stringify({ id, ...req }) + "\n");
  };
  // DEV: a freshly spawned worker loads `rsc-render/` FROM DISK, which lags the live
  // process by every hot update the dev server has not yet re-emitted in full. Replay
  // them, in order, before this worker can answer anything — the worker reads stdin
  // sequentially, so queueing the ops here is enough to order them ahead of the first
  // render. `pendingHot` is the dev server's own list (it re-sends it on every push and
  // empties it once the full re-emit has landed on disk), so the replay is never a
  // guess about what disk holds.
  for (const update of pendingHot) {
    worker.call({ op: "invalidate", chunk: update.chunk, ids: update.ids }).catch((error) => {
      console.error(
        "next-server: replaying a dev hot update into a respawned react-server worker failed:",
        error && error.stack ? error.stack : String(error),
      );
    });
  }
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

// Prespawn the pool, THEN prime + validate this process's SSR bundle: the worker
// children evaluate `rsc-render/server.mjs` while this process evaluates the SSR
// bundle, so the two big imports overlap instead of the first render paying the
// worker's whole cold start on top of a boot that already sat through the SSR
// import. A worker that fails to boot surfaces exactly as before — its first
// call rejects and `nextWorker` respawns on demand.
workerPool = Array.from({ length: POOL_SIZE }, spawnWorker);
bootMark("worker pool prespawned");
await getSsrRenderers();
bootMark("ssr bundle primed");

// --- DEV hot updates (`POST /__diffpack_dev/hot`) ---------------------------------
// The dev server's chunk-granular invalidation channel, the Next analogue of the
// TanStack control endpoint (`dev_server::hmr_reload_server`). Per edit it renders a
// register-only micro-chunk holding ONLY the changed modules of a server graph and
// POSTs its path here. Applying one is: import the micro-chunk (a URL Node has never
// seen, so it really evaluates and re-registers those factories into the LIVE runtime),
// then `serverInvalidate(ids)` — which drops the cache for the changed modules and
// everything that imports them up to the entry, re-runs exactly that path, and
// republishes the entry's exports. Everything else (React, the app's dependencies)
// stays cached, so the React singleton and the process both survive.
//
// `pendingHot` (declared with the worker pool above) mirrors the react-server updates
// the dev server has not yet re-emitted to disk in full, so a respawned worker can be
// caught up — see `spawnWorker`.

/// Resolve the live HMR runtime of the bundle loaded in THIS process. Emitted only in
/// dev builds (`EmitOptions.hmr`), so its absence is a hard error, never a fallback.
function hotRuntime(graph) {
  const runtime = globalThis.__diffpack_hmr_runtime;
  if (!runtime || typeof runtime.serverInvalidate !== "function") {
    throw new Error(
      `diffpack dev: the ${graph} bundle exposes no __diffpack_hmr_runtime.serverInvalidate; it was not emitted with HMR enabled, so a hot update cannot be applied`,
    );
  }
  return runtime;
}

/// Apply one micro-chunk to the SSR bundle living in THIS process and rebind the
/// cached render functions to the entry's fresh exports.
async function applySsrHotUpdate(update) {
  await import(pathToFileURL(update.chunk).href);
  await hotRuntime("ssr").serverInvalidate(update.ids, []);
  // `serverInvalidate` re-runs the entry and publishes its fresh exports here; the
  // namespace object the original `import` returned is bound to the OLD run.
  const fresh = globalThis.__diffpack_ssr_entry;
  if (!fresh) {
    throw new Error(
      "diffpack dev: the ssr runtime did not republish globalThis.__diffpack_ssr_entry after serverInvalidate; the entry re-run failed",
    );
  }
  __ssrCache = { fns: pickRender(fresh) };
}

// --- Spawn the react-server child for a flight (render or action) ----------------
// fd 3 is a status/params sidechannel the render op writes `{status,params}` to
// (guarded on the child side): a 404 renders its flight AND exits 0, carrying its
// HTTP status only over fd 3 — so we must NOT infer failure from a non-zero exit for
// a 404 (the child never exits non-zero for one). Resolves `{flight,status,params}`.
// Render + action always route through the persistent worker pool (dev AND prod), so
// no request pays a Node cold start; other ops fall through to a one-shot spawn.
async function runReactServer(args, stdinBody, reqCtxOverride) {
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
    const msg = await nextWorker().call({
      op: "render",
      pathname: args[1],
      manifestPath: args[2],
      reqCtx,
    });
    return {
      flight: Buffer.from(msg.flight || "", "base64"),
      status: msg.status || 200,
      params: msg.params || {},
      redirect: msg.redirect,
      notFound: msg.notFound,
      // next/cache: the tag set this page read, so the prerender manifest / tagToPaths map
      // can register the pathname under those tags (revalidateTag → pathname).
      tags: msg.tags || [],
      // Set-Cookie strings a top-level cookies().set()/draftMode() write produced.
      setCookies: msg.setCookies || [],
    };
  }
  if (op === "action") {
    const msg = await nextWorker().call({
      op: "action",
      actionId: args[1],
      manifestPath: args[2],
      body: stdinBody != null ? String(stdinBody) : "",
      // The request context (url/headers/cookie) so the action's cookies()/headers()/
      // draftMode() reads resolve against the real request.
      reqCtx: reqCtxOverride || {},
    });
    return {
      flight: Buffer.from(msg.flight || "", "base64"),
      status: msg.status || 200,
      // next/cache invalidations the action requested (revalidatePath/revalidateTag).
      revalidated: msg.revalidated || { tags: [], paths: [] },
      // Set-Cookie strings the action wrote via cookies().set()/draftMode().
      setCookies: msg.setCookies || [],
    };
  }
  // Unknown op falls through to a one-shot spawn (defensive; not reached today).
  return new Promise((resolve, reject) => {
    // See `spawnWorker`: an emitted-chunk entry takes the flag.
    const child = spawn(process.execPath, ["--enable-source-maps", rscRenderEntry, ...args], {
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

// Static-asset response policy, matching what `next start` puts on a `public/` file:
//
//   Cache-Control: public, max-age=0
//   Last-Modified: <mtime>
//   ETag: W/"<size-hex>-<mtime-ms-hex>"          (the `send` package's format)
//
// `max-age=0` is NOT "do not cache" — it is "cache it, then revalidate", which is the
// only correct policy for diffpack's client build, whose URLs (`/client.js`,
// `/client.chunk-7.js`) are NOT content-hashed: a rebuild changes the bytes behind the
// same URL, so `immutable` would serve stale code. With a validator the browser keeps
// the body and each later navigation costs one conditional request answered `304` with
// no body. Without one (what this server used to send) EVERY navigation re-downloaded
// the whole client bundle — 8.3 MB per document on cal.com.
const STATIC_CACHE_CONTROL = "public, max-age=0";

function staticValidators(info) {
  const mtimeMs = Math.floor(info.mtimeMs);
  return {
    etag: `W/"${info.size.toString(16)}-${mtimeMs.toString(16)}"`,
    lastModified: new Date(mtimeMs).toUTCString(),
    mtimeMs,
  };
}

// Whether the request's validators still match the file, i.e. a `304` is owed.
// `If-None-Match` wins outright when present (per RFC 9110); otherwise
// `If-Modified-Since` is compared at second granularity, since that header carries none.
function staticIsFresh(req, validators) {
  const noneMatch = req.headers["if-none-match"];
  if (noneMatch) {
    return noneMatch
      .split(",")
      .some((candidate) => candidate.trim().replace(/^W\//, "") === validators.etag.replace(/^W\//, ""));
  }
  const modifiedSince = req.headers["if-modified-since"];
  if (!modifiedSince) return false;
  const since = Date.parse(modifiedSince);
  return Number.isFinite(since) && Math.floor(validators.mtimeMs / 1000) * 1000 <= since;
}

// Text-ish types worth gzipping. Images/fonts/wasm are already compressed, and gzipping
// them burns CPU for nothing — the same set `next start`'s compression middleware skips.
const COMPRESSIBLE = /^(text\/|application\/(javascript|json|manifest\+json)|image\/svg\+xml)/;

// Below this, a gzip frame's own overhead swamps the saving. Next's threshold too.
const COMPRESS_MIN_BYTES = 1024;

function acceptsGzip(req) {
  const accept = req.headers["accept-encoding"];
  return typeof accept === "string" && /(^|,)\s*gzip\s*(;|,|$)/.test(accept);
}

// Compressing the client bundle costs ~300 ms of CPU, and this orchestrator is ONE
// event loop serving every render — paying that per request (12 Playwright workers each
// opening a fresh browser) is far worse than the bytes it saves. A build output does not
// change while the server runs, so each asset is compressed at most once and the frame
// is kept, keyed by the same (size, mtime) pair the ETag is derived from: a rebuild
// changes the key and the stale frame is simply never read again.
const GZIP_CACHE = new Map();
// Enough for a whole client build's text assets; past it, later files stream through a
// per-request gzip rather than evicting (no asset is ever served WRONG, only slower).
const GZIP_CACHE_MAX_BYTES = 128 * 1024 * 1024;
let gzipCacheBytes = 0;

function cachedGzip(filePath, validators) {
  const hit = GZIP_CACHE.get(filePath);
  if (hit && hit.etag === validators.etag) return hit.body;
  return null;
}

function rememberGzip(filePath, validators, body) {
  const previous = GZIP_CACHE.get(filePath);
  if (previous) gzipCacheBytes -= previous.body.length;
  if (gzipCacheBytes + body.length > GZIP_CACHE_MAX_BYTES) {
    if (previous) GZIP_CACHE.delete(filePath);
    return;
  }
  GZIP_CACHE.set(filePath, { etag: validators.etag, body });
  gzipCacheBytes += body.length;
}

// Append Set-Cookie strings to a response-headers object WITHOUT clobbering any already
// there (e.g. middleware set-cookies). node's writeHead accepts an array for set-cookie,
// so we normalize to an array and concat. A render/action/route can add cookies via
// next/headers cookies().set() or draftMode(); those ride here alongside middleware ones.
function mergeSetCookie(headers, cookies) {
  if (!cookies || !cookies.length) return;
  const existing = headers["set-cookie"];
  const base = existing == null ? [] : Array.isArray(existing) ? existing.slice() : [existing];
  headers["set-cookie"] = base.concat(cookies);
}

// --- next.config redirects / rewrites / headers ----------------------------------
// Rules evaluated from `next.config.*` at build time (next-config-manifest.json).
// Compiled once here; applied per request after middleware, before route/render.
function compilePattern(source) {
  const keys = [];
  let re = "^";
  for (const part of source.split("/")) {
    if (part === "") continue;
    re += "/";
    const m = part.match(/^:([A-Za-z0-9_]+)([*+])?$/);
    if (m) {
      keys.push(m[1]);
      re += m[2] ? "(.*)" : "([^/]+)";
    } else {
      re += part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    }
  }
  re += "/?$";
  return { re: new RegExp(re), keys };
}
function matchCompiled(compiled, pathname) {
  const m = compiled.re.exec(pathname);
  if (!m) return null;
  const params = {};
  compiled.keys.forEach((k, i) => (params[k] = m[i + 1] || ""));
  return params;
}
function substitutePattern(dest, params) {
  return dest.replace(/:([A-Za-z0-9_]+)[*+]?/g, (_, k) => (params[k] != null ? params[k] : ""));
}
const nextConfig = (() => {
  const p = join(outputDir, "next-config-manifest.json");
  const cfg = existsSync(p)
    ? JSON.parse(readFileSync(p, "utf8"))
    : { redirects: [], rewrites: [], headers: [] };
  for (const list of ["redirects", "rewrites", "headers"]) {
    for (const rule of cfg[list] || []) rule.__compiled = compilePattern(rule.source || "");
  }
  return cfg;
})();

// --- next.config routing surface: basePath / assetPrefix / trailingSlash / i18n --------
// Extracted at build time into next-config-manifest.json; applied here as pure O(path)
// string normalization BEFORE the render pipeline, so every downstream matcher stays
// prefix/locale-agnostic. Zero new per-request node work; zero new deps.
const routing = {
  basePath: typeof nextConfig.basePath === "string" ? nextConfig.basePath : "",
  assetPrefix: typeof nextConfig.assetPrefix === "string" ? nextConfig.assetPrefix : "",
  trailingSlash: Boolean(nextConfig.trailingSlash),
  i18n: nextConfig.i18n || null,
};
if (routing.i18n) {
  // next.config `i18n` is a Pages-Router feature that app-router `next build` IGNORES.
  // diffpack honors it as an explicit opt-in to a locale-prefix routing EXTENSION (NOT
  // next-build behavior) — announced once, loudly, so it is never a silent divergence.
  console.warn(
    `next-server: next.config i18n is set (locales ${JSON.stringify(routing.i18n.locales)}, ` +
      `default ${JSON.stringify(routing.i18n.defaultLocale)}). This is a diffpack ` +
      `locale-routing EXTENSION; app-router \`next build\` does not consume next.config i18n.`,
  );
}

// stripPrefix / addBasePath: a lean reimplementation of Next's removePathPrefix +
// addPathPrefix (MIT, Copyright (c) Vercel), covering the exact-match and path-boundary
// cases a naive slice/concat would get wrong. `stripPrefix` returns the app-relative
// remainder, or null when `pathname` is not under `prefix` at all (`${prefix}` -> "/").
function stripPrefix(pathname, prefix) {
  if (!prefix) return pathname;
  if (pathname === prefix) return "/";
  if (pathname.startsWith(prefix + "/")) return pathname.slice(prefix.length);
  return null;
}
// Re-apply basePath (+ any active non-default locale segment) to an outgoing redirect
// Location so the browser round-trips through the same prefixed URL space. A non
// leading-slash Location (a full URL, e.g. a middleware redirect off-site) passes through.
function addBasePath(location, localeSeg) {
  if (typeof location !== "string" || !location.startsWith("/")) return location;
  const withLocale = localeSeg ? localeSeg + (location === "/" ? "" : location) : location;
  return routing.basePath + withLocale;
}
// Whether a path participates in trailingSlash normalization: not the root, and its last
// segment has no file extension (so `/client.js`, `/rsc.css`, favicons stay untouched).
function trailingEligible(pathname) {
  if (pathname === "/") return false;
  const last = pathname.slice(pathname.lastIndexOf("/") + 1);
  return !last.includes(".");
}

// --- i18n Accept-Language locale detection (diffpack locale-routing EXTENSION) --------
// Read one cookie value from a Cookie header (returns null when absent).
function readCookie(header, name) {
  if (typeof header !== "string" || !header) return null;
  for (const pair of header.split(";")) {
    const eq = pair.indexOf("=");
    if (eq === -1) continue;
    if (pair.slice(0, eq).trim() === name) return decodeURIComponent(pair.slice(eq + 1).trim());
  }
  return null;
}
// Parse an Accept-Language header into lowercased BCP-47 tags, highest quality first.
function parseAcceptLanguage(header) {
  if (typeof header !== "string" || !header) return [];
  return header
    .split(",")
    .map((part) => {
      const [tag, ...params] = part.trim().split(";");
      let q = 1;
      for (const p of params) {
        const m = /^q=([0-9.]+)$/.exec(p.trim());
        if (m) q = parseFloat(m[1]);
      }
      return { tag: tag.trim().toLowerCase(), q: Number.isFinite(q) ? q : 0 };
    })
    .filter((e) => e.tag && e.tag !== "*" && e.q > 0)
    .sort((a, b) => b.q - a.q)
    .map((e) => e.tag);
}
// The best configured locale for a request, mirroring Next's pages-router detection order:
// an explicit `NEXT_LOCALE` cookie (when it names a configured locale) wins; otherwise the
// highest-quality Accept-Language tag matching a configured locale (exact, then primary
// subtag). Falls back to the default locale when nothing matches.
function detectPreferredLocale(i18n, cookieHeader, acceptLanguage) {
  const locales = i18n.locales;
  const lower = locales.map((l) => l.toLowerCase());
  const cookieLocale = readCookie(cookieHeader, "NEXT_LOCALE");
  if (cookieLocale) {
    const idx = lower.indexOf(cookieLocale.toLowerCase());
    if (idx !== -1) return locales[idx];
  }
  for (const tag of parseAcceptLanguage(acceptLanguage)) {
    let idx = lower.indexOf(tag);
    if (idx === -1) {
      const primary = tag.split("-")[0];
      idx = lower.findIndex((l) => l === primary || l.split("-")[0] === primary);
    }
    if (idx !== -1) return locales[idx];
  }
  return i18n.defaultLocale;
}

// --- `has` / `missing` conditions on a redirect/rewrite/header rule -------------------
// A rule may be conditional on a request header, cookie, query parameter or host
// (`has`: all must match; `missing`: none may). Ignoring them does not make a rule
// inert — it makes it fire UNCONDITIONALLY, which is strictly worse than not
// supporting it: cal.com's `/api/auth/:path*` -> `/404` redirect is gated on a
// `callbackUrl` query, so every one of its auth API requests was redirected to /404.
//
// Semantics mirror Next's `matchHas` (next/dist/shared/lib/router/utils/prepare-
// destination.ts, MIT, Copyright (c) Vercel): a bare `{type,key}` matches when the
// value is merely PRESENT (and binds it as a param); with a `value` the whole value
// must match `^value$`, and any named capture groups become destination params.
// `host` compares the hostname with the port removed.
//
// Next keeps only ASCII letters in a param name (`getSafeParamName`), so the name is
// substitutable in a `:param` destination.
function safeParamName(name) {
  let out = "";
  for (const ch of String(name)) if (/[A-Za-z]/.test(ch)) out += ch;
  return out;
}
function hasItemValue(item, req, url) {
  switch (item.type) {
    case "header":
      return req.headers[String(item.key).toLowerCase()];
    case "cookie":
      return readCookie(req.headers.cookie, item.key) ?? undefined;
    case "query": {
      const all = url.searchParams.getAll(item.key);
      if (all.length === 0) return undefined;
      return all.length === 1 ? all[0] : all;
    }
    case "host": {
      const host = req.headers.host;
      return host ? String(host).split(":", 1)[0].toLowerCase() : undefined;
    }
    default:
      return undefined;
  }
}
// Returns the params a matching set of conditions contributes, or null when the rule
// must not apply. A rule with neither `has` nor `missing` always matches (no params).
function matchHas(rule, req, url) {
  const has = Array.isArray(rule.has) ? rule.has : [];
  const missing = Array.isArray(rule.missing) ? rule.missing : [];
  if (!has.length && !missing.length) return {};
  const params = {};
  const hasMatch = (item) => {
    const value = hasItemValue(item, req, url);
    if (!item.value && value) {
      params[safeParamName(item.key)] = Array.isArray(value) ? value[value.length - 1] : value;
      return true;
    }
    if (value) {
      const matcher = new RegExp(`^${item.value}$`);
      const target = Array.isArray(value) ? value[value.length - 1] : value;
      const matches = String(target).match(matcher);
      if (matches) {
        if (matches.groups) for (const k of Object.keys(matches.groups)) params[k] = matches.groups[k];
        else if (item.type === "host" && matches[0]) params.host = matches[0];
        return true;
      }
    }
    return false;
  };
  if (!has.every(hasMatch)) return null;
  if (missing.some(hasMatch)) return null;
  return params;
}

// Apply next.config redirects (short-circuit) + rewrites (mutate url) and COLLECT
// matching response headers. Returns { redirect } to short-circuit, or { headers }.
// `req` is needed for the `has`/`missing` conditions above.
function applyNextConfig(url, req) {
  for (const r of nextConfig.redirects) {
    const params = matchCompiled(r.__compiled, url.pathname);
    if (!params) continue;
    const conditionParams = matchHas(r, req, url);
    if (!conditionParams) continue;
    const all = { ...params, ...conditionParams };
    return {
      redirect: {
        status: r.permanent ? 308 : r.statusCode || 307,
        location: substitutePattern(r.destination, all),
      },
    };
  }
  for (const r of nextConfig.rewrites) {
    const params = matchCompiled(r.__compiled, url.pathname);
    if (!params) continue;
    const conditionParams = matchHas(r, req, url);
    if (!conditionParams) continue;
    url.pathname = substitutePattern(r.destination, { ...params, ...conditionParams });
    break;
  }
  const headers = [];
  for (const h of nextConfig.headers) {
    const params = matchCompiled(h.__compiled, url.pathname);
    if (!params) continue;
    const conditionParams = matchHas(h, req, url);
    if (!conditionParams) continue;
    const all = { ...params, ...conditionParams };
    for (const kv of h.headers || []) {
      headers.push([kv.key, substitutePattern(kv.value, all)]);
    }
  }
  return { headers };
}

// --- Route handlers (`route.ts` HTTP endpoints) ----------------------------------
// The orchestrator matches a request path against the handler routes (queried once
// from the worker at boot) and, on a match, dispatches the request to the react-server
// worker's `route` op instead of page rendering.
let __manifest = null;
async function getManifest() {
  if (__manifest === null) {
    try {
      const msg = await nextWorker().call({ op: "routes" });
      const routes = msg.routes || {};
      __manifest = {
        handlers: (routes.handlers || []).map((r) => ({ segments: r.segments, methods: new Set(r.methods) })),
        // A hybrid app's `pages/api/**` endpoints. They are dispatched through the same
        // `route` op as an app-router `route.ts`, so they only need their patterns here.
        pagesApi: (routes.pagesApi || []).map((r) => ({ segments: r.segments })),
        hasMiddleware: !!routes.hasMiddleware,
      };
    } catch {
      __manifest = { handlers: [], pagesApi: [], hasMiddleware: false };
    }
  }
  return __manifest;
}
// Segment matcher (mirrors the react-server entry's): returns captured params or null.
function matchHandlerSegments(segments, parts) {
  const params = {};
  let i = 0;
  for (const seg of segments) {
    if (seg.k === "static") {
      if (parts[i] !== seg.v) return null;
      i += 1;
    } else if (seg.k === "dynamic") {
      if (i >= parts.length) return null;
      params[seg.v] = decodeURIComponent(parts[i]);
      i += 1;
    } else if (seg.k === "catchall") {
      if (i >= parts.length) return null;
      params[seg.v] = parts.slice(i).map(decodeURIComponent);
      i = parts.length;
    } else if (seg.k === "optcatchall") {
      params[seg.v] = parts.slice(i).map(decodeURIComponent);
      i = parts.length;
    } else {
      return null;
    }
  }
  return i === parts.length ? params : null;
}
// Whether `filePath` is a REGULAR FILE that can be streamed as a static asset. A
// directory exists but cannot be read, and a broken symlink does not resolve — both must
// fall through to route rendering rather than being answered with a truncated 200.
function isStaticFile(filePath) {
  try {
    return statSync(filePath).isFile();
  } catch {
    return false;
  }
}

// Answer a static asset with the same cache contract `next start` gives a `public/`
// file: validators + `max-age=0` (so a repeat navigation revalidates into a bodiless
// `304` instead of re-downloading), gzip for compressible types, and a STREAM rather
// than `readFileSync` — this is the single-threaded orchestrator, and reading 8 MB
// synchronously per request stalls every concurrent render behind it.
function serveStaticAsset(req, res, filePath) {
  let info;
  try {
    info = statSync(filePath);
  } catch {
    res.writeHead(404);
    res.end();
    return;
  }
  const validators = staticValidators(info);
  const type = MIME[extname(filePath)] || "application/octet-stream";
  if (staticIsFresh(req, validators)) {
    res.writeHead(304, {
      "cache-control": STATIC_CACHE_CONTROL,
      "last-modified": validators.lastModified,
      etag: validators.etag,
    });
    res.end();
    return;
  }
  const headers = {
    "content-type": type,
    "cache-control": STATIC_CACHE_CONTROL,
    "last-modified": validators.lastModified,
    etag: validators.etag,
  };
  const compressible = COMPRESSIBLE.test(type) && info.size >= COMPRESS_MIN_BYTES;
  if (compressible) headers.vary = "Accept-Encoding";
  const gzip = compressible && acceptsGzip(req);
  if (!gzip) {
    headers["content-length"] = info.size;
    res.writeHead(200, headers);
    pipeline(createReadStream(filePath), res, ignoreAbortedTransfer);
    return;
  }
  headers["content-encoding"] = "gzip";
  const cached = cachedGzip(filePath, validators);
  if (cached) {
    headers["content-length"] = cached.length;
    res.writeHead(200, headers);
    res.end(cached);
    return;
  }
  // First request for this asset: compress once, answer from the frame, and keep it so
  // no later request pays the CPU again. A gzip frame's length is not known before it
  // exists, so this one response is chunked; every later one declares a length.
  res.writeHead(200, headers);
  const frames = [];
  const compressor = createGzip();
  compressor.on("data", (chunk) => frames.push(Buffer.from(chunk)));
  compressor.on("end", () => rememberGzip(filePath, validators, Buffer.concat(frames)));
  pipeline(createReadStream(filePath), compressor, res, ignoreAbortedTransfer);
}

// A client that navigated away mid-transfer aborts the socket; that is not a server
// fault and must not take the process down.
function ignoreAbortedTransfer() {}

// Which KIND of endpoint owns this path, or null when none does. The two kinds are
// served by DIFFERENT GRAPHS, which is why this cannot collapse to a boolean:
//   "handler"  -> an `app/**/route.ts`, react-server layer, dispatched to the worker.
//   "pages-api" -> a `pages/api/**` module, Next's `api-node` layer (no `react-server`
//                  export condition), bundled into the SSR graph and called in THIS
//                  process. Getting this wrong resolves `react-dom/server` to React's
//                  "not supported in React Server Components" stub — see the header of
//                  src/next_runtime/pages_api.js.
// `app/**/route.ts` wins a path both could answer, which is Next's own precedence.
async function routeEndpointKind(pathname) {
  const parts = pathname.split("/").filter(Boolean);
  const manifest = await getManifest();
  for (const h of manifest.handlers) {
    if (matchHandlerSegments(h.segments, parts)) return "handler";
  }
  for (const h of manifest.pagesApi) {
    if (matchHandlerSegments(h.segments, parts)) return "pages-api";
  }
  return null;
}

// Run middleware for a request and interpret its NextResponse (via Next's
// `x-middleware-*` protocol). Returns an action for the request handler:
//   { kind: "response", status, headers, body }  -> send it, short-circuit
//   { kind: "redirect", status, location, setCookies }
//   { kind: "rewrite", pathname, setCookies, requestHeaders }
//   { kind: "next", setCookies, requestHeaders }  -> continue to route/render
//   null -> no middleware / not applicable
async function runMiddleware(reqCtx) {
  const msg = await nextWorker().call({ op: "middleware", reqCtx });
  const mw = msg.middlewareResult;
  if (!mw) return null;
  const headers = mw.headers || [];
  const setCookies = headers.filter(([k]) => k.toLowerCase() === "set-cookie").map(([, v]) => v);
  const get = (name) => {
    const hit = headers.find(([k]) => k.toLowerCase() === name);
    return hit ? hit[1] : undefined;
  };
  // Request-header overrides (NextResponse.next({ request: { headers } })).
  const requestHeaders = [];
  const overrides = get("x-middleware-override-headers");
  if (overrides) {
    for (const name of overrides.split(",").map((s) => s.trim()).filter(Boolean)) {
      const v = get("x-middleware-request-" + name);
      if (v !== undefined) requestHeaders.push([name, v]);
    }
  }
  const location = get("location");
  if (location && mw.status >= 300 && mw.status < 400) {
    return { kind: "redirect", status: mw.status, location, setCookies };
  }
  const rewrite = get("x-middleware-rewrite");
  if (rewrite) {
    return { kind: "rewrite", pathname: new URL(rewrite, "http://localhost").pathname, setCookies, requestHeaders };
  }
  if (get("x-middleware-next")) {
    return { kind: "next", setCookies, requestHeaders };
  }
  // A plain Response (e.g. NextResponse.json(...) / new Response(...)): short-circuit.
  return {
    kind: "response",
    status: mw.status || 200,
    headers: headers.filter(([k]) => !k.toLowerCase().startsWith("x-middleware")),
    body: mw.body,
  };
}

// --- Prerender cache (static / SSG / ISR) ---------------------------------------
// `build-app production` prerenders static/SSG/ISR routes to `static/<stem>.html` +
// `.rsc` and records them in `prerender-manifest.json`. We serve those straight off
// disk — no per-request render. An ISR entry (revalidate = N seconds) is served from
// cache too, but once the cached file is older than N seconds the next request gets
// the STALE copy immediately AND kicks a background regeneration (stale-while-
// revalidate), so no request ever waits on the render.
const staticDir = join(outputDir, "static");
const prerenderCache = new Map(); // pathname -> { path, file, revalidate|null, tags:[], params:{} }
// next/cache on-demand revalidation. `tagToPaths` maps a cache tag to the set of cached
// pathnames that read it (captured per page at prerender time — no per-request work).
// `forcedStale` holds pathnames an action / route handler invalidated (revalidatePath /
// revalidateTag); the next request to such a path serves the stale copy and kicks the
// existing background regen (stale-while-revalidate), exactly like an expired ISR entry.
const tagToPaths = new Map(); // tag -> Set<pathname>
const forcedStale = new Set(); // pathname
function registerTags(pathname, tags) {
  for (const tag of tags || []) {
    let set = tagToPaths.get(tag);
    if (!set) {
      set = new Set();
      tagToPaths.set(tag, set);
    }
    set.add(pathname);
  }
}
(() => {
  const manifestPath = join(staticDir, "prerender-manifest.json");
  if (!existsSync(manifestPath)) return;
  let manifest;
  try {
    manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  } catch {
    return; // a malformed manifest is non-fatal — fall back to dynamic rendering.
  }
  for (const e of manifest.entries || []) {
    if (e && e.path && e.file) {
      const tags = Array.isArray(e.tags) ? e.tags : [];
      prerenderCache.set(e.path, { path: e.path, file: e.file, revalidate: e.revalidate ?? null, tags, params: e.params || {} });
      registerTags(e.path, tags);
    }
  }
  if (prerenderCache.size) {
    const isr = [...prerenderCache.values()].filter((e) => e.revalidate != null).length;
    console.log(`next-server: ${prerenderCache.size} prerendered page(s) cached (${isr} ISR) from ${manifestPath}`);
  }
})();

// Apply a batch of on-demand invalidations (from a Server Action or Route Handler that
// called next/cache revalidatePath/revalidateTag). Paths are `<type>:<pathname>`: a `page`
// invalidates that exact cached pathname; a `layout` (or a dynamic route path) invalidates
// every cached pathname at or under that prefix. Tags invalidate every pathname registered
// under the tag. Marking a pathname forcedStale is all that is needed — the next request
// serves stale + triggers the existing background regen. Returns the count invalidated.
function applyRevalidation(revalidated) {
  if (!revalidated) return 0;
  let count = 0;
  const markStale = (pathname) => {
    if (prerenderCache.has(pathname) && !forcedStale.has(pathname)) {
      forcedStale.add(pathname);
      count++;
    }
  };
  for (const raw of revalidated.paths || []) {
    const idx = raw.indexOf(":");
    const type = idx === -1 ? "page" : raw.slice(0, idx);
    const pathname = idx === -1 ? raw : raw.slice(idx + 1);
    if (type === "layout") {
      // A layout (or dynamic route) path invalidates its whole subtree.
      const prefix = pathname === "/" ? "/" : pathname + "/";
      for (const p of prerenderCache.keys()) {
        if (p === pathname || p.startsWith(prefix)) markStale(p);
      }
    } else {
      // A page path: exact match, OR (for a dynamic route path like /blog/[slug]) any
      // concrete cached child. A literal cached pathname just matches exactly.
      if (prerenderCache.has(pathname)) {
        markStale(pathname);
      } else if (pathname.includes("[")) {
        const prefix = pathname.slice(0, pathname.indexOf("["));
        for (const p of prerenderCache.keys()) {
          if (p.startsWith(prefix)) markStale(p);
        }
      }
    }
  }
  for (const tag of revalidated.tags || []) {
    const set = tagToPaths.get(tag);
    if (set) for (const p of set) markStale(p);
  }
  if (count) console.log(`next-server: on-demand revalidation marked ${count} cached page(s) stale`);
  return count;
}

// Background ISR regeneration guard — one in-flight regen per page stem.
const revalidating = new Set();
function triggerRevalidate(entry) {
  if (revalidating.has(entry.file)) return;
  revalidating.add(entry.file);
  (async () => {
    try {
      const reqCtx = JSON.stringify({ url: "http://localhost" + entry.path, headers: [], cookie: "" });
      const r = await runReactServer(["render", entry.path, clientManifestPath], reqCtx);
      // Never overwrite a good cache entry with an error tree.
      if (r.redirect || r.notFound) return;
      const flightBuf = Buffer.from(r.flight);
      const html = await (await getRenderFlightToDocument())(
        new Uint8Array(flightBuf),
        serverConsumerManifest,
        flightBuf.toString("base64"),
        r.params || {},
        { pathname: entry.path, search: "" },
      );
      // Write via a temp file + rename so a concurrent reader never sees a half-written
      // page (rename is atomic on the same filesystem).
      const htmlPath = join(staticDir, `${entry.file}.html`);
      const rscPath = join(staticDir, `${entry.file}.rsc`);
      writeFileSync(`${htmlPath}.tmp`, html);
      renameSync(`${htmlPath}.tmp`, htmlPath);
      writeFileSync(`${rscPath}.tmp`, flightBuf);
      renameSync(`${rscPath}.tmp`, rscPath);
      // next/cache: the regenerated page's tag set can differ from last time, so refresh
      // tagToPaths for this pathname (drop the stale registrations, add the fresh ones),
      // then clear the forcedStale flag — the fresh file is now on disk, so the next
      // request is a HIT again (until it expires or is invalidated anew).
      const freshTags = Array.isArray(r.tags) ? r.tags : [];
      for (const set of tagToPaths.values()) set.delete(entry.path);
      entry.tags = freshTags;
      registerTags(entry.path, freshTags);
      forcedStale.delete(entry.path);
    } catch (error) {
      console.error(`[diffpack] ISR revalidate of ${entry.path} failed:`, error && error.message ? error.message : error);
    } finally {
      revalidating.delete(entry.file);
    }
  })();
}

// Serve a prerendered page from disk (HTML, or the raw .rsc flight for a soft-nav
// `?__rsc=1` request). Applies config headers + middleware set-cookies. For an ISR
// entry past its TTL, serves the stale copy now and schedules a background regen.
// Returns true if served, false if the file is missing (fall through to a render).
function servePrerendered(entry, isRsc, res, configHeaders, mwSetCookies) {
  const filePath = join(staticDir, `${entry.file}.${isRsc ? "rsc" : "html"}`);
  let stat;
  try {
    stat = statSync(filePath);
  } catch {
    return false;
  }
  let cacheState = "HIT";
  // next/cache on-demand: an action / route handler marked this pathname forcedStale
  // (revalidatePath / revalidateTag). Serve the stale copy now and kick a background regen
  // — identical machinery to an expired ISR entry, so no request ever blocks.
  if (forcedStale.has(entry.path)) {
    cacheState = "STALE";
    triggerRevalidate(entry);
  } else if (entry.revalidate != null && Date.now() - stat.mtimeMs >= entry.revalidate * 1000) {
    cacheState = "STALE";
    triggerRevalidate(entry);
  }
  const headers = {
    "content-type": isRsc ? "text/x-component" : "text/html; charset=utf-8",
    "x-diffpack-cache": cacheState,
  };
  // A soft navigation to a PRERENDERED route is answered from `.rsc` with no render, so
  // the matched params have to come off the manifest entry the prerender recorded them
  // on — the client Router reads this header to update `useParams()`.
  if (isRsc) headers["x-diffpack-params"] = encodeURIComponent(JSON.stringify(entry.params || {}));
  for (const [k, v] of configHeaders) headers[k] = v;
  if (mwSetCookies.length) headers["set-cookie"] = mwSetCookies;
  res.writeHead(200, headers);
  res.end(readFileSync(filePath));
  return true;
}

// --- Runtime next/image optimizer (`/_next/image`) --------------------------------
// Next's default loader points EVERY optimizable image here, so the emitted HTML is
// byte-faithful to Next. Cost is kept off the static-page path a different way: a
// request whose (src, width) the BUILD already precomputed is answered straight from
// the emitted variant file (see `buildVariantFile`) with no decode and no spawn.
//
// For everything else — a REMOTE src, or a width/quality the build did not precompute —
// the actual resize/re-encode is done NATIVELY by shelling to the diffpack binary
// (`optimize-image`, the same `image` crate as the build-time variants) — no Node
// image dependency. Optimized bytes are cached on disk (keyed by src+w+q+format) so a
// repeated request never re-optimizes or re-spawns.
const imagesConfig = (() => {
  const img = (nextConfig && nextConfig.images) || {};
  const deviceSizes = Array.isArray(img.deviceSizes) && img.deviceSizes.length
    ? img.deviceSizes
    : [640, 750, 828, 1080, 1200, 1920, 2048, 3840];
  const imageSizes = Array.isArray(img.imageSizes) && img.imageSizes.length
    ? img.imageSizes
    : [16, 32, 48, 64, 96, 128, 256, 384];
  return {
    deviceSizes,
    imageSizes,
    allSizes: [...imageSizes, ...deviceSizes].sort((a, b) => a - b),
    remotePatterns: Array.isArray(img.remotePatterns) ? img.remotePatterns : [],
    domains: Array.isArray(img.domains) ? img.domains : [],
    qualities: Array.isArray(img.qualities) && img.qualities.length ? img.qualities : null,
    minimumCacheTTL: typeof img.minimumCacheTTL === "number" ? img.minimumCacheTTL : 60,
    dangerouslyAllowSVG: Boolean(img.dangerouslyAllowSVG),
  };
})();
const imageCacheDir = join(outputDir, ".diffpack-image-cache");
// Port of Next's remote-pattern matcher (protocol/hostname/port/pathname wildcards +
// legacy `domains`). A remote src is allowed only if it matches — otherwise 400 (never
// a silent fetch of an unconfigured host).
function imageWildcardMatch(pattern, value) {
  if (!pattern) return true;
  const rx = "^" + pattern
    .replace(/[.+?^${}()|[\]\\]/g, "\\$&")
    .replace(/\*\*/g, " ")
    .replace(/\*/g, "[^.\\/]*")
    .replace(/ /g, ".*") + "$";
  return new RegExp(rx).test(value);
}
function remoteHostAllowed(u) {
  if (imagesConfig.domains.includes(u.hostname)) return true;
  return imagesConfig.remotePatterns.some((p) => {
    if (p.protocol && p.protocol.replace(/:$/, "") !== u.protocol.replace(/:$/, "")) return false;
    if (p.hostname && !imageWildcardMatch(p.hostname, u.hostname)) return false;
    if (p.port && p.port !== u.port) return false;
    if (p.pathname && !imageWildcardMatch(p.pathname, u.pathname)) return false;
    return true;
  });
}
const IMAGE_EXT_MIME = {
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  gif: "image/gif",
  webp: "image/webp",
  avif: "image/avif",
  svg: "image/svg+xml",
};

// --- Build-emitted variants: the pre-optimized answer to a `/_next/image` request ----
// The emitted HTML uses Next's optimizer URL shape for every optimizable image (there
// is no "prefer a build-time file" branch in Next, so the shim has none either). The
// PIXELS are still computed at build time: `diffpack build-app` writes a responsive
// ladder for every `/public` raster (`_diffpack-image/`, indexed by the manifest below)
// and for every static image import (`/assets/<name>-<hash>-<w>.<ext>`, named by
// convention next to the original). This lookup answers a request straight from those
// files, so a prerendered page costs ZERO runtime re-encodes; only a width/quality the
// build did not precompute reaches the native optimizer.
//
// Build variants are encoded at the default quality (75), so a request asking for a
// different `q` deliberately misses and gets a real re-encode.
const IMAGE_BUILD_QUALITY = 75;
const variantManifest = (() => {
  const path = join(publicDir, "_diffpack-image", "variants.json");
  if (!existsSync(path)) return {};
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch (error) {
    // A corrupt manifest is a build defect, not something to paper over: say so loudly
    // and fall back to real optimization (correct output, just slower).
    console.error(`next/image: cannot read the build variant manifest ${path}: ${error.message}`);
    return {};
  }
})();
// The static-import variant ladders, read off disk ONCE. Static-import assets are
// emitted by the client build (not the next adapter), so they are not in the manifest
// above; their ladder is recovered from the emitted file names, `<stem>-<w><ext>` next
// to `<stem><ext>`. A single readdir at boot beats a per-request stat sweep.
const assetVariantLadders = (() => {
  const index = new Map();
  const dir = join(publicDir, "assets");
  if (!existsSync(dir)) return index;
  for (const name of readdirSync(dir)) {
    const m = /^(.*)-(\d+)(\.(?:png|jpe?g))$/i.exec(name);
    if (!m) continue;
    const key = m[1] + m[3];
    const widths = index.get(key) || [];
    widths.push(Number(m[2]));
    index.set(key, widths);
  }
  for (const widths of index.values()) widths.sort((a, b) => a - b);
  return index;
})();
// Resolve `src` at width `w` to a build-emitted variant file, or null when the build
// did not precompute it. Never invents a file: every branch checks existsSync.
function buildVariantFile(src, w, q) {
  if (q !== IMAGE_BUILD_QUALITY) return null;
  const entry = variantManifest[src];
  if (entry) {
    // A width at or above the intrinsic width resolves to the intrinsic-width variant:
    // the optimizer never upscales, so those are the exact bytes it would produce.
    const url = entry.widths[String(w)] || (w >= entry.width ? entry.widths[String(entry.width)] : null);
    if (url) {
      const file = join(publicDir, url.replace(/^\/+/, ""));
      if (file.startsWith(publicDir) && existsSync(file)) return file;
    }
    return null;
  }
  // Static image imports live under the build's own hashed asset dir, where the variant
  // name is `<original stem>-<w><ext>`. Scoped to `/assets/` precisely because the app
  // cannot put files there — a `/public` file literally named `hero-640.png` must never
  // be mistaken for a variant of `hero.png`.
  const m = /^\/assets\/([^/]+)(\.(?:png|jpe?g))$/i.exec(src);
  if (!m) return null;
  const ladder = assetVariantLadders.get(m[1] + m[2]);
  if (!ladder) return null;
  // The ladder's top entry IS the image's intrinsic width (the build emits every
  // standard size below the intrinsic, then the intrinsic itself), so a request at or
  // above it resolves there: the optimizer never upscales, making that variant the
  // exact bytes it would have produced.
  const top = ladder[ladder.length - 1];
  const pick = ladder.includes(w) ? w : w >= top ? top : null;
  if (pick === null) return null;
  const file = join(publicDir, "assets", `${m[1]}-${pick}${m[2]}`);
  if (file.startsWith(publicDir) && existsSync(file)) return file;
  return null;
}

// Run the native optimizer (diffpack `optimize-image`) over `input`, returning the
// optimized bytes. A missing DIFFPACK_BIN, a non-zero exit, or a spawn failure is a
// hard rejection carrying the cause — never a silent passthrough of the un-optimized
// image (which would defeat the point and hide a misconfiguration).
function runNativeOptimizer(input, width, quality, format) {
  const bin = process.env.DIFFPACK_BIN;
  if (!bin) {
    return Promise.reject(
      new Error(
        "next/image optimizer: DIFFPACK_BIN is not set, so the native resize cannot run. " +
          "Start the app via `diffpack start` / `diffpack dev` (both pass it).",
      ),
    );
  }
  return new Promise((resolve, reject) => {
    const child = spawn(
      bin,
      ["optimize-image", "--width", String(width), "--quality", String(quality), "--format", format],
      { stdio: ["pipe", "pipe", "pipe"] },
    );
    const out = [];
    const err = [];
    child.stdout.on("data", (c) => out.push(c));
    child.stderr.on("data", (c) => err.push(c));
    child.on("error", (e) => reject(new Error(`next/image optimizer: cannot spawn ${bin}: ${e.message}`)));
    child.on("close", (code) => {
      if (code === 0) resolve(Buffer.concat(out));
      else reject(new Error(`next/image optimizer exited ${code}: ${Buffer.concat(err).toString().trim()}`));
    });
    child.stdin.on("error", () => {}); // EPIPE if the child died early; surfaced via `close`.
    child.stdin.end(input);
  });
}
// Serve `GET /_next/image?url=&w=&q=`. Validates params exactly like Next (bad `url`/
// `w`/`q` -> 400), fetches the source (a local file under public/, or an allow-listed
// remote), runs the native optimizer, caches the result on disk, and responds with a
// long-lived cache header. SVG/GIF are passed through un-optimized (Next does the same
// unless `dangerouslyAllowSVG`); an animated/undecodable payload surfaces the
// optimizer's error as a 500 rather than a wrong image.
async function serveOptimizedImage(url, res) {
  const src = url.searchParams.get("url");
  const wRaw = url.searchParams.get("w");
  const qRaw = url.searchParams.get("q");
  if (!src) return void res.writeHead(400, { "content-type": "text/plain" }).end('"url" parameter is required');
  const w = Number(wRaw);
  if (!wRaw || !Number.isInteger(w) || w <= 0 || !imagesConfig.allSizes.includes(w)) {
    return void res.writeHead(400, { "content-type": "text/plain" }).end('"w" parameter (width) is not allowed');
  }
  const q = qRaw == null ? 75 : Number(qRaw);
  if (!Number.isInteger(q) || q < 1 || q > 100 || (imagesConfig.qualities && !imagesConfig.qualities.includes(q))) {
    return void res.writeHead(400, { "content-type": "text/plain" }).end('"q" parameter (quality) is not allowed');
  }

  const isRemote = /^https?:\/\//i.test(src);
  // Build-emitted variant fast path: the exact bytes the optimizer would produce, already
  // on disk. No file read of the original, no spawn, no cache write.
  if (!isRemote && src.startsWith("/")) {
    const variant = buildVariantFile(src.split("?")[0], w, q);
    if (variant) {
      const vext = extname(variant).slice(1).toLowerCase();
      return void res
        .writeHead(200, {
          "content-type": IMAGE_EXT_MIME[vext] || "application/octet-stream",
          "cache-control": `public, max-age=${imagesConfig.minimumCacheTTL}, must-revalidate`,
          "content-disposition": "inline",
          "x-diffpack-image-cache": "BUILD",
        })
        .end(readFileSync(variant));
    }
  }
  // Resolve the source bytes + extension.
  let bytes;
  let ext;
  try {
    if (isRemote) {
      const remoteUrl = new URL(src);
      if (!remoteHostAllowed(remoteUrl)) {
        return void res
          .writeHead(400, { "content-type": "text/plain" })
          .end(`"url" parameter is not allowed: host '${remoteUrl.hostname}' is not configured under images`);
      }
      const resp = await fetch(src);
      if (!resp.ok) {
        return void res.writeHead(resp.status === 404 ? 404 : 502, { "content-type": "text/plain" }).end("upstream image error");
      }
      bytes = Buffer.from(await resp.arrayBuffer());
      const ct = (resp.headers.get("content-type") || "").split(";")[0].trim();
      ext = Object.keys(IMAGE_EXT_MIME).find((k) => IMAGE_EXT_MIME[k] === ct) || extname(new URL(src).pathname).slice(1).toLowerCase() || "jpeg";
    } else {
      if (!src.startsWith("/")) {
        return void res.writeHead(400, { "content-type": "text/plain" }).end('"url" parameter must be an absolute path or an allow-listed URL');
      }
      const rel = src.split("?")[0].replace(/^\/+/, "");
      const filePath = join(publicDir, rel);
      // Path-traversal guard: the resolved file must stay inside public/.
      if (!filePath.startsWith(publicDir) || !isStaticFile(filePath)) {
        return void res.writeHead(404, { "content-type": "text/plain" }).end("image not found");
      }
      bytes = readFileSync(filePath);
      ext = extname(filePath).slice(1).toLowerCase();
    }
  } catch (e) {
    return void res.writeHead(500, { "content-type": "text/plain" }).end(`next/image: ${e.message}`);
  }

  // SVG/GIF: pass through un-optimized (byte-faithful to Next, which skips them unless
  // dangerouslyAllowSVG). SVG additionally needs the opt-in + a hardened CSP.
  if (ext === "svg" || ext === "gif") {
    if (ext === "svg" && !imagesConfig.dangerouslyAllowSVG) {
      return void res.writeHead(400, { "content-type": "text/plain" }).end('"url" parameter (svg) requires images.dangerouslyAllowSVG');
    }
    const headers = {
      "content-type": IMAGE_EXT_MIME[ext],
      "cache-control": `public, max-age=${imagesConfig.minimumCacheTTL}, must-revalidate`,
    };
    if (ext === "svg") headers["content-security-policy"] = "script-src 'none'; frame-src 'none'; sandbox;";
    return void res.writeHead(200, headers).end(bytes);
  }

  // Output format: PNG preserves alpha, everything else re-encodes to JPEG. (webp/avif
  // encode is deliberately not pulled in — it would add a heavy dep; the resize/recompress
  // is the honest optimization here.)
  const outFormat = ext === "png" ? "png" : "jpeg";
  const outMime = outFormat === "png" ? "image/png" : "image/jpeg";
  const key = createHash("sha1").update(`${src}|${w}|${q}|${outFormat}`).digest("hex");
  const cachePath = join(imageCacheDir, `${key}.${outFormat}`);
  const cacheHeaders = () => ({
    "content-type": outMime,
    "cache-control": `public, max-age=${imagesConfig.minimumCacheTTL}, must-revalidate`,
    "content-disposition": "inline",
  });
  if (existsSync(cachePath)) {
    const h = cacheHeaders();
    h["x-diffpack-image-cache"] = "HIT";
    return void res.writeHead(200, h).end(readFileSync(cachePath));
  }
  let optimized;
  try {
    optimized = await runNativeOptimizer(bytes, w, q, outFormat);
  } catch (e) {
    return void res.writeHead(500, { "content-type": "text/plain" }).end(`next/image: ${e.message}`);
  }
  try {
    mkdirSync(imageCacheDir, { recursive: true });
    // Write via temp + rename so a concurrent reader never sees a half-written file.
    const tmp = `${cachePath}.${process.pid}.tmp`;
    writeFileSync(tmp, optimized);
    renameSync(tmp, cachePath);
  } catch {
    // A cache-write failure is non-fatal: still serve the freshly optimized bytes.
  }
  const h = cacheHeaders();
  h["x-diffpack-image-cache"] = "MISS";
  res.writeHead(200, h).end(optimized);
}

const server = createServer(async (req, res) => {
  // A peer that resets the connection mid-stream emits `error` on the request and the
  // response. An `error` event with no listener THROWS, which — with a streaming SSR
  // response in flight — would take the server down for a client-side disconnect.
  req.on("error", (error) => {
    console.error("next-server: request socket error:", error && error.message ? error.message : error);
  });
  res.on("error", (error) => {
    console.error("next-server: response socket error:", error && error.message ? error.message : error);
  });
  try {
    const url = new URL(req.url, "http://localhost");
    // DEV hot update: the dev server's chunk-granular invalidation channel. Applied
    // BEFORE the reply, so by the time `diffpack dev` pushes the browser update both
    // server graphs in this process tree already run the new code — the next document
    // request cannot observe the old one. Any failure is a 500 with the reason; the dev
    // server surfaces it in the error overlay rather than silently serving stale HTML.
    if (req.method === "POST" && url.pathname === "/__diffpack_dev/hot") {
      if (!DEV) {
        res.writeHead(404).end("hot updates are dev-only");
        return;
      }
      const body = [];
      for await (const chunk of req) body.push(Buffer.from(chunk));
      let applied = { ssr: 0, reactServer: 0 };
      try {
        const msg = JSON.parse(Buffer.concat(body).toString("utf8") || "{}");
        // The dev server owns the truth about what disk still lacks; it re-sends the
        // whole list every push, so this is a replace, not an append.
        pendingHot = Array.isArray(msg.pendingReactServer) ? msg.pendingReactServer : [];
        if (msg.ssr) {
          await applySsrHotUpdate(msg.ssr);
          applied.ssr = msg.ssr.ids.length;
        }
        if (msg.reactServer) {
          const reply = await nextWorker().call({
            op: "invalidate",
            chunk: msg.reactServer.chunk,
            ids: msg.reactServer.ids,
          });
          applied.reactServer = reply.invalidated || 0;
        }
      } catch (error) {
        res
          .writeHead(500, { "content-type": "text/plain" })
          .end(error && error.stack ? error.stack : String(error));
        return;
      }
      res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify({ ok: true, applied }));
      return;
    }
    // Server actions.
    if (req.method === "POST" && url.pathname === "/_action/") {
      const id = req.headers["x-diffpack-action-id"];
      if (!id) {
        res.writeHead(400).end("missing x-diffpack-action-id");
        return;
      }
      const body = [];
      for await (const chunk of req) body.push(Buffer.from(chunk));
      // The action's request context so cookies()/headers()/draftMode() reads resolve.
      const actionReqCtx = {
        url: "http://localhost" + req.url,
        method: req.method,
        headers: Object.entries(req.headers).map(([k, v]) => [k, Array.isArray(v) ? v.join(", ") : String(v)]),
        cookie: req.headers.cookie || "",
      };
      const { flight, revalidated, setCookies } = await runReactServer(
        ["action", id, clientManifestPath],
        Buffer.concat(body),
        actionReqCtx,
      );
      // next/cache: bust any prerendered pages the action invalidated (revalidatePath /
      // revalidateTag) so the next request to them re-renders in the background.
      applyRevalidation(revalidated);
      const actionHeaders = { "content-type": "text/x-component" };
      // Server-side cookie writes (cookies().set()/delete(), draftMode().enable()/disable())
      // the action collected — delivered on the action's 200 response.
      mergeSetCookie(actionHeaders, setCookies);
      res.writeHead(200, actionHeaders);
      res.end(flight);
      return;
    }
    // --- next.config routing normalization (basePath / assetPrefix / trailingSlash / i18n)
    // Strip the configured prefixes up front so the whole pipeline below (static-serve,
    // middleware, next.config, route handlers, prerender cache, render) stays prefix- and
    // locale-agnostic; the prefix (+ locale) is re-applied to every redirect Location. The
    // `/_action/` POST above is intentionally handled first (its transport is a fixed,
    // unprefixed endpoint), so it never hits the basePath gate.
    let localeSeg = ""; // e.g. "/fr" when i18n peeled a non-default locale from the path
    let reqLocale = routing.i18n ? routing.i18n.defaultLocale : undefined;
    {
      // (1) relative assetPrefix (assets are baked as `${assetPrefix}${basePath}/x`; a
      //     full-URL/CDN assetPrefix never reaches this server, so nothing to strip).
      if (routing.assetPrefix && routing.assetPrefix.startsWith("/")) {
        const s = stripPrefix(url.pathname, routing.assetPrefix);
        if (s !== null) url.pathname = s;
      }
      // (2) basePath gate + strip: a non-prefixed page/asset path is a hard 404 (matching
      //     Next), a matching one is stripped to app-relative for every matcher below.
      if (routing.basePath) {
        const stripped = stripPrefix(url.pathname, routing.basePath);
        if (stripped === null) {
          res.writeHead(404, { "content-type": "text/plain" }).end("not found");
          return;
        }
        url.pathname = stripped;
      }
      // (3) i18n locale detection (diffpack EXTENSION). Peel a leading `/{locale}` segment
      //     that is one of the configured locales; the default locale is served UNPREFIXED
      //     (Pages-Router convention), so a bare path implies the default. `reqLocale` is
      //     handed to the render/route ops; a NON-default locale is re-added to redirects.
      if (routing.i18n) {
        const seg = url.pathname.split("/")[1] || "";
        if (routing.i18n.locales.includes(seg)) {
          reqLocale = seg;
          if (seg !== routing.i18n.defaultLocale) localeSeg = "/" + seg;
          url.pathname = url.pathname.slice(seg.length + 1) || "/";
        } else if (
          routing.i18n.localeDetection &&
          url.pathname === "/" &&
          !url.searchParams.has("__rsc")
        ) {
          // Root visit with no locale prefix: detect the visitor's preferred locale
          // (NEXT_LOCALE cookie, then Accept-Language). A NON-default detected locale gets a
          // 307 to `/{locale}` (Next's pages-router behavior — detection redirects only from
          // `/`); the default stays unprefixed. The Location carries basePath so the browser
          // round-trips the same URL space.
          const preferred = detectPreferredLocale(
            routing.i18n,
            req.headers.cookie || "",
            req.headers["accept-language"],
          );
          if (preferred !== routing.i18n.defaultLocale) {
            res.writeHead(307, {
              location: routing.basePath + "/" + preferred + url.search,
            });
            res.end();
            return;
          }
        }
      }
      // (4) trailingSlash: a 308 to the canonical slash form (query preserved, assets +
      //     root exempt), Location carrying basePath (+ locale). Internal `?__rsc=1` flight
      //     fetches are exempt (the client Router owns the displayed URL, not the fetch).
      if (!url.searchParams.has("__rsc") && trailingEligible(url.pathname)) {
        const has = url.pathname.endsWith("/");
        if (routing.trailingSlash && !has) {
          res.writeHead(308, { location: addBasePath(url.pathname + "/", localeSeg) + url.search });
          res.end();
          return;
        }
        if (!routing.trailingSlash && has) {
          res.writeHead(308, {
            location: addBasePath(url.pathname.replace(/\/+$/, ""), localeSeg) + url.search,
          });
          res.end();
          return;
        }
      }
    }
    // Runtime next/image optimizer (dynamic/remote fallback). Checked here — AFTER the
    // basePath/assetPrefix normalization (so a `${basePath}/_next/image` request has been
    // stripped to `/_next/image`) and BEFORE static serving / rendering. Static pages
    // reference build-time variants under /_diffpack-image/, so they never reach this.
    if (req.method === "GET" && url.pathname === "/_next/image") {
      await serveOptimizedImage(url, res);
      return;
    }
    // Static assets from the client build's public/ (checked before route render so
    // /client.js, /rsc.css, etc. are served, not treated as app-router paths).
    if (req.method === "GET") {
      const name = url.pathname.replace(/^\//, "");
      const filePath = join(publicDir, name);
      // A DIRECTORY is not a static asset. `existsSync` says yes for one, and the app
      // route that shares its name then died between the header and the body: the 200 +
      // content-type went out, `readFileSync` threw EISDIR, and the response was never
      // ended — an empty body under a 200, with the page never rendered. cal.com ships
      // `public/apps/`, so `/apps` (a real page) answered exactly that.
      if (name && filePath.startsWith(publicDir) && isStaticFile(filePath)) {
        serveStaticAsset(req, res, filePath);
        return;
      }
    }
    // Middleware (`middleware.ts`): runs before route handlers / page render for
    // non-asset, non-action requests. `next()` continues (applying request-header
    // overrides + response set-cookies), `redirect()`/`rewrite()`/a plain Response act
    // per Next's `x-middleware-*` protocol.
    let mwSetCookies = [];
    let mwRequestHeaders = [];
    if ((await getManifest()).hasMiddleware) {
      const mw = await runMiddleware({
        url: "http://localhost" + req.url,
        method: req.method,
        headers: Object.entries(req.headers).map(([k, v]) => [k, Array.isArray(v) ? v.join(", ") : String(v)]),
        cookie: req.headers.cookie || "",
      });
      if (mw) {
        if (mw.kind === "redirect") {
          const h = { location: addBasePath(mw.location, localeSeg) };
          if (mw.setCookies.length) h["set-cookie"] = mw.setCookies;
          res.writeHead(mw.status, h);
          res.end();
          return;
        }
        if (mw.kind === "response") {
          const h = {};
          for (const [k, v] of mw.headers) h[k] = v;
          res.writeHead(mw.status, h);
          res.end(mw.body ? Buffer.from(mw.body, "base64") : undefined);
          return;
        }
        if (mw.kind === "rewrite") url.pathname = mw.pathname;
        mwSetCookies = mw.setCookies || [];
        mwRequestHeaders = mw.requestHeaders || [];
      }
    }
    // next.config redirects/rewrites/headers (evaluated at build time). A matching
    // redirect short-circuits with a 3xx; a matching rewrite mutates url.pathname (so
    // the render/route dispatch below sees the destination); matching header rules are
    // collected and merged onto whatever response we ultimately send.
    const nc = applyNextConfig(url, req);
    if (nc.redirect) {
      const h = { location: addBasePath(nc.redirect.location, localeSeg) };
      if (mwSetCookies.length) h["set-cookie"] = mwSetCookies;
      res.writeHead(nc.redirect.status, h);
      res.end();
      return;
    }
    const configHeaders = nc.headers;
    // Endpoints (`app/**/route.ts` and a hybrid app's `pages/api/**`): a request whose
    // path matches one is served by it (any method), not by page rendering. Which graph
    // runs it depends on the kind — see `routeEndpointKind`.
    const endpointKind = await routeEndpointKind(url.pathname);
    if (endpointKind) {
      const bodyChunks = [];
      if (req.method !== "GET" && req.method !== "HEAD") {
        for await (const chunk of req) bodyChunks.push(Buffer.from(chunk));
      }
      const bodyBuf = Buffer.concat(bodyChunks);
      const reqCtx = {
        url: "http://localhost" + req.url,
        method: req.method,
        headers: Object.entries(req.headers).map(([k, v]) => [
          k,
          Array.isArray(v) ? v.join(", ") : String(v),
        ]),
        cookie: req.headers.cookie || "",
        body: bodyBuf.length ? bodyBuf.toString("base64") : undefined,
        bodyIsBase64: bodyBuf.length > 0,
        locale: reqLocale,
      };
      for (const h of mwRequestHeaders) reqCtx.headers.push(h);
      let result = null;
      if (endpointKind === "pages-api") {
        // Pages-router API route: run it in THIS process, out of the SSR bundle. Same
        // `{ status, headers, body(base64), setCookies }` shape a route handler returns.
        result = await (await getPagesApiHandler())(url.pathname, req.method, reqCtx);
      } else {
        const msg = await nextWorker().call({
          op: "route",
          pathname: url.pathname,
          method: req.method,
          reqCtx,
        });
        // next/cache: a route handler can call revalidatePath/revalidateTag — bust the
        // matching prerendered pages so the next request to them re-renders.
        applyRevalidation(msg.revalidated);
        result = msg.routeResult;
      }
      if (result) {
        const headers = {};
        for (const [k, v] of result.headers || []) headers[k] = v;
        for (const [k, v] of configHeaders) headers[k] = v;
        if (mwSetCookies.length) headers["set-cookie"] = mwSetCookies;
        // Cookies the handler wrote (next/headers cookies().set()/draftMode() OR the
        // Response's own Set-Cookie headers) — appended so middleware cookies survive too.
        mergeSetCookie(headers, result.setCookies);
        res.writeHead(result.status || 200, headers);
        res.end(result.body ? Buffer.from(result.body, "base64") : undefined);
        return;
      }
      // No handler produced a response (unexpected): fall through to 404 below.
    }
    // Prerendered cache: a static / SSG / ISR page served straight from disk with no
    // render (ISR revalidates in the background when stale). Takes precedence over the
    // dynamic render path below, but not over middleware / next.config / route handlers.
    if (req.method === "GET") {
      // The prerender cache is keyed by the app-relative route path (no trailing slash);
      // a trailingSlash-canonical request (`/about/`) still hits it.
      const cacheKey =
        url.pathname !== "/" && url.pathname.endsWith("/")
          ? url.pathname.replace(/\/+$/, "")
          : url.pathname;
      const entry = prerenderCache.get(cacheKey);
      if (entry && servePrerendered(entry, url.searchParams.has("__rsc"), res, configHeaders, mwSetCookies)) {
        return;
      }
    }
    // Any other GET is an app-router route: the react-server render STREAMS the flight
    // (shell first, then each Suspense boundary as its async Server Component resolves)
    // and the SSR bundle streams the HTML document out as those chunks arrive — a fast
    // TTFB even when the page has a slow data dependency behind `<Suspense>`.
    if (req.method === "GET") {
      // The per-request context the react-server render establishes (an
      // AsyncLocalStorage carrying the request url/headers/cookie, read by
      // next/headers cookies()/headers() inside async Server Components). Array-valued
      // headers (e.g. set-cookie) are joined so `new Headers([[k,v]...])` accepts them.
      const reqCtxObj = {
        url: "http://localhost" + req.url,
        headers: [
          ...Object.entries(req.headers).map(([k, v]) => [k, Array.isArray(v) ? v.join(", ") : String(v)]),
          ...mwRequestHeaders,
        ],
        cookie: req.headers.cookie || "",
        // A `?__rsc=1` fetch is a client soft navigation — the only context in which an
        // intercepting route renders its overlay instead of the full page.
        softNav: url.searchParams.has("__rsc"),
        // The i18n-detected request locale (diffpack locale-routing extension), or
        // undefined when no next.config `i18n` is configured.
        locale: reqLocale,
      };
      // Strict-CSP nonce for every script the document emits (see scriptNonceFromHeaders).
      const scriptNonce = scriptNonceFromHeaders(reqCtxObj.headers);
      // Kick off the streaming render. `meta` (status/params + any TOP-LEVEL
      // redirect/notFound) settles on the first chunk; flight chunks flow into a queue
      // exposed as an async iterator.
      const worker = nextWorker();
      let metaResolve;
      let metaReject;
      const metaP = new Promise((resolve, reject) => {
        metaResolve = resolve;
        metaReject = reject;
      });
      const queue = [];
      let waiters = [];
      let ended = false;
      let endMeta = null;
      let streamError = null;
      const wake = () => {
        const w = waiters;
        waiters = [];
        for (const f of w) f();
      };
      worker.callStream(
        { op: "render-stream", pathname: url.pathname, manifestPath: clientManifestPath, reqCtx: reqCtxObj },
        {
          onMeta: (m) => metaResolve(m),
          onChunk: (b64) => {
            queue.push(b64);
            wake();
          },
          onEnd: (m) => {
            // The END meta carries this route's client-reference chunk list (references
            // are discovered as React serializes, so `meta` is too early for it).
            endMeta = m || {};
            // redirect()/notFound() thrown BEHIND a Suspense boundary (after the shell
            // already flushed) can't unwind the streamed response. Never silent: log it.
            // `lateControl` is what makes this a real report: a redirect the META carried
            // was turned into a real 307 above, and warning about THAT one would be noise
            // on every single redirecting route.
            if (m && m.metaSent && m.lateControl && (m.redirect || m.notFound)) {
              const what = m.redirect ? `redirect(${m.redirect})` : "notFound()";
              console.error(
                `[diffpack] next: ${what} after the shell flushed on ${url.pathname}; the response was already streamed and cannot be changed.`,
              );
            }
            ended = true;
            wake();
          },
          onError: (e) => {
            streamError = new Error(String(e));
            metaReject(streamError);
            wake();
          },
        },
      );
      async function* flightChunks() {
        let i = 0;
        for (;;) {
          if (streamError) throw streamError;
          if (i < queue.length) {
            yield queue[i];
            i += 1;
            continue;
          }
          if (ended) return;
          await new Promise((resolve) => waiters.push(resolve));
        }
      }
      const meta = await metaP;
      // Server-side redirect(): issue a REAL HTTP redirect (do NOT SSR the errored
      // flight tree). Over the soft-nav channel (?__rsc=1) hand the client Router a
      // JSON redirect it follows via history + a re-fetch.
      if (meta.redirect) {
        // Cookies set BEFORE a top-level redirect() (e.g. a login flow that writes a
        // session cookie then redirects) travel with the redirect response.
        if (url.searchParams.has("__rsc")) {
          const rh = { "content-type": "application/json" };
          if (mwSetCookies.length) rh["set-cookie"] = mwSetCookies;
          mergeSetCookie(rh, meta.setCookies);
          res.writeHead(200, rh);
          res.end(JSON.stringify({ __redirect: addBasePath(meta.redirect, localeSeg) }));
          return;
        }
        const rh = { location: addBasePath(meta.redirect, localeSeg) };
        if (mwSetCookies.length) rh["set-cookie"] = mwSetCookies;
        mergeSetCookie(rh, meta.setCookies);
        res.writeHead(meta.status || 307, rh);
        res.end();
        return;
      }
      // Server-side notFound(): render the real not-found tree (buffered — an error
      // path) and serve it 404. The render is asked for BY FLAG (`reqCtx.notFound`) at
      // the pathname the visitor requested, never by a sentinel pathname: an app with a
      // catch-all route matches every sentinel too, and rendering that catch-all page is
      // how the 404 document ended up empty.
      if (meta.notFound) {
        const nf = await runReactServer(
          ["render", url.pathname, clientManifestPath],
          JSON.stringify({ ...reqCtxObj, notFound: true }),
        );
        // The not-found tree cannot itself redirect or 404 — if it reports either, the
        // react-server entry did not honour the flag and we would be about to serve the
        // WRONG document under a 404. Fail loudly instead.
        if (nf.notFound || nf.redirect) {
          throw new Error(
            `next-server: the not-found document render for ${url.pathname} itself signalled ` +
              `${nf.redirect ? `redirect(${nf.redirect})` : "notFound()"} — the react-server entry ` +
              `did not honour reqCtx.notFound and rendered a matched route instead of app/not-found`,
          );
        }
        const nfDoc = await (await getRenderFlightToDocument())(
          new Uint8Array(nf.flight),
          serverConsumerManifest,
          nf.flight.toString("base64"),
          {},
          { pathname: url.pathname, search: url.search },
          scriptNonce,
          nf.chunks || [],
        );
        const nfHeaders = { "content-type": "text/html; charset=utf-8" };
        for (const [k, v] of configHeaders) nfHeaders[k] = v;
        if (mwSetCookies.length) nfHeaders["set-cookie"] = mwSetCookies;
        // Cookies written before the top-level notFound() throw still apply to the 404.
        mergeSetCookie(nfHeaders, meta.setCookies);
        res.writeHead(404, nfHeaders);
        res.end(nfDoc);
        return;
      }
      // Soft-navigation: the client Router fetches `?__rsc=1` for the RAW flight of the
      // target route and diff-renders it in place. When the render was an intercepting
      // overlay, tell the client via `x-diffpack-intercept` so it portals the flight over
      // the current page (masking the URL) instead of swapping the document.
      if (url.searchParams.has("__rsc")) {
        const rscHeaders = { "content-type": "text/x-component" };
        // The matched route's dynamic params travel WITH the flight. A soft navigation
        // changes the route, so `useParams()` must change with it — the client Router
        // has no way to derive `{ uid: "…" }` from a URL it does not know the segment
        // pattern of. Without this the params context stayed frozen on whatever the
        // document was first loaded with, and cal.com's `/booking/[uid]` page (reached
        // by `router.push` after a booking) parsed `uid: undefined` and rendered its 500
        // page. Percent-encoded because a param value may hold non-ASCII bytes.
        rscHeaders["x-diffpack-params"] = encodeURIComponent(JSON.stringify(meta.params || {}));
        if (meta.intercept) rscHeaders["x-diffpack-intercept"] = "1";
        // DEV splits the islands into per-island chunks, so a soft navigation can reach
        // references the loaded page never needed. The client Router loads these before
        // it decodes the flight (the RSC seam's require is synchronous). Buffered rather
        // than streamed, because the list is only complete once the flight is — and the
        // client buffers this response anyway (`fetchFlight` reads it to completion).
        if (DEV) {
          const parts = [];
          for await (const b64 of flightChunks()) parts.push(Buffer.from(b64, "base64"));
          const chunkList = (endMeta && endMeta.chunks) || [];
          if (chunkList.length) rscHeaders["x-diffpack-chunks"] = encodeURIComponent(JSON.stringify(chunkList));
          if (mwSetCookies.length) rscHeaders["set-cookie"] = mwSetCookies;
          mergeSetCookie(rscHeaders, meta.setCookies);
          res.writeHead(200, rscHeaders);
          res.end(Buffer.concat(parts));
          return;
        }
        if (mwSetCookies.length) rscHeaders["set-cookie"] = mwSetCookies;
        mergeSetCookie(rscHeaders, meta.setCookies);
        res.writeHead(200, rscHeaders);
        for await (const b64 of flightChunks()) res.write(Buffer.from(b64, "base64"));
        res.end();
        return;
      }
      const docHeaders = { "content-type": "text/html; charset=utf-8" };
      for (const [k, v] of configHeaders) docHeaders[k] = v;
      if (mwSetCookies.length) docHeaders["set-cookie"] = mwSetCookies;
      // Top-level cookies().set()/draftMode() writes captured before the shell flushed.
      mergeSetCookie(docHeaders, meta.setCookies);
      // DEV serves BUFFERED (whole flight inlined as __DIFFPACK_FLIGHT__): the
      // DEVELOPMENT react-server-dom-webpack client is stricter and trips a spurious
      // "Connection closed" on the incremental __DF_FLIGHT stream (the production client
      // reconstructs it fine). Dev has no streaming-TTFB requirement, so the buffered
      // document is the reliable path; PRODUCTION keeps streaming for a fast first byte.
      if (DEV) {
        const parts = [];
        for await (const b64 of flightChunks()) parts.push(Buffer.from(b64, "base64"));
        const flightBuf = Buffer.concat(parts);
        const html = await (await getRenderFlightToDocument())(
          new Uint8Array(flightBuf),
          serverConsumerManifest,
          flightBuf.toString("base64"),
          meta.params || {},
          { pathname: url.pathname, search: url.search },
          scriptNonce,
          // Drained above, so the END meta has landed: every split chunk this route's
          // client references live in, for the browser entry to load before it hydrates.
          (endMeta && endMeta.chunks) || [],
        );
        res.writeHead(meta.status || 200, docHeaders);
        res.end(html);
        return;
      }
      await (await getRenderFlightToStream())(
        flightChunks(),
        serverConsumerManifest,
        meta.params || {},
        { pathname: url.pathname, search: url.search },
        res,
        docHeaders,
        meta.status,
        scriptNonce,
      );
      return;
    }
    res.writeHead(404).end("not found");
  } catch (error) {
    failRequest(res, error);
  }
});

// instrumentation.ts boot hook: if the build emitted `<out>/instrumentation.mjs` (the
// bundled boot-entry that CALLS the app's register()), import it exactly once here,
// before we start accepting connections. Importing runs register() as the module's
// top-level side effect, and the await blocks listen until it resolves. This is the
// OpenTelemetry/Sentry-style startup hook; it touches request latency zero times. A
// missing register() is a hard error thrown by the generated wrapper (never a no-op).
const instrumentationPath = join(outputDir, "instrumentation.mjs");
if (existsSync(instrumentationPath)) {
  await import(pathToFileURL(instrumentationPath).href);
  bootMark("instrumentation ran");
}

server.listen(port, () => {
  const actual = server.address().port;
  bootMark("listening");
  console.log(`next-server listening on http://localhost:${actual}`);
});
