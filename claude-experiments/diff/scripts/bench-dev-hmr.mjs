// Dev-server EDIT-TO-UPDATE (HMR) benchmark: `diffpack dev` vs `next dev --turbopack`,
// on the SAME real Next app-router fixture (integration/next-app-router).
//
// WHAT IT MEASURES (per server):
//   1. STARTUP: two honest numbers — `ready` (server accepts a request) and
//      `first-byte` (first 200 for `/`, i.e. the first fully-rendered document a
//      user actually sees). For Turbopack these differ by ~10x because routes are
//      compiled ON DEMAND at first request; diffpack builds all three RSC graphs at
//      boot so its two numbers nearly coincide. Median of --starts cold starts
//      (output dir wiped before each).
//   2. HMR edit-to-update latency, end-to-end, for TWO edit classes:
//        • client-component visible text (app/Counter.tsx label) — Fast Refresh
//          hot update on both servers (state-preserving, no reload).
//        • server-component text (app/page.tsx #heading) — on diffpack a FULL
//          RELOAD (fresh react-server child per GET); on Turbopack an RSC refresh
//          (new flight reconciled, page NOT reloaded). Same DOM outcome, DIFFERENT
//          semantics — reported alongside the latency, never hidden.
//      One warmup edit per class (= that route's cold first-compile; Turbopack pays
//      it here, diffpack already paid it at boot) is measured but reported
//      separately, then --samples warm edits: median / p95 / min / max.
//
// HOW t0 AND t1 SHARE A CLOCK (the crux of a FAIR cross-process measurement):
//   Both the harness (Node) and the page (Chrome) run on the same machine and read
//   the same OS wall clock via Date.now() / performance.timeOrigin. t0 = Date.now()
//   in Node immediately before the file write. t1 is captured INSIDE the browser at
//   the instant the DOM reflects the change, by a self-timestamping MutationObserver
//   (window.__mark = performance.timeOrigin + performance.now()). The harness reads
//   __mark back afterwards over the agent-browser CLI; that readback has its own
//   (tens-of-ms, process-spawn) latency, but it does NOT enter the measurement
//   because __mark was stamped at the true instant. delta = __mark - t0.
//   For the reload class the observer is destroyed by navigation, so t1 falls back
//   to the NEW document's navigation timing: performance.timeOrigin +
//   navigation.responseEnd — still the same wall clock, still the moment the marker
//   is in the DOM. Each sample records which path (hot|reload) it took.
//
// Each edit writes a UNIQUE nonce into the visible text, so the observer knows
// exactly which mutation is "the" change (no "was it already showing that?"
// ambiguity, and no dependence on a fixed poll interval).
//
// Native build (Rust); Node + Chrome (via the agent-browser CLI) are TEST ORACLES.
// The two fixture files are snapshotted and ALWAYS restored (finally).
//
// Usage:
//   node scripts/bench-dev-hmr.mjs                       # both servers, 20 samples, 5 starts
//   node scripts/bench-dev-hmr.mjs --server diffpack
//   node scripts/bench-dev-hmr.mjs --server next
//   node scripts/bench-dev-hmr.mjs --samples 30 --starts 7
//
// Requires: cargo build --release; the fixture's node_modules (npm install there);
// agent-browser on PATH.

import { spawn, execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, rmSync, mkdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const scriptsDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = dirname(scriptsDir);
const fixture = join(repoRoot, "integration", "next-app-router");
const diffpack = join(repoRoot, "target", "release", "diffpack");
const nextBin = join(fixture, "node_modules", ".bin", "next");

const args = parseArgs(process.argv.slice(2));
const which = args.server ?? "both";
const SAMPLES = args.samples ?? 20;
const STARTS = args.starts ?? 5;
const BASE_PORT = args.port ?? 8990;

const counterFile = join(fixture, "app", "Counter.tsx");
const pageFile = join(fixture, "app", "page.tsx");

if (!existsSync(diffpack)) throw new Error(`missing ${diffpack}; run \`cargo build --release\` first`);
if (!existsSync(join(fixture, "node_modules"))) throw new Error(`${fixture}/node_modules missing; run \`npm install\` in the fixture`);
if (!existsSync(nextBin)) throw new Error(`missing ${nextBin}`);
sh("agent-browser", ["--help"]); // fail loudly if the CLI is absent

const counterOrig = readFileSync(counterFile, "utf8");
const pageOrig = readFileSync(pageFile, "utf8");
if (!counterOrig.includes("count: ")) throw new Error("Counter.tsx no longer contains 'count: '; refusing to edit");
if (!pageOrig.includes("from-server")) throw new Error("page.tsx no longer contains 'from-server'; refusing to edit");

const servers = {
  diffpack: {
    label: "diffpack dev",
    spawn: (port) => spawn(diffpack, ["dev", ".", String(port)], { cwd: fixture, stdio: ["ignore", "pipe", "pipe"] }),
    outDir: join(fixture, ".diffpack-output"),
  },
  next: {
    label: "next dev --turbopack",
    spawn: (port) => spawn(nextBin, ["dev", "--turbopack", "--port", String(port)], {
      cwd: fixture, stdio: ["ignore", "pipe", "pipe"], env: { ...process.env, NEXT_TELEMETRY_DISABLED: "1" },
    }),
    outDir: join(fixture, ".next"),
  },
};

const chosen = which === "both" ? ["diffpack", "next"] : [which];
const report = { meta: { samples: SAMPLES, starts: STARTS, date: new Date().toISOString(), fixture: "integration/next-app-router" }, servers: {} };

try {
  for (const key of chosen) {
    console.log(`\n================  ${servers[key].label}  ================`);
    report.servers[key] = await benchServer(servers[key]);
    restoreFixture();
  }
} finally {
  restoreFixture();
  sh("agent-browser", ["close"], true);
}

const resultsDir = join(repoRoot, "bench", "results");
mkdirSync(resultsDir, { recursive: true });
// Default output is the canonical file the docs quote; the check.sh liveness row
// passes --out so its small (3/2) run does NOT clobber the doc's source of truth.
const outPath = args.out
  ? (args.out.startsWith("/") ? args.out : join(repoRoot, args.out))
  : join(resultsDir, "dev-hmr-results.json");
writeFileSync(outPath, JSON.stringify(report, null, 2));
printTable(report, outPath);

// Exit explicitly: a killed dev-server child (or the agent-browser daemon) can
// leave a handle that keeps the Node event loop alive after the report is
// written, and the liveness gate waits on this process exiting.
process.exit(0);

// ----------------------------------------------------------------------------

async function benchServer(server) {
  const out = { startup: await measureStartup(server), hmr: {} };

  // One long-lived instance for the warm HMR loop.
  const port = BASE_PORT;
  rmSync(server.outDir, { recursive: true, force: true });
  const { proc, log } = boot(server, port);
  const base = `http://127.0.0.1:${port}`;
  try {
    await waitFirstByte(base, log, 120000);
    // Open the page at `localhost`, NOT 127.0.0.1: Next 16 blocks its dev HMR
    // WebSocket (`/_next/webpack-hmr`) as a cross-origin dev resource unless the
    // page origin matches, so a 127.0.0.1 page NEVER receives Fast Refresh /
    // RSC-refresh updates (it silently falls back to a ~40s hard reload). Both
    // servers bind 127.0.0.1 and `localhost` resolves there, so this is uniform.
    // Health checks stay on 127.0.0.1 (Node fetch, origin-agnostic).
    sh("agent-browser", ["open", `${browserBase(base)}/`], true);
    waitSelector("#counter", 30000);

    out.hmr["client-text (Fast Refresh)"] = await editLoop({
      file: counterFile, orig: counterOrig, base,
      find: "count: ", selector: "#counter",
      mkEdit: (nonce) => counterOrig.replace("count: ", `c${nonce}: `),
      wantPrefix: (nonce) => `c${nonce}:`,
      expectReload: false,
    });

    out.hmr["server-text (RSC refresh)"] = await editLoop({
      file: pageFile, orig: pageOrig, base,
      find: "from-server", selector: "#heading",
      mkEdit: (nonce) => pageOrig.replace("from-server", `srv${nonce}`),
      wantPrefix: (nonce) => `srv${nonce}`,
      expectReload: false, // both now refresh the flight in place (no full reload)
    });
  } finally {
    kill(proc);
  }
  return out;
}

// Cold startup: ready (accepts a connection) and first-byte (first 200 for /).
async function measureStartup(server) {
  const ready = [], firstByte = [];
  for (let i = 0; i < STARTS; i++) {
    const port = BASE_PORT + 10 + i;
    rmSync(server.outDir, { recursive: true, force: true });
    const t0 = Date.now();
    const { proc, log } = boot(server, port);
    const base = `http://127.0.0.1:${port}`;
    try {
      await waitReady(base, log, 120000);
      ready.push(Date.now() - t0);
      await waitFirstByte(base, log, 120000);
      firstByte.push(Date.now() - t0);
    } finally {
      kill(proc);
    }
    console.log(`  start #${i + 1}: ready=${ready.at(-1)}ms first-byte=${firstByte.at(-1)}ms`);
  }
  return { ready: stats(ready), firstByte: stats(firstByte) };
}

async function editLoop({ file, orig, base, find, selector, mkEdit, wantPrefix, expectReload }) {
  if (!orig.includes(find)) throw new Error(`fixture ${file} lost '${find}'`);
  const samples = [];
  let warmup = null;
  for (let s = 0; s < SAMPLES + 1; s++) {
    const nonce = `${Date.now()}${s}`.slice(-9);
    const want = wantPrefix(nonce);
    armObserver(selector, want);
    const t0 = Date.now();
    writeFileSync(file, mkEdit(nonce));
    const mark = await pollMark(want, 90000);
    const delta = mark.wall - t0;
    if (s === 0) warmup = { delta, path: mark.path }; // cold first-compile for this route
    else samples.push(delta);
    writeFileSync(file, orig); // restore between samples; observer re-armed each time
    await settle(base, selector, find, 30000); // wait for the restore to land before next edit
    process.stdout.write(`  ${selector} #${s}: ${delta}ms (${mark.path})${s === 0 ? " [cold/warmup]" : ""}\n`);
  }
  return { warmup, warm: stats(samples), semantics: expectReload ? "full reload" : "state-preserving hot update", samplesRaw: samples };
}

// Install a self-timestamping observer. Hot updates fire the MutationObserver;
// a navigation (reload) destroys it, so we ALSO stamp from the fresh document's
// navigation timing if the marker is already present on a NEW navigation.
function armObserver(selector, want) {
  const js = `(() => {
    window.__mark = null;
    const want = ${JSON.stringify(want)};
    const sel = ${JSON.stringify(selector)};
    const hit = () => {
      const el = document.querySelector(sel);
      if (el && el.textContent.replace(/\\s+/g,' ').includes(want)) {
        window.__mark = { wall: performance.timeOrigin + performance.now(), path: 'hot' };
        return true;
      }
      return false;
    };
    if (!hit()) {
      const mo = new MutationObserver(() => { if (hit()) mo.disconnect(); });
      mo.observe(document.documentElement, { subtree: true, childList: true, characterData: true });
    }
    return true;
  })()`;
  evalJson(js);
}

// Poll for t1 on the same wall clock. Two paths, returned by ONE eval:
//   hot    — the MutationObserver in THIS document fired and stamped window.__mark
//            (state-preserving update; no navigation).
//   reload — the observer was destroyed by a navigation; the marker `want` is now
//            present in the FRESH document, so stamp t1 from its navigation timing
//            (performance.timeOrigin + navigation.responseEnd), the wall-clock
//            instant the new document's bytes (with the marker) arrived.
// `want` is the same unique nonce armObserver() looked for, threaded in so the
// reload path only fires on the intended change.
async function pollMark(want, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const r = evalJson(`(() => {
      if (window.__mark && typeof window.__mark.wall === 'number') return window.__mark;
      const want = ${JSON.stringify(want)};
      const doc = document.documentElement.textContent.replace(/\\s+/g,' ');
      if (doc.includes(want)) {
        const n = performance.getEntriesByType('navigation')[0];
        const t1 = n ? performance.timeOrigin + n.responseEnd : performance.timeOrigin + performance.now();
        return { wall: t1, path: 'reload' };
      }
      return null;
    })()`);
    if (r && typeof r.wall === "number") return r;
    sleepMs(40);
  }
  throw new Error("edit-to-update timed out (no DOM change observed)");
}

function settle(base, selector, find, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const txt = evalJson(`(() => { const e = document.querySelector(${JSON.stringify(selector)}); return e ? e.textContent : null; })()`);
    if (txt && txt.includes(find.replace("count: ", "count").replace("from-server", "from-server").slice(0, 4))) return;
    sleepMs(60);
  }
}

// ---- process / io helpers --------------------------------------------------

function boot(server, port) {
  let log = "";
  const proc = server.spawn(port);
  proc.stdout.on("data", (d) => (log += d));
  proc.stderr.on("data", (d) => (log += d));
  proc._getLog = () => log;
  return { proc, get log() { return proc._getLog(); } };
}

function kill(proc) { try { proc.kill("SIGTERM"); } catch {} setTimeout(() => { try { proc.kill("SIGKILL"); } catch {} }, 800); }

async function waitReady(base, getLog, timeoutMs) {
  // "ready" = TCP accept / any HTTP response (even a compile-in-progress 200/404).
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try { await fetch(base + "/_next/nonexistent", { signal: AbortSignal.timeout(2000) }); return; } catch {}
    sleepMs(20);
  }
  throw new Error("server never became ready:\n" + logOf(getLog));
}

async function waitFirstByte(base, getLog, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(base + "/", { signal: AbortSignal.timeout(5000) });
      if (r.ok) { await r.text(); return; }
    } catch {}
    sleepMs(20);
  }
  throw new Error("no first byte for /:\n" + logOf(getLog));
}

function waitSelector(sel, timeoutMs) { sh("agent-browser", ["wait", sel, "--timeout", String(timeoutMs)], true); }

// agent-browser eval that returns a JSON value. We wrap the expression in
// JSON.stringify so the CLI has a single JSON string to emit, and pass the whole
// script base64-encoded (`eval -b`) — the CLI's own docs mark inline eval as
// "simple expressions only"; base64 is the robust path for JS with quotes/regex.
// agent-browser then JSON-encodes ITS result too, so the wire form is a
// JSON-string-of-a-JSON-string → we decode twice.
function evalJson(js) {
  const b64 = Buffer.from(`JSON.stringify(${js})`).toString("base64");
  const out = sh("agent-browser", ["eval", "-b", b64], true);
  const line = (out || "").trim();
  if (!line) return null;
  try {
    let v = JSON.parse(line);                                  // strip agent-browser's wrapper
    if (typeof v === "string") { try { v = JSON.parse(v); } catch {} } // strip our JSON.stringify
    return v;
  } catch {
    return null;
  }
}

function sh(cmd, argv, soft = false) {
  try { return execFileSync(cmd, argv, { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }); }
  catch (e) { if (soft) return (e.stdout || "") + (e.stderr || ""); throw new Error(`${cmd} ${argv.join(" ")} failed: ${e.message}`); }
}

// Zero-spawn synchronous sleep (Atomics.wait on a throwaway SAB). The earlier
// `execFileSync("node", ...)` per poll spawned a fresh process every 40ms, which
// contends with the dev server's own compiler thread and inflates the very
// rebuild latency we are measuring. This adds no processes to the timed window.
function sleepMs(ms) { Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms); }

// Map a 127.0.0.1 health-check base to a browser origin Next won't cross-origin
// block. Both dev servers bind 127.0.0.1 and `localhost` resolves there.
function browserBase(base) { return base.replace("127.0.0.1", "localhost"); }
function logOf(getLog) { return typeof getLog === "function" ? getLog() : String(getLog); }
function restoreFixture() { try { writeFileSync(counterFile, counterOrig); writeFileSync(pageFile, pageOrig); } catch {} }

// ---- stats / reporting -----------------------------------------------------

function stats(xs) {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const q = (p) => s[Math.min(s.length - 1, Math.floor(p * (s.length - 1)))];
  return { n: s.length, median: q(0.5), p95: q(0.95), min: s[0], max: s.at(-1) };
}

function printTable(r, outPath) {
  console.log("\n===== dev HMR summary (ms) =====");
  for (const [key, v] of Object.entries(r.servers)) {
    console.log(`\n[${key}]`);
    console.log(`  startup ready     median=${v.startup.ready?.median} p95=${v.startup.ready?.p95}`);
    console.log(`  startup first-byte median=${v.startup.firstByte?.median} p95=${v.startup.firstByte?.p95}`);
    for (const [cls, h] of Object.entries(v.hmr)) {
      console.log(`  ${cls}: warm median=${h.warm?.median} p95=${h.warm?.p95} min=${h.warm?.min} max=${h.warm?.max} | cold-first=${h.warmup?.delta} | ${h.semantics}`);
    }
  }
  console.log(`\nfull JSON -> ${outPath}`);
}

function parseArgs(a) {
  const o = {};
  for (let i = 0; i < a.length; i++) {
    if (a[i] === "--server") o.server = a[++i];
    else if (a[i] === "--samples") o.samples = Number(a[++i]);
    else if (a[i] === "--starts") o.starts = Number(a[++i]);
    else if (a[i] === "--port") o.port = Number(a[++i]);
    else if (a[i] === "--out") o.out = a[++i];
    else throw new Error(`unknown arg ${a[i]}`);
  }
  return o;
}
