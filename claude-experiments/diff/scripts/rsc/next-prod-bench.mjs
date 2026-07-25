// Production benchmark: `diffpack build-app production` + `diffpack start` vs
// `next build` + `next start`, on the SAME app. Measures three things that matter for
// a deployed dynamic app:
//   1. BUILD time (wall clock).
//   2. Server MEMORY — steady-state RSS summed across the whole server process tree
//      (diffpack: the orchestrator + its react-server worker pool; next: next-server
//      + any workers). Sampled repeatedly under load; we report the peak.
//   3. Request LATENCY — median/p95 of a warm document request.
//
//   node next-prod-bench.mjs <app-dir> [--requests N] [--server diffpack|next]
//
// The app must build under BOTH toolchains (a stock app-router app). Fixture-safe.
import { spawn, execFileSync, execSync } from "node:child_process";
import { connect } from "node:net";
import { rmSync } from "node:fs";
import { resolve } from "node:path";

const appDir = resolve(process.argv[2] || ".");
const arg = (name, def) => {
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : def;
};
const REQUESTS = Number(arg("--requests", "60"));
const only = arg("--server", null);
// The route whose TTFB we measure — a page with a slow async Server Component behind a
// Suspense boundary, where streaming SSR flushes the shell long before the data.
const TTFB_PATH = arg("--ttfb-path", "/slow");
const DIFFPACK = process.env.DIFFPACK ||
  "/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/diff/target/release/diffpack";
const REPO = "/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/diff";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// The set of pid + all descendants (one `ps` snapshot).
function processTree(rootPid) {
  let out;
  try {
    out = execSync("ps -Ao pid=,ppid=", { encoding: "utf8" });
  } catch {
    return [rootPid];
  }
  const kids = new Map();
  for (const line of out.split("\n")) {
    const m = line.trim().match(/^(\d+)\s+(\d+)$/);
    if (!m) continue;
    const [pid, ppid] = [Number(m[1]), Number(m[2])];
    if (!kids.has(ppid)) kids.set(ppid, []);
    kids.get(ppid).push(pid);
  }
  const all = [];
  const stack = [rootPid];
  const seen = new Set();
  while (stack.length) {
    const p = stack.pop();
    if (seen.has(p)) continue;
    seen.add(p);
    all.push(p);
    for (const k of kids.get(p) || []) stack.push(k);
  }
  return all;
}

// Fair cross-process memory: sum macOS `phys_footprint` (dirty + compressed private
// memory — the number that counts against a process's memory limit) over the tree.
// Unlike RSS, it does NOT double-count the shared Node binary/libs across processes,
// so it is a fair comparison between diffpack's multi-process server and next's single
// process. Returns MB (0 if `footprint` is unavailable).
function treeFootprintMb(rootPid) {
  let total = 0;
  for (const pid of processTree(rootPid)) {
    try {
      const out = execSync(`footprint ${pid} 2>/dev/null`, { encoding: "utf8" });
      const m = out.match(/phys_footprint:\s*([0-9.]+)\s*([MK])/i) || out.match(/Footprint:\s*([0-9.]+)\s*([MK])/i);
      if (m) total += Number(m[1]) / (m[2].toUpperCase() === "K" ? 1024 : 1);
    } catch {}
  }
  return total;
}

// Sum RSS (KB) of `pid` and every descendant, via one `ps` snapshot of the tree.
function treeRssKb(rootPid) {
  let out;
  try {
    out = execSync("ps -Ao pid=,ppid=,rss=", { encoding: "utf8" });
  } catch {
    return 0;
  }
  const kids = new Map();
  const rss = new Map();
  for (const line of out.split("\n")) {
    const m = line.trim().match(/^(\d+)\s+(\d+)\s+(\d+)$/);
    if (!m) continue;
    const [pid, ppid, r] = [Number(m[1]), Number(m[2]), Number(m[3])];
    rss.set(pid, r);
    if (!kids.has(ppid)) kids.set(ppid, []);
    kids.get(ppid).push(pid);
  }
  let total = 0;
  const stack = [rootPid];
  const seen = new Set();
  while (stack.length) {
    const p = stack.pop();
    if (seen.has(p)) continue;
    seen.add(p);
    total += rss.get(p) || 0;
    for (const k of kids.get(p) || []) stack.push(k);
  }
  return total;
}

async function waitReady(port, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(`http://127.0.0.1:${port}/`, { signal: AbortSignal.timeout(2000) });
      if (r.status < 500) return;
    } catch {}
    await sleep(150);
  }
  throw new Error("server never became ready");
}

async function measureLatency(port, n) {
  // warmup
  for (let i = 0; i < 5; i++) await fetch(`http://127.0.0.1:${port}/`).then((r) => r.text());
  const samples = [];
  for (let i = 0; i < n; i++) {
    const t = performance.now();
    const r = await fetch(`http://127.0.0.1:${port}/`);
    await r.text();
    samples.push(performance.now() - t);
  }
  samples.sort((a, b) => a - b);
  return {
    median: samples[Math.floor(samples.length / 2)],
    p95: samples[Math.floor(samples.length * 0.95)],
    min: samples[0],
  };
}

// Time-to-first-byte vs full-response on a streaming route, over a raw socket (fetch
// buffers, hiding the streaming gap). TTFB = when the shell's first bytes arrive; full
// = when the slow Suspense boundary has streamed in. A large full-minus-TTFB gap on the
// SAME server is the streaming win: the user sees the shell without waiting for data.
function ttfbOnce(port, path) {
  return new Promise((resolve, reject) => {
    const sock = connect(port, "127.0.0.1", () => {
      t0 = performance.now();
      sock.write(`GET ${path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n`);
    });
    let t0 = 0;
    let ttfb = null;
    let bytes = 0;
    sock.on("data", (b) => {
      if (ttfb === null) ttfb = performance.now() - t0;
      bytes += b.length;
    });
    sock.on("end", () => resolve({ ttfb, full: performance.now() - t0, bytes }));
    sock.on("error", reject);
  });
}

async function measureTtfb(port, path, n) {
  await ttfbOnce(port, path).catch(() => {}); // warmup
  const ttfbs = [];
  const fulls = [];
  let bytes = 0;
  for (let i = 0; i < n; i++) {
    const r = await ttfbOnce(port, path);
    ttfbs.push(r.ttfb);
    fulls.push(r.full);
    bytes = r.bytes;
  }
  ttfbs.sort((a, b) => a - b);
  fulls.sort((a, b) => a - b);
  return {
    ttfb: ttfbs[Math.floor(ttfbs.length / 2)],
    full: fulls[Math.floor(fulls.length / 2)],
    bytes,
  };
}

async function benchServer(label, buildCmd, startCmd, port) {
  console.log(`\n================ ${label} ================`);
  // BUILD
  rmSync(`${appDir}/.diffpack-output`, { recursive: true, force: true });
  rmSync(`${appDir}/.next`, { recursive: true, force: true });
  const t0 = Date.now();
  execFileSync(buildCmd.cmd, buildCmd.args, { cwd: appDir, stdio: "ignore", env: { ...process.env } });
  const buildMs = Date.now() - t0;
  console.log(`  build: ${buildMs} ms`);
  // START
  const proc = spawn(startCmd.cmd, startCmd.args, { cwd: appDir, stdio: "ignore", env: { ...process.env } });
  try {
    await waitReady(port, 60000);
    await sleep(500);
    // LATENCY (under some load) + MEMORY sampling
    let peakRssKb = 0;
    const memTimer = setInterval(() => {
      peakRssKb = Math.max(peakRssKb, treeRssKb(proc.pid));
    }, 100);
    const latency = await measureLatency(port, REQUESTS);
    // Streaming TTFB on the slow route (best-effort: only if the route exists).
    let ttfb = null;
    try {
      const probe = await fetch(`http://127.0.0.1:${port}${TTFB_PATH}`, { signal: AbortSignal.timeout(5000) });
      await probe.text();
      if (probe.status < 400) ttfb = await measureTtfb(port, TTFB_PATH, 15);
    } catch {}
    // a burst of concurrency to catch peak worker memory
    await Promise.all(Array.from({ length: 20 }, () => fetch(`http://127.0.0.1:${port}/`).then((r) => r.text())));
    await sleep(300);
    clearInterval(memTimer);
    peakRssKb = Math.max(peakRssKb, treeRssKb(proc.pid));
    const footprintMb = treeFootprintMb(proc.pid);
    console.log(`  latency: median=${latency.median.toFixed(1)}ms p95=${latency.p95.toFixed(1)}ms min=${latency.min.toFixed(1)}ms`);
    if (ttfb) {
      console.log(`  streaming ${TTFB_PATH}: TTFB=${ttfb.ttfb.toFixed(1)}ms  full=${ttfb.full.toFixed(1)}ms  (shell arrives ${(ttfb.full / ttfb.ttfb).toFixed(1)}x sooner than the full response)`);
    } else {
      console.log(`  streaming ${TTFB_PATH}: (route not present on this server — skipped)`);
    }
    console.log(`  server memory: ${footprintMb.toFixed(0)} MB phys_footprint (fair) | ${(peakRssKb / 1024).toFixed(0)} MB summed RSS (over-counts shared pages)`);
    return { label, buildMs, latency, ttfb, rssMb: peakRssKb / 1024, footprintMb };
  } finally {
    try { proc.kill("SIGTERM"); } catch {}
    await sleep(300);
    try { proc.kill("SIGKILL"); } catch {}
    execSync("pkill -9 -f 'next-server|server.mjs serve|next start|next-server-prod' 2>/dev/null || true");
  }
}

const results = [];
if (!only || only === "diffpack") {
  results.push(
    await benchServer(
      "diffpack (build production + start)",
      { cmd: DIFFPACK, args: ["build-app", ".", "production"] },
      { cmd: DIFFPACK, args: ["start", ".diffpack-output", "8850"] },
      8850,
    ),
  );
}
if (!only || only === "next") {
  const nextBin = `${appDir}/node_modules/.bin/next`;
  results.push(
    await benchServer(
      "next (build + start)",
      { cmd: nextBin, args: ["build"] },
      { cmd: nextBin, args: ["start", "-p", "8851"] },
      8851,
    ),
  );
}

if (results.length === 2) {
  const [d, n] = results;
  console.log("\n=============== PRODUCTION: diffpack vs next ===============");
  console.log(`  build time:    diffpack ${d.buildMs}ms   vs   next ${n.buildMs}ms   (${(n.buildMs / d.buildMs).toFixed(1)}x faster)`);
  console.log(`  req latency:   diffpack ${d.latency.median.toFixed(1)}ms  vs  next ${n.latency.median.toFixed(1)}ms  (${(n.latency.median / d.latency.median).toFixed(1)}x)`);
  if (d.ttfb && n.ttfb) {
    console.log(`  stream TTFB:   diffpack ${d.ttfb.ttfb.toFixed(1)}ms  vs  next ${n.ttfb.ttfb.toFixed(1)}ms  (${(n.ttfb.ttfb / d.ttfb.ttfb).toFixed(1)}x — first byte on the slow route)`);
  }
  console.log(`  memory (fair): diffpack ${d.footprintMb.toFixed(0)}MB  vs  next ${n.footprintMb.toFixed(0)}MB phys_footprint  (${(d.footprintMb / n.footprintMb).toFixed(2)}x — <1 means less)`);
}
