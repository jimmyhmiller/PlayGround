// Comprehensive diffpack-vs-Vite benchmark on a real Vite app. Vite 8 already bundles
// with Rolldown (Rust), so this compares diffpack against the current Rust-bundled Vite.
// Measures, for each toolchain: production BUILD (wall time + peak build memory + output
// size), PREVIEW serve (memory + request latency), and DEV cold start (time-to-first-byte
// + dev-server memory).
//
//   node scripts/rsc/vite-bench.mjs <app-dir> [--vite-bin path]
//
// A second Vite (e.g. classic-Rollup Vite 7) can be compared by pointing --vite-bin at it.
import { spawn, spawnSync, execSync } from "node:child_process";
import { rmSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const DP = process.env.DIFFPACK || resolve(REPO, "target/release/diffpack");
const appDir = resolve(process.argv[2] || "integration/vite-real");
const arg = (n) => { const i = process.argv.indexOf(n); return i >= 0 ? process.argv[i + 1] : null; };
const viteBin = arg("--vite-bin") || resolve(appDir, "node_modules/.bin/vite");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function treeFootprintMb(rootPid) {
  const out = execSync("ps -Ao pid=,ppid=", { encoding: "utf8" });
  const kids = new Map();
  for (const line of out.split("\n")) {
    const m = line.trim().match(/^(\d+)\s+(\d+)$/);
    if (!m) continue;
    const [pid, ppid] = [Number(m[1]), Number(m[2])];
    if (!kids.has(ppid)) kids.set(ppid, []);
    kids.get(ppid).push(pid);
  }
  const all = []; const st = [rootPid]; const seen = new Set();
  while (st.length) { const p = st.pop(); if (seen.has(p)) continue; seen.add(p); all.push(p); for (const k of kids.get(p) || []) st.push(k); }
  let total = 0;
  for (const pid of all) { try { const o = execSync(`footprint ${pid} 2>/dev/null`, { encoding: "utf8" }); const m = o.match(/phys_footprint:\s*([0-9.]+)\s*([MK])/i); if (m) total += Number(m[1]) / (m[2].toUpperCase() === "K" ? 1024 : 1); } catch {} }
  return Math.round(total);
}
const duMb = (p) => { try { return Number(execSync(`du -sm "${resolve(appDir, p)}"`, { encoding: "utf8" }).split("\t")[0]); } catch { return 0; } };

// Build with `/usr/bin/time -l` -> { seconds, peakMb }. macOS reports max RSS in bytes.
function timedBuild(cmd, args) {
  const r = spawnSync("/usr/bin/time", ["-l", cmd, ...args], { cwd: appDir, encoding: "utf8" });
  const err = (r.stderr || "") + (r.stdout || "");
  const real = err.match(/([0-9.]+)\s+real/);
  const rss = err.match(/(\d+)\s+maximum resident set size/);
  return { seconds: real ? Number(real[1]) : 0, peakMb: rss ? Math.round(Number(rss[1]) / 1048576) : 0, ok: r.status === 0 };
}

async function waitReady(port) {
  for (let i = 0; i < 100; i++) { try { const r = await fetch(`http://127.0.0.1:${port}/`, { signal: AbortSignal.timeout(2000) }); if (r.status < 500) { await r.text(); return true; } } catch {} await sleep(200); }
  return false;
}
async function latency(port, n = 20) {
  const s = [];
  for (let i = 0; i < n; i++) { const t = performance.now(); try { await (await fetch(`http://127.0.0.1:${port}/`)).text(); } catch {} s.push(performance.now() - t); }
  s.sort((a, b) => a - b); return s[Math.floor(n / 2)];
}
function killTree(proc) { try { proc.kill("SIGKILL"); } catch {} }

async function servePhase(label, cmd, args, port) {
  const proc = spawn(cmd, args, { cwd: appDir, stdio: "ignore" });
  const ready = await waitReady(port);
  if (!ready) { killTree(proc); return { mem: 0, lat: 0, ready: false }; }
  await sleep(500);
  const lat = await latency(port);
  const mem = treeFootprintMb(proc.pid);
  killTree(proc); await sleep(300);
  return { mem, lat, ready: true };
}
async function devColdStart(cmd, args, port) {
  const t0 = Date.now();
  const proc = spawn(cmd, args, { cwd: appDir, stdio: "ignore" });
  const ready = await waitReady(port);
  const firstByteMs = Date.now() - t0;
  await sleep(400);
  const mem = ready ? treeFootprintMb(proc.pid) : 0;
  killTree(proc); await sleep(300);
  return { firstByteMs, mem, ready };
}

async function bench(label, cfg) {
  console.log(`\n===== ${label} =====`);
  rmSync(resolve(appDir, cfg.outDir), { recursive: true, force: true });
  const b = timedBuild(cfg.build[0], cfg.build.slice(1));
  const out = duMb(cfg.outDir);
  console.log(`  build:   ${b.seconds.toFixed(2)}s   peak ${b.peakMb} MB   output ${out} MB${b.ok ? "" : "  (BUILD FAILED)"}`);
  const prev = await servePhase("preview", cfg.preview[0], cfg.preview.slice(1), cfg.previewPort);
  console.log(`  preview: serve mem ${prev.mem} MB   req ${prev.lat.toFixed(1)} ms${prev.ready ? "" : "  (serve failed)"}`);
  const dev = await devColdStart(cfg.dev[0], cfg.dev.slice(1), cfg.devPort);
  console.log(`  dev:     cold start ${dev.firstByteMs} ms   dev mem ${dev.mem} MB${dev.ready ? "" : "  (dev failed)"}`);
  execSync("pkill -9 -f 'diffpack (dev|preview)|node_modules/.bin/vite' 2>/dev/null || true");
  return { label, build: b, out, prev, dev };
}

const results = [];
results.push(await bench("diffpack", {
  build: [DP, "build", ".", "--out-dir", "dist-diffpack"], outDir: "dist-diffpack",
  preview: [DP, "preview", "dist-diffpack", "8960"], previewPort: 8960,
  dev: [DP, "dev", ".", "8961"], devPort: 8961,
}));
results.push(await bench("vite (Vite 8 / Rolldown)", {
  build: [viteBin, "build"], outDir: "dist",
  preview: [viteBin, "preview", "--port", "8962", "--strictPort", "--host", "127.0.0.1"], previewPort: 8962,
  dev: [viteBin, "--port", "8963", "--strictPort", "--host", "127.0.0.1"], devPort: 8963,
}));

const [d, v] = results;
console.log(`\n=============== VITE BENCH: diffpack vs vite (${appDir.split("/").pop()}) ===============`);
const x = (a, b) => (a && b ? (b / a).toFixed(1) + "x" : "n/a");
console.log(`  build time:   diffpack ${d.build.seconds.toFixed(2)}s vs vite ${v.build.seconds.toFixed(2)}s   (${x(d.build.seconds, v.build.seconds)} faster)`);
console.log(`  build memory: diffpack ${d.build.peakMb}MB vs vite ${v.build.peakMb}MB   (${(d.build.peakMb / (v.build.peakMb || 1)).toFixed(2)}x)`);
console.log(`  output size:  diffpack ${d.out}MB vs vite ${v.out}MB`);
console.log(`  preview mem:  diffpack ${d.prev.mem}MB vs vite ${v.prev.mem}MB   (${(d.prev.mem / (v.prev.mem || 1)).toFixed(2)}x)`);
console.log(`  dev cold:     diffpack ${d.dev.firstByteMs}ms vs vite ${v.dev.firstByteMs}ms   (${x(d.dev.firstByteMs, v.dev.firstByteMs)} faster)`);
console.log(`  dev memory:   diffpack ${d.dev.mem}MB vs vite ${v.dev.mem}MB   (${(d.dev.mem / (v.dev.mem || 1)).toFixed(2)}x)`);
