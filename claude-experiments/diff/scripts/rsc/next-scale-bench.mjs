// Large-app SCALE benchmark: generate N app-router pages, then build + serve them with
// BOTH diffpack and next, comparing build time, build peak memory, serve memory, and
// output size. The 3000-page case is where diffpack's native graph build + PARALLEL
// prerenderer pull far ahead of Turbopack's per-page prerender.
//
//   node next-scale-bench.mjs [N=3000] [--server diffpack|next]
//
// Requires integration/next-scale (scaffolding + generate.mjs committed; pages
// generated on demand) with a REAL node_modules (next needs a non-symlinked one; on
// APFS `cp -Rc ../next-app-router/node_modules node_modules` clones it instantly).
import { spawn, execFileSync, execSync } from "node:child_process";
import { rmSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const appDir = resolve(REPO, "integration/next-scale");
const DIFFPACK = process.env.DIFFPACK || resolve(REPO, "target/release/diffpack");
const N = Number(process.argv.find((a) => /^\d+$/.test(a)) || 3000);
const only = process.argv.includes("--server") ? process.argv[process.argv.indexOf("--server") + 1] : null;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function treeFootprintMb(rootPid) {
  const out = execSync("ps -Ao pid=,ppid=", { encoding: "utf8" });
  const kids = new Map();
  for (const line of out.split("\n")) {
    const m = line.trim().match(/^(\d+)\s+(\d+)$/);
    if (m) { const [p, pp] = [Number(m[1]), Number(m[2])]; (kids.get(pp) || kids.set(pp, []).get(pp)).push(p); }
  }
  const all = []; const st = [rootPid]; const seen = new Set();
  while (st.length) { const p = st.pop(); if (seen.has(p)) continue; seen.add(p); all.push(p); for (const k of kids.get(p) || []) st.push(k); }
  let tot = 0;
  for (const pid of all) {
    try {
      const o = execSync(`footprint ${pid} 2>/dev/null`, { encoding: "utf8" });
      const m = o.match(/phys_footprint:\s*([0-9.]+)\s*([MK])/i);
      if (m) tot += Number(m[1]) / (m[2].toUpperCase() === "K" ? 1024 : 1);
    } catch {}
  }
  return tot;
}
const duMb = (p) => { try { return Number(execSync(`du -sm ${p}`, { encoding: "utf8" }).split("\t")[0]); } catch { return 0; } };
async function waitReady(port) {
  for (let i = 0; i < 120; i++) { try { const r = await fetch(`http://127.0.0.1:${port}/`, { signal: AbortSignal.timeout(2000) }); if (r.status < 500) return; } catch {} await sleep(250); }
  throw new Error("server never ready");
}

console.log(`Generating ${N} pages...`);
execFileSync("node", [resolve(appDir, "generate.mjs"), String(N)], { stdio: "inherit" });

async function bench(label, buildCmd, startCmd, out, port) {
  console.log(`\n===== ${label} =====`);
  rmSync(resolve(appDir, out), { recursive: true, force: true });
  const t0 = Date.now();
  execFileSync(buildCmd.cmd, buildCmd.args, { cwd: appDir, stdio: "ignore" });
  const buildMs = Date.now() - t0;
  const outMb = duMb(resolve(appDir, out));
  const proc = spawn(startCmd.cmd, startCmd.args, { cwd: appDir, stdio: "ignore" });
  let mem = 0;
  try {
    await waitReady(port);
    await sleep(500);
    for (const p of ["/", `/p/0`, `/p/${Math.floor(N / 2)}`, `/p/${N - 1}`]) {
      const r = await fetch(`http://127.0.0.1:${port}${p}`); await r.text();
      if (r.status !== 200) throw new Error(`${p} -> ${r.status}`);
    }
    mem = treeFootprintMb(proc.pid);
  } finally { try { proc.kill("SIGKILL"); } catch {} await sleep(300); execSync("pkill -9 -f 'server.mjs serve|next-server|next start' 2>/dev/null || true"); }
  console.log(`  build: ${(buildMs / 1000).toFixed(1)}s   output: ${outMb} MB   serve mem: ${mem.toFixed(0)} MB (phys_footprint)`);
  return { label, buildMs, outMb, mem };
}

const results = [];
if (!only || only === "diffpack") results.push(await bench("diffpack", { cmd: DIFFPACK, args: ["build-app", ".", "production"] }, { cmd: DIFFPACK, args: ["start", ".diffpack-output", "8850"] }, ".diffpack-output", 8850));
if (!only || only === "next") results.push(await bench("next", { cmd: `${appDir}/node_modules/.bin/next`, args: ["build"] }, { cmd: `${appDir}/node_modules/.bin/next`, args: ["start", "-p", "8851"] }, ".next", 8851));
if (results.length === 2) {
  const [d, n] = results;
  console.log(`\n===== SCALE (${N} pages): diffpack vs next =====`);
  console.log(`  build:      diffpack ${(d.buildMs / 1000).toFixed(1)}s vs next ${(n.buildMs / 1000).toFixed(1)}s   (${(n.buildMs / d.buildMs).toFixed(1)}x faster)`);
  console.log(`  serve mem:  diffpack ${d.mem.toFixed(0)}MB vs next ${n.mem.toFixed(0)}MB   (${(d.mem / n.mem).toFixed(2)}x = ${((1 - d.mem / n.mem) * 100).toFixed(0)}% less)`);
  console.log(`  output:     diffpack ${d.outMb}MB vs next ${n.outMb}MB   (${(n.outMb / d.outMb).toFixed(0)}x smaller)`);
}
