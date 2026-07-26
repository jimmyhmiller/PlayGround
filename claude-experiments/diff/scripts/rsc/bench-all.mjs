// The permanent diffpack-vs-Next benchmark suite. Runs every benchmark in sequence with
// MEMORY reported throughout, so one command produces the full head-to-head picture:
//
//   1. PROD  — the full-featured app-router fixture (integration/next-app-router):
//              build time, request latency, streaming TTFB, and server memory.
//   2. SCALE — a generated 3000-page app (integration/next-scale): build time,
//              serve memory, and on-disk output size.
//   3. DEV   — the edit flow (integration/next-app-router): cold startup, edit-to-update
//              HMR latency for a client-component and a server-component edit, and
//              dev-server memory.
//
// Each sub-benchmark builds + serves the SAME app under BOTH `diffpack` and `next` and
// prints its own comparison; diffpack must stay faster AND lighter on every axis.
//
//   node scripts/rsc/bench-all.mjs                     # all three
//   node scripts/rsc/bench-all.mjs --only prod,scale   # a subset
//   node scripts/rsc/bench-all.mjs --pages 5000        # scale page count
//
// Native build (Rust); node + Chrome (agent-browser) are the oracles. Requires a
// release binary (cargo build --release) and the fixtures' node_modules.
import { execFileSync } from "node:child_process";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const argv = process.argv.slice(2);
const arg = (name) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : null; };
const only = (arg("--only") || "prod,scale,dev").split(",").map((s) => s.trim());
const pages = arg("--pages") || "3000";
const want = (k) => only.includes(k);

function run(label, args) {
  console.log(`\n\n############################## ${label} ##############################`);
  try {
    execFileSync("node", args, { stdio: "inherit", cwd: REPO });
  } catch (e) {
    console.error(`\n[${label}] benchmark exited non-zero: ${e.message}`);
  }
}

if (want("prod")) run("PROD (full-featured app)", ["scripts/rsc/next-prod-bench.mjs", "integration/next-app-router"]);
if (want("scale")) run(`SCALE (${pages} pages)`, ["scripts/rsc/next-scale-bench.mjs", pages]);
if (want("dev")) run("DEV (edit flow / HMR)", ["scripts/bench-dev-hmr.mjs"]);

console.log("\n\n===== benchmark suite complete (each section above compares diffpack vs next, memory included) =====");
