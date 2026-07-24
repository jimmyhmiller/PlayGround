/** The timing-independence property against a REAL app (Next.js Learn
 * dashboard): for every seeded condition profile (injected latency ranges),
 * every flow passes. Needs postgres + the app built; starts the app itself
 * if :3000 is not already serving.
 *
 * Usage: npx tsx scripts/realapp-timing-property.ts
 */
import { spawn } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { globFiles, loadConfig, loadSeeds, loadWorldHandle } from "../src/config.js";
import { runFlowFile } from "../src/runner/run.js";
import { renderReport } from "../src/runner/trace.js";

const BASE = join(dirname(fileURLToPath(import.meta.url)), "..");
const APP = join(BASE, "apps", "nextjs-dashboard");

const PROFILES: Array<{ latencyMs: [number, number]; seed: number } | null> = [
  null, // no injected conditions — the baseline
  { latencyMs: [50, 200], seed: 11 },
  { latencyMs: [300, 800], seed: 12 },
  { latencyMs: [900, 1600], seed: 13 },
];

async function appUp(): Promise<boolean> {
  try {
    const res = await fetch("http://localhost:3000/login", { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}

let appProc: ReturnType<typeof spawn> | null = null;
if (!(await appUp())) {
  console.log("starting nextjs-dashboard (next start)…");
  appProc = spawn("npx", ["next", "start"], { cwd: APP, stdio: "ignore" });
  for (let i = 0; i < 60 && !(await appUp()); i++) await new Promise((r) => setTimeout(r, 500));
  if (!(await appUp())) throw new Error("nextjs-dashboard did not come up on :3000 — run `npx next build` in apps/nextjs-dashboard first");
}

const baseConfig = await loadConfig(APP);
const seeds = await loadSeeds(baseConfig);
const world = await loadWorldHandle(baseConfig);
const browser = await chromium.launch({ headless: true });
const flows = await globFiles(APP, baseConfig.flows);

let failures = 0;
for (const profile of PROFILES) {
  const label = profile ? `latency +${profile.latencyMs[0]}–${profile.latencyMs[1]}ms seed ${profile.seed}` : "no injected conditions";
  console.log(`\nprofile: ${label}`);
  const config = {
    ...baseConfig,
    stepBudgetMs: 45000, // physics scales with the profile; semantics don't
    rerunsOnFailure: 0, // a property failure must surface the profile itself
    ...(profile ? { conditions: profile } : {}),
  };
  for (const file of flows) {
    const { trace } = await runFlowFile(file, { config, world, seeds, browser });
    const total = trace.steps.reduce((n, s) => n + s.durationMs, 0);
    if (trace.status === "pass") {
      console.log(`  ✓ ${trace.flow} (${total}ms)`);
    } else {
      failures++;
      console.log(`  ✗ ${trace.flow} — TIMING LEAKED INTO SEMANTICS under ${label}`);
      console.log(renderReport(trace));
    }
  }
}
console.log(`\nproperty ${failures === 0 ? "HOLDS" : "VIOLATED"}: ${failures} failure(s) across ${PROFILES.length} profiles × ${flows.length} flows`);
await browser.close();
appProc?.kill();
process.exit(failures ? 1 : 0);
