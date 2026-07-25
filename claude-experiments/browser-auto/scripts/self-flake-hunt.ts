/**
 * Self-flake-hunt: bat is a flake-hunting tool, so its OWN timing-sensitive
 * paths must be hunted for flakes. Each self-test in the vitest suite runs
 * once — which cannot detect intermittency. This harness runs the paths that
 * navigate + observe (normal flows, --fast replay, fallback replay, runtime
 * loops, tabs) MANY times and fails if ANY single iteration fails.
 *
 * A 50%-per-invocation flake is invisible to a run-once suite but certain to
 * surface here within a few iterations. Plain repetition is deliberate: some
 * races (e.g. tracker-attach vs the app's first fetch) only appear WITHOUT
 * injected latency, so we run a no-latency pass plus latency-perturbed passes.
 *
 * Usage: npx tsx scripts/self-flake-hunt.ts [iterations]
 */
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { startShopServer } from "../fixtures/shop/server.js";
import { world } from "../fixtures/shop/world.js";
import { replayStep, runFlowFile } from "../src/runner/run.js";
import { localWorldHandle } from "../src/runner/world-handle.js";
import { renderReport, type FlowTrace } from "../src/runner/trace.js";
import { loadSeeds, type BatConfig } from "../src/config.js";
import type { RunDeps } from "../src/runner/run.js";

const BASE = join(dirname(fileURLToPath(import.meta.url)), "..");
const iterations = Number(process.argv[2] ?? 10);
const FLOW = (n: string) => join(BASE, "fixtures/shop/e2e/flows", n);

process.env.BAT_TEST = "1";
const shop = await startShopServer();
const browser = await chromium.launch({ headless: true });
const root = await mkdtemp(join(tmpdir(), "bat-flakehunt-"));
const baseConfig: BatConfig = {
  baseUrl: shop.url,
  world: { module: join(BASE, "fixtures/shop/world.ts") },
  seeds: join(BASE, "fixtures/shop/e2e/world/*.seed.ts"),
  flows: join(BASE, "fixtures/shop/e2e/flows/**/*.flow"),
  stepBudgetMs: Number(process.env.BUDGET ?? 12000),
  headless: true,
  root,
};
const seeds = await loadSeeds(baseConfig);

interface Outcome {
  ok: boolean;
  detail?: string;
}

// each scenario is a timing-sensitive path that navigates + observes
const scenarios: Array<{ name: string; run: (deps: RunDeps) => Promise<Outcome> }> = [
  {
    name: "buy.flow (normal)",
    run: async (deps) => check(await runFlowFile(FLOW("buy.flow"), deps).then((r) => r.trace)),
  },
  // --fast replay of every mid-flow step: restores checkpoint/snapshot then runs one step.
  // This is where a tracker-attach-after-navigation race lives.
  ...[2, 3, 4, 5].map((step) => ({
    name: `buy.flow --fast replay :${step}`,
    run: async (deps: RunDeps) => {
      await runFlowFile(FLOW("buy.flow"), deps); // seed a run with checkpoints
      const r = await replayStep(FLOW("buy.flow"), step, deps, { fast: true });
      return check(r.trace);
    },
  })),
  ...[3, 4].map((step) => ({
    name: `buy.flow fallback replay :${step}`,
    run: async (deps: RunDeps) => check((await replayStep(FLOW("buy.flow"), step, deps, { fast: false })).trace),
  })),
  {
    name: "manage-cart.flow (runtime for-each loop)",
    run: async (deps) => check(await runFlowFile(FLOW("manage-cart.flow"), deps).then((r) => r.trace)),
  },
  {
    name: "search.flow (debounce under fake clock)",
    run: async (deps) => check(await runFlowFile(FLOW("search.flow"), deps).then((r) => r.trace)),
  },
  {
    name: "interactions.flow (tabs/dialog/download/iframe/drag)",
    run: async (deps) => check(await runFlowFile(FLOW("interactions.flow"), deps).then((r) => r.trace)),
  },
];

// timing perturbations: plain (catches tracker-vs-first-fetch races) + latency
const allPerturbations: Array<{ label: string; conditions?: BatConfig["conditions"] }> = [
  { label: "plain" },
  { label: "latency+50-400", conditions: { latencyMs: [50, 400], seed: 1 } },
  { label: "latency+300-900", conditions: { latencyMs: [300, 900], seed: 2 } },
];
// PLAIN_ONLY: the tracker-vs-first-fetch race only appears without latency
const perturbations = process.env.PLAIN_ONLY ? allPerturbations.slice(0, 1) : allPerturbations;

function check(trace: FlowTrace): Outcome {
  if (trace.status === "pass") return { ok: true };
  const failed = trace.steps.find((s) => s.status === "fail");
  return {
    ok: false,
    detail:
      renderReport(trace) +
      (failed?.settle ? `\nsettle: ${JSON.stringify(failed.settle)}` : "") +
      (failed ? `\nnetwork: ${failed.requests.map((r) => `${r.method} ${new URL(r.url).pathname} ${r.status ?? r.failure} stream=${r.streaming}`).join(" | ")}` : ""),
  };
}

let total = 0;
let failures = 0;
const seen = new Set<string>();
for (let i = 0; i < iterations; i++) {
  for (const pert of perturbations) {
    const config = { ...baseConfig, ...(pert.conditions ? { conditions: pert.conditions } : {}), rerunsOnFailure: 0 };
    const deps: RunDeps = { config, world: localWorldHandle(world), seeds, browser };
    for (const sc of scenarios) {
      total++;
      const label = `${sc.name} [${pert.label}]`;
      let outcome: Outcome;
      try {
        outcome = await sc.run(deps);
      } catch (e) {
        outcome = { ok: false, detail: e instanceof Error ? (e.stack ?? e.message) : String(e) };
      }
      if (!outcome.ok) {
        failures++;
        if (!seen.has(label)) {
          seen.add(label);
          console.log(`\n✗ FLAKE: ${label}  (iteration ${i + 1})`);
          console.log(outcome.detail ?? "(no detail)");
        } else {
          console.log(`✗ ${label} (iteration ${i + 1}) — repeat`);
        }
      }
    }
  }
  process.stdout.write(`iter ${i + 1}/${iterations} done (${failures}/${total} failed so far)\n`);
}

console.log(`\n${"=".repeat(60)}`);
console.log(`self-flake-hunt: ${failures} failure(s) across ${total} runs`);
console.log(failures === 0 ? "CLEAN — no flakes detected" : `FLAKY — ${[...seen].length} distinct path(s) flaked`);
await browser.close();
await shop.close();
process.exit(failures ? 1 : 0);
