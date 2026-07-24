import { mkdir, mkdtemp } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, type Browser } from "playwright";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { setShopTiming, startShopServer, type ShopTiming } from "../fixtures/shop/server.js";
import { world } from "../fixtures/shop/world.js";
import { naiveBuyJourney } from "./naive-runner.js";
import { runFlowFile } from "./runner/run.js";
import { localWorldHandle } from "./runner/world-handle.js";
import { renderReport } from "./runner/trace.js";
import { loadSeeds, type BatConfig } from "./config.js";
import { applyWorld } from "./world/adapter.js";
import { composeWorld } from "./world/algebra.js";
import type { RunDeps } from "./runner/run.js";

const FIXTURE = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures", "shop");
const REPO_RUNS = join(dirname(fileURLToPath(import.meta.url)), "..", ".bat-test-runs");
const BUY_FLOW = join(FIXTURE, "e2e/flows/buy.flow");

/**
 * THE timing-independence property. Flakiness is, by definition, an outcome
 * that depends on timing the test did not mean to encode. So:
 *
 *   ∀ timing profiles P (server latency ranges, toast lifetimes):
 *     bat(buy.flow, P) = pass
 *
 * while the comparison arm — the same journey written as raw Playwright with
 * explicit fixed tolerances — has an outcome that is a function of P.
 */

// deliberately spans two orders of magnitude, including sub-toast-lifetime
// latencies and latencies far beyond the naive runner's 1.5s tolerances
const PROFILES: ShopTiming[] = [
  { apiLatencyMs: [0, 5], toastMs: 60, seed: 101 },
  { apiLatencyMs: [5, 40], toastMs: 400, seed: 102 },
  { apiLatencyMs: [80, 250], toastMs: 150, seed: 103 },
  { apiLatencyMs: [200, 600], toastMs: 90, seed: 104 },
  { apiLatencyMs: [500, 1000], toastMs: 250, seed: 105 },
  { apiLatencyMs: [900, 1400], toastMs: 60, seed: 106 },
  { apiLatencyMs: [1800, 2300], toastMs: 150, seed: 107 },
  { apiLatencyMs: [2500, 3000], toastMs: 100, seed: 108 },
];
const FAST = PROFILES.slice(0, 3);
const BEYOND_TOLERANCE = PROFILES.slice(6); // > the naive runner's 1.5s waits

let browser: Browser;
let shop: Awaited<ReturnType<typeof startShopServer>>;
let deps: RunDeps;

beforeAll(async () => {
  process.env.BAT_TEST = "1";
  shop = await startShopServer();
  browser = await chromium.launch({ headless: true });
  await mkdir(REPO_RUNS, { recursive: true });
  const root = await mkdtemp(join(REPO_RUNS, "bat-timing-"));
  const config: BatConfig = {
    baseUrl: shop.url,
    world: { module: join(FIXTURE, "world.ts") },
    seeds: join(FIXTURE, "e2e/world/*.seed.ts"),
    flows: join(FIXTURE, "e2e/flows/**/*.flow"),
    stepBudgetMs: 30000, // physics scales with the profile; semantics don't
    headless: true,
    rerunsOnFailure: 0, // a property failure must surface the profile, not rerun it
    root,
  };
  const seeds = await loadSeeds(config);
  deps = { config, world: localWorldHandle(world), seeds, browser };
}, 60000);

afterAll(async () => {
  setShopTiming({ apiLatencyMs: [50, 400], toastMs: 150, seed: 0 });
  await browser?.close();
  await shop?.close();
});

describe("timing-independence property", () => {
  it("bat passes buy.flow under EVERY timing profile", async () => {
    for (const profile of PROFILES) {
      setShopTiming(profile);
      const { trace } = await runFlowFile(BUY_FLOW, deps);
      if (trace.status !== "pass") {
        throw new Error(
          `bat failed under profile ${JSON.stringify(profile)} — timing leaked into semantics:\n${renderReport(trace)}`,
        );
      }
    }
  }, 300000);

  it("the fixed-tolerance runner's outcome is a function of the timing profile", async () => {
    // seed the world once for the naive arm (it has no world machinery)
    const seedMod = (await import("../fixtures/shop/e2e/world/catalog.seed.js")) as Record<string, unknown>;
    const catalog = (seedMod.default ?? seedMod) as never;

    for (const profile of FAST) {
      setShopTiming(profile);
      await applyWorld(world, composeWorld([catalog]));
      const result = await naiveBuyJourney(browser, shop.url);
      expect(result.ok, `naive should pass fast profile ${JSON.stringify(profile)}`).toBe(true);
    }
    for (const profile of BEYOND_TOLERANCE) {
      setShopTiming(profile);
      await applyWorld(world, composeWorld([catalog]));
      const result = await naiveBuyJourney(browser, shop.url);
      expect(result.ok, `naive should fail beyond-tolerance profile ${JSON.stringify(profile)}`).toBe(false);
    }
    // same app, same journey, same correctness — outcome flipped by timing
    // alone. That flip IS flakiness; bat's arm above has no such dependence.
  }, 300000);
});
