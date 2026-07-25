import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { startShopServer } from "../fixtures/shop/server.js";
import { world } from "../fixtures/shop/world.js";
import { runFlowFile, replayStep } from "../src/runner/run.js";
import { localWorldHandle } from "../src/runner/world-handle.js";
import { renderReport } from "../src/runner/trace.js";
import { loadSeeds, type BatConfig } from "../src/config.js";

const BASE = join(dirname(fileURLToPath(import.meta.url)), "..");
process.env.BAT_TEST = "1";
const shop = await startShopServer();
const browser = await chromium.launch({ headless: true });
const root = await mkdtemp(join(tmpdir(), "bat-repro-"));
const config: BatConfig = {
  baseUrl: shop.url,
  world: { module: join(BASE, "fixtures/shop/world.ts") },
  seeds: join(BASE, "fixtures/shop/e2e/world/*.seed.ts"),
  flows: join(BASE, "fixtures/shop/e2e/flows/**/*.flow"),
  stepBudgetMs: 10000,
  headless: true,
  root,
};
const seeds = await loadSeeds(config);
const deps = { config, world: localWorldHandle(world), seeds, browser };
const BUY = join(BASE, "fixtures/shop/e2e/flows/buy.flow");

let fails = 0;
for (let i = 0; i < 12; i++) {
  await runFlowFile(BUY, deps);
  const result = await replayStep(BUY, 4, deps, { fast: true });
  if (result.trace.status !== "pass") {
    fails++;
    console.log(`\n=== iter ${i} FAIL (tier: ${result.tier}) ===`);
    console.log(renderReport(result.trace));
    const step4 = result.trace.steps[3]!;
    console.log("settle:", JSON.stringify(step4.settle));
    console.log("requests:", step4.requests.map((r) => `${r.method} ${new URL(r.url).pathname} ${r.status ?? r.failure} fin#${r.finishSeq} stream=${r.streaming}`).join(" | "));
    console.log("console:", JSON.stringify(step4.consoleErrors));
  } else {
    console.log(`iter ${i}: pass`);
  }
}
console.log(`\nfails: ${fails}/12`);
await browser.close();
await shop.close();
process.exit(0);
