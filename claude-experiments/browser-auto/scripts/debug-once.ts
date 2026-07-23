/** Run one flow against the fixture and print the full report (failures are
 * auto-diagnosed: test-fault vs app-fault, with rerun evidence).
 * Usage: npx tsx scripts/debug-once.ts <flow-file>
 */
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { startShopServer } from "../fixtures/shop/server.js";
import { world } from "../fixtures/shop/world.js";
import { runFlowFile } from "../src/runner/run.js";
import { localWorldHandle } from "../src/runner/world-handle.js";
import { loadSeeds, type BatConfig } from "../src/config.js";
import { renderReport } from "../src/runner/trace.js";

const BASE = join(dirname(fileURLToPath(import.meta.url)), "..");
const flowFile = process.argv[2]!;

process.env.BAT_TEST = "1";
const shop = await startShopServer();
const browser = await chromium.launch({ headless: true });
const root = await mkdtemp(join(tmpdir(), "bat-debug-"));
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

const { trace } = await runFlowFile(join(BASE, flowFile), deps);
console.log(renderReport(trace));
await browser.close();
await shop.close();
process.exit(0);
