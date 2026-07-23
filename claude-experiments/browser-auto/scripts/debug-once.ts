/** Run one flow (or a hunt) against the fixture and print the full report.
 * Usage: npx tsx scripts/debug-once.ts <flow-file> [hunt-runs]
 */
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { startShopServer } from "../fixtures/shop/server.js";
import { world } from "../fixtures/shop/world.js";
import { runFlowFile } from "../src/runner/run.js";
import { huntFlow } from "../src/runner/hunt.js";
import { localWorldHandle } from "../src/runner/world-handle.js";
import { loadSeeds, type BatConfig } from "../src/config.js";
import { renderReport } from "../src/runner/trace.js";

const BASE = join(dirname(fileURLToPath(import.meta.url)), "..");
const flowFile = process.argv[2]!;
const huntRuns = process.argv[3] ? Number(process.argv[3]) : 0;

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

if (huntRuns > 0) {
  const report = await huntFlow(join(BASE, flowFile), deps, { runs: huntRuns });
  console.log(report.reportText);
} else {
  const { trace } = await runFlowFile(join(BASE, flowFile), deps);
  console.log(renderReport(trace));
}
await browser.close();
await shop.close();
process.exit(0);
